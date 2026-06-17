"""AdamW optimizer.

Decoupled weight-decay variant of Adam (Loshchilov & Hutter 2019).
`lr` / `beta1` / `beta2` / `eps` / `weight_decay` are public mut fields,
not `make`-time args.

Per-param `apply_decay` flag is collected at init time (visitor reads it
from each Param's `decay` bit) and stored in a parallel `apply_decay:
List[Bool]` table — Linear says weight=True, bias=False, LayerNorm says
γ=False, β=False.

GPU step state (`step_dev`, `bc_dev`) lives on device — CUDA-graph
capture friendly. CPU path keeps host-side scalars.
"""

from std.math import sqrt
from std.sys import has_nvidia_gpu_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB, USE_GROUPED_GPU_OPTIMIZER
from ..core import ParamVisitor

# Grouped multi-tensor helpers shared with Adam (NVIDIA-only path). The
# descriptor collector + flat-index helper + uploaders are algorithm-
# agnostic; only the per-element update kernel differs (AdamW folds in the
# decoupled weight-decay term), so it lives here.
from .adam import (
    _find_param,
    _grouped_zero_kernel,
    _upload_u64,
    _upload_i32,
    _MTDescriptorCollector,
)
from ..core.module import Module, mptr
from ..core.optimizer import Optimizer
from ..core.grad_clip import (
    clip_grads_auto,
    clip_grads_auto_gpu,
    clip_grads_grouped_gpu,
    GradClipState,
)
from ..core.saveable import Saveable
from ..core.save_scalar import _expect_kv_line
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (lifted from v1).
# ──────────────────────────────────────────────────────────────────────


def _adamw_step_prep_kernel(
    step_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    if Int(global_idx.x) == 0:
        step_ptr[0] = step_ptr[0] + UInt32(1)
        var b1_new = bc_ptr[0] * beta1
        var b2_new = bc_ptr[1] * beta2
        bc_ptr[0] = b1_new
        bc_ptr[1] = b2_new
        bc_ptr[2] = Scalar[DT](1.0) - b1_new
        bc_ptr[3] = Scalar[DT](1.0) - b2_new


def _adamw_update_kernel(
    param: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    m_off: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_off: UnsafePointer[Scalar[DT], MutAnyOrigin],
    bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_elems: Int,
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    weight_decay: Scalar[DT],
    apply_decay: Int,
):
    var i = Int(global_idx.x)
    if i < n_elems:
        var one: Scalar[DT] = 1.0
        var bc1 = bc_ptr[2]
        var bc2 = bc_ptr[3]
        var p = param[i]
        var g = grad[i]
        var m_new = beta1 * m_off[i] + (one - beta1) * g
        var v_new = beta2 * v_off[i] + (one - beta2) * g * g
        m_off[i] = m_new
        v_off[i] = v_new
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        var update = lr * m_hat / (sqrt(v_hat) + eps)
        if apply_decay != 0:
            update = update + lr * weight_decay * p
        param[i] = p - update


def _zero_fill_kernel(
    ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_elems: Int,
):
    var i = Int(global_idx.x)
    if i < n_elems:
        ptr[i] = 0.0


# ──────────────────────────────────────────────────────────────────────
# Multi-tensor ("grouped") GPU update kernel — NVIDIA only.
#
# AdamW counterpart of `adam._grouped_adam_update_kernel`: ONE launch
# updates every param tensor via a 1-D grid over the dense moment slab.
# Each flat element maps back to its param `p` (and local offset) via
# `_find_param` over the dense prefix-sum offset table, then dereferences
# the host-captured device address `param_addrs[p]` / `grad_addrs[p]`. The
# ONLY difference vs the Adam grouped kernel is the decoupled weight-decay
# term, gated per-param by `apply_decay_arr[p]` (Linear weight=1, bias=0,
# LayerNorm/BN γ/β=0). Math is identical to the per-tensor
# `_adamw_update_kernel`, so the grouped path is bit-identical to it on
# the same inputs.
#
# NVIDIA-only: dereferencing a host-captured device address in-kernel is
# valid on CUDA but silently dropped on Apple Metal — the caller gates
# this under `has_nvidia_gpu_accelerator()` and Apple keeps the per-tensor
# `_AdamWGPUStepVisitor` path.
# ──────────────────────────────────────────────────────────────────────


def _grouped_adamw_update_kernel(
    param_addrs: UnsafePointer[UInt64, MutAnyOrigin],
    grad_addrs: UnsafePointer[UInt64, MutAnyOrigin],
    moment_offs: UnsafePointer[Int32, MutAnyOrigin],
    apply_decay_arr: UnsafePointer[Int32, MutAnyOrigin],
    n_params: Int,
    total: Int,
    m_base: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_base: UnsafePointer[Scalar[DT], MutAnyOrigin],
    bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    weight_decay: Scalar[DT],
):
    var flat = Int(global_idx.x)
    if flat < total:
        var p = _find_param(moment_offs, n_params, flat)
        var local = flat - Int(moment_offs[p])
        var param = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(param_addrs[p])
        )
        var grad = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(grad_addrs[p])
        )
        var one: Scalar[DT] = 1.0
        var bc1 = bc_ptr[2]
        var bc2 = bc_ptr[3]
        var g = grad[local]
        var m_new = beta1 * m_base[flat] + (one - beta1) * g
        var v_new = beta2 * v_base[flat] + (one - beta2) * g * g
        m_base[flat] = m_new
        v_base[flat] = v_new
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        var pv = param[local]
        var update = lr * m_hat / (sqrt(v_hat) + eps)
        if Int(apply_decay_arr[p]) != 0:
            update = update + lr * weight_decay * pv
        param[local] = pv - update


# ──────────────────────────────────────────────────────────────────────
# CPU visitors.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _AdamWCPUInitVisitor[
    om: Origin[mut=True],
    ov: Origin[mut=True],
    oo: Origin[mut=True],
    oa: Origin[mut=True],
](ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], Self.om]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], Self.ov]
    var offsets_ptr: UnsafePointer[List[Int], Self.oo]
    var apply_decay_ptr: UnsafePointer[List[Bool], Self.oa]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var zero: Scalar[DT] = 0.0
        self.offsets_ptr[].append(len(self.m_flat_ptr[]))
        self.apply_decay_ptr[].append(apply_decay)
        for _ in range(n_elems):
            self.m_flat_ptr[].append(zero)
            self.v_flat_ptr[].append(zero)


@fieldwise_init
struct _AdamWCPUStepVisitor[
    om: Origin[mut=True],
    ov: Origin[mut=True],
    oo: Origin[mut=True],
    oa: Origin[mut=True],
](ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], Self.om]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], Self.ov]
    var offsets_ptr: UnsafePointer[List[Int], Self.oo]
    var apply_decay_ptr: UnsafePointer[List[Bool], Self.oa]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var weight_decay: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var decay_flag = self.apply_decay_ptr[][self.idx]
        var p_ptr = mptr(param.ptr)
        var g_ptr = mptr(grad.ptr)
        var m_ptr = self.m_flat_ptr[].unsafe_ptr() + off
        var v_ptr = self.v_flat_ptr[].unsafe_ptr() + off
        var b1_v = SIMD[DT, CPU_SIMD_W](self.beta1)
        var b2_v = SIMD[DT, CPU_SIMD_W](self.beta2)
        var omb1_v = SIMD[DT, CPU_SIMD_W](1.0) - b1_v
        var omb2_v = SIMD[DT, CPU_SIMD_W](1.0) - b2_v
        var bc1_v = SIMD[DT, CPU_SIMD_W](self.bc1)
        var bc2_v = SIMD[DT, CPU_SIMD_W](self.bc2)
        var lr_v = SIMD[DT, CPU_SIMD_W](self.lr)
        var eps_v = SIMD[DT, CPU_SIMD_W](self.eps)
        var wd_scalar: Scalar[DT] = self.weight_decay if decay_flag else 0.0
        var wd_v = SIMD[DT, CPU_SIMD_W](wd_scalar)
        var i = 0
        while i + CPU_SIMD_W <= n_elems:
            var g = g_ptr.load[width=CPU_SIMD_W](i)
            var m_old = m_ptr.load[width=CPU_SIMD_W](i)
            var v_old = v_ptr.load[width=CPU_SIMD_W](i)
            var m_new = b1_v * m_old + omb1_v * g
            var v_new = b2_v * v_old + omb2_v * g * g
            m_ptr.store(i, m_new)
            v_ptr.store(i, v_new)
            var m_hat = m_new / bc1_v
            var v_hat = v_new / bc2_v
            var p = p_ptr.load[width=CPU_SIMD_W](i)
            var update = lr_v * (m_hat / (sqrt(v_hat) + eps_v) + wd_v * p)
            p_ptr.store(i, p - update)
            i += CPU_SIMD_W
        var one: Scalar[DT] = 1.0
        while i < n_elems:
            var g = g_ptr[i]
            var m_old = m_ptr[i]
            var v_old = v_ptr[i]
            var m_new = self.beta1 * m_old + (one - self.beta1) * g
            var v_new = self.beta2 * v_old + (one - self.beta2) * g * g
            m_ptr[i] = m_new
            v_ptr[i] = v_new
            var m_hat = m_new / self.bc1
            var v_hat = v_new / self.bc2
            var update = self.lr * m_hat / (sqrt(v_hat) + self.eps)
            if decay_flag:
                update = update + self.lr * self.weight_decay * p_ptr[i]
            p_ptr[i] = p_ptr[i] - update
            i += 1
        self.idx += 1


struct _ZeroGradCPUVisitor(ParamVisitor):
    def __init__(out self):
        pass

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var g_ptr = mptr(grad.ptr)
        var zero: Scalar[DT] = 0.0
        for i in range(n_elems):
            g_ptr[i] = zero


# ──────────────────────────────────────────────────────────────────────
# GPU visitors.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _AdamWGPUInitVisitor[
    oo: Origin[mut=True], oa: Origin[mut=True], ot: Origin[mut=True]
](ParamVisitor):
    var offsets_ptr: UnsafePointer[List[Int], Self.oo]
    var apply_decay_ptr: UnsafePointer[List[Bool], Self.oa]
    var total_ptr: UnsafePointer[Int, Self.ot]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.offsets_ptr[].append(self.total_ptr[])
        self.apply_decay_ptr[].append(apply_decay)
        self.total_ptr[] = self.total_ptr[] + n_elems


@fieldwise_init
struct _AdamWGPUStepVisitor[oo: Origin[mut=True], oa: Origin[mut=True]](
    ParamVisitor
):
    var ctx: DeviceContext
    var m_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var v_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var bc_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], Self.oo]
    var apply_decay_ptr: UnsafePointer[List[Bool], Self.oa]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var weight_decay: Scalar[DT]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var decay_flag = self.apply_decay_ptr[][self.idx]
        var m_off = self.m_base + off
        var v_off = self.v_base + off
        var param_w_ptr = mptr(param.ptr)
        var grad_w_ptr = mptr(grad.ptr)
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_adamw_update_kernel](
            param_w_ptr,
            grad_w_ptr,
            m_off,
            v_off,
            self.bc_base,
            n_elems,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            Int(1) if decay_flag else Int(0),
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        self.idx += 1


@fieldwise_init
struct _ZeroGradGPUVisitor(ParamVisitor):
    var ctx: DeviceContext

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var grad_w_ptr = mptr(grad.ptr)
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad_w_ptr,
            n_elems,
            grid_dim=n_blocks,
            block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────
# AdamW.
# ──────────────────────────────────────────────────────────────────────


struct AdamW(Optimizer, Saveable):
    # CPU storage
    var m_flat: List[Scalar[DT]]
    var v_flat: List[Scalar[DT]]
    # GPU storage
    var m_dev: Optional[DeviceBuffer[DT]]
    var v_dev: Optional[DeviceBuffer[DT]]
    var step_dev: Optional[DeviceBuffer[DType.uint32]]
    var bc_dev: Optional[DeviceBuffer[DT]]
    # Shared
    var offsets: List[Int]
    var apply_decay: List[Bool]
    var total_size: Int
    # Host-side step state (CPU path).
    var step_count: Int
    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]
    # Public mut hyperparameters.
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var weight_decay: Scalar[DT]

    # Phase B.3 — global L2 grad-norm clip threshold. `0.0` (default) means
    # disabled — `AdamW.step` skips the clip pipeline entirely, preserving
    # bit-identity with pre-B.3 behaviour. Set on the optimizer instance
    # after `make` (e.g. `optim.max_grad_norm = 1.0`, nanoGPT's default).
    # GPU path lazy-allocates `_clip_state` on the first clipped GPU step.
    # Mirrors `Adam`. See `mojo_rl/nn/core/grad_clip.mojo`.
    var max_grad_norm: Scalar[DT]
    # Lazy-allocated GPU clip state (None on CPU or before first clipped
    # GPU step). `n_params` is exactly `len(self.offsets)`.
    var _clip_state: Optional[GradClipState]

    # Multi-tensor ("grouped") GPU optimizer descriptors. Built ONCE in
    # `make[target='gpu']` on NVIDIA only (Apple Metal can't deref
    # host-captured device addresses in-kernel → keeps the per-tensor
    # path). The grouped update / zero kernels read these device-resident
    # arrays to update every param tensor in ONE launch. `_mt_apply_decay`
    # is the per-param 1/0 decay flag (AdamW-specific). None / 0 on
    # CPU + Apple.
    var _mt_param_addrs: Optional[DeviceBuffer[DType.uint64]]
    var _mt_grad_addrs: Optional[DeviceBuffer[DType.uint64]]
    var _mt_moment_offs: Optional[DeviceBuffer[DType.int32]]
    var _mt_apply_decay: Optional[DeviceBuffer[DType.int32]]
    var _mt_n_params: Int

    var ts: TargetStorage

    def __init__(out self):
        self.m_flat = List[Scalar[DT]]()
        self.v_flat = List[Scalar[DT]]()
        self.m_dev = None
        self.v_dev = None
        self.step_dev = None
        self.bc_dev = None
        self.offsets = List[Int]()
        self.apply_decay = List[Bool]()
        self.total_size = 0
        self.step_count = 0
        self.beta1_pow_t = Scalar[DT](1.0)
        self.beta2_pow_t = Scalar[DT](1.0)
        self.lr = Scalar[DT](0.001)
        self.beta1 = Scalar[DT](0.9)
        self.beta2 = Scalar[DT](0.999)
        self.eps = Scalar[DT](1e-8)
        self.weight_decay = Scalar[DT](0.01)
        self.max_grad_norm = Scalar[DT](0.0)
        self._clip_state = None
        self._mt_param_addrs = None
        self._mt_grad_addrs = None
        self._mt_moment_offs = None
        self._mt_apply_decay = None
        self._mt_n_params = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, M: Module
    ](mut model: M, ctx: Optional[DeviceContext] = None,) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "AdamW: target must be 'cpu' or 'gpu'"
        var opt = Self()
        comptime if target == "cpu":
            var visitor = _AdamWCPUInitVisitor(
                m_flat_ptr=UnsafePointer(to=opt.m_flat),
                v_flat_ptr=UnsafePointer(to=opt.v_flat),
                offsets_ptr=UnsafePointer(to=opt.offsets),
                apply_decay_ptr=UnsafePointer(to=opt.apply_decay),
            )
            model.for_each_param[target, type_of(visitor)](
                String(""),
                visitor,
            )
            opt.total_size = len(opt.m_flat)
            opt.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["AdamW.make[target='gpu']"](ctx)
            var visitor = _AdamWGPUInitVisitor(
                offsets_ptr=UnsafePointer(to=opt.offsets),
                apply_decay_ptr=UnsafePointer(to=opt.apply_decay),
                total_ptr=UnsafePointer(to=opt.total_size),
            )
            model.for_each_param[target, type_of(visitor)](
                String(""),
                visitor,
            )
            var m_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            var v_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            m_real.enqueue_fill(0.0)
            v_real.enqueue_fill(0.0)
            opt.m_dev = m_real^
            opt.v_dev = v_real^
            var step_real = ctx_v.enqueue_create_buffer[DType.uint32](1)
            step_real.enqueue_fill(0)
            var bc_real = ctx_v.enqueue_create_buffer[DT](4)
            var bc_init_host = ctx_v.enqueue_create_host_buffer[DT](4)
            ctx_v.synchronize()
            bc_init_host.unsafe_ptr()[0] = Scalar[DT](1.0)
            bc_init_host.unsafe_ptr()[1] = Scalar[DT](1.0)
            bc_init_host.unsafe_ptr()[2] = Scalar[DT](0.0)
            bc_init_host.unsafe_ptr()[3] = Scalar[DT](0.0)
            ctx_v.enqueue_copy(bc_real, bc_init_host)
            opt.step_dev = step_real^
            opt.bc_dev = bc_real^
            opt.ts = TargetStorage.make_gpu(ctx_v)
            # Build the multi-tensor descriptor arrays (NVIDIA only). On
            # Apple these stay None and step/zero_grad take the per-tensor
            # path (Metal can't deref host-captured device addresses
            # in-kernel). Mirrors `Adam.make`, plus the per-param decay
            # flags collected by the init visitor above.
            comptime if has_nvidia_gpu_accelerator() and USE_GROUPED_GPU_OPTIMIZER:
                var pa = List[UInt64]()
                var ga = List[UInt64]()
                var coll = _MTDescriptorCollector(
                    param_addrs_ptr=UnsafePointer(to=pa),
                    grad_addrs_ptr=UnsafePointer(to=ga),
                )
                model.for_each_param[target, type_of(coll)](String(""), coll)
                # Moment offsets = the init walker's dense per-param prefix
                # sums (same walk order), narrowed to int32; apply_decay
                # flags narrowed to 1/0 int32 (same walk order).
                var mo = List[Int32]()
                for i in range(len(opt.offsets)):
                    mo.append(Int32(opt.offsets[i]))
                var ad = List[Int32]()
                for i in range(len(opt.apply_decay)):
                    ad.append(Int32(1) if opt.apply_decay[i] else Int32(0))
                opt._mt_n_params = len(pa)
                if opt._mt_n_params > 0:
                    opt._mt_param_addrs = _upload_u64(ctx_v, pa)
                    opt._mt_grad_addrs = _upload_u64(ctx_v, ga)
                    opt._mt_moment_offs = _upload_i32(ctx_v, mo)
                    opt._mt_apply_decay = _upload_i32(ctx_v, ad)
        return opt^

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def get_lr(self) -> Scalar[DT]:
        return self.lr

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        assert_tag_for["AdamW", target](self.ts.target_tag)
        comptime if target == "cpu":
            var v = _ZeroGradCPUVisitor()
            model.for_each_param[target, _ZeroGradCPUVisitor](String(""), v)
        else:
            var ctx = self.ts.ctx.value()
            comptime if has_nvidia_gpu_accelerator() and USE_GROUPED_GPU_OPTIMIZER:
                # Grouped: one launch zeros every grad tensor (shares Adam's
                # zero kernel). (`_mt_n_params == 0` → no params, no-op.)
                if self._mt_n_params > 0:
                    var n_blocks = (self.total_size + TPB - 1) // TPB
                    ctx.enqueue_function[_grouped_zero_kernel](
                        rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                            self._mt_grad_addrs.value().unsafe_ptr()
                        ),
                        rebind[UnsafePointer[Int32, MutAnyOrigin]](
                            self._mt_moment_offs.value().unsafe_ptr()
                        ),
                        self._mt_n_params,
                        self.total_size,
                        grid_dim=n_blocks,
                        block_dim=TPB,
                    )
            else:
                var v = _ZeroGradGPUVisitor(ctx=ctx)
                model.for_each_param[target, _ZeroGradGPUVisitor](String(""), v)

    def step[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        assert_tag_for["AdamW", target](self.ts.target_tag)

        # Phase B.3 — global grad-norm clip. No-op when `max_grad_norm == 0`
        # (the default sentinel) → bit-identical to pre-clip behaviour.
        comptime if target == "cpu":
            _ = clip_grads_auto[M, target](model, self.max_grad_norm)
        else:
            if self.max_grad_norm > Scalar[DT](0.0):
                comptime if (
                    has_nvidia_gpu_accelerator() and USE_GROUPED_GPU_OPTIMIZER
                ):
                    # Grouped clip: 3 launches total via the multi-tensor
                    # descriptor arrays (flat grid → big params get many
                    # blocks). Lazy-alloc state WITH total_size for the
                    # block-partials scratch.
                    if self._mt_n_params > 0:
                        if not self._clip_state:
                            self._clip_state = GradClipState.make(
                                self.ts.ctx.value(),
                                self._mt_n_params,
                                self.total_size,
                            )
                        clip_grads_grouped_gpu(
                            self.ts.ctx.value(),
                            self._clip_state.value(),
                            rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                                self._mt_grad_addrs.value().unsafe_ptr()
                            ),
                            rebind[UnsafePointer[Int32, MutAnyOrigin]](
                                self._mt_moment_offs.value().unsafe_ptr()
                            ),
                            self._mt_n_params,
                            self.total_size,
                            self.max_grad_norm,
                        )
                else:
                    # Per-Param path (Apple / non-grouped).
                    if not self._clip_state:
                        self._clip_state = GradClipState.make(
                            self.ts.ctx.value(),
                            len(self.offsets),
                        )
                    clip_grads_auto_gpu[M](
                        model,
                        self.ts.ctx.value(),
                        self._clip_state.value(),
                        self.max_grad_norm,
                    )

        comptime if target == "cpu":
            self.step_count += 1
            self.beta1_pow_t = self.beta1_pow_t * self.beta1
            self.beta2_pow_t = self.beta2_pow_t * self.beta2
            var bc1: Scalar[DT] = Scalar[DT](1.0) - self.beta1_pow_t
            var bc2: Scalar[DT] = Scalar[DT](1.0) - self.beta2_pow_t
            var visitor = _AdamWCPUStepVisitor(
                m_flat_ptr=UnsafePointer(to=self.m_flat),
                v_flat_ptr=UnsafePointer(to=self.v_flat),
                offsets_ptr=UnsafePointer(to=self.offsets),
                apply_decay_ptr=UnsafePointer(to=self.apply_decay),
                idx=0,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                bc1=bc1,
                bc2=bc2,
            )
            model.for_each_param[target, type_of(visitor)](String(""), visitor)
        else:
            var ctx = self.ts.ctx.value()
            var step_ptr: UnsafePointer[
                UInt32, MutAnyOrigin
            ] = self.step_dev.value().unsafe_ptr()
            var bc_ptr: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.bc_dev.value().unsafe_ptr()
            ctx.enqueue_function[_adamw_step_prep_kernel](
                step_ptr,
                bc_ptr,
                self.beta1,
                self.beta2,
                grid_dim=1,
                block_dim=1,
            )
            comptime if has_nvidia_gpu_accelerator() and USE_GROUPED_GPU_OPTIMIZER:
                # Grouped: ONE launch updates every param tensor, reading
                # the device-resident descriptor arrays + device bias-
                # correction (bc_ptr). Bit-identical to the per-tensor path
                # (same m_dev/v_dev dense layout, same math + decay term).
                # (`_mt_n_params == 0` → no params, no-op.)
                if self._mt_n_params > 0:
                    var n_blocks = (self.total_size + TPB - 1) // TPB
                    ctx.enqueue_function[_grouped_adamw_update_kernel](
                        rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                            self._mt_param_addrs.value().unsafe_ptr()
                        ),
                        rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                            self._mt_grad_addrs.value().unsafe_ptr()
                        ),
                        rebind[UnsafePointer[Int32, MutAnyOrigin]](
                            self._mt_moment_offs.value().unsafe_ptr()
                        ),
                        rebind[UnsafePointer[Int32, MutAnyOrigin]](
                            self._mt_apply_decay.value().unsafe_ptr()
                        ),
                        self._mt_n_params,
                        self.total_size,
                        mptr(self.m_dev.value().unsafe_ptr()),
                        mptr(self.v_dev.value().unsafe_ptr()),
                        bc_ptr,
                        self.lr,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        self.weight_decay,
                        grid_dim=n_blocks,
                        block_dim=TPB,
                    )
            else:
                var visitor = _AdamWGPUStepVisitor(
                    ctx=ctx,
                    m_base=self.m_dev.value().unsafe_ptr(),
                    v_base=self.v_dev.value().unsafe_ptr(),
                    bc_base=bc_ptr,
                    offsets_ptr=UnsafePointer(to=self.offsets),
                    apply_decay_ptr=UnsafePointer(to=self.apply_decay),
                    idx=0,
                    lr=self.lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
                model.for_each_param[target, type_of(visitor)](
                    String(""), visitor
                )

    # ─────────────────────────── Saveable (CPU only) ───────────────────────────
    # Same envelope as `Adam` plus a `<prefix>.weight_decay=<float>` line.
    # `apply_decay` flags are topology-derived (rebuilt by `make` from the
    # model) and NOT serialized — `make[target, M](model)` MUST run before
    # `load` so the in-memory optimizer is sized and decay-flagged.

    def save(self, mut out: String, prefix: String) raises:
        out += prefix + ".lr=" + String(self.lr) + "\n"
        out += prefix + ".beta1=" + String(self.beta1) + "\n"
        out += prefix + ".beta2=" + String(self.beta2) + "\n"
        out += prefix + ".eps=" + String(self.eps) + "\n"
        out += prefix + ".weight_decay=" + String(self.weight_decay) + "\n"
        out += prefix + ".step_count=" + String(self.step_count) + "\n"
        out += prefix + ".beta1_pow_t=" + String(self.beta1_pow_t) + "\n"
        out += prefix + ".beta2_pow_t=" + String(self.beta2_pow_t) + "\n"
        out += prefix + ".m_flat#size=" + String(self.total_size) + "\n"
        var m_ptr = mptr(self.m_flat.unsafe_ptr())
        for k in range(self.total_size):
            out += String(m_ptr[k]) + "\n"
        out += prefix + ".v_flat#size=" + String(self.total_size) + "\n"
        var v_ptr = mptr(self.v_flat.unsafe_ptr())
        for k in range(self.total_size):
            out += String(v_ptr[k]) + "\n"

    def load(
        mut self,
        lines: List[String],
        mut idx: Int,
        prefix: String,
    ) raises:
        self.lr = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".lr")))
        idx += 1
        self.beta1 = Scalar[DT](
            atof(_expect_kv_line(lines, idx, prefix + ".beta1"))
        )
        idx += 1
        self.beta2 = Scalar[DT](
            atof(_expect_kv_line(lines, idx, prefix + ".beta2"))
        )
        idx += 1
        self.eps = Scalar[DT](
            atof(_expect_kv_line(lines, idx, prefix + ".eps"))
        )
        idx += 1
        self.weight_decay = Scalar[DT](
            atof(_expect_kv_line(lines, idx, prefix + ".weight_decay"))
        )
        idx += 1
        self.step_count = atol(
            _expect_kv_line(lines, idx, prefix + ".step_count")
        )
        idx += 1
        self.beta1_pow_t = Scalar[DT](
            atof(_expect_kv_line(lines, idx, prefix + ".beta1_pow_t"))
        )
        idx += 1
        self.beta2_pow_t = Scalar[DT](
            atof(_expect_kv_line(lines, idx, prefix + ".beta2_pow_t"))
        )
        idx += 1
        AdamW._load_flat_section(
            lines,
            idx,
            prefix + ".m_flat",
            self.m_flat,
            self.total_size,
        )
        AdamW._load_flat_section(
            lines,
            idx,
            prefix + ".v_flat",
            self.v_flat,
            self.total_size,
        )

    # ─── GPU checkpoint sync (mirrors Adam) ──────────────────────────────

    def sync_to_host(mut self) raises:
        """D2H device buffers → host fields. Call before `save` on GPU."""
        var ctx = self.ts.ctx.value()
        if len(self.m_flat) != self.total_size:
            self.m_flat = List[Scalar[DT]](
                length=self.total_size, fill=Scalar[DT](0.0)
            )
            self.v_flat = List[Scalar[DT]](
                length=self.total_size, fill=Scalar[DT](0.0)
            )
        ctx.enqueue_copy(self.m_flat.unsafe_ptr(), self.m_dev.value())
        ctx.enqueue_copy(self.v_flat.unsafe_ptr(), self.v_dev.value())
        var step_host = ctx.enqueue_create_host_buffer[DType.uint32](1)
        ctx.enqueue_copy(step_host, self.step_dev.value())
        var bc_host = ctx.enqueue_create_host_buffer[DT](4)
        ctx.enqueue_copy(bc_host, self.bc_dev.value())
        ctx.synchronize()
        self.step_count = Int(step_host.unsafe_ptr()[0])
        self.beta1_pow_t = bc_host.unsafe_ptr()[0]
        self.beta2_pow_t = bc_host.unsafe_ptr()[1]

    def upload_from_host(mut self) raises:
        """H2D host fields → device buffers. Call after `load` on GPU."""
        var ctx = self.ts.ctx.value()
        ctx.enqueue_copy(self.m_dev.value(), self.m_flat.unsafe_ptr())
        ctx.enqueue_copy(self.v_dev.value(), self.v_flat.unsafe_ptr())
        var step_host = ctx.enqueue_create_host_buffer[DType.uint32](1)
        var bc_host = ctx.enqueue_create_host_buffer[DT](4)
        ctx.synchronize()
        step_host.unsafe_ptr()[0] = UInt32(self.step_count)
        bc_host.unsafe_ptr()[0] = self.beta1_pow_t
        bc_host.unsafe_ptr()[1] = self.beta2_pow_t
        bc_host.unsafe_ptr()[2] = Scalar[DT](1.0) - self.beta1_pow_t
        bc_host.unsafe_ptr()[3] = Scalar[DT](1.0) - self.beta2_pow_t
        ctx.enqueue_copy(self.step_dev.value(), step_host)
        ctx.enqueue_copy(self.bc_dev.value(), bc_host)
        ctx.synchronize()

    @staticmethod
    def _load_flat_section(
        lines: List[String],
        mut idx: Int,
        expected_prefix: String,
        mut target: List[Scalar[DT]],
        expected_size: Int,
    ) raises:
        if idx >= len(lines):
            raise Error(
                "AdamW.load: out of input at `"
                + expected_prefix
                + "#size=...` (idx "
                + String(idx)
                + ")"
            )
        var header = lines[idx]
        var expected_header = expected_prefix + "#size=" + String(expected_size)
        if header != expected_header:
            raise Error(
                "AdamW.load: section header mismatch. Expected `"
                + expected_header
                + "`, got `"
                + header
                + "`"
            )
        idx += 1
        var t_ptr = mptr(target.unsafe_ptr())
        for k in range(expected_size):
            if idx >= len(lines):
                raise Error(
                    "AdamW.load: short read for `"
                    + expected_prefix
                    + "` at element "
                    + String(k)
                    + " of "
                    + String(expected_size)
                )
            t_ptr[k] = Scalar[DT](atof(lines[idx]))
            idx += 1
