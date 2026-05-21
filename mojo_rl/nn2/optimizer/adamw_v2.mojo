"""AdamWV2 — retrofit (Phase D).

Decoupled weight-decay variant of Adam (Loshchilov & Hutter 2019).
Same math + kernels as v1; the trait surface drops algorithm-specific
hyperparams from `make`. `lr` / `beta1` / `beta2` / `eps` /
`weight_decay` are public mut fields.

Per-param `apply_decay` flag is collected at init time (visitor reads it
from each Param's `decay` bit) and stored in a parallel `apply_decay:
List[Bool]` table — Linear says weight=True, bias=False, LayerNorm says
γ=False, β=False.

GPU step state (`step_dev`, `bc_dev`) lives on device — CUDA-graph
capture friendly. CPU path keeps host-side scalars.
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import ParamVisitor
from ..core.module_v2 import ModuleV2
from ..core.optimizer_v2 import OptimizerV2
from ..core.target_storage import TargetStorage, assert_tag_for


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
# CPU visitors.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _AdamWCPUInitVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]

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
struct _AdamWCPUStepVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]
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
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
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
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        var zero: Scalar[DT] = 0.0
        for i in range(n_elems):
            g_ptr[i] = zero


# ──────────────────────────────────────────────────────────────────────
# GPU visitors.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _AdamWGPUInitVisitor(ParamVisitor):
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]
    var total_ptr: UnsafePointer[Int, MutAnyOrigin]

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
struct _AdamWGPUStepVisitor(ParamVisitor):
    var ctx: DeviceContext
    var m_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var v_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var bc_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]
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
        var param_w_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var grad_w_ptr  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_adamw_update_kernel](
            param_w_ptr, grad_w_ptr, m_off, v_off, self.bc_base, n_elems,
            self.lr, self.beta1, self.beta2, self.eps,
            self.weight_decay,
            Int(1) if decay_flag else Int(0),
            grid_dim=n_blocks, block_dim=TPB,
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
        var grad_w_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad_w_ptr, n_elems, grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────
# AdamWV2.
# ──────────────────────────────────────────────────────────────────────


struct AdamWV2(OptimizerV2):
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
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, M: ModuleV2](mut model: M) raises -> Self:
        comptime assert target == "cpu", (
            "AdamWV2.make[target='gpu', M] requires a DeviceContext"
        )
        var opt = Self()
        var visitor = _AdamWCPUInitVisitor(
            m_flat_ptr=UnsafePointer(to=opt.m_flat),
            v_flat_ptr=UnsafePointer(to=opt.v_flat),
            offsets_ptr=UnsafePointer(to=opt.offsets),
            apply_decay_ptr=UnsafePointer(to=opt.apply_decay),
        )
        model.for_each_param[target, _AdamWCPUInitVisitor](String(""), visitor)
        opt.total_size = len(opt.m_flat)
        opt.ts = TargetStorage.make_cpu()
        return opt^

    @staticmethod
    def make[target: StaticString, M: ModuleV2](
        mut model: M, ctx: DeviceContext,
    ) raises -> Self:
        comptime assert target == "gpu", (
            "AdamWV2.make[target='cpu', M](model, ctx) — drop ctx for CPU"
        )
        var opt = Self()
        var visitor = _AdamWGPUInitVisitor(
            offsets_ptr=UnsafePointer(to=opt.offsets),
            apply_decay_ptr=UnsafePointer(to=opt.apply_decay),
            total_ptr=UnsafePointer(to=opt.total_size),
        )
        model.for_each_param[target, _AdamWGPUInitVisitor](String(""), visitor)
        var m_real = ctx.enqueue_create_buffer[DT](opt.total_size)
        var v_real = ctx.enqueue_create_buffer[DT](opt.total_size)
        m_real.enqueue_fill(0.0)
        v_real.enqueue_fill(0.0)
        opt.m_dev = m_real^
        opt.v_dev = v_real^
        var step_real = ctx.enqueue_create_buffer[DType.uint32](1)
        step_real.enqueue_fill(0)
        var bc_real = ctx.enqueue_create_buffer[DT](4)
        var bc_init_host = ctx.enqueue_create_host_buffer[DT](4)
        ctx.synchronize()
        bc_init_host.unsafe_ptr()[0] = Scalar[DT](1.0)
        bc_init_host.unsafe_ptr()[1] = Scalar[DT](1.0)
        bc_init_host.unsafe_ptr()[2] = Scalar[DT](0.0)
        bc_init_host.unsafe_ptr()[3] = Scalar[DT](0.0)
        ctx.enqueue_copy(bc_real, bc_init_host)
        opt.step_dev = step_real^
        opt.bc_dev = bc_real^
        opt.ts = TargetStorage.make_gpu(ctx)
        return opt^

    def zero_grad[
        target: StaticString,
        M: ModuleV2,
    ](mut self, mut model: M) raises:
        assert_tag_for["AdamWV2", target](self.ts.target_tag)
        comptime if target == "cpu":
            var v = _ZeroGradCPUVisitor()
            model.for_each_param[target, _ZeroGradCPUVisitor](String(""), v)
        else:
            var v = _ZeroGradGPUVisitor(ctx=self.ts.ctx.value())
            model.for_each_param[target, _ZeroGradGPUVisitor](String(""), v)

    def step[
        target: StaticString,
        M: ModuleV2,
    ](mut self, mut model: M) raises:
        assert_tag_for["AdamWV2", target](self.ts.target_tag)
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
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                weight_decay=self.weight_decay,
                bc1=bc1, bc2=bc2,
            )
            model.for_each_param[target, _AdamWCPUStepVisitor](String(""), visitor)
        else:
            var ctx = self.ts.ctx.value()
            var step_ptr: UnsafePointer[UInt32, MutAnyOrigin] = (
                self.step_dev.value().unsafe_ptr()
            )
            var bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = (
                self.bc_dev.value().unsafe_ptr()
            )
            ctx.enqueue_function[_adamw_step_prep_kernel](
                step_ptr, bc_ptr, self.beta1, self.beta2,
                grid_dim=1, block_dim=1,
            )
            var visitor = _AdamWGPUStepVisitor(
                ctx=ctx,
                m_base=self.m_dev.value().unsafe_ptr(),
                v_base=self.v_dev.value().unsafe_ptr(),
                bc_base=bc_ptr,
                offsets_ptr=UnsafePointer(to=self.offsets),
                apply_decay_ptr=UnsafePointer(to=self.apply_decay),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                weight_decay=self.weight_decay,
            )
            model.for_each_param[target, _AdamWGPUStepVisitor](String(""), visitor)
