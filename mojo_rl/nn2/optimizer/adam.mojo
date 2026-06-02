"""Adam optimizer.

`make[target, M]` does not take algorithm-specific hyperparams;
they live as public mut fields on the optimizer struct:

    var opt = Adam.make[target="cpu", M=MyModel](model)
    opt.lr = Scalar[DT](3e-4)   # poke a public field after construction
    opt.step["cpu", M=MyModel](model)

The mut-field pattern lets external schedules (cosine LR, SAC alpha
annealing) update hyperparams per-step without rebuilding the optimizer.

Internals:
  * flat `m_flat`/`v_flat` Lists (CPU) or DeviceBuffers (GPU); offsets
    table maps each param's walk-order index to its start.
  * Visitors reach into Adam's storage via raw pointers (sidesteps
    Mojo's field-destroyed-mid-life rejection).
  * GPU kernel `_adam_update_kernel` — standard fp32 Adam math.
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import ParamVisitor
from ..core.grad_clip import (
    clip_grads_auto,
    clip_grads_auto_gpu,
    GradClipState,
)
from ..core.module import Module
from ..core.optimizer import Optimizer
from ..core.saveable import Saveable
from ..core.save_scalar import _expect_kv_line
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (lifted from v1).
# ──────────────────────────────────────────────────────────────────────


def _adam_step_prep_kernel(
    step_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    # Single-thread on-device step bump + bias-correction update. Keeps the
    # Adam step counter off the host so the GPU step is CUDA-graph capturable
    # (no host scalar baked into the update kernel's args). Mirrors AdamW's
    # `_adamw_step_prep_kernel`. `bc_ptr` layout: [beta1_pow_t, beta2_pow_t,
    # bias_correction1, bias_correction2].
    if Int(global_idx.x) == 0:
        step_ptr[0] = step_ptr[0] + UInt32(1)
        var b1_new = bc_ptr[0] * beta1
        var b2_new = bc_ptr[1] * beta2
        bc_ptr[0] = b1_new
        bc_ptr[1] = b2_new
        bc_ptr[2] = Scalar[DT](1.0) - b1_new
        bc_ptr[3] = Scalar[DT](1.0) - b2_new


def _adam_update_kernel(
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
):
    var i = Int(global_idx.x)
    if i < n_elems:
        var one: Scalar[DT] = 1.0
        var bc1 = bc_ptr[2]
        var bc2 = bc_ptr[3]
        var g = grad[i]
        var m_new = beta1 * m_off[i] + (one - beta1) * g
        var v_new = beta2 * v_off[i] + (one - beta2) * g * g
        m_off[i] = m_new
        v_off[i] = v_new
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        param[i] = param[i] - lr * m_hat / (sqrt(v_hat) + eps)


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
struct _AdamCPUInitVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]

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
        for _ in range(n_elems):
            self.m_flat_ptr[].append(zero)
            self.v_flat_ptr[].append(zero)


@fieldwise_init
struct _AdamCPUStepVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var bias_correction1: Scalar[DT]
    var bias_correction2: Scalar[DT]

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
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        var m_ptr = self.m_flat_ptr[].unsafe_ptr() + off
        var v_ptr = self.v_flat_ptr[].unsafe_ptr() + off
        var b1_v = SIMD[DT, CPU_SIMD_W](self.beta1)
        var b2_v = SIMD[DT, CPU_SIMD_W](self.beta2)
        var omb1_v = SIMD[DT, CPU_SIMD_W](1.0) - b1_v
        var omb2_v = SIMD[DT, CPU_SIMD_W](1.0) - b2_v
        var bc1_v = SIMD[DT, CPU_SIMD_W](self.bias_correction1)
        var bc2_v = SIMD[DT, CPU_SIMD_W](self.bias_correction2)
        var lr_v = SIMD[DT, CPU_SIMD_W](self.lr)
        var eps_v = SIMD[DT, CPU_SIMD_W](self.eps)
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
            p_ptr.store(i, p - lr_v * m_hat / (sqrt(v_hat) + eps_v))
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
            var m_hat = m_new / self.bias_correction1
            var v_hat = v_new / self.bias_correction2
            p_ptr[i] = p_ptr[i] - self.lr * m_hat / (sqrt(v_hat) + self.eps)
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
struct _AdamGPUInitVisitor(ParamVisitor):
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
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
        self.total_ptr[] = self.total_ptr[] + n_elems


@fieldwise_init
struct _AdamGPUStepVisitor(ParamVisitor):
    var ctx: DeviceContext
    var m_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var v_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var bc_base: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var idx: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]

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
        var m_off = self.m_base + off
        var v_off = self.v_base + off
        var param_w_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var grad_w_ptr  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_adam_update_kernel](
            param_w_ptr, grad_w_ptr, m_off, v_off, self.bc_base, n_elems,
            self.lr, self.beta1, self.beta2, self.eps,
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
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad_w_ptr, n_elems, grid_dim=n_blocks, block_dim=TPB,
        )


@fieldwise_init
struct _AdamGPUSlabCollector(ParamVisitor):
    """Walks the model's params WITHOUT launching any kernel — it records the
    param/grad slab base pointers and verifies the params (and grads) are laid
    out contiguously in walk order (`param[idx].ptr == base + Σ n_elems`).

    nn2 models keep their params in one contiguous slab (stateless layers are
    `LayoutTensor` views into it) and the Adam moment slab `m_flat`/`v_flat` is
    built with the SAME walk-order offsets — so when contiguity holds, the
    whole Adam update (and `zero_grad`) is a single element-wise pass over
    `[0, total_size)` instead of one kernel per param tensor. `_adam_update_kernel`
    applies plain Adam with NO weight-decay term, so a slab-wide launch is
    bit-identical to the per-tensor launches (same elements, same math).

    `contiguous` stays True only if every tensor abuts the previous one; any
    gap (e.g. a model whose params are NOT one slab) flips it False and the
    caller falls back to the per-tensor visitor — so this is safe for any model.
    """

    var idx: Int
    var running: Int  # cumulative n_elems seen so far (walk-order offset)
    var param_base: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var grad_base: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var contiguous: Bool

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
        var p_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        if self.idx == 0:
            self.param_base = p_ptr
            self.grad_base = g_ptr
        else:
            if p_ptr != self.param_base.value() + self.running:
                self.contiguous = False
            if g_ptr != self.grad_base.value() + self.running:
                self.contiguous = False
        self.running += n_elems
        self.idx += 1


# ──────────────────────────────────────────────────────────────────────
# Adam.
# ──────────────────────────────────────────────────────────────────────


struct Adam(Optimizer, Saveable):
    # CPU storage
    var m_flat: List[Scalar[DT]]
    var v_flat: List[Scalar[DT]]
    # GPU storage
    var m_dev: Optional[DeviceBuffer[DT]]
    var v_dev: Optional[DeviceBuffer[DT]]
    # On-device step + bias-correction state (GPU path only). Keeps the step
    # counter off the host so the GPU `step` is CUDA-graph capturable. `bc_dev`
    # layout: [beta1_pow_t, beta2_pow_t, bias_correction1, bias_correction2].
    var step_dev: Optional[DeviceBuffer[DType.uint32]]
    var bc_dev: Optional[DeviceBuffer[DT]]
    # Shared bookkeeping
    var offsets: List[Int]
    var total_size: Int
    # Host-side step state (CPU path; GPU uses step_dev/bc_dev instead).
    var step_count: Int

    # Public mut hyperparams. Defaults match Kingma & Ba 2014.
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]

    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]

    # Phase B.3 — global L2 grad-norm clip threshold. `0.0` (default)
    # means disabled — `Adam.step` skips the clip pipeline entirely,
    # preserving bit-identity with pre-B.3 behaviour. Set on the
    # optimizer instance after `make`, typically wired from the trainer's
    # Config struct (`SACConfig.max_grad_norm` etc.).
    #
    # GPU path: lazy-allocates `_clip_state` (per-Param partials buffer +
    # scale scalar + norm scalar) on the first GPU `step` with
    # `max_grad_norm > 0`. Three on-device passes, zero D2H.
    # See `mojo_rl/nn2/core/grad_clip.mojo`.
    var max_grad_norm: Scalar[DT]

    # Lazy-allocated GPU clip state (None on CPU or before first clipped
    # GPU step). `n_params` is exactly `len(self.offsets)` — known after
    # the init walker has run.
    var _clip_state: Optional[GradClipState]

    var ts: TargetStorage

    def __init__(out self):
        self.m_flat = List[Scalar[DT]]()
        self.v_flat = List[Scalar[DT]]()
        self.m_dev = None
        self.v_dev = None
        self.step_dev = None
        self.bc_dev = None
        self.offsets = List[Int]()
        self.total_size = 0
        self.step_count = 0
        self.lr = Scalar[DT](0.001)
        self.beta1 = Scalar[DT](0.9)
        self.beta2 = Scalar[DT](0.999)
        self.eps = Scalar[DT](1e-8)
        self.beta1_pow_t = Scalar[DT](1.0)
        self.beta2_pow_t = Scalar[DT](1.0)
        self.max_grad_norm = Scalar[DT](0.0)
        self._clip_state = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory — no hyperparams. User sets `opt.lr`
        etc. after. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Adam: target must be 'cpu' or 'gpu'"
        )
        var opt = Self()
        comptime if target == "cpu":
            var visitor = _AdamCPUInitVisitor(
                m_flat_ptr=UnsafePointer(to=opt.m_flat),
                v_flat_ptr=UnsafePointer(to=opt.v_flat),
                offsets_ptr=UnsafePointer(to=opt.offsets),
            )
            model.for_each_param[target, _AdamCPUInitVisitor](
                String(""), visitor,
            )
            opt.total_size = len(opt.m_flat)
            opt.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Adam.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            var visitor = _AdamGPUInitVisitor(
                offsets_ptr=UnsafePointer(to=opt.offsets),
                total_ptr=UnsafePointer(to=opt.total_size),
            )
            model.for_each_param[target, _AdamGPUInitVisitor](
                String(""), visitor,
            )
            var m_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            var v_real = ctx_v.enqueue_create_buffer[DT](opt.total_size)
            m_real.enqueue_fill(0.0)
            v_real.enqueue_fill(0.0)
            opt.m_dev = m_real^
            opt.v_dev = v_real^
            # On-device step counter (0) + bias-correction buffer
            # [beta1_pow_t=1, beta2_pow_t=1, bc1=0, bc2=0].
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
        return opt^

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        assert_tag_for["Adam", target](self.ts.target_tag)
        comptime if target == "cpu":
            var v = _ZeroGradCPUVisitor()
            model.for_each_param[target, _ZeroGradCPUVisitor](String(""), v)
        else:
            var ctx = self.ts.ctx.value()
            # Batched path: one `_zero_fill_kernel` over the whole grad slab
            # when params are contiguous (the common nn2 case). Falls back to
            # the per-tensor visitor otherwise. Bit-identical either way.
            var collector = _AdamGPUSlabCollector(
                idx=0, running=0, param_base=None, grad_base=None,
                contiguous=True,
            )
            model.for_each_param[target, _AdamGPUSlabCollector](
                String(""), collector
            )
            if (
                collector.contiguous
                and collector.running == self.total_size
                and self.total_size > 0
            ):
                var n_blocks = (self.total_size + TPB - 1) // TPB
                ctx.enqueue_function[_zero_fill_kernel](
                    collector.grad_base.value(),
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
        assert_tag_for["Adam", target](self.ts.target_tag)

        # Phase B.3 — global grad-norm clip. No-op when `max_grad_norm == 0`
        # (the default sentinel) → bit-identical to pre-B.3 behaviour.
        comptime if target == "cpu":
            _ = clip_grads_auto[M, target](model, self.max_grad_norm)
        else:
            if self.max_grad_norm > Scalar[DT](0.0):
                # Lazy-allocate clip state on first clipped GPU step.
                # `len(self.offsets)` is the Param count after the init
                # walker ran in `make[target='gpu']`.
                if not self._clip_state:
                    self._clip_state = GradClipState.make(
                        self.ts.ctx.value(), len(self.offsets),
                    )
                clip_grads_auto_gpu[M](
                    model,
                    self.ts.ctx.value(),
                    self._clip_state.value(),
                    self.max_grad_norm,
                )

        comptime if target == "cpu":
            # Host step + bias-correction (CPU path only).
            self.step_count += 1
            self.beta1_pow_t = self.beta1_pow_t * self.beta1
            self.beta2_pow_t = self.beta2_pow_t * self.beta2
            var bc1: Scalar[DT] = Scalar[DT](1.0) - self.beta1_pow_t
            var bc2: Scalar[DT] = Scalar[DT](1.0) - self.beta2_pow_t
            var visitor = _AdamCPUStepVisitor(
                m_flat_ptr=UnsafePointer(to=self.m_flat),
                v_flat_ptr=UnsafePointer(to=self.v_flat),
                offsets_ptr=UnsafePointer(to=self.offsets),
                idx=0,
                lr=self.lr, beta1=self.beta1, beta2=self.beta2, eps=self.eps,
                bias_correction1=bc1, bias_correction2=bc2,
            )
            model.for_each_param[target, _AdamCPUStepVisitor](String(""), visitor)
        else:
            var ctx = self.ts.ctx.value()
            var step_ptr: UnsafePointer[UInt32, MutAnyOrigin] = (
                self.step_dev.value().unsafe_ptr()
            )
            var bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = (
                self.bc_dev.value().unsafe_ptr()
            )
            # On-device step bump + bias-correction. No host scalar feeds the
            # update kernel → CUDA-graph capturable.
            ctx.enqueue_function[_adam_step_prep_kernel](
                step_ptr, bc_ptr, self.beta1, self.beta2,
                grid_dim=1, block_dim=1,
            )
            # Batched path: one `_adam_update_kernel` over the whole param/grad/
            # moment slab when params are contiguous (the common nn2 case),
            # replacing one kernel PER param tensor. The moment slab is built
            # with the same walk-order offsets, so element i of the param slab
            # pairs with element i of `m_dev`/`v_dev` — identical to the
            # per-tensor launches (plain Adam, no per-tensor weight decay).
            # Falls back to the per-tensor visitor if a model isn't contiguous.
            var collector = _AdamGPUSlabCollector(
                idx=0, running=0, param_base=None, grad_base=None,
                contiguous=True,
            )
            model.for_each_param[target, _AdamGPUSlabCollector](
                String(""), collector
            )
            if (
                collector.contiguous
                and collector.running == self.total_size
                and self.total_size > 0
            ):
                var n_blocks = (self.total_size + TPB - 1) // TPB
                ctx.enqueue_function[_adam_update_kernel](
                    collector.param_base.value(),
                    collector.grad_base.value(),
                    self.m_dev.value().unsafe_ptr(),
                    self.v_dev.value().unsafe_ptr(),
                    bc_ptr,
                    self.total_size,
                    self.lr, self.beta1, self.beta2, self.eps,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                var visitor = _AdamGPUStepVisitor(
                    ctx=ctx,
                    m_base=self.m_dev.value().unsafe_ptr(),
                    v_base=self.v_dev.value().unsafe_ptr(),
                    bc_base=bc_ptr,
                    offsets_ptr=UnsafePointer(to=self.offsets),
                    idx=0,
                    lr=self.lr, beta1=self.beta1, beta2=self.beta2,
                    eps=self.eps,
                )
                model.for_each_param[target, _AdamGPUStepVisitor](
                    String(""), visitor
                )

    # ─────────────────────────── Saveable (CPU only) ───────────────────────────
    # Saved fields:
    #   <prefix>.lr=<float>
    #   <prefix>.beta1=<float>
    #   <prefix>.beta2=<float>
    #   <prefix>.eps=<float>
    #   <prefix>.step_count=<int>
    #   <prefix>.beta1_pow_t=<float>
    #   <prefix>.beta2_pow_t=<float>
    #   <prefix>.m_flat#size=<N>
    #   v0
    #   ...
    #   <prefix>.v_flat#size=<N>
    #   v0
    #   ...
    # Topology-derived state (offsets, total_size) is NOT saved; the
    # caller must call `Adam.make[target, M](model)` BEFORE `load` so
    # the in-memory Adam has been sized to the model. `load` validates
    # the saved m_flat/v_flat size matches in-memory `total_size`.
    # GPU mirrors (m_dev/v_dev) are not saved either; trainer must
    # re-upload after load.

    def save(self, mut out: String, prefix: String) raises:
        out += prefix + ".lr=" + String(self.lr) + "\n"
        out += prefix + ".beta1=" + String(self.beta1) + "\n"
        out += prefix + ".beta2=" + String(self.beta2) + "\n"
        out += prefix + ".eps=" + String(self.eps) + "\n"
        out += prefix + ".step_count=" + String(self.step_count) + "\n"
        out += prefix + ".beta1_pow_t=" + String(self.beta1_pow_t) + "\n"
        out += prefix + ".beta2_pow_t=" + String(self.beta2_pow_t) + "\n"
        out += prefix + ".m_flat#size=" + String(self.total_size) + "\n"
        var m_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.m_flat.unsafe_ptr()
        )
        for k in range(self.total_size):
            out += String(m_ptr[k]) + "\n"
        out += prefix + ".v_flat#size=" + String(self.total_size) + "\n"
        var v_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.v_flat.unsafe_ptr()
        )
        for k in range(self.total_size):
            out += String(v_ptr[k]) + "\n"

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        self.lr = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".lr")))
        idx += 1
        self.beta1 = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".beta1")))
        idx += 1
        self.beta2 = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".beta2")))
        idx += 1
        self.eps = Scalar[DT](atof(_expect_kv_line(lines, idx, prefix + ".eps")))
        idx += 1
        self.step_count = atol(_expect_kv_line(lines, idx, prefix + ".step_count"))
        idx += 1
        self.beta1_pow_t = Scalar[DT](atof(
            _expect_kv_line(lines, idx, prefix + ".beta1_pow_t")
        ))
        idx += 1
        self.beta2_pow_t = Scalar[DT](atof(
            _expect_kv_line(lines, idx, prefix + ".beta2_pow_t")
        ))
        idx += 1
        Adam._load_flat_section(
            lines, idx, prefix + ".m_flat", self.m_flat, self.total_size,
        )
        Adam._load_flat_section(
            lines, idx, prefix + ".v_flat", self.v_flat, self.total_size,
        )

    # ─── GPU checkpoint sync (Phase 2 — GPU checkpointing) ───────────────
    # On the GPU path the host fields (`m_flat`/`v_flat`/`step_count`/
    # `beta*_pow_t`) are NOT maintained — the live state lives in the device
    # buffers (`m_dev`/`v_dev`/`step_dev`/`bc_dev`). These bridge device ↔
    # host so the EXISTING CPU `save`/`load` serializer runs unchanged on a
    # GPU optimizer, giving a byte-identical (interchangeable) checkpoint:
    # GPU-saved files load on a CPU trainer and vice-versa.

    def sync_to_host(mut self) raises:
        """D2H device buffers → host fields. Call before `save` on GPU.
        Sizes `m_flat`/`v_flat` if empty, copies moments, the step counter
        (`step_dev`), and the bias-correction powers (`bc_dev[0:2]` =
        β₁ᵗ/β₂ᵗ)."""
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
        """H2D host fields → device buffers. Call after `load` on GPU.
        Restores moments + the step counter, and recomputes `bc_dev` =
        [β₁ᵗ, β₂ᵗ, 1−β₁ᵗ, 1−β₂ᵗ] from the loaded `beta*_pow_t` (matching
        `_adam_step_prep_kernel`'s layout)."""
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
                "Adam.load: out of input at `" + expected_prefix
                + "#size=...` (idx " + String(idx) + ")"
            )
        var header = lines[idx]
        var expected_header = (
            expected_prefix + "#size=" + String(expected_size)
        )
        if header != expected_header:
            raise Error(
                "Adam.load: section header mismatch. Expected `"
                + expected_header + "`, got `" + header + "`"
            )
        idx += 1
        var t_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            target.unsafe_ptr()
        )
        for k in range(expected_size):
            if idx >= len(lines):
                raise Error(
                    "Adam.load: short read for `" + expected_prefix
                    + "` at element " + String(k) + " of "
                    + String(expected_size)
                )
            t_ptr[k] = Scalar[DT](atof(lines[idx]))
            idx += 1
