"""AdamW — Loshchilov & Hutter 2019, decoupled weight decay.

Differs from Adam in two ways:

  1. **Decoupled weight decay**: the L2 penalty is applied *after* the
     Adam-normalized step, not folded into the gradient:

         p_t = p_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + λ * p_{t-1})

     This is mathematically distinct from Adam-with-L2 and converges
     better in practice (LayerNorm + AdamW is the modern default).

  2. **Per-param `apply_decay`**: weight-decay is skipped for params the
     visitor reports as `apply_decay=False`. Layer-local convention —
     `Linear` reports `weight=True, bias=False`. AdamW reads the flag at
     init time and stores it in a parallel `apply_decay: List[Bool]`
     table indexed by walk order. No name-match filter; convention lives
     in the layer.

Phase 4 also migrates the optimizer state (`step_count`, `beta1_pow_t`,
`beta2_pow_t`, `bc1`, `bc2`) to `DeviceBuffer` so that AdamW.step is
CUDA-graph-capturable end-to-end:

  - `step_dev: DeviceBuffer[UInt32]`  — single u32, step counter
  - `bc_dev:   DeviceBuffer[DT]` (4 floats)  — [β₁^t, β₂^t, bc1, bc2]
  - `_adamw_step_prep_kernel(grid=1, block=1)` bumps the counter +
    recomputes bc1, bc2 in-place each `step()`.
  - The per-param `_adamw_update_kernel` reads bc1/bc2 from `bc_dev`.

CPU mode keeps host-side scalars — CUDA-graph capture doesn't apply
there. Bit-exact parity with the GPU path is achievable because the
update math is identical; only the storage location differs.
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module, ParamVisitor, Optimizer,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────


def _adamw_step_prep_kernel(
    step_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    bc_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    """Single-thread kernel: bump step counter, recompute β₁^t, β₂^t, bc1, bc2.

    Layout of `bc_ptr` (4 floats):
      [0]: β₁^t   running product
      [1]: β₂^t   running product
      [2]: bc1   = 1 - β₁^t
      [3]: bc2   = 1 - β₂^t
    """
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
    apply_decay: Int,   # 0 or 1 — kernels can't take Bool
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


# ──────────────────────────────────────────────────────────────────────────
# CPU visitors
# ──────────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _AdamWCPUInitVisitor(ParamVisitor):
    var m_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var v_flat_ptr: UnsafePointer[List[Scalar[DT]], MutAnyOrigin]
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
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

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var decay_flag = self.apply_decay_ptr[][self.idx]
        var p_ptr = param.ptr
        var g_ptr = grad.ptr
        var one: Scalar[DT] = 1.0
        for i in range(n_elems):
            var g = g_ptr[i]
            var m_old = self.m_flat_ptr[][off + i]
            var v_old = self.v_flat_ptr[][off + i]
            var m_new = self.beta1 * m_old + (one - self.beta1) * g
            var v_new = self.beta2 * v_old + (one - self.beta2) * g * g
            self.m_flat_ptr[][off + i] = m_new
            self.v_flat_ptr[][off + i] = v_new
            var m_hat = m_new / self.bc1
            var v_hat = v_new / self.bc2
            var update = self.lr * m_hat / (sqrt(v_hat) + self.eps)
            if decay_flag:
                update = update + self.lr * self.weight_decay * p_ptr[i]
            p_ptr[i] = p_ptr[i] - update
        self.idx += 1


struct _ZeroGradCPUVisitor(ParamVisitor):
    def __init__(out self):
        pass

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var g_ptr = grad.ptr
        var zero: Scalar[DT] = 0.0
        for i in range(n_elems):
            g_ptr[i] = zero


# ──────────────────────────────────────────────────────────────────────────
# GPU visitors
# ──────────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _AdamWGPUInitVisitor(ParamVisitor):
    var offsets_ptr: UnsafePointer[List[Int], MutAnyOrigin]
    var apply_decay_ptr: UnsafePointer[List[Bool], MutAnyOrigin]
    var total_ptr: UnsafePointer[Int, MutAnyOrigin]

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
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

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var off = self.offsets_ptr[][self.idx]
        var decay_flag = self.apply_decay_ptr[][self.idx]
        var m_off = self.m_base + off
        var v_off = self.v_base + off
        var param_w = rebind[TileTensor[DT, L, MutAnyOrigin]](param)
        var grad_w  = rebind[TileTensor[DT, L, MutAnyOrigin]](grad)
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_adamw_update_kernel](
            param_w.ptr, grad_w.ptr, m_off, v_off, self.bc_base, n_elems,
            self.lr, self.beta1, self.beta2, self.eps,
            self.weight_decay,
            Int(1) if decay_flag else Int(0),
            grid_dim=n_blocks, block_dim=TPB,
        )
        self.idx += 1


@fieldwise_init
struct _ZeroGradGPUVisitor(ParamVisitor):
    var ctx: DeviceContext

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var grad_w = rebind[TileTensor[DT, L, MutAnyOrigin]](grad)
        comptime TPB = 128
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_zero_fill_kernel](
            grad_w.ptr, n_elems, grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────────
# AdamW — method-level target.
# ──────────────────────────────────────────────────────────────────────────


struct AdamW(Optimizer):
    # CPU storage
    var m_flat: List[Scalar[DT]]
    var v_flat: List[Scalar[DT]]
    # GPU storage
    var m_dev: Optional[DeviceBuffer[DT]]
    var v_dev: Optional[DeviceBuffer[DT]]
    # Device-side step state — [β₁^t, β₂^t, bc1, bc2] + step counter.
    var step_dev: Optional[DeviceBuffer[DType.uint32]]
    var bc_dev: Optional[DeviceBuffer[DT]]
    var ctx: Optional[DeviceContext]
    # Shared
    var offsets: List[Int]
    var apply_decay: List[Bool]
    var total_size: Int
    # Host-side step state — used when target=="cpu", else unused (defaults
    # fine to leave as initial values, the GPU path doesn't read them).
    var step_count: Int
    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]
    # Hyperparameters
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var weight_decay: Scalar[DT]
    var _target_tag: Int8

    def __init__(out self):
        self.m_flat = List[Scalar[DT]]()
        self.v_flat = List[Scalar[DT]]()
        self.m_dev = None
        self.v_dev = None
        self.step_dev = None
        self.bc_dev = None
        self.ctx = None
        self.offsets = List[Int]()
        self.apply_decay = List[Bool]()
        self.total_size = 0
        self.step_count = 0
        self.beta1_pow_t = 1.0
        self.beta2_pow_t = 1.0
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 0.01
        self._target_tag = TARGET_UNINIT

    # ------------------------------------------------------------------
    # Factories — `make[target, M]`. Optimizer trait requires the 4-hyper
    # signature, so we expose `weight_decay` via a setter post-make.
    # ------------------------------------------------------------------

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
    ) raises -> Self:
        """CPU factory with default weight_decay=0.01."""
        return Self.make_with_wd[target](
            model, lr, beta1, beta2, eps, weight_decay=0.01,
        )

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        ctx: DeviceContext,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
    ) raises -> Self:
        """GPU factory with default weight_decay=0.01."""
        return Self.make_with_wd[target](
            model, ctx, lr, beta1, beta2, eps, weight_decay=0.01,
        )

    @staticmethod
    def make_with_wd[target: StaticString, M: Module](
        mut model: M,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        weight_decay: Scalar[DT] = 0.01,
    ) raises -> Self:
        """CPU factory with explicit weight_decay."""
        comptime assert target == "cpu", (
            "AdamW.make_with_wd[target='gpu'] requires a DeviceContext"
        )
        var adam = Self()
        adam.lr = lr
        adam.beta1 = beta1
        adam.beta2 = beta2
        adam.eps = eps
        adam.weight_decay = weight_decay
        var visitor = _AdamWCPUInitVisitor(
            m_flat_ptr=UnsafePointer(to=adam.m_flat),
            v_flat_ptr=UnsafePointer(to=adam.v_flat),
            offsets_ptr=UnsafePointer(to=adam.offsets),
            apply_decay_ptr=UnsafePointer(to=adam.apply_decay),
        )
        model.for_each_param[target](String(""), visitor)
        adam.total_size = len(adam.m_flat)
        adam._target_tag = TARGET_CPU
        return adam^

    @staticmethod
    def make_with_wd[target: StaticString, M: Module](
        mut model: M,
        ctx: DeviceContext,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        weight_decay: Scalar[DT] = 0.01,
    ) raises -> Self:
        """GPU factory with explicit weight_decay."""
        comptime assert target == "gpu", (
            "AdamW.make_with_wd[target='cpu'](model, ctx) — drop ctx for CPU"
        )
        var adam = Self()
        adam.lr = lr
        adam.beta1 = beta1
        adam.beta2 = beta2
        adam.eps = eps
        adam.weight_decay = weight_decay
        adam.ctx = ctx
        # Pass 1: compute offsets + total_size + apply_decay.
        var visitor = _AdamWGPUInitVisitor(
            offsets_ptr=UnsafePointer(to=adam.offsets),
            apply_decay_ptr=UnsafePointer(to=adam.apply_decay),
            total_ptr=UnsafePointer(to=adam.total_size),
        )
        model.for_each_param[target](String(""), visitor)
        var m_real = ctx.enqueue_create_buffer[DT](adam.total_size)
        var v_real = ctx.enqueue_create_buffer[DT](adam.total_size)
        m_real.enqueue_fill(0.0)
        v_real.enqueue_fill(0.0)
        adam.m_dev = m_real^
        adam.v_dev = v_real^
        # Device-side step state: [β₁^t, β₂^t, bc1, bc2] starts at [1, 1, 0, 0].
        # After first _adamw_step_prep_kernel: [β₁, β₂, 1-β₁, 1-β₂].
        var step_real = ctx.enqueue_create_buffer[DType.uint32](1)
        step_real.enqueue_fill(0)
        var bc_real = ctx.enqueue_create_buffer[DT](4)
        var bc_init_host = ctx.enqueue_create_host_buffer[DT](4)
        ctx.synchronize()
        bc_init_host.unsafe_ptr()[0] = 1.0
        bc_init_host.unsafe_ptr()[1] = 1.0
        bc_init_host.unsafe_ptr()[2] = 0.0
        bc_init_host.unsafe_ptr()[3] = 0.0
        ctx.enqueue_copy(bc_real, bc_init_host)
        adam.step_dev = step_real^
        adam.bc_dev = bc_real^
        adam._target_tag = TARGET_GPU
        return adam^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "AdamW: method called with [target='" + String(target)
                + "'] but optimizer was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    # ------------------------------------------------------------------
    # zero_grad — clears all grads via for_each_param.
    # ------------------------------------------------------------------

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        self._assert_tag[target]()
        comptime if target == "cpu":
            var v = _ZeroGradCPUVisitor()
            model.for_each_param[target](String(""), v)
        else:
            var v = _ZeroGradGPUVisitor(ctx=self.ctx.value())
            model.for_each_param[target](String(""), v)

    # ------------------------------------------------------------------
    # step — bump (β₁^t, β₂^t, bc1, bc2) then per-param update.
    # ------------------------------------------------------------------

    def step[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        self._assert_tag[target]()
        comptime if target == "cpu":
            self.step_count += 1
            self.beta1_pow_t = self.beta1_pow_t * self.beta1
            self.beta2_pow_t = self.beta2_pow_t * self.beta2
            var bc1: Scalar[DT] = 1.0 - self.beta1_pow_t
            var bc2: Scalar[DT] = 1.0 - self.beta2_pow_t
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
            model.for_each_param[target](String(""), visitor)
        else:
            var ctx = self.ctx.value()
            # Launder pointers through MutAnyOrigin (different fields, different
            # buffers — but Mojo's analyzer can't see that from `self.*`).
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
            model.for_each_param[target](String(""), visitor)
