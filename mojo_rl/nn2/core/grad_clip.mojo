"""Global grad-norm clipping over a Module — Phase B.3 (CPU + GPU).

Two-pass walker. Pre-clip norm is returned for diagnostics.

CPU path:
  1. `_GradSumSqVisitorCPU` accumulates ‖grad‖² across every IsParam-typed
     field via the standard `model.for_each_param[target, V]` dispatch.
  2. If √(sum) > max_norm, `_GradScaleVisitorCPU` scales each grad in
     place by max_norm / norm. Otherwise (or when max_norm ≤ 0) — no-op.

GPU path (no D2H during training step):

  1. `_GradSumSqVisitorGPU` launches `_sum_sq_partial_kernel` per Param,
     writing each per-Param `Σ g·g` into a pre-allocated `partials`
     device buffer (one slot per Param). Tree reduction inside a single
     block via `block.sum` — same primitive LayerNorm uses (no atomics,
     works on Apple M1+ and NVIDIA).
  2. `_compute_scale_kernel` (single-thread) sums `partials`, takes
     `sqrt`, computes `scale = min(1, max_norm / max(norm, eps))`, writes
     to a `scale_buf` device scalar. Also writes the raw norm to
     `norm_buf` for callers that want to D2H it on a log cadence.
  3. `_GradScaleVisitorGPU` launches `_grad_scale_apply_kernel` per
     Param, reading `scale_buf[0]` and multiplying its grad in place.

Three on-device passes, zero D2H. `scale = 1` when `norm ≤ max_norm` is
a no-op, semantically identical to the host-branch CPU path — and lets
us drop the host-side `if`.

Integration: `Adam.step` calls `clip_grads_auto[target]` before the
update visitor, gated by `Adam.max_grad_norm > 0`. Adam owns the
`GradClipState` and lazy-allocates it on first GPU call when
`len(self.offsets)` (= N_PARAMS) is known.

Caller convention: walker invoked AFTER all backward passes that wrote
into `model.<param>.grad` and BEFORE the optimizer applies the update.
For trainers with multiple disjoint models (SAC: actor + critic1 +
critic2), each Adam clips its own model independently — no cross-model
"global" norm. Matches deep_agents/'s per-optimizer clipping convention.
"""

from std.math import sqrt
from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT, TPB
from .module import mptr
from .module import Module
from .param_visitor import ParamVisitor


# ──────────────────────────────────────────────────────────────────────
# CPU visitors — unchanged from B.3.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _GradSumSqVisitorCPU(ParamVisitor):
    var sum_sq: Scalar[DT]

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
        for i in range(n_elems):
            var g = g_ptr[i]
            self.sum_sq += g * g


@fieldwise_init
struct _GradScaleVisitorCPU(ParamVisitor):
    var scale: Scalar[DT]

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
        for i in range(n_elems):
            g_ptr[i] = g_ptr[i] * self.scale


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level so enqueue_function can bind them.
# ──────────────────────────────────────────────────────────────────────


comptime GC_TPB: Int = 128  # single-block reduction width; mirrors LN_TPB.


def _sum_sq_partial_kernel(
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    partials: UnsafePointer[Scalar[DT], MutAnyOrigin],
    slot: Int,
    n_elems: Int,
):
    """Single-block, GC_TPB-thread tree reduction. Each thread strides
    over `grad[0..n_elems]`, accumulates `g²`, then `block.sum` produces
    the per-block total. Thread 0 writes to `partials[slot]`.

    Same pattern LayerNorm uses for per-row reductions. Capped at one
    block per param — for nn2's typical param sizes (hidden² ≤ 65K) the
    strided loop is ~512 iters per thread, well within budget."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < n_elems:
        var g = grad[k]
        my_sum += g * g
        k += GC_TPB
    var total = block.sum[block_size=GC_TPB, broadcast=False](val=my_sum)
    if t == 0:
        partials[slot] = total[0]


def _compute_scale_kernel(
    partials: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_params: Int,
    scale_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    norm_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    max_norm: Scalar[DT],
    eps: Scalar[DT],
):
    """Single-thread sum + scale computation.

    `scale = min(1, max_norm / max(norm, eps))`. Apply unconditionally
    in the next pass — `scale = 1` when `norm ≤ max_norm` is a no-op,
    same semantics as the host-branch CPU formulation.
    """
    var t = Int(thread_idx.x)
    if t == 0:
        var s: Scalar[DT] = 0.0
        for i in range(n_params):
            s += partials[i]
        var norm = sqrt(s)
        norm_buf[0] = norm
        var denom = norm if norm > eps else eps
        var ratio = max_norm / denom
        scale_buf[0] = ratio if ratio < Scalar[DT](1.0) else Scalar[DT](1.0)


def _grad_scale_kernel(
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    scale_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_elems: Int,
):
    """`grad[i] *= scale_buf[0]`. One thread per element."""
    var i = Int(global_idx.x)
    if i < n_elems:
        grad[i] = grad[i] * scale_buf[0]


# ──────────────────────────────────────────────────────────────────────
# GPU clip state — owns the per-param partials + scale + norm buffers.
# Adam lazy-allocates on first GPU step with max_grad_norm > 0.
# ──────────────────────────────────────────────────────────────────────


struct GradClipState(Movable & ImplicitlyDestructible):
    var partials:  Optional[DeviceBuffer[DT]]   # [N_PARAMS]
    var scale_buf: Optional[DeviceBuffer[DT]]   # [1]
    var norm_buf:  Optional[DeviceBuffer[DT]]   # [1] — for D2H on log cadence
    var n_params:  Int

    def __init__(out self):
        self.partials = None
        self.scale_buf = None
        self.norm_buf = None
        self.n_params = 0

    @staticmethod
    def make(ctx: DeviceContext, n_params: Int) raises -> Self:
        var s = Self()
        s.partials  = ctx.enqueue_create_buffer[DT](n_params)
        s.scale_buf = ctx.enqueue_create_buffer[DT](1)
        s.norm_buf  = ctx.enqueue_create_buffer[DT](1)
        s.n_params  = n_params
        return s^


# ──────────────────────────────────────────────────────────────────────
# GPU visitors — Approach A (stateful: ctx + device buffer field).
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct _GradSumSqVisitorGPU(ParamVisitor):
    var ctx: DeviceContext
    var partials: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var slot: Int

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
        self.ctx.enqueue_function[_sum_sq_partial_kernel](
            g_ptr, self.partials, self.slot, n_elems,
            grid_dim=1, block_dim=GC_TPB,
        )
        self.slot += 1


@fieldwise_init
struct _GradScaleVisitorGPU(ParamVisitor):
    var ctx: DeviceContext
    var scale_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]

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
        var n_blocks = (n_elems + TPB - 1) // TPB
        self.ctx.enqueue_function[_grad_scale_kernel](
            g_ptr, self.scale_buf, n_elems,
            grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────
# Entry points.
# ──────────────────────────────────────────────────────────────────────


def clip_grads_auto[
    M: Module,
    target: StaticString,
](mut model: M, max_norm: Scalar[DT]) raises -> Scalar[DT]:
    """CPU-target entry point. Kept for callsites that don't carry a
    clip state (e.g. Adam CPU path).

    Returns pre-clip global L2 norm. When `max_norm ≤ 0` returns 0 (the
    disabled sentinel — callers can short-circuit).

    The GPU dispatch lives in `clip_grads_auto_gpu` because the GPU path
    requires a pre-allocated `GradClipState` for the on-device partials
    buffer. Mojo nightly doesn't support default-valued comptime params
    or trait-typed defaults in the same overload, so we expose two
    entry points.
    """
    if max_norm <= Scalar[DT](0.0):
        return Scalar[DT](0.0)

    comptime if target == "cpu":
        var sum_visitor = _GradSumSqVisitorCPU(sum_sq=Scalar[DT](0.0))
        model.for_each_param[target, _GradSumSqVisitorCPU](
            String(""), sum_visitor
        )
        var norm = sqrt(sum_visitor.sum_sq)
        if norm > max_norm:
            var scale_visitor = _GradScaleVisitorCPU(
                scale=max_norm / norm
            )
            model.for_each_param[target, _GradScaleVisitorCPU](
                String(""), scale_visitor
            )
        return norm
    else:
        # GPU callers must use `clip_grads_auto_gpu` with a state buffer.
        raise Error(
            "clip_grads_auto[target='gpu']: GPU path requires a"
            " GradClipState — call `clip_grads_auto_gpu` from `Adam.step`"
            " instead. This overload exists only for the CPU dispatch."
        )


def clip_grads_auto_gpu[M: Module](
    mut model: M,
    ctx: DeviceContext,
    mut state: GradClipState,
    max_norm: Scalar[DT],
) raises:
    """GPU path. Three on-device passes, zero D2H.

    Pre-requisite: `state.n_params == count_params(model)` and all three
    `state.*` buffers are allocated. Caller (Adam) lazy-allocates `state`
    when it first runs the clip with `max_grad_norm > 0`.

    Does NOT return the pre-clip norm — it stays on-device in
    `state.norm_buf`. Callers wanting to log the norm can D2H that buffer
    on whatever cadence they choose (e.g. once per `flush_train_log`).
    """
    if max_norm <= Scalar[DT](0.0):
        return

    # Pass 1: per-Param partial sum_sq into partials[slot].
    var sum_visitor = _GradSumSqVisitorGPU(
        ctx=ctx,
        partials=state.partials.value().unsafe_ptr(),
        slot=0,
    )
    model.for_each_param[target="gpu", V=_GradSumSqVisitorGPU](
        String(""), sum_visitor
    )

    # Pass 2: compute scale on device.
    ctx.enqueue_function[_compute_scale_kernel](
        state.partials.value().unsafe_ptr(),
        state.n_params,
        state.scale_buf.value().unsafe_ptr(),
        state.norm_buf.value().unsafe_ptr(),
        max_norm,
        Scalar[DT](1e-12),
        grid_dim=1, block_dim=1,
    )

    # Pass 3: apply scale to every Param's grad.
    var scale_visitor = _GradScaleVisitorGPU(
        ctx=ctx,
        scale_buf=state.scale_buf.value().unsafe_ptr(),
    )
    model.for_each_param[target="gpu", V=_GradScaleVisitorGPU](
        String(""), scale_visitor
    )
