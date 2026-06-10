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
from std.gpu import global_idx, block_idx, block_dim, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT, TPB
from .module import mptr
from .module import Module
from .param_visitor import ParamVisitor
from .graph_node import GraphNode
from ..combinators.compute_graph import ComputeGraph


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
# GPU GROUPED (multi-tensor) kernels — NVIDIA-only path. Reuse the Adam
# grouped optimizer's device descriptor arrays (`grad_addrs` + cumulative
# `moment_offs`) to clip EVERY param in 3 launches total (vs the per-Param
# path's 2·N_PARAMS launches). The win is twofold: (1) launch overhead
# collapses, (2) the sum-of-squares reduction is a single FLAT grid over
# all params' elements, so a big param (e.g. the 3136×512 head) gets many
# blocks instead of the per-Param path's single 128-thread block — that
# single-block reduction over a ~1.6M-element tensor was the profile's
# `_sum_sq_partial_kernel` 425µs tail.
#
# NOT bit-identical to the per-Param path: the reduction is regrouped, so
# the L2 norm differs by fp-rounding (~1e-6). The CPU / Apple per-Param
# path is unchanged and stays bit-identical. Same tradeoff the grouped
# Adam update already makes.
# ──────────────────────────────────────────────────────────────────────


def _find_param_gc(
    moment_offs: UnsafePointer[Int32, MutAnyOrigin], n_params: Int, flat: Int
) -> Int:
    """Map a flat element index → owning Param via the cumulative offset
    table (`moment_offs[p]` = first flat index of Param p). Linear scan —
    n_params is tiny. Mirrors Adam's `_find_param` (duplicated here to
    avoid a grad_clip→adam import cycle)."""
    var p = 0
    while p + 1 < n_params and Int(moment_offs[p + 1]) <= flat:
        p += 1
    return p


def _grouped_sumsq_kernel(
    grad_addrs: UnsafePointer[UInt64, MutAnyOrigin],
    moment_offs: UnsafePointer[Int32, MutAnyOrigin],
    n_params: Int,
    total: Int,
    block_partials: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Flat 1-D grid over ALL params' elements. Each thread reads its grad
    element `g` (resolving owning Param via `moment_offs`), squares it; a
    block-wide `block.sum` reduces the block's chunk and thread 0 writes
    the partial to `block_partials[block_idx]`. Threads past `total`
    contribute 0."""
    var flat = Int(global_idx.x)
    var my_sum: Scalar[DT] = 0.0
    if flat < total:
        var p = _find_param_gc(moment_offs, n_params, flat)
        var local = flat - Int(moment_offs[p])
        var grad = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(grad_addrs[p])
        )
        var g = grad[local]
        my_sum = g * g
    var total_blk = block.sum[block_size=TPB, broadcast=False](val=my_sum)
    if Int(thread_idx.x) == 0:
        block_partials[Int(block_idx.x)] = total_blk[0]


def _grouped_compute_scale_kernel(
    block_partials: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_blocks: Int,
    scale_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    norm_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    max_norm: Scalar[DT],
    eps: Scalar[DT],
):
    """Single-block reduction of the per-block partials → global ‖grad‖²,
    then `scale = min(1, max_norm / max(norm, eps))`. Writes scale + raw
    norm. Same scale formula as `_compute_scale_kernel`."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < n_blocks:
        my_sum += block_partials[k]
        k += GC_TPB
    var s = block.sum[block_size=GC_TPB, broadcast=False](val=my_sum)
    if t == 0:
        var norm = sqrt(s[0])
        norm_buf[0] = norm
        var denom = norm if norm > eps else eps
        var ratio = max_norm / denom
        scale_buf[0] = ratio if ratio < Scalar[DT](1.0) else Scalar[DT](1.0)


def _grouped_scale_apply_kernel(
    grad_addrs: UnsafePointer[UInt64, MutAnyOrigin],
    moment_offs: UnsafePointer[Int32, MutAnyOrigin],
    n_params: Int,
    total: Int,
    scale_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Flat 1-D grid over ALL params' elements: `grad[local] *= scale`."""
    var flat = Int(global_idx.x)
    if flat < total:
        var p = _find_param_gc(moment_offs, n_params, flat)
        var local = flat - Int(moment_offs[p])
        var grad = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(grad_addrs[p])
        )
        grad[local] = grad[local] * scale_buf[0]


# ──────────────────────────────────────────────────────────────────────
# GPU clip state — owns the per-param partials + scale + norm buffers.
# Adam lazy-allocates on first GPU step with max_grad_norm > 0.
# ──────────────────────────────────────────────────────────────────────


struct GradClipState(Movable & ImplicitlyDestructible):
    var partials: Optional[DeviceBuffer[DT]]  # [N_PARAMS] (per-Param path)
    var scale_buf: Optional[DeviceBuffer[DT]]  # [1]
    var norm_buf: Optional[DeviceBuffer[DT]]  # [1] — for D2H on log cadence
    var n_params: Int
    # Grouped (NVIDIA) path scratch: one partial per flat-grid block. Sized
    # to ceil(total_elems / TPB), allocated only when `make` is given total>0.
    var block_partials: Optional[DeviceBuffer[DT]]
    var n_blocks: Int

    def __init__(out self):
        self.partials = None
        self.scale_buf = None
        self.norm_buf = None
        self.n_params = 0
        self.block_partials = None
        self.n_blocks = 0

    @staticmethod
    def make(ctx: DeviceContext, n_params: Int, total: Int = 0) raises -> Self:
        var s = Self()
        s.partials = ctx.enqueue_create_buffer[DT](n_params)
        s.scale_buf = ctx.enqueue_create_buffer[DT](1)
        s.norm_buf = ctx.enqueue_create_buffer[DT](1)
        s.n_params = n_params
        # Grouped path: per-block partials buffer (total>0 → grouped clip).
        if total > 0:
            var nb = (total + TPB - 1) // TPB
            s.block_partials = ctx.enqueue_create_buffer[DT](nb)
            s.n_blocks = nb
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
            g_ptr,
            self.partials,
            self.slot,
            n_elems,
            grid_dim=1,
            block_dim=GC_TPB,
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
            g_ptr,
            self.scale_buf,
            n_elems,
            grid_dim=n_blocks,
            block_dim=TPB,
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
            var scale_visitor = _GradScaleVisitorCPU(scale=max_norm / norm)
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


def clip_grads_auto_gpu[
    M: Module
](
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
        grid_dim=1,
        block_dim=1,
    )

    # Pass 3: apply scale to every Param's grad.
    var scale_visitor = _GradScaleVisitorGPU(
        ctx=ctx,
        scale_buf=state.scale_buf.value().unsafe_ptr(),
    )
    model.for_each_param[target="gpu", V=_GradScaleVisitorGPU](
        String(""), scale_visitor
    )


# ──────────────────────────────────────────────────────────────────────
# ComputeGraph entry points — a `ComputeGraph` exposes the same
# `for_each_param[target, V]` walk as a `Module` but does NOT conform to
# `Module`, so the `M: Module`-bounded entry points above can't take it.
# These mirror `clip_grads_auto` (CPU) / `clip_grads_auto_gpu` (GPU
# per-Param) exactly, walking the graph's params instead. Used by
# `Adam.step_graph` when `max_grad_norm > 0` (graph-owned-params trainers
# such as LeWM, whose single-token CLS readout concentrates gradients and
# needs clipping the mean-pooled variant got away without). The GPU path
# is per-Param only (graph-made Adam builds no grouped descriptors).
# ──────────────────────────────────────────────────────────────────────


def clip_grads_graph_cpu[
    OUT: Int, *NODES: GraphNode
](mut g: ComputeGraph[OUT, *NODES], max_norm: Scalar[DT]) raises -> Scalar[DT]:
    """CPU graph clip. Returns pre-clip global L2 norm (0 when disabled)."""
    if max_norm <= Scalar[DT](0.0):
        return Scalar[DT](0.0)
    var sum_visitor = _GradSumSqVisitorCPU(sum_sq=Scalar[DT](0.0))
    g.for_each_param[target="cpu", V=_GradSumSqVisitorCPU](
        String(""), sum_visitor
    )
    var norm = sqrt(sum_visitor.sum_sq)
    if norm > max_norm:
        var scale_visitor = _GradScaleVisitorCPU(scale=max_norm / norm)
        g.for_each_param[target="cpu", V=_GradScaleVisitorCPU](
            String(""), scale_visitor
        )
    return norm


def clip_grads_graph_gpu[
    OUT: Int, *NODES: GraphNode
](
    mut g: ComputeGraph[OUT, *NODES],
    ctx: DeviceContext,
    mut state: GradClipState,
    max_norm: Scalar[DT],
) raises:
    """GPU graph clip — per-Param, three on-device passes, zero D2H.
    Mirrors `clip_grads_auto_gpu` but walks `g.for_each_param`. `state`
    must be `make`-d with `n_params == count(graph params)`."""
    if max_norm <= Scalar[DT](0.0):
        return
    var sum_visitor = _GradSumSqVisitorGPU(
        ctx=ctx, partials=state.partials.value().unsafe_ptr(), slot=0,
    )
    g.for_each_param[target="gpu", V=_GradSumSqVisitorGPU](
        String(""), sum_visitor
    )
    ctx.enqueue_function[_compute_scale_kernel](
        state.partials.value().unsafe_ptr(),
        state.n_params,
        state.scale_buf.value().unsafe_ptr(),
        state.norm_buf.value().unsafe_ptr(),
        max_norm,
        Scalar[DT](1e-12),
        grid_dim=1,
        block_dim=1,
    )
    var scale_visitor = _GradScaleVisitorGPU(
        ctx=ctx, scale_buf=state.scale_buf.value().unsafe_ptr(),
    )
    g.for_each_param[target="gpu", V=_GradScaleVisitorGPU](
        String(""), scale_visitor
    )


def clip_grads_grouped_gpu(
    ctx: DeviceContext,
    mut state: GradClipState,
    grad_addrs: UnsafePointer[UInt64, MutAnyOrigin],
    moment_offs: UnsafePointer[Int32, MutAnyOrigin],
    n_params: Int,
    total: Int,
    max_norm: Scalar[DT],
) raises:
    """GPU grouped (multi-tensor) clip — NVIDIA-only. Three launches total:
      1. `_grouped_sumsq_kernel`  — FLAT grid over all `total` grad elements,
         block-reduced into `state.block_partials` (one slot per block).
      2. `_grouped_compute_scale_kernel` — single block reduces the partials
         → ‖grad‖² → scale.
      3. `_grouped_scale_apply_kernel` — FLAT grid scales every grad in place.

    Reuses the Adam grouped descriptor arrays (`grad_addrs` + `moment_offs`);
    `total` is the dense element count (== Adam.total_size). `state` must be
    `make`-d with `total > 0` so `block_partials` exists. Numerically
    equivalent to (not bit-identical with) the per-Param path. Zero D2H."""
    if max_norm <= Scalar[DT](0.0) or total <= 0 or n_params <= 0:
        return

    var n_blocks = (total + TPB - 1) // TPB

    # Pass 1: flat-grid sum of squares → per-block partials.
    ctx.enqueue_function[_grouped_sumsq_kernel](
        grad_addrs,
        moment_offs,
        n_params,
        total,
        state.block_partials.value().unsafe_ptr(),
        grid_dim=n_blocks,
        block_dim=TPB,
    )

    # Pass 2: reduce partials → scale (single block).
    ctx.enqueue_function[_grouped_compute_scale_kernel](
        state.block_partials.value().unsafe_ptr(),
        n_blocks,
        state.scale_buf.value().unsafe_ptr(),
        state.norm_buf.value().unsafe_ptr(),
        max_norm,
        Scalar[DT](1e-12),
        grid_dim=1,
        block_dim=GC_TPB,
    )

    # Pass 3: flat-grid scale-apply over all grads.
    ctx.enqueue_function[_grouped_scale_apply_kernel](
        grad_addrs,
        moment_offs,
        n_params,
        total,
        state.scale_buf.value().unsafe_ptr(),
        grid_dim=n_blocks,
        block_dim=TPB,
    )
