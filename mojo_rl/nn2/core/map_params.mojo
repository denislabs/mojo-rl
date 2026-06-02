"""`map_params` / `polyak_update` — two-tree parameter walker.

`polyak_update` mutates a target model toward an online model:

    target = (1 - tau) * target + tau * online

elementwise across every leaf parameter. Used for SAC's soft-update of
target critic nets each gradient step (tau ≈ 0.005).

Implementation builds on `named_params`: walks both models, validates
the resulting `List[NamedParam]`s match leaf-for-leaf (same count, same
names, same sizes), then runs the linear interpolation.

CPU path: scalar loop. GPU path: one-thread-per-element kernel launched
per leaf, using the named-params raw pointers (which point at the live
Param storage — CPU `List` or GPU `DeviceBuffer` depending on how the
Param was made).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from .module import Module
from .named_params import NamedParam, named_params
from .target_tag import TARGET_GPU, target_tag_for


def _polyak_kernel(
    online: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_net: UnsafePointer[Scalar[DT], MutAnyOrigin],
    one_minus_tau: Scalar[DT],
    tau: Scalar[DT],
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        target_net[idx] = one_minus_tau * target_net[idx] + tau * online[idx]


def polyak_update[
    target: StaticString,
    M: Module,
](
    mut online: M,
    mut target_net: M,
    tau: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Mutate `target_net` toward `online` by `tau`.

    `target = (1 - tau) * target + tau * online` per leaf parameter.

    Validates structure parity via `named_params`. Raises if the two
    walks disagree on count, name, or size (a sign of a typo or a model
    architecture mismatch).

    Block A: GPU path. The named-params raw pointers come from the
    Param wrapper's `param_ptr_for[target]` which returns the right
    storage's `.unsafe_ptr()`. The polyak kernel reads/writes through
    those pointers directly.
    """
    var online_ps = named_params[target, M](online)
    var target_ps = named_params[target, M](target_net)

    if len(online_ps) != len(target_ps):
        raise Error(
            "polyak_update: param count mismatch (online="
            + String(len(online_ps))
            + ", target="
            + String(len(target_ps))
            + ")"
        )

    var one_minus_tau: Scalar[DT] = Scalar[DT](1.0) - tau

    # Structure-parity validation (all targets). Done up front so the GPU
    # execution below can launch a single batched kernel without interleaving
    # host-side checks.
    for i in range(len(online_ps)):
        ref op = online_ps[i]
        ref tp = target_ps[i]
        if op.n_elems != tp.n_elems:
            raise Error(
                "polyak_update: param size mismatch at index "
                + String(i)
                + " (online '"
                + op.name
                + "' n="
                + String(op.n_elems)
                + ", target '"
                + tp.name
                + "' n="
                + String(tp.n_elems)
                + ")"
            )
        if op.name != tp.name:
            raise Error(
                "polyak_update: param name mismatch at index "
                + String(i)
                + " (online '"
                + op.name
                + "', target '"
                + tp.name
                + "')"
            )

    comptime if target == "cpu":
        for i in range(len(online_ps)):
            ref op = online_ps[i]
            ref tp = target_ps[i]
            var n = op.n_elems
            var op_ptr = op.param_ptr
            var tp_ptr = tp.param_ptr
            for k in range(n):
                tp_ptr[k] = one_minus_tau * tp_ptr[k] + tau * op_ptr[k]
    else:
        # GPU: launch one-thread-per-element via a kernel. The named-params
        # pointers are already on-device because the Param wrapper's
        # `param_ptr_for["gpu"]` returns the DeviceBuffer's underlying pointer.
        comptime assert target == "gpu", (
            "polyak_update: target must be 'cpu' or 'gpu'"
        )
        if not ctx:
            raise Error(
                "polyak_update[target='gpu']: ctx is required — "
                "thread the SAC trainer's DeviceContext through "
                "polyak_step / polyak_update to avoid per-call "
                "DeviceContext() construction (Apple Metal command-"
                "queue exhaustion)."
            )
        # Batched path: one slab-wide `_polyak_kernel` when BOTH param trees
        # are contiguous in walk order (the common nn2 case — params live in
        # one slab) — replaces one kernel per leaf (~3/critic → 1). The blend
        # is per-element, so a slab-wide grid is bit-identical. Any gap (a
        # model whose params aren't one slab) flips `contiguous` False and we
        # fall back to per-leaf launches.
        var contiguous = len(online_ps) > 0
        var running = 0
        for i in range(len(online_ps)):
            ref op = online_ps[i]
            ref tp = target_ps[i]
            if i > 0:
                if op.param_ptr != online_ps[0].param_ptr + running:
                    contiguous = False
                if tp.param_ptr != target_ps[0].param_ptr + running:
                    contiguous = False
            running += op.n_elems
        if contiguous and running > 0:
            _polyak_launch_gpu_slab(
                online_ps[0].param_ptr,
                target_ps[0].param_ptr,
                one_minus_tau,
                tau,
                running,
                ctx.value(),
            )
        else:
            for i in range(len(online_ps)):
                _polyak_launch_gpu(
                    online_ps[i], target_ps[i], one_minus_tau, tau,
                    ctx.value(),
                )


def _polyak_launch_gpu(
    op: NamedParam,
    tp: NamedParam,
    one_minus_tau: Scalar[DT],
    tau: Scalar[DT],
    ctx: DeviceContext,
) raises:
    """Helper to launch the polyak kernel for one leaf. Extracted from
    the loop body so the comptime-for body doesn't carry a kernel
    constructor (keeps the inliner happy).

    Takes ctx explicitly: constructing a fresh `DeviceContext()` per leaf
    per train step exhausts Apple Metal command-queue resources within a
    few hundred SAC train steps."""
    var n = op.n_elems
    var n_blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[_polyak_kernel](
        op.param_ptr, tp.param_ptr, one_minus_tau, tau, n,
        grid_dim=n_blocks, block_dim=TPB,
    )


def _polyak_launch_gpu_slab(
    online_base: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_base: UnsafePointer[Scalar[DT], MutAnyOrigin],
    one_minus_tau: Scalar[DT],
    tau: Scalar[DT],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Single slab-wide polyak launch over `[0, n)` of the contiguous online /
    target param slabs — replaces one per-leaf launch. `_polyak_kernel` is a
    per-element blend (no cross-element interaction), so a slab-wide grid is
    bit-identical to the per-leaf launches. Mirrors `_polyak_launch_gpu`'s
    extraction so the kernel constructor stays out of `polyak_update`'s body."""
    var n_blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[_polyak_kernel](
        online_base, target_base, one_minus_tau, tau, n,
        grid_dim=n_blocks, block_dim=TPB,
    )


def hard_copy_params[
    target: StaticString,
    M: Module,
](
    mut online: M,
    mut target_net: M,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Copy `online` → `target_net` verbatim (tau=1.0). Used to initialize
    target nets to the online net's state immediately after `make`."""
    polyak_update[target, M](online, target_net, Scalar[DT](1.0), ctx)
