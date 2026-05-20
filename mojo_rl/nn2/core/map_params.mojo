"""`map_params` / `polyak_update` — two-tree parameter walker.

Phase 7.2.

`polyak_update` mutates a target model toward an online model:

    target = (1 - tau) * target + tau * online

elementwise across every leaf parameter. Used for SAC's soft-update of
target critic nets each gradient step (tau ≈ 0.005).

Implementation builds on `named_params`: walks both models, validates
the resulting `List[NamedParam]`s match leaf-for-leaf (same count, same
names, same sizes), then runs the linear interpolation. The two trees
must have identical shape — same Module types in the same order with
the same param sizes — which is the natural invariant for online/target
pairs that were `make`d from the same comptime Module alias.

CPU only for Phase 7 (SAC Pendulum is CPU-only). GPU port follows the
same shape (kernel per leaf) when the first GPU SAC env lands.
"""

from layout import TileTensor, TensorLayout

from ..constants import DT
from .module import Module
from .named_params import NamedParam, named_params


def polyak_update[
    target: StaticString,
    M: Module,
](
    mut online: M,
    mut target_net: M,
    tau: Scalar[DT],
) raises:
    """Mutate `target_net` toward `online` by `tau`.

    `target = (1 - tau) * target + tau * online` per leaf parameter.

    Validates structure parity via `named_params`. Raises if the two
    walks disagree on count, name, or size (a sign of a typo or a model
    architecture mismatch).
    """
    comptime assert target == "cpu", (
        "polyak_update only supports target='cpu' in Phase 7"
    )

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

        var n = op.n_elems
        var op_ptr = op.param_ptr
        var tp_ptr = tp.param_ptr
        for k in range(n):
            tp_ptr[k] = one_minus_tau * tp_ptr[k] + tau * op_ptr[k]


def hard_copy_params[
    target: StaticString,
    M: Module,
](
    mut online: M,
    mut target_net: M,
) raises:
    """Copy `online` → `target_net` verbatim (tau=1.0). Used to initialize
    target nets to the online net's state immediately after `make`."""
    polyak_update[target, M](online, target_net, Scalar[DT](1.0))
