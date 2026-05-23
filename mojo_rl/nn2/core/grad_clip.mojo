"""Global grad-norm clipping over a Module — Phase B.3.

Two-pass walker:
  1. `_GradSumSqVisitor` accumulates ‖grad‖² across every IsParam-typed
     field of the model (per the standard `model.for_each_param[target, V]`
     dispatch).
  2. If √(sum) > max_norm, `_GradScaleVisitor` scales each grad in place
     by max_norm / norm. Otherwise (or when max_norm <= 0) — no-op.

Integration point: called from `Adam.step` *before* the m/v update
visitor, gated by `Adam.max_grad_norm`. Sentinel `max_grad_norm == 0.0`
(the default) → no clip code path runs — bit-identity preserved.

CPU only for B.3. The GPU path raises when `max_grad_norm > 0` so callers
don't silently miss the clip. When `max_grad_norm == 0` (default), the
GPU path is a no-op (matches pre-B.3 behaviour) — no D2H, no kernel
launch.

Caller convention (matches `Adam.step`): the walker is invoked AFTER all
backward passes that wrote into `model.<param>.grad` and BEFORE the
optimizer applies the update. For trainers with multiple disjoint models
(SAC: actor + critic1 + critic2), each Adam clips its own model
independently — there is no cross-model "global" norm. This matches
`deep_agents/`'s per-optimizer clipping convention.
"""

from std.math import sqrt
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .module import Module
from .param_visitor import ParamVisitor


@fieldwise_init
struct _GradSumSqVisitor(ParamVisitor):
    """Accumulate ∑ g·g over every Param.grad in the walked model."""
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
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        for i in range(n_elems):
            var g = g_ptr[i]
            self.sum_sq += g * g


@fieldwise_init
struct _GradScaleVisitor(ParamVisitor):
    """Scale every Param.grad in place by `scale`."""
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
        var g_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        for i in range(n_elems):
            g_ptr[i] = g_ptr[i] * self.scale


def clip_grads_auto[
    M: Module,
    target: StaticString,
](mut model: M, max_norm: Scalar[DT]) raises -> Scalar[DT]:
    """Global L2-norm clip on `model.<param>.grad` (in place).

    Returns the *pre-clip* norm — useful for diagnostics / logging. When
    `max_norm <= 0` the call short-circuits and returns 0; callers use
    that as the "disabled" sentinel.

    Args:
        model: Any Module whose params have already had grads accumulated.
        max_norm: L2 threshold. ≤0 means disabled (no-op, returns 0).

    Returns:
        The pre-clip global grad norm if clipping was active; 0 when
        disabled. Caller can use this to detect grad explosions even when
        clipping is on.
    """
    if max_norm <= Scalar[DT](0.0):
        return Scalar[DT](0.0)

    comptime if target == "cpu":
        var sum_visitor = _GradSumSqVisitor(sum_sq=Scalar[DT](0.0))
        model.for_each_param[target, _GradSumSqVisitor](
            String(""), sum_visitor
        )
        var norm = sqrt(sum_visitor.sum_sq)
        if norm > max_norm:
            var scale_visitor = _GradScaleVisitor(scale=max_norm / norm)
            model.for_each_param[target, _GradScaleVisitor](
                String(""), scale_visitor
            )
        return norm
    else:
        raise Error(
            "clip_grads_auto: GPU path is TODO (Phase B.3 ships CPU only)."
            " Set max_grad_norm=0 on the Adam optimizer for GPU training,"
            " or run training on CPU."
        )
