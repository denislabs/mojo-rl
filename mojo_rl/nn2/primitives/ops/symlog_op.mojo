"""SymlogOp — `ElementOp` for the symlog transform.

`y = sign(x) · log(1 + |x|)`, `dy/dx = 1 / (1 + |x|)`. Used by
DreamerV3 (reward / return rescaling) and TD-MPC2 (distributional value
head encoding). Input-cache op (`owns_cache = False`): backward needs
the original `x`, so the orchestrator's input slab is aliased through
`Elementwise._cached_input_ptr` and read back as `c` in the backward
kernel.

The hand-written `Symlog[DIM]` math is mirrored exactly so
`Elementwise[DIM, SymlogOp]` is bit-identical.
"""

from std.math import log
from std.math import abs as math_abs

from ...constants import DT
from ...core.element_op import ElementOp


struct SymlogOp(ElementOp):
    """Symlog activation with input-cache backward (`owns_cache=False`)."""

    comptime owns_cache = False

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        var zero: Scalar[DT] = 0.0
        var one: Scalar[DT] = 1.0
        var abs_x = x if x >= zero else -x
        var sgn: Scalar[DT] = one if x >= zero else -one
        return sgn * log(one + abs_x)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        var zero_v = SIMD[DT, W](0)
        var one_v = SIMD[DT, W](1)
        var pos_v = SIMD[DT, W](1)
        var neg_v = SIMD[DT, W](-1)
        var abs_x = math_abs(x)
        var sgn = x.ge(zero_v).select(pos_v, neg_v)
        return sgn * log(one_v + abs_x)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached INPUT (x). dy/dx = 1 / (1 + |x|).
        var one: Scalar[DT] = 1.0
        var abs_x = c if c >= Scalar[DT](0) else -c
        return go / (one + abs_x)

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var one_v = SIMD[DT, W](1)
        return go / (one_v + math_abs(c))
