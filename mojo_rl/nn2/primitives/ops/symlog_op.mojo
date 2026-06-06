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

from std.math import abs as math_abs

from ...constants import DT
from ...core.element_op import ElementOp
from .symlog_math import symlog_simd
from .symlog_math import symlog as _symlog_scalar


struct SymlogOp(ElementOp):
    """Symlog activation with input-cache backward (`owns_cache=False`)."""

    comptime owns_cache = False

    @staticmethod
    def display_label() -> String:
        return String("Symlog")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return _symlog_scalar(x)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return symlog_simd[W](x)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached INPUT (x). dy/dx = 1 / (1 + |x|). Kept as an
        # explicit division (not go * reciprocal) to stay bit-identical.
        var one: Scalar[DT] = 1.0
        var abs_x = c if c >= Scalar[DT](0) else -c
        return go / (one + abs_x)

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var one_v = SIMD[DT, W](1)
        return go / (one_v + math_abs(c))
