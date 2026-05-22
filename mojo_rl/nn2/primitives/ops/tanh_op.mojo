"""TanhOp — `ElementOp` for tanh activation.

Output-cache op: `owns_cache = True`. Forward writes `y = tanh(x)` to
the cache; backward reads `y` back as `c` and returns `go ⊙ (1 − y²)`.
Mirrors the math the hand-written `Tanh[DIM]` leaf used pre-Phase-1.3,
so `Elementwise[DIM, TanhOp]` is bit-identical to `Tanh[DIM]`.
"""

from std.math import tanh

from ...constants import DT
from ...core.element_op import ElementOp


struct TanhOp(ElementOp):
    """tanh activation with `y`-caching backward."""

    comptime owns_cache = True

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return tanh(x)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return tanh(x)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached y. dy/dx = 1 - y².
        return go * (Scalar[DT](1.0) - c * c)

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var one = SIMD[DT, W](1.0)
        return go * (one - c * c)
