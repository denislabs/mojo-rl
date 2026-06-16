"""SigmoidOp — `ElementOp` for the logistic sigmoid activation.

`y = 1 / (1 + exp(-x))`,  `dy/dx = y · (1 - y)`.

Output-cache op (`owns_cache = True`): forward writes `y` to the cache;
backward reads `y` back as `c` and returns `go · y · (1 - y)`. Same
contract as `TanhOp` — the derivative depends only on the output, so we
own the cache instead of aliasing the input slab.

The forward keeps `exp(-x)` unguarded: for `x ≪ 0` the value saturates
to `+inf` and the division yields `0`; for `x ≫ 0` it underflows to `0`
and the division yields `1`. Both endpoints are mathematically correct
and IEEE-754-safe, matching the behaviour of `SwishOp`'s internal
sigmoid.
"""

from std.math import exp

from ...constants import DT
from ...core.element_op import ElementOp


struct SigmoidOp(ElementOp):
    """Sigmoid activation with `y`-caching backward."""

    comptime owns_cache = True

    @staticmethod
    def display_label() -> String:
        return String("Sigmoid")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        var one: Scalar[DT] = 1.0
        return one / (one + exp(-x))

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        var one = SIMD[DT, W](1.0)
        return one / (one + exp(-x))

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached y. dy/dx = y · (1 - y).
        var one: Scalar[DT] = 1.0
        return go * c * (one - c)

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var one = SIMD[DT, W](1.0)
        return go * c * (one - c)
