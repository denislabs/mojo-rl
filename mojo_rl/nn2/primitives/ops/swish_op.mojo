"""SwishOp — `ElementOp` for the Swish (SiLU) activation.

`y = x · sigmoid(x)`,  `dy/dx = sigmoid(x) + y · (1 - sigmoid(x))`
                       `      = sigmoid(x) · (1 + x · (1 - sigmoid(x)))`

Input-cache op (`owns_cache = False`): backward needs the original `x`
to recompute `sigmoid(x)` (Swish is not invertible in closed form, so
storing `y` would not let us recover the gradient).  The orchestrator's
input slab is aliased through `Elementwise._cached_input_ptr` and read
back as `c` in the backward kernel.

Used by MBPO dynamics ensemble (4×LinearSwish trunk) and any other
SiLU-flavoured architecture.  Mirrors the math the nn1 `Swish` /
`LinearSwish` leaves use so an `Elementwise[DIM, SwishOp]` is
math-identical to nn1's Swish leaf modulo nn1/nn2 numerical conventions.
"""

from std.math import exp

from ...constants import DT
from ...core.element_op import ElementOp


struct SwishOp(ElementOp):
    """Swish / SiLU activation with input-cache backward (`owns_cache=False`)."""

    comptime owns_cache = False

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        var sig = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
        return x * sig

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        var one_v = SIMD[DT, W](1.0)
        var sig = one_v / (one_v + exp(-x))
        return x * sig

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached INPUT (x).  d_swish = sig · (1 + x · (1 - sig)).
        var sig = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-c))
        var d_swish = sig * (Scalar[DT](1.0) + c * (Scalar[DT](1.0) - sig))
        return go * d_swish

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var one_v = SIMD[DT, W](1.0)
        var sig = one_v / (one_v + exp(-c))
        var d_swish = sig * (one_v + c * (one_v - sig))
        return go * d_swish
