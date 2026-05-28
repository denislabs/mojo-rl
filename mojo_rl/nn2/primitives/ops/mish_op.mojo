"""MishOp — `ElementOp` for the Mish activation.

`y = x · tanh(softplus(x))`,  `softplus(x) = log(1 + exp(x))`.

Backward:
    `dy/dx = tanh_sp + x · sigmoid(x) · (1 - tanh_sp²)`

Input-cache op (`owns_cache = False`): backward needs the original `x`
to recompute `softplus(x)` and `sigmoid(x)`. The orchestrator's input
slab is aliased through `Elementwise._cached_input_ptr` and read back
as `c` in the backward kernel — same contract as `SwishOp` / `ReLUOp`.

Numerical stability: forward uses the branchless stable softplus
`sp = max(x, 0) + log(1 + exp(-|x|))`, which keeps the exp argument in
[-∞, 0] and never overflows. Mirrors the math of the hand-written
`Mish[DIM]` leaf in `mojo_rl/nn/model/mish.mojo` (which used a scalar
`if x > 20.0` branch); the branchless form is identical algebraically
and avoids a per-lane branch in the SIMD path.
"""

from std.math import exp, log, tanh

from ...constants import DT
from ...core.element_op import ElementOp


struct MishOp(ElementOp):
    """Mish activation with input-cache backward (`owns_cache=False`)."""

    comptime owns_cache = False

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        var zero: Scalar[DT] = 0.0
        var one: Scalar[DT] = 1.0
        var mx = x if x > zero else zero
        var ax = x if x > zero else -x
        var sp = mx + log(one + exp(-ax))
        return x * tanh(sp)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        var zero = SIMD[DT, W](0.0)
        var one = SIMD[DT, W](1.0)
        var pos = x.gt(zero)
        var mx = pos.select(x, zero)
        var ax = pos.select(x, -x)
        var sp = mx + log(one + exp(-ax))
        return x * tanh(sp)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached INPUT (x). Recompute softplus + tanh_sp + sigmoid.
        var zero: Scalar[DT] = 0.0
        var one: Scalar[DT] = 1.0
        var mx = c if c > zero else zero
        var ax = c if c > zero else -c
        var sp = mx + log(one + exp(-ax))
        var t = tanh(sp)
        var sig = one / (one + exp(-c))
        var d_mish = t + c * sig * (one - t * t)
        return go * d_mish

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var zero = SIMD[DT, W](0.0)
        var one = SIMD[DT, W](1.0)
        var pos = c.gt(zero)
        var mx = pos.select(c, zero)
        var ax = pos.select(c, -c)
        var sp = mx + log(one + exp(-ax))
        var t = tanh(sp)
        var sig = one / (one + exp(-c))
        var d_mish = t + c * sig * (one - t * t)
        return go * d_mish
