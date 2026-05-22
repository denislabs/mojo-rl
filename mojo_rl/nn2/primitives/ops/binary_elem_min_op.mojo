"""BinaryElemMinOp — `BinaryElementOp` for element-wise min.

  output[i]   = min(x[i], y[i])
  cache[i]    = 1.0 if x[i] <  y[i] else 0.0     (ties → in1 wins, matching
                                                  the pre-Phase-4.5 standalone
                                                  BinaryElemMin behavior)
  grad_in0[i] = grad_output[i] if cache[i] > 0.5 else 0
  grad_in1[i] = grad_output[i] if cache[i] <= 0.5 else 0

The SIMD `.lt()` path returns lane-wise Bool (see
`feedback_simd_gt_scalar_bool`); `.select()` then dispatches the per-lane
arithmetic without scalarising.
"""

from ...constants import DT
from ...core.binary_element_op import BinaryElementOp


struct BinaryElemMinOp(BinaryElementOp):
    """Element-wise min with mask cache (owns_cache = True)."""

    comptime owns_cache = True

    @staticmethod
    def forward_scalar(x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
        return x if x < y else y

    @staticmethod
    def forward_simd[W: Int](
        x: SIMD[DT, W], y: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        return x.lt(y).select(x, y)

    @staticmethod
    def cache_scalar(x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
        return Scalar[DT](1.0) if x < y else Scalar[DT](0.0)

    @staticmethod
    def cache_simd[W: Int](
        x: SIMD[DT, W], y: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var one_v = SIMD[DT, W](1.0)
        var zero_v = SIMD[DT, W](0.0)
        return x.lt(y).select(one_v, zero_v)

    @staticmethod
    def backward_scalar_x(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        var zero: Scalar[DT] = 0.0
        return go if c > Scalar[DT](0.5) else zero

    @staticmethod
    def backward_simd_x[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var zero_v = SIMD[DT, W](0.0)
        var half_v = SIMD[DT, W](0.5)
        return c.gt(half_v).select(go, zero_v)

    @staticmethod
    def backward_scalar_y(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        var zero: Scalar[DT] = 0.0
        return go if c <= Scalar[DT](0.5) else zero

    @staticmethod
    def backward_simd_y[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var zero_v = SIMD[DT, W](0.0)
        var half_v = SIMD[DT, W](0.5)
        # NB: `<= 0.5` rather than `< 0.5` so ties (c=0 case from
        # forward) credit in1, matching the pre-Phase-4.5 standalone
        # implementation.
        return c.le(half_v).select(go, zero_v)
