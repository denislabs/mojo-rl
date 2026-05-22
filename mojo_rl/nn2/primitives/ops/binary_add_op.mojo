"""BinaryAddOp — `BinaryElementOp` for element-wise addition.

  output[i]   = x[i] + y[i]
  grad_in0[i] = grad_output[i]
  grad_in1[i] = grad_output[i]

No cache (owns_cache = False) — backward needs only grad_output.
"""

from ...constants import DT
from ...core.binary_element_op import BinaryElementOp


struct BinaryAddOp(BinaryElementOp):
    """Element-wise addition (no cache)."""

    comptime owns_cache = False

    @staticmethod
    def forward_scalar(x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
        return x + y

    @staticmethod
    def forward_simd[W: Int](
        x: SIMD[DT, W], y: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        return x + y

    @staticmethod
    def cache_scalar(x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
        return Scalar[DT](0.0)

    @staticmethod
    def cache_simd[W: Int](
        x: SIMD[DT, W], y: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        return SIMD[DT, W](0.0)

    @staticmethod
    def backward_scalar_x(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        return go

    @staticmethod
    def backward_simd_x[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        return go

    @staticmethod
    def backward_scalar_y(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        return go

    @staticmethod
    def backward_simd_y[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        return go
