"""BinaryAddOp — `BinaryElementOp` for element-wise addition.

  output[i]   = x[i] + y[i]
  grad_in0[i] = grad_output[i]
  grad_in1[i] = grad_output[i]

No cache (owns_cache = False). Mirror of `BinarySubOp` (the only difference is
the `+` forward and the `+go` y-gradient). Used by the TD3 target-y graph
(`a_plus_n = a_sp + noise_clip`).
"""

from ...constants import DT
from ...core.binary_element_op import BinaryElementOp


struct BinaryAddOp(BinaryElementOp):
    """Element-wise addition (no cache)."""

    comptime owns_cache = False

    @staticmethod
    def display_label() -> String:
        return String("Add")

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
