"""StopGradOp — `ElementOp` that severs gradient flow.

Forward is identity (`y = x`). Backward returns zero regardless of
upstream gradient or cached value. Used in twin-critic actor losses,
target-value blocks, and anywhere a tensor needs to be treated as a
constant by the backward pass.

`owns_cache = False`: we still go through the input-alias cache path
because that's what `Elementwise` knows, but neither `c` nor `go` is
read in the backward — both are ignored and grad_input is filled with
zeros. The cache write itself is harmless (idempotent write of `x` to
its own slab — see Elementwise's GPU forward kernel comment).
"""

from ...constants import DT
from ...core.element_op import ElementOp


struct StopGradOp(ElementOp):
    """Identity forward, zero backward."""

    comptime owns_cache = False

    @staticmethod
    def display_label() -> String:
        return String("StopGrad")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return x

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return x

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        return Scalar[DT](0.0)

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        return SIMD[DT, W](0)
