"""CenterHalfOp — `ElementOp` that centers [0,1] inputs to [-0.5, 0.5].

The DreamerV3 reference feeds the image encoder `imgs/255 - 0.5` (centered to
[-0.5, 0.5]) — see `references/dreamerv3-main/dreamerv3/rssm.py` Encoder. Our
pixel envs already emit obs in [0,1], so the centering reduces to a constant
shift `y = x - 0.5`. The decoder TARGET stays in [0,1] (the reference centers
only the encoder input, not the reconstruction target), so this op belongs only
at the encoder's front (prepended to `DreamerEncoderCNN`).

Backward is trivial: f(x) = x - 0.5 ⇒ f'(x) = 1, so the gradient passes through
unchanged (`owns_cache=False`, cached value ignored).
"""

from ...constants import DT
from ...core.element_op import ElementOp


comptime _HALF: Scalar[DT] = 0.5


struct CenterHalfOp(ElementOp):
    """Centering shift y = x - 0.5 (derivative 1, gradient pass-through)."""

    comptime owns_cache = False

    @staticmethod
    def display_label() -> String:
        return String("Center")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return x - _HALF

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return x - SIMD[DT, W](_HALF)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # f'(x) = 1 → gradient passes through unchanged (cache `c` ignored).
        return go

    @staticmethod
    def backward_simd[W: Int](c: SIMD[DT, W], go: SIMD[DT, W]) -> SIMD[DT, W]:
        return go
