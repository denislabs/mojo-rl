"""Conv → (BN) → ReLU block aliases for nn.storage.

Storage-surface port of `nn/models/conv.mojo`. Compositional aliases — each
expands to a `Sequential` of existing storage primitives, so they get correct
forward / vjp / walkers for free and need no bespoke kernels. Only change vs
legacy: `ReLU` imported from `primitives/activations.mojo`.

Spatial-shape convention (matches `Conv2D`):
    OH = (H + 2*P - K) // S + 1
    OW = (W + 2*P - K) // S + 1

  - `Conv2DReLU[IC, OC, K, S, P, H, W]`           Conv → ReLU
  - `Conv2DBatchNormReLU[IC, OC, K, S, P, H, W]`  Conv → BN → ReLU
"""

from ..primitives.conv2d import Conv2D
from ..primitives.batch_norm_2d import (
    BatchNorm2D, BN2D_DEFAULT_EPS, BN2D_DEFAULT_MOM,
)
from ..primitives.activations import ReLU
from ..combinators.sequential import Sequential


comptime Conv2DReLU[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
] = Sequential[
    Conv2D[IC, OC, K, S, P, H, W],
    ReLU[OC * ((H + 2 * P - K) // S + 1) * ((W + 2 * P - K) // S + 1)],
]


comptime Conv2DBatchNormReLU[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    EPS: Float64 = BN2D_DEFAULT_EPS,
] = Sequential[
    Conv2D[IC, OC, K, S, P, H, W],
    BatchNorm2D[
        OC, (H + 2 * P - K) // S + 1, (W + 2 * P - K) // S + 1,
        BN2D_DEFAULT_MOM, EPS,
    ],
    ReLU[OC * ((H + 2 * P - K) // S + 1) * ((W + 2 * P - K) // S + 1)],
]
