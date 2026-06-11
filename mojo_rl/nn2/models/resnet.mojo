"""Residual (ResNet) block aliases for nn2.

Compositional aliases — each expands to a `Sequential` / `Residual` /
`ProjectedResidual` of existing primitives, so they get correct forward /
vjp / walkers / `set_attr` propagation for free and need no bespoke
kernels.

Spatial-shape convention (matches `Conv2D`):
    OH = (H + 2*P - K) // S + 1
    OW = (W + 2*P - K) // S + 1

Provided blocks:
  - `ResBlockConv2DBN[C, K, P, H, W]`             identity-skip ResNet
        block (stride-1, dims preserved): the standard
        `ReLU(x + BN(Conv(ReLU(BN(Conv(x))))))`. Requires `P = (K-1)//2`
        so output spatial == input spatial (else the inner `Residual`'s
        `IN == OUT` assert fires).
  - `ResBlockDownsampleBN[IC, OC, K, P, H, W]`    downsampling ResNet
        block (stride-2 main path, 1×1-stride-2 BN projection skip).
        Sized for the canonical `K=3, P=1` transition; main and skip
        paths both map H → (H-1)//2 + 1.
"""

from ..primitives.conv2d import Conv2D
from ..primitives.batch_norm_2d import (
    BatchNorm2D, BN2D_DEFAULT_EPS, BN2D_DEFAULT_MOM,
)
from ..primitives.relu import ReLU
from ..combinators.sequential import Sequential
from ..combinators.residual import Residual
from ..combinators.projected_residual import ProjectedResidual


# Identity-skip ResNet block (stride 1, spatial preserved).
#   y = ReLU( x + BN2(Conv2(ReLU(BN1(Conv1(x))))) )
# Pass P = (K-1)//2 so OH == H, OW == W (required by the inner Residual).
comptime ResBlockConv2DBN[
    C: Int, K: Int, P: Int, H: Int, W: Int,
    EPS: Float64 = BN2D_DEFAULT_EPS,
] = Sequential[
    Residual[
        Sequential[
            Conv2D[C, C, K, 1, P, H, W],
            BatchNorm2D[C, H, W, BN2D_DEFAULT_MOM, EPS],
            ReLU[C * H * W],
            Conv2D[C, C, K, 1, P, H, W],
            BatchNorm2D[C, H, W, BN2D_DEFAULT_MOM, EPS],
        ]
    ],
    ReLU[C * H * W],
]


# Downsampling ResNet block (stride-2 main path + 1×1-stride-2 BN skip).
#   y = ReLU( Skip(x) + BN2(Conv2(ReLU(BN1(Conv1_s2(x))))) )
# Canonical K=3, P=1: both paths map H → (H-1)//2 + 1.
comptime ResBlockDownsampleBN[
    IC: Int, OC: Int, K: Int, P: Int, H: Int, W: Int,
] = Sequential[
    ProjectedResidual[
        Sequential[
            Conv2D[IC, OC, K, 2, P, H, W],
            BatchNorm2D[OC, (H + 2 * P - K) // 2 + 1, (W + 2 * P - K) // 2 + 1],
            ReLU[
                OC
                * ((H + 2 * P - K) // 2 + 1)
                * ((W + 2 * P - K) // 2 + 1)
            ],
            Conv2D[
                OC, OC, K, 1, P,
                (H + 2 * P - K) // 2 + 1,
                (W + 2 * P - K) // 2 + 1,
            ],
            BatchNorm2D[OC, (H + 2 * P - K) // 2 + 1, (W + 2 * P - K) // 2 + 1],
        ],
        Sequential[
            Conv2D[IC, OC, 1, 2, 0, H, W],
            BatchNorm2D[OC, (H - 1) // 2 + 1, (W - 1) // 2 + 1],
        ],
    ],
    ReLU[OC * ((H + 2 * P - K) // 2 + 1) * ((W + 2 * P - K) // 2 + 1)],
]
