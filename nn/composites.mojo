"""Pre-built composite model architectures.

These combine autodiff primitives, combinators, and Sequential into
ready-to-use model definitions. All conform to the Model trait and
work with Trainer, NetworkState, optimizers, and losses.

Usage:
    from nn.composites import ResBlock, ResNet, LeNet, FFN
"""

from .model import Sequential, Parallel, Residual, Repeat
from .model import Linear, LinearReLU
from .autodiff import (
    AutoDiffChain,
    Dense,
    DenseReLU,
    DenseTanh,
    DenseSigmoid,
    MatMul,
    BiasAdd,
    ReLUOp,
    Flatten,
    Conv2D,
    MaxPool2D,
)

# =============================================================================
# ResNet Variants
# =============================================================================

# ResBlock: residual block with two dense layers
#   x -> DenseReLU[dim,dim] -> Dense[dim,dim] -> (+x) -> out
comptime ResBlock[dim: Int] = Residual[
    Sequential[DenseReLU[dim, dim], Dense[dim, dim]]
]

# ResNet: input projection + stacked residual blocks + output projection
#   DenseReLU[in_d, dim] -> Repeat[depth, ResBlock[dim]] -> Dense[dim, out_d]
comptime ResNet[in_d: Int, dim: Int, out_d: Int, depth: Int] = Sequential[
    DenseReLU[in_d, dim],
    Repeat[depth, ResBlock[dim]],
    Dense[dim, out_d],
]

# =============================================================================
# Multi-Head Architectures
# =============================================================================

# Example 2-branch multi-head (for documentation; actual multi-head models
# use Parallel[...] directly since branch specs vary per use case).
#
# Usage pattern:
#   comptime MyMultiHead = Parallel[DenseReLU[in_d, 32], DenseTanh[in_d, 16]]
#   comptime MyClassifier = Sequential[MyMultiHead, DenseReLU[48, 32], Dense[32, out_d]]

# =============================================================================
# CNN Architectures
# =============================================================================

# LeNet-5 style for 8x8 single-channel input (e.g., downsampled MNIST)
# Conv2D[1,6,3,1,0,8,8] (8x8->6x6x6) -> ReLU -> MaxPool (->3x3x6)
# Conv2D[6,16,3,1,0,3,3] (3x3->1x1x16) -> ReLU
# Flatten -> Dense[16,10]
comptime LeNet8x8 = Sequential[
    AutoDiffChain[
        Conv2D[1, 6, 3, 1, 0, 8, 8],
        ReLUOp[6 * 6 * 6],
        MaxPool2D[6, 6, 6, 2],
    ],
    AutoDiffChain[
        Conv2D[6, 16, 3, 1, 0, 3, 3],
        ReLUOp[16 * 1 * 1],
    ],
    AutoDiffChain[
        Flatten[16],
        MatMul[16, 10],
        BiasAdd[10],
    ],
]

# Nature DQN CNN (Mnih et al., 2015) for Atari-style 84×84 4-frame input
# Conv2D[4,32,8,4,0,84,84] (84→20, 32ch) → ReLU
# Conv2D[32,64,4,2,0,20,20] (20→9, 64ch) → ReLU
# Conv2D[64,64,3,1,0,9,9] (9→7, 64ch) → ReLU
# Flatten → Dense[3136, 512] → ReLU → Dense[512, out_d]
comptime NatureDQN[out_d: Int] = Sequential[
    AutoDiffChain[
        Conv2D[4, 32, 8, 4, 0, 84, 84],
        ReLUOp[32 * 20 * 20],
    ],
    AutoDiffChain[
        Conv2D[32, 64, 4, 2, 0, 20, 20],
        ReLUOp[64 * 9 * 9],
    ],
    AutoDiffChain[
        Conv2D[64, 64, 3, 1, 0, 9, 9],
        ReLUOp[64 * 7 * 7],
    ],
    AutoDiffChain[
        Flatten[64 * 7 * 7],
        MatMul[64 * 7 * 7, 512],
        BiasAdd[512],
        ReLUOp[512],
    ],
    AutoDiffChain[
        MatMul[512, out_d],
        BiasAdd[out_d],
    ],
]

# =============================================================================
# Feed-Forward Networks
# =============================================================================

# FFN: feed-forward network (two dense layers with ReLU)
comptime FFN[dim: Int, ff_dim: Int] = Sequential[
    DenseReLU[dim, ff_dim], Dense[ff_dim, dim]
]
