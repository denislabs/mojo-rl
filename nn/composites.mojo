"""Pre-built composite model architectures.

These combine Model-level layers, combinators, and Sequential into
ready-to-use model definitions. All conform to the Model trait and
work with Trainer, NetworkState, optimizers, and losses.

Usage:
    from nn.composites import ResBlock, ResNet, LeNet8x8, NatureDQN, FFN
"""

from .model import (
    Sequential,
    Parallel,
    Residual,
    Repeat,
    Linear,
    LinearReLU,
    Conv2DReLU,
    MaxPoolLayer,
    FlattenLayer,
)

# =============================================================================
# ResNet Variants
# =============================================================================

# ResBlock: residual block with two dense layers
#   x -> LinearReLU[dim,dim] -> Linear[dim,dim] -> (+x) -> out
comptime ResBlock[dim: Int] = Residual[
    Sequential[LinearReLU[dim, dim], Linear[dim, dim]]
]

# ResNet: input projection + stacked residual blocks + output projection
#   LinearReLU[in_d, dim] -> Repeat[depth, ResBlock[dim]] -> Linear[dim, out_d]
comptime ResNet[in_d: Int, dim: Int, out_d: Int, depth: Int] = Sequential[
    LinearReLU[in_d, dim],
    Repeat[depth, ResBlock[dim]],
    Linear[dim, out_d],
]

# =============================================================================
# Multi-Head Architectures
# =============================================================================

# Example 2-branch multi-head (for documentation; actual multi-head models
# use Parallel[...] directly since branch specs vary per use case).
#
# Usage pattern:
#   comptime MyMultiHead = Parallel[LinearReLU[in_d, 32], LinearTanh[in_d, 16]]
#   comptime MyClassifier = Sequential[MyMultiHead, LinearReLU[48, 32], Linear[32, out_d]]

# =============================================================================
# CNN Architectures
# =============================================================================

# LeNet-5 style for 8x8 single-channel input (e.g., downsampled MNIST)
# Conv2DReLU[1,6,3,1,0,8,8] (8x8->6x6x6) -> MaxPool (->3x3x6)
# Conv2DReLU[6,16,3,1,0,3,3] (3x3->1x1x16)
# Flatten -> Linear[16,10]
comptime LeNet8x8 = Sequential[
    Conv2DReLU[1, 6, 3, 1, 0, 8, 8],
    MaxPoolLayer[6, 6, 6, 2],
    Conv2DReLU[6, 16, 3, 1, 0, 3, 3],
    FlattenLayer[16],
    Linear[16, 10],
]

# Nature DQN CNN (Mnih et al., 2015) for Atari-style 84x84 4-frame input
# Conv2DReLU[4,32,8,4,0,84,84] (84->20, 32ch)
# Conv2DReLU[32,64,4,2,0,20,20] (20->9, 64ch)
# Conv2DReLU[64,64,3,1,0,9,9] (9->7, 64ch)
# Flatten -> LinearReLU[3136, 512] -> Linear[512, out_d]
comptime NatureDQN[out_d: Int] = Sequential[
    Conv2DReLU[4, 32, 8, 4, 0, 84, 84],
    Conv2DReLU[32, 64, 4, 2, 0, 20, 20],
    Conv2DReLU[64, 64, 3, 1, 0, 9, 9],
    FlattenLayer[64 * 7 * 7],
    LinearReLU[64 * 7 * 7, 512],
    Linear[512, out_d],
]

# =============================================================================
# Feed-Forward Networks
# =============================================================================

# FFN: feed-forward network (two dense layers with ReLU)
comptime FFN[dim: Int, ff_dim: Int] = Sequential[
    LinearReLU[dim, ff_dim], Linear[ff_dim, dim]
]
