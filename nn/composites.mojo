"""Pre-built composite model architectures (Phase 9).

These combine autodiff primitives, combinators, and Sequential into
ready-to-use model definitions. All conform to the Model trait and
work with Trainer, NetworkState, optimizers, and losses.

Usage:
    from nn.composites import ResBlock, ResNet, LeNet, FFN
"""

from .model.sequential import Sequential
from .autodiff import (
    AutoDiffChain,
    AutoFused,
    Dense,
    DenseReLU,
    DenseTanh,
    DenseSigmoid,
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    LayerNormOp,
    Flatten,
    Conv2D,
    MaxPool2D,
    Embedding,
    ScaledDotProductAttention,
    Residual,
    Parallel,
    Repeat,
)

# =============================================================================
# 9.1 ResNet Variants
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
# 9.2 Multi-Head Architectures
# =============================================================================

# Example 2-branch multi-head (for documentation; actual multi-head models
# use Parallel[...] directly since branch specs vary per use case).
#
# Usage pattern:
#   comptime MyMultiHead = Parallel[DenseReLU[in_d, 32], DenseTanh[in_d, 16]]
#   comptime MyClassifier = Sequential[MyMultiHead, DenseReLU[48, 32], Dense[32, out_d]]

# =============================================================================
# 9.3 CNN Architectures
# =============================================================================

# SimpleCNN: single conv layer + pool + dense output
#   Conv2D -> ReLU -> MaxPool -> Flatten -> Dense
#
# Parameters:
#   in_ch: input channels
#   in_h, in_w: input spatial dimensions
#   out_d: output dimension (e.g., number of classes)
#
# Note: comptime aliases with arithmetic expressions in type params work,
# but complex CNN architectures are better defined explicitly in user code
# to keep dimension calculations clear.

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

# =============================================================================
# 9.4 Transformer Architectures
# =============================================================================

# FFN: feed-forward network (two dense layers with ReLU)
comptime FFN[dim: Int, ff_dim: Int] = Sequential[
    DenseReLU[dim, ff_dim], Dense[ff_dim, dim]
]
