"""Pre-built composite model architectures.

These combine Model-level layers, combinators, and Sequential into
ready-to-use model definitions. All conform to the Model trait and
work with Trainer, NetworkState, optimizers, and losses.

Usage:
    from mojo_rl.nn.composites import (
        ResBlock, ResNet, LeNet8x8, NatureDQN, FFN,
        MultiHeadAttention, TransformerFFN, TransformerBlock,
    )
"""

from .model import (
    Sequential,
    Parallel,
    Residual,
    Repeat,
    Tokenwise,
    Linear,
    LinearReLU,
    LayerNorm,
    Conv2DReLU,
    MaxPoolLayer,
    FlattenLayer,
)
from .autodiff import AutoDiffChain
from .autodiff.primitives import ScaledDotProductAttention, Embedding, BiasAdd

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

# =============================================================================
# Transformer
# =============================================================================
#
# Pre-LN GPT/ViT-style transformer block. Standard layout:
#
#   y = x + MHA(LayerNorm(x))
#   z = y + FFN(LayerNorm(y))
#
# All sublayers are per-token (shared weights across the seq_len positions),
# so QKV projections, output projection, FFN, and LayerNorm are all wrapped
# in `Tokenwise[seq_len, ...]`. Attention itself is a single op operating on
# the full sequence.
#
# Causal vs non-causal is selected at compile time by the `causal` flag on
# `ScaledDotProductAttention` (defaults to False = bidirectional / ViT). For
# decoder/GPT use, pass `causal=True`.
#
# Input/output (per sample): seq_len * dim.
# QKV projection: per-token Linear[dim, 3*dim] — same weights every position.
# Attention: ScaledDotProductAttention[dim, n_heads, seq_len, causal].
# Output projection: per-token Linear[dim, dim].
# FFN: two per-token Linears with ReLU between, hidden width ff_dim
#      (typically 4 * dim).

# MultiHeadAttention: per-token QKV projection → attention → per-token output projection.
# IN_DIM = OUT_DIM = seq_len * dim.
comptime MultiHeadAttention[
    dim: Int, n_heads: Int, seq_len: Int, causal: Bool = False
] = Sequential[
    Tokenwise[seq_len, Linear[dim, 3 * dim]],
    AutoDiffChain[ScaledDotProductAttention[dim, n_heads, seq_len, causal]],
    Tokenwise[seq_len, Linear[dim, dim]],
]

# TransformerFFN: per-token feed-forward network for use inside a transformer.
# Same as FFN[dim, ff_dim] but applied tokenwise (shared weights across positions).
# IN_DIM = OUT_DIM = seq_len * dim.
comptime TransformerFFN[seq_len: Int, dim: Int, ff_dim: Int] = Sequential[
    Tokenwise[seq_len, LinearReLU[dim, ff_dim]],
    Tokenwise[seq_len, Linear[ff_dim, dim]],
]

# TransformerBlock: pre-LN transformer encoder/decoder layer.
#   y = x + MHA(LN(x))
#   z = y + FFN(LN(y))
# IN_DIM = OUT_DIM = seq_len * dim.
comptime TransformerBlock[
    dim: Int,
    n_heads: Int,
    seq_len: Int,
    ff_dim: Int,
    causal: Bool = False,
] = Sequential[
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim]],
            MultiHeadAttention[dim, n_heads, seq_len, causal],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim]],
            TransformerFFN[seq_len, dim, ff_dim],
        ]
    ],
]

# =============================================================================
# GPT — char-level or token-level decoder transformer
# =============================================================================
#
# Standard GPT-2 style decoder stack. Pre-LN, causal attention by default,
# learnable token + position embeddings.
#
# Input layout per sample: seq_len * vocab one-hots (one-hot per token, row-
# major). For a 4-token sample with vocab=65, the input is laid out as
# [token_0_one_hot[0..65], token_1_one_hot[0..65], ...]. This matches the
# Tokenwise reinterpretation: at the (BATCH * seq_len, vocab) view, each row
# is one one-hot vector that the underlying Embedding op looks up to a dim-
# vector embedding.
#
# Output: seq_len * vocab logits per sample. Each token position has its own
# logit row over the vocabulary, ready for cross-entropy against the next-
# token target.
#
# Pipeline:
#   1. Per-token Embedding[vocab, dim]                  — token embedding
#   2. BiasAdd[seq_len * dim]                           — position embedding (learnable)
#   3. n_layers × TransformerBlock(causal=True)         — transformer stack
#   4. Per-token LayerNorm[dim]                         — final pre-LN
#   5. Per-token Linear[dim, vocab]                     — language-modelling head
#
# `causal=True` is the default; pass causal=False for prefix-LM-style models.
# `ff_mult=4` matches GPT-2; the FFN inner dim becomes 4 * embed_dim.
comptime GPT[
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int = 4,
    causal: Bool = True,
] = Sequential[
    Tokenwise[seq_len, AutoDiffChain[Embedding[vocab, embed_dim]]],
    AutoDiffChain[BiasAdd[seq_len * embed_dim]],
    Repeat[
        n_layers,
        TransformerBlock[embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal],
        False,
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim]],
    Tokenwise[seq_len, Linear[embed_dim, vocab]],
]
