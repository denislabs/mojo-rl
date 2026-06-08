"""Pre-built convolutional / residual block aliases for nn2.

These are *compositional* aliases — each expands to a `Sequential` /
`Residual` / `ProjectedResidual` of existing primitives, so they get
correct forward / vjp / walkers / `set_attr` propagation for free and
need no bespoke kernels. (Hand-fused single-kernel variants can later
drop in behind the same names; see `FusedConv2DBatchNormReLU` in
`composites_fused.mojo` once added.)

Spatial-shape convention (matches `Conv2D`):
    OH = (H + 2*P - K) // S + 1
    OW = (W + 2*P - K) // S + 1

Provided blocks:
  - `Conv2DReLU[IC, OC, K, S, P, H, W]`           Conv → ReLU
  - `Conv2DBatchNormReLU[IC, OC, K, S, P, H, W]`  Conv → BN → ReLU
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

from .primitives.conv2d import Conv2D
from .primitives.batch_norm_2d import BatchNorm2D
from .primitives.relu import ReLU
from .primitives.linear import Linear
from .primitives.tied_linear import TiedLinear
from .primitives.layer_norm import LayerNorm
from .primitives.gelu import GELU
from .primitives.embedding import Embedding
from .primitives.bias_add import BiasAdd
from .primitives.transpose_2d import Transpose2D
from .primitives.token_mean import TokenMean
from .primitives.attention import ScaledDotProductAttention
from .primitives.dropout import Dropout
from .combinators.sequential import Sequential
from .combinators.residual import Residual
from .combinators.projected_residual import ProjectedResidual
from .combinators.repeat import Repeat
from .combinators.tokenwise import Tokenwise
from .primitives.qkv_to_major import QKVToMajor


# ──────────────────────────────────────────────────────────────────────
# Conv → (BN) → ReLU
# ──────────────────────────────────────────────────────────────────────


comptime Conv2DReLU[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
] = Sequential[
    Conv2D[IC, OC, K, S, P, H, W],
    ReLU[OC * ((H + 2 * P - K) // S + 1) * ((W + 2 * P - K) // S + 1)],
]


comptime Conv2DBatchNormReLU[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
] = Sequential[
    Conv2D[IC, OC, K, S, P, H, W],
    BatchNorm2D[OC, (H + 2 * P - K) // S + 1, (W + 2 * P - K) // S + 1],
    ReLU[OC * ((H + 2 * P - K) // S + 1) * ((W + 2 * P - K) // S + 1)],
]


# ──────────────────────────────────────────────────────────────────────
# Residual blocks
# ──────────────────────────────────────────────────────────────────────


# Identity-skip ResNet block (stride 1, spatial preserved).
#   y = ReLU( x + BN2(Conv2(ReLU(BN1(Conv1(x))))) )
# Pass P = (K-1)//2 so OH == H, OW == W (required by the inner Residual).
comptime ResBlockConv2DBN[
    C: Int, K: Int, P: Int, H: Int, W: Int,
] = Sequential[
    Residual[
        Sequential[
            Conv2D[C, C, K, 1, P, H, W],
            BatchNorm2D[C, H, W],
            ReLU[C * H * W],
            Conv2D[C, C, K, 1, P, H, W],
            BatchNorm2D[C, H, W],
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


# ──────────────────────────────────────────────────────────────────────
# Transformer (pre-LN GPT/ViT-style)
# ──────────────────────────────────────────────────────────────────────
#
# Direct port of nn/composites.mojo, minus the gen-1 `AutoDiffChain[...]`
# wrappers — nn2 ops (Embedding, ScaledDotProductAttention) are already
# Modules. All per-token sublayers (QKV/out projection, FFN, LayerNorm,
# embedding, LM head) are wrapped in `Tokenwise[seq_len, ...]` so weights
# are shared across positions; attention itself spans the full sequence.
# `causal=False` → bidirectional (ViT); `causal=True` → decoder (GPT).
# Input/output per sample: seq_len * dim.


# MultiHeadAttention: per-token QKV proj → attention → per-token out proj.
#
# CRITICAL layout note: ScaledDotProductAttention reads its input qkv-MAJOR —
# `[all-Q tokens | all-K tokens | all-V tokens]`. The QKV projection
# `Tokenwise[Linear[dim, 3*dim]]` produces token-MAJOR `[tok0:q,k,v | tok1:q,k,v
# | …]`. Feeding token-major straight into SDPA scrambles the position axis (the
# causal mask hits the wrong tokens → future leakage; this was the nn2/gen-1 GPT
# bug). `QKVToMajor[seq_len, dim]` rearranges token-major → qkv-major in between.
#
# `use_max` selects the SDPA GPU path: True (default) = batched-GEMM; False =
# serial custom kernels (bit-identical; CPU ignores it).
# MultiHeadAttentionXL: expanded ("XL") attention with an INDEPENDENT
# head_dim, decoupled from `dim`. The QKV/out projections + SDPA run at the
# inner width `n_heads * head_dim` (can exceed `dim`), then project back to
# `dim`. The legacy LeWM predictor used this (16 heads × 64 = 1024 inner over
# emb=192). `MultiHeadAttention` below is the special case head_dim=dim/n_heads
# (inner == dim), so this is a strict generalization — bit-identical there.
comptime MultiHeadAttentionXL[
    dim: Int, n_heads: Int, head_dim: Int, seq_len: Int,
    causal: Bool = False, use_max: Bool = True,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, 3 * n_heads * head_dim]],
    QKVToMajor[seq_len, n_heads * head_dim],
    ScaledDotProductAttention[
        n_heads * head_dim, n_heads, seq_len, causal, use_max
    ],
    Tokenwise[seq_len, Linear[n_heads * head_dim, dim]],
]


comptime MultiHeadAttention[
    dim: Int, n_heads: Int, seq_len: Int, causal: Bool = False,
    use_max: Bool = True,
] = MultiHeadAttentionXL[
    dim, n_heads, dim // n_heads, seq_len, causal, use_max
]


# TransformerFFN: per-token Linear → GELU → per-token Linear.
# GELU is pointwise, so applying it to the flat (BATCH, seq_len*ff_dim)
# tensor is identical to per-token — no Tokenwise wrapper needed.
comptime TransformerFFN[seq_len: Int, dim: Int, ff_dim: Int] = Sequential[
    Tokenwise[seq_len, Linear[dim, ff_dim]],
    GELU[seq_len * ff_dim],
    Tokenwise[seq_len, Linear[ff_dim, dim]],
]


# TransformerBlock: pre-LN encoder/decoder layer.
#   y = x + MHA(LN(x));  z = y + FFN(LN(y)).  IN_DIM == OUT_DIM == seq_len*dim.
comptime TransformerBlock[
    dim: Int, n_heads: Int, seq_len: Int, ff_dim: Int, causal: Bool = False,
    use_max: Bool = True,
] = Sequential[
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim]],
            MultiHeadAttention[dim, n_heads, seq_len, causal, use_max],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim]],
            TransformerFFN[seq_len, dim, ff_dim],
        ]
    ],
]


# GPT: token Embedding → learnable position BiasAdd → N×TransformerBlock
#      (causal) → final LayerNorm → LM head. Input: seq_len one-hots of
#      width vocab; output: seq_len * vocab logits.
comptime GPT[
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int = 4,
    causal: Bool = True,
    use_max: Bool = True,
] = Sequential[
    Tokenwise[seq_len, Embedding[vocab, embed_dim]],
    BiasAdd[seq_len * embed_dim],
    Repeat[
        n_layers,
        TransformerBlock[
            embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal, use_max
        ],
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim]],
    Tokenwise[seq_len, Linear[embed_dim, vocab]],
]


# PatchEmbed: image → patch tokens.
#   Conv2D (channel-major patches) → Transpose2D (→ patch-major).
#   (BATCH, in_channels*img_h*img_w) → (BATCH, n_patches*embed_dim).
comptime PatchEmbed[
    in_channels: Int,
    img_h: Int,
    img_w: Int,
    patch_size: Int,
    embed_dim: Int,
    n_patches: Int,
] = Sequential[
    Conv2D[in_channels, embed_dim, patch_size, patch_size, 0, img_h, img_w],
    Transpose2D[embed_dim, n_patches],
]


# ViT: Vision Transformer encoder + classification head (non-causal).
#   PatchEmbed → position BiasAdd → N×TransformerBlock → LayerNorm →
#   TokenMean (mean-pool patches) → Linear head.
comptime ViT[
    in_channels: Int,
    img_h: Int,
    img_w: Int,
    patch_size: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    n_patches: Int,
    n_classes: Int,
    ff_mult: Int = 4,
    use_max: Bool = True,
] = Sequential[
    PatchEmbed[in_channels, img_h, img_w, patch_size, embed_dim, n_patches],
    BiasAdd[n_patches * embed_dim],
    Repeat[
        n_layers,
        TransformerBlock[
            embed_dim, n_heads, n_patches, ff_mult * embed_dim, False, use_max
        ],
    ],
    Tokenwise[n_patches, LayerNorm[embed_dim]],
    TokenMean[n_patches, embed_dim],
    Linear[embed_dim, n_classes],
]


# ──────────────────────────────────────────────────────────────────────
# Transformer + Dropout variants (nanoGPT-style)
# ──────────────────────────────────────────────────────────────────────
#
# Same shapes as the plain transformer aliases, with the three nanoGPT
# dropout points: (1) after token+position embedding, (2) after MHA's output
# projection, (3) after the FFN's output projection. nn2 `Dropout[DIM, p,
# SEED]` carries a host-side per-forward counter so the mask refreshes each
# step; toggle eval mode with `net.set_attr["training"](0.0)` (and back to
# 1.0 for training) — propagates to every Dropout leaf, ignored elsewhere.
#
# Within a block MHA-dropout uses seed_base+1, FFN-dropout seed_base+2, input
# dropout seed_base+0. Repeat reuses one TransformerBlockDrop type, so all
# layers share these seeds (masks correlated across depth — same as gen-1;
# the per-step counter still refreshes them each iteration). Defined as
# separate aliases so the plain `GPT`/etc. callers are unaffected.


# MultiHeadAttentionDrop: MHA + output-projection dropout. QKVToMajor fixes the
# token-major→qkv-major layout (see MultiHeadAttention note above for why).
comptime MultiHeadAttentionDrop[
    dim: Int, n_heads: Int, seq_len: Int, causal: Bool,
    dropout_p: Float64, seed: UInt64, use_max: Bool = True,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, 3 * dim]],
    QKVToMajor[seq_len, dim],
    ScaledDotProductAttention[dim, n_heads, seq_len, causal, use_max],
    Tokenwise[seq_len, Linear[dim, dim]],
    Dropout[seq_len * dim, dropout_p, seed],
]


# TransformerFFNDrop: TransformerFFN + output dropout.
comptime TransformerFFNDrop[
    seq_len: Int, dim: Int, ff_dim: Int, dropout_p: Float64, seed: UInt64,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, ff_dim]],
    GELU[seq_len * ff_dim],
    Tokenwise[seq_len, Linear[ff_dim, dim]],
    Dropout[seq_len * dim, dropout_p, seed],
]


# TransformerBlockDrop: pre-LN block with MHA + FFN sublayer dropout.
comptime TransformerBlockDrop[
    dim: Int, n_heads: Int, seq_len: Int, ff_dim: Int, causal: Bool,
    dropout_p: Float64, seed_base: UInt64, use_max: Bool = True,
] = Sequential[
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim]],
            MultiHeadAttentionDrop[
                dim, n_heads, seq_len, causal, dropout_p,
                seed_base + UInt64(1), use_max,
            ],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim]],
            TransformerFFNDrop[
                seq_len, dim, ff_dim, dropout_p, seed_base + UInt64(2)
            ],
        ]
    ],
]


# GPTDrop: GPT with the three dropout points (input, MHA-out, FFN-out).
comptime GPTDrop[
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int = 4,
    causal: Bool = True,
    dropout_p: Float64 = 0.2,
    seed_base: UInt64 = UInt64(0xC0FFEE),
    use_max: Bool = True,
] = Sequential[
    Tokenwise[seq_len, Embedding[vocab, embed_dim]],
    BiasAdd[seq_len * embed_dim],
    Dropout[seq_len * embed_dim, dropout_p, seed_base],
    Repeat[
        n_layers,
        TransformerBlockDrop[
            embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal,
            dropout_p, seed_base, use_max,
        ],
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim]],
    Tokenwise[seq_len, Linear[embed_dim, vocab]],
]


# GPTDropTied: GPTDrop with the LM head WEIGHT-TIED to the token embedding
# (nanoGPT's `lm_head.weight = wte.weight`). Identical to `GPTDrop` except
# the final projection is a bias-less `TiedLinear` that borrows the
# embedding's `[vocab, embed]` table (used transposed) instead of owning a
# separate `Linear[embed, vocab]`. After `make`, call `gpt_wire_tie` (see
# `composites_gpt.mojo`) once to point the head at the embedding's buffers;
# then the standard `Trainer.train_*` loop trains it with no per-step tying
# code — the shared weight gets one gradient (both leaves accumulate into
# it) and one optimizer update (reflection sees it only via the embedding).
comptime GPTDropTied[
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int = 4,
    causal: Bool = True,
    dropout_p: Float64 = 0.2,
    seed_base: UInt64 = UInt64(0xC0FFEE),
    use_max: Bool = True,
] = Sequential[
    Tokenwise[seq_len, Embedding[vocab, embed_dim]],
    BiasAdd[seq_len * embed_dim],
    Dropout[seq_len * embed_dim, dropout_p, seed_base],
    Repeat[
        n_layers,
        TransformerBlockDrop[
            embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal,
            dropout_p, seed_base, use_max,
        ],
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim]],
    Tokenwise[seq_len, TiedLinear[embed_dim, vocab]],
]
