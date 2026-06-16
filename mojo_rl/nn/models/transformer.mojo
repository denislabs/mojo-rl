"""Shared transformer building blocks (pre-LN GPT/ViT-style) for nn.

Direct port of nn/composites.mojo, minus the gen-1 `AutoDiffChain[...]`
wrappers — nn ops (Embedding, ScaledDotProductAttention) are already
Modules. All per-token sublayers (QKV/out projection, FFN, LayerNorm,
embedding, LM head) are wrapped in `Tokenwise[seq_len, ...]` so weights
are shared across positions; attention itself spans the full sequence.
`causal=False` → bidirectional (ViT); `causal=True` → decoder (GPT).
Input/output per sample: seq_len * dim.

These are the reusable, dropout-free pieces shared by both `models.gpt`
and `models.vit`:
  - `MultiHeadAttentionXL` / `MultiHeadAttention`
  - `TransformerFFN`
  - `TransformerBlock`
"""

from ..primitives.linear import Linear
from ..primitives.layer_norm import LayerNorm
from ..primitives.gelu import GELU
from ..primitives.attention import ScaledDotProductAttention
from ..primitives.qkv_to_major import QKVToMajor
from ..combinators.sequential import Sequential
from ..combinators.residual import Residual
from ..combinators.tokenwise import Tokenwise


# MultiHeadAttention: per-token QKV proj → attention → per-token out proj.
#
# CRITICAL layout note: ScaledDotProductAttention reads its input qkv-MAJOR —
# `[all-Q tokens | all-K tokens | all-V tokens]`. The QKV projection
# `Tokenwise[Linear[dim, 3*dim]]` produces token-MAJOR `[tok0:q,k,v | tok1:q,k,v
# | …]`. Feeding token-major straight into SDPA scrambles the position axis (the
# causal mask hits the wrong tokens → future leakage; this was the nn/gen-1 GPT
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
