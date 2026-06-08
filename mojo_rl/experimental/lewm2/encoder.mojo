"""LeWM nn2 nets — encoder, action embedder, AR predictor, pred projector.

Pure `Sequential` aliases over existing nn2 composites + the Phase A/B
primitives. No new compute. Mirrors the legacy `experimental/lewm/`
architecture (docs/LEWM_PORT_PLAN.md §3) on nn2.

- `LeWMEncoder`  : ViT body (PatchEmbed → pos BiasAdd → N×TransformerBlock →
                   per-token LayerNorm → TokenMean) + JEPA projector MLP
                   (Linear → BatchNorm1D → GELU → Linear). Maps a flattened
                   image (B, IN_CH·IMG·IMG) → (B, EMB). Run at BATCH=B·T.
- `ActionEmbedder`: per-step Conv1d-k1 stack (Tokenwise LinearSwish×2 + Linear)
                   mapping (B, T·ACT) → (B, T·EMB).
- `ARPredictor`  : RepeatConditional[DEPTH, ConditionalTransformerBlock] over
                   the H-token context (the learnable position BiasAdd is a
                   separate node in the JEPA graph, applied to x before this —
                   it can't chain into an ARITY=2 module).
- `PredProj`     : per-token projector on the predictor output (Tokenwise).
"""

from ...nn2.combinators import Sequential, Repeat, Tokenwise, RepeatConditional
from ...nn2.composites import PatchEmbed, TransformerBlock
from ...nn2.primitives.linear import Linear
from ...nn2.primitives.linear_swish import LinearSwish
from ...nn2.primitives.batch_norm_1d import BatchNorm1D
from ...nn2.primitives.gelu import GELU
from ...nn2.primitives.layer_norm import LayerNorm
from ...nn2.primitives.token_mean import TokenMean
from ...nn2.primitives.bias_add import BiasAdd
from ...nn2.primitives.conditional_transformer_block import (
    ConditionalTransformerBlock,
)


# Projector MLP: (B, HIDDEN) → (B, EMB). BatchNorm1D matches the reference.
comptime LeWMProjector[
    HIDDEN: Int, PROJ_H: Int, EMB: Int
] = Sequential[
    Linear[HIDDEN, PROJ_H],
    BatchNorm1D[PROJ_H],
    GELU[PROJ_H],
    Linear[PROJ_H, EMB],
]


# Encoder: image → embedding. Run at effective BATCH = B·T (one image per row).
comptime LeWMEncoder[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    N_PATCHES: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    FF_MULT: Int = 4,
] = Sequential[
    PatchEmbed[IN_CH, IMG, IMG, PATCH, HIDDEN, N_PATCHES],
    BiasAdd[N_PATCHES * HIDDEN],
    Repeat[
        ENC_LAYERS,
        TransformerBlock[HIDDEN, ENC_HEADS, N_PATCHES, FF_MULT * HIDDEN, False],
    ],
    Tokenwise[N_PATCHES, LayerNorm[HIDDEN]],
    TokenMean[N_PATCHES, HIDDEN],
    LeWMProjector[HIDDEN, PROJ_H, EMB],
]


# Action embedder: (B, T·ACT) → (B, T·EMB). Conv1d-k1 ≡ per-token Linear.
#   LinearSwish (Linear+SiLU) ×2 then a bare Linear, all Tokenwise over T.
comptime ActionEmbedder[
    T: Int, ACT: Int, SMOOTHED: Int, EMB: Int, MLP_SCALE: Int = 4
] = Sequential[
    Tokenwise[T, LinearSwish[ACT, SMOOTHED]],
    Tokenwise[T, LinearSwish[SMOOTHED, MLP_SCALE * EMB]],
    Tokenwise[T, Linear[MLP_SCALE * EMB, EMB]],
]


# AR predictor: learnable position embedding over the H-token context, then a
# DEPTH-stack of AdaLN-zero conditional blocks. ARITY=2: forward(x, c) where x
# is the context embeddings (B, H·EMB) and c the action conditioning.
comptime ARPredictor[
    EMB: Int, HEADS: Int, H: Int, FF: Int, DEPTH: Int, HEAD_DIM: Int = 0
] = RepeatConditional[
    DEPTH, ConditionalTransformerBlock[EMB, HEADS, H, FF, HEAD_DIM]
]


# Predictor output projector: per-token MLP (B, H·EMB) → (B, H·EMB).
comptime PredProj[
    H: Int, EMB: Int, PROJ_H: Int
] = Tokenwise[
    H,
    Sequential[
        Linear[EMB, PROJ_H],
        BatchNorm1D[PROJ_H],
        GELU[PROJ_H],
        Linear[PROJ_H, EMB],
    ],
]
