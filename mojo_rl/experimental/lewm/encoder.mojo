"""LeWM Encoder — ViT-derived image encoder + Projector head.

Maps a single image `(B, in_channels * img_h * img_w)` to a JEPA embedding
`(B, embed_dim)`. Reference uses ViT-tiny + a separate Projector MLP
(`references/le-wm-main/train.py:82-110`).

mojo-rl already has a CIFAR-validated `ViT` composite (Phase B in
`docs/TRANSFORMER_VIT.md`, ≥70% top-1). We reuse all of it *except* the
classification head: instead of `Linear[hidden, n_classes]`, we attach a
JEPA-style projector — `Linear[hidden, 2048] → BatchNorm1D → GELU →
Linear[2048, embed_dim]`. This matches the reference projector and lets the
Trainer's full-recipe AdamW+cosine schedule supervise it.

Encoder differences from reference:

  - TokenMean over patches (instead of CLS token). See
    `docs/LEWM_PORT_PLAN.md` §3.3 — accepted simplification for the POC.
  - No `interpolate_pos_encoding`: position embedding is a fixed-shape
    learnable `BiasAdd[n_patches * hidden_dim]`. Fine as long as we train
    + plan at the same image size.

The output's PARAM_SIZE absorbs the full ViT param count plus the projector
head; CACHE_SIZE / WORKSPACE_SIZE_PER_SAMPLE scale with depth × seq_len.

Usage (Phase 2 POC config — 32×32 images, tiny ViT, embed=128):

    comptime ENC = LeWMEncoder[
        in_channels=3, img_h=32, img_w=32, patch_size=4,
        hidden_dim=128, n_heads=4, n_layers=4, n_patches=64,
        embed_dim=128, ff_mult=4, projector_hidden=512,
    ]

The composite is a `Model`, so `NetworkState[ENC, ...]` etc. work directly.
"""

from ...nn.model import (
    Sequential,
    Linear,
    LayerNorm,
    BatchNorm1D,
    Tokenwise,
)
from ...nn.model.autodiff_layers import GELU, TokenMean, Transpose2D
from ...nn.model.conv2d_layer import Conv2DLayer
from ...nn.autodiff import AutoDiffChain
from ...nn.autodiff.primitives import BiasAdd


# =============================================================================
# PatchEmbed — mirror of `composites.PatchEmbed`. Inlined here so the encoder
# is self-contained and doesn't drag in the entire composites.mojo module
# (which also pulls Transformer / GPT / classifier-head aliases we don't
# need). Same wiring as the canonical ViT path.
# =============================================================================

comptime _LeWMPatchEmbed[
    in_channels: Int,
    img_h: Int,
    img_w: Int,
    patch_size: Int,
    hidden_dim: Int,
    n_patches: Int,
] = Sequential[
    Conv2DLayer[in_channels, hidden_dim, patch_size, patch_size, 0, img_h, img_w],
    Transpose2D[hidden_dim, n_patches],
]


# =============================================================================
# Projector MLP — replaces ViT's classification head.
#
#   Linear[hidden, projector_hidden] → BatchNorm1D → GELU → Linear[projector_hidden, embed_dim]
#
# BatchNorm1D matches reference (`norm_fn=torch.nn.BatchNorm1d` in
# `train.py`). At inference time the BN uses running stats — relevant when
# we use the encoder as a goal-image cost model in MPC.
# =============================================================================

comptime _LeWMProjector[
    hidden_dim: Int,
    projector_hidden: Int,
    embed_dim: Int,
] = Sequential[
    Linear[hidden_dim, projector_hidden],
    BatchNorm1D[projector_hidden],
    GELU[projector_hidden],
    Linear[projector_hidden, embed_dim],
]


# =============================================================================
# LeWMEncoder — full image-to-embedding pipeline.
#
# Layout:
#   1. PatchEmbed         : Conv2D + Transpose2D, channel-major → patch-major
#   2. Pos embedding      : learnable BiasAdd[n_patches * hidden_dim]
#   3. Transformer stack  : n_layers × non-causal TransformerBlock
#   4. Final per-token LN : Tokenwise[n_patches, LayerNorm[hidden_dim]]
#   5. TokenMean          : (B, n_patches * hidden_dim) → (B, hidden_dim)
#   6. Projector MLP      : (B, hidden_dim) → (B, embed_dim)
#
# We re-implement the transformer stack inline using the same
# `TransformerBlock` from the canonical ViT — non-causal, ff_mult=4 default,
# matches the existing nanoGPT-style init recipe used by `vit_cifar_training_gpu.mojo`.
# =============================================================================

from ...nn.composites import TransformerBlock
from ...nn.model import Repeat

comptime LeWMEncoder[
    in_channels: Int,
    img_h: Int,
    img_w: Int,
    patch_size: Int,
    hidden_dim: Int,
    n_heads: Int,
    n_layers: Int,
    n_patches: Int,
    embed_dim: Int,
    ff_mult: Int = 4,
    projector_hidden: Int = 2048,
] = Sequential[
    _LeWMPatchEmbed[in_channels, img_h, img_w, patch_size, hidden_dim, n_patches],
    AutoDiffChain[BiasAdd[n_patches * hidden_dim]],
    Repeat[
        n_layers,
        TransformerBlock[hidden_dim, n_heads, n_patches, ff_mult * hidden_dim, False],
        False,
    ],
    Tokenwise[n_patches, LayerNorm[hidden_dim]],
    TokenMean[n_patches, hidden_dim],
    _LeWMProjector[hidden_dim, projector_hidden, embed_dim],
]
