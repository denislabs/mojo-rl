"""LeWM configuration trait + concrete env configs.

Bundles the 20 comptime parameters that previously sprawled across
LeWMGPUState / LeWMTrainer / CEMPlanner / LeWMEvalSuite signatures into
a single `CONFIG: LeWMConfig` template arg. Mirrors the
`OffPolicyConfig` / `DDPGConfig` pattern from
`mojo_rl/deep_agents/core/configs/`.

The trait also carries an `EncoderModel: Model` field so the image
encoder can be swapped per-config without touching LeWMGPUState — e.g.,
ViT today, a CNN encoder in a future `LeWMPongCNNConfig`.

Concrete configs:
  - `LeWMPongViTConfig`  — Atari Pong (4-stack 84×84 grayscale, 14×14
    patches → 36 patches, ViT encoder).
  - `LeWMPushTViTConfig` — PushT pixels (RGB 224×224, 14×14 patches →
    256 patches, ViT encoder; matches the LeWM paper recipe).

Driver usage::

    train_lewm_offline_gpu[LeWMPongViTConfig[batch=16, depth=6]](
        buffer_path="...", num_steps=8000, ...
    )

Naming convention: struct params are snake_case (driver-facing); trait
fields are UPPERCASE (consumer-facing). The asymmetry is a Mojo
quirk — `comptime X = Self.X` is rejected as a redefinition when both
share a name, so we rename one side. The `DDPGConfig` pattern
(`OBS`/`obs_dim`) does the same trick.
"""

from ...nn.model import (
    Sequential,
    Linear,
    BatchNorm1D,
    Tokenwise,
    Model,
)
from ...nn.model.autodiff_layers import GELU

from .encoder import LeWMEncoder


# =============================================================================
# LeWMConfig trait
# =============================================================================


trait LeWMConfig(Movable, ImplicitlyDestructible):
    """Compile-time configuration for the LeWM offline trainer.

    Every concrete config exposes the 20 dimensional fields below plus a
    swappable `EncoderModel`. Consumers (LeWMGPUState, LeWMTrainer,
    CEMPlanner, LeWMEvalSuite) read each field via `Self.CONFIG.X`.
    """

    comptime NAME: String

    # ── Workload dimensions ──────────────────────────────────────────
    comptime BATCH: Int
    comptime T: Int          # window length on the observation axis
    comptime H: Int          # predictor context length (≤ T)
    comptime N_PREDS: Int    # # predicted steps per window

    # ── Pixel + action dimensions ────────────────────────────────────
    comptime IN_CH: Int
    comptime IMG: Int        # IMG × IMG square frames
    comptime PATCH: Int      # ViT patch size — encoder-specific but
                              # quoted here so trainer knows the shape
    comptime N_PATCHES: Int  # (IMG / PATCH) ** 2
    comptime ACT: Int        # raw action dim (one-hot for discrete envs)
    comptime SMOOTHED: Int   # action embedder hidden width

    # ── Latent + predictor ───────────────────────────────────────────
    comptime EMB: Int        # JEPA embedding dim
    comptime PROJ_H: Int     # projector hidden width
    comptime HIDDEN: Int     # encoder hidden width
    comptime ENC_HEADS: Int
    comptime ENC_LAYERS: Int
    comptime PRED_HEADS: Int
    comptime PRED_FF: Int    # predictor FFN hidden width
    comptime DEPTH: Int      # # stacked conditional blocks

    # ── SIGReg ───────────────────────────────────────────────────────
    comptime SIG_NUM_PROJ: Int
    comptime SIG_KNOTS: Int

    # ── Swappable network types ──────────────────────────────────────
    comptime EncoderModel: Model


# =============================================================================
# LeWMPongViTConfig — Atari Pong defaults (84×84 grayscale, 4-stack)
# =============================================================================


struct LeWMPongViTConfig[
    batch: Int = 4,
    t: Int = 4,
    h: Int = 3,
    n_preds: Int = 1,
    in_ch: Int = 4,
    img: Int = 84,
    patch: Int = 14,
    n_patches: Int = 36,
    hidden: Int = 32,
    enc_heads: Int = 2,
    enc_layers: Int = 1,
    emb: Int = 32,
    proj_h: Int = 64,
    act: Int = 3,
    smoothed: Int = 16,
    pred_heads: Int = 2,
    pred_ff: Int = 64,
    depth: Int = 2,
    sig_num_proj: Int = 64,
    sig_knots: Int = 5,
](LeWMConfig):
    """LeWM config for Atari Pong with the ViT encoder.

    Defaults match `examples/lewm/lewm_pong_pixel_train_gpu_smoke.mojo`
    (tiny smoke). Override per-driver, e.g.::

        train_lewm_offline_gpu[
            LeWMPongViTConfig[batch=16, t=6, depth=6, hidden=128, emb=128]
        ](...)
    """

    comptime NAME: String = "LeWM-Pong-ViT"

    comptime BATCH: Int = Self.batch
    comptime T: Int = Self.t
    comptime H: Int = Self.h
    comptime N_PREDS: Int = Self.n_preds
    comptime IN_CH: Int = Self.in_ch
    comptime IMG: Int = Self.img
    comptime PATCH: Int = Self.patch
    comptime N_PATCHES: Int = Self.n_patches
    comptime HIDDEN: Int = Self.hidden
    comptime ENC_HEADS: Int = Self.enc_heads
    comptime ENC_LAYERS: Int = Self.enc_layers
    comptime EMB: Int = Self.emb
    comptime PROJ_H: Int = Self.proj_h
    comptime ACT: Int = Self.act
    comptime SMOOTHED: Int = Self.smoothed
    comptime PRED_HEADS: Int = Self.pred_heads
    comptime PRED_FF: Int = Self.pred_ff
    comptime DEPTH: Int = Self.depth
    comptime SIG_NUM_PROJ: Int = Self.sig_num_proj
    comptime SIG_KNOTS: Int = Self.sig_knots

    comptime EncoderModel = LeWMEncoder[
        Self.in_ch, Self.img, Self.img, Self.patch,
        Self.hidden, Self.enc_heads, Self.enc_layers, Self.n_patches,
        Self.emb, 2, Self.proj_h,
    ]


# =============================================================================
# LeWMPushTViTConfig — PushT pixels (224×224 RGB) per LeWM paper recipe
# =============================================================================


struct LeWMPushTViTConfig[
    batch: Int = 4,
    t: Int = 4,
    h: Int = 3,
    n_preds: Int = 1,
    in_ch: Int = 3,
    img: Int = 224,
    patch: Int = 14,
    n_patches: Int = 256,
    hidden: Int = 96,
    enc_heads: Int = 4,
    enc_layers: Int = 2,
    emb: Int = 96,
    proj_h: Int = 256,
    act: Int = 10,         # FRAMESKIP(5) * ACTION_DIM(2)
    smoothed: Int = 32,
    pred_heads: Int = 4,
    pred_ff: Int = 256,
    depth: Int = 2,
    sig_num_proj: Int = 1024,
    sig_knots: Int = 17,
](LeWMConfig):
    """LeWM config for PushT pixels with the ViT encoder.

    Defaults match `examples/lewm/lewm_pusht_pixel_train_gpu_smoke.mojo`
    (tiny smoke). Override per-driver to match the paper recipe (batch=16,
    t=6, depth=6, hidden=192, emb=192, pred_heads=16, pred_ff=2048).
    """

    comptime NAME: String = "LeWM-PushT-ViT"

    comptime BATCH: Int = Self.batch
    comptime T: Int = Self.t
    comptime H: Int = Self.h
    comptime N_PREDS: Int = Self.n_preds
    comptime IN_CH: Int = Self.in_ch
    comptime IMG: Int = Self.img
    comptime PATCH: Int = Self.patch
    comptime N_PATCHES: Int = Self.n_patches
    comptime HIDDEN: Int = Self.hidden
    comptime ENC_HEADS: Int = Self.enc_heads
    comptime ENC_LAYERS: Int = Self.enc_layers
    comptime EMB: Int = Self.emb
    comptime PROJ_H: Int = Self.proj_h
    comptime ACT: Int = Self.act
    comptime SMOOTHED: Int = Self.smoothed
    comptime PRED_HEADS: Int = Self.pred_heads
    comptime PRED_FF: Int = Self.pred_ff
    comptime DEPTH: Int = Self.depth
    comptime SIG_NUM_PROJ: Int = Self.sig_num_proj
    comptime SIG_KNOTS: Int = Self.sig_knots

    comptime EncoderModel = LeWMEncoder[
        Self.in_ch, Self.img, Self.img, Self.patch,
        Self.hidden, Self.enc_heads, Self.enc_layers, Self.n_patches,
        Self.emb, 2, Self.proj_h,
    ]
