"""EfficientZero V2 network composites — comptime aliases.

Four building blocks that show up across the EZ-V2 representation, dynamics,
prediction, and SimSiam-consistency networks. Implemented as comptime
aliases over the standard `nn.model` layers + autodiff combinators, so they
participate in the existing AutoFused / Trainer / NetworkState machinery
without any new infrastructure.

Components (paper App. G):

  • `ImproveResidualBlock[DIM]` — Pre-LN residual block

        x ──► LayerNorm ──► Linear ──► ReLU ──► Linear ──► +x ──►

    The LN comes *before* the residual transform (transformer-style),
    which keeps activations bounded as we stack many blocks across the
    K-step unroll. Plain Post-LN (residual then LN) lets the unroll
    amplify activations exponentially, hence the rep network audit
    (Phase G post-mortem 2026-05-04) settled on Pre-LN here too.

  • `ActionEmbedding[ACT, EMBED=64]` — discrete one-hot → 64-d embed

        a (one-hot) ──► Linear[ACT, EMBED] ──► LayerNorm ──► ReLU ──►

    Used inside the dynamics network: `g(z, embed(a)) → z'`. EMBED=64
    matches paper Table 3 across all discrete envs.

  • `ProjectionMLP[HIDDEN, PROJ=1024]` — SimSiam projector

        h ──► Linear[HIDDEN, PROJ] ──► ReLU
              ──► Linear[PROJ, PROJ]   ──► ReLU
              ──► Linear[PROJ, PROJ]   ──► LayerNorm ──►

    The trailing LayerNorm is critical for SimSiam-style consistency: it
    forces the projection onto a unit-ish-norm manifold so the cosine
    consistency loss has a meaningful denominator and the network can't
    cheat by collapsing all features to zero magnitude.

  • `PredictionMLP[PROJ=1024, BOTTLENECK=512]` — SimSiam predictor

        p ──► Linear[PROJ, BOTTLENECK] ──► ReLU
              ──► Linear[BOTTLENECK, PROJ] ──►

    Asymmetric bottleneck (512 < 1024) — only applied on the *online*
    branch of the consistency loss (paper Eq. 4: stop-gradient on the
    target branch, predictor only on the dynamics branch). The
    asymmetry is the load-bearing collapse defence in SimSiam.

The reward-prefix LSTM head (paper App. G) is intentionally not defined
here yet: the EZ-V2 plan defers it until after CartPole converges with the
plain reward head (risk register row "Reward-prefix LSTM hidden reset every
5 steps adds another off-by-one").
"""

from mojo_rl.nn.model import (
    Sequential,
    Linear,
    LinearReLU,
    LayerNorm,
    ReLU,
    Residual,
)


# ═════════════════════════════════════════════════════════════════════════
# ImproveResidualBlock — Pre-LN residual
# ═════════════════════════════════════════════════════════════════════════

# x ↦ x + Linear(ReLU(Linear(LayerNorm(x))))
#
# `LinearReLU[DIM, DIM]` is `Linear + ReLU` fused into one autodiff op,
# matching the layout `Linear → ReLU → Linear` from the paper.
comptime ImproveResidualBlock[DIM: Int] = Residual[
    Sequential[
        LayerNorm[DIM],
        LinearReLU[DIM, DIM],
        Linear[DIM, DIM],
    ]
]


# ═════════════════════════════════════════════════════════════════════════
# ActionEmbedding — discrete one-hot → 64-d
# ═════════════════════════════════════════════════════════════════════════


comptime ActionEmbedding[
    ACT: Int,
    EMBED: Int = 64,
] = Sequential[
    Linear[ACT, EMBED],
    LayerNorm[EMBED],
    ReLU[EMBED],
]


# ═════════════════════════════════════════════════════════════════════════
# ProjectionMLP — SimSiam projector
# ═════════════════════════════════════════════════════════════════════════
#
# Reference shape (`ez_dmc_state.py:518-527`,
# `dmc_state.yaml: proj_hid_shape=512, proj_shape=128`):
#
#     Linear(HIDDEN, PROJ_HID) → LN → ReLU
#         → Linear(PROJ_HID, PROJ_HID) → LN → ReLU
#         → Linear(PROJ_HID, PROJ) → LN
#
# Reference uses inner width `proj_hid=512` and output width `proj=128`
# (expand-then-contract). Earlier "uniform PROJ" alone wasn't enough
# collapse defence on HalfCheetah: even with per-layer LN, the encoder
# transiently learned (`L_V` dropping to 0.41) then re-collapsed under
# the consistency-loss pull, leaving `L_V` oscillating around log(2).
# Adding the wider inner hidden gives the projector enough capacity to
# carry a non-trivial cosine alignment that's also state-discriminative.
#
# Per-layer LayerNorm before every ReLU is **load-bearing for SimSiam
# collapse defence**. The original SimSiam paper (Chen & He 2021, App.
# D) identifies BatchNorm/LayerNorm at every projector layer as the
# critical structural defence against the trivial all-same-direction
# fixed point. Earlier `LinearReLU` only had a single trailing LN, which
# let the encoder collapse: `L_G → -0.999` within ~250 train steps on
# HalfCheetah, dragging `L_V` / `L_R` to `log(2) = 0.69` (heads predicting
# marginal of the two-hot target since latents carried no state info).
# Found 2026-05-13.

comptime ProjectionMLP[
    HIDDEN: Int,
    PROJ: Int = 1024,
    PROJ_HID: Int = PROJ,
] = Sequential[
    Linear[HIDDEN, PROJ_HID],
    LayerNorm[PROJ_HID],
    ReLU[PROJ_HID],
    Linear[PROJ_HID, PROJ_HID],
    LayerNorm[PROJ_HID],
    ReLU[PROJ_HID],
    Linear[PROJ_HID, PROJ],
    LayerNorm[PROJ],
]


# ═════════════════════════════════════════════════════════════════════════
# PredictionMLP — SimSiam predictor (asymmetric bottleneck)
# ═════════════════════════════════════════════════════════════════════════
#
# Reference shape (`ez_dmc_state.py:528-533`):
#
#     Linear(PROJ, BOTTLENECK) → LN → ReLU → Linear(BOTTLENECK, PROJ)
#
# Same per-layer LN rationale as ProjectionMLP above — landed 2026-05-13.

comptime PredictionMLP[
    PROJ: Int = 1024,
    BOTTLENECK: Int = 512,
] = Sequential[
    Linear[PROJ, BOTTLENECK],
    LayerNorm[BOTTLENECK],
    ReLU[BOTTLENECK],
    Linear[BOTTLENECK, PROJ],
]


# ═════════════════════════════════════════════════════════════════════════
# RewardPrefixHeadMLP — post-LSTM MLP for the EZ-V1 reward-prefix head
# ═════════════════════════════════════════════════════════════════════════
#
# The full reward-prefix head is **LSTMCell + this MLP**, applied at every
# unroll step k:
#
#     (h[k], c[k]) = LSTMCell.step_forward(z_dyn[k], h[k-1], c[k-1])
#     reward_prefix_logits[k] = RewardPrefixHeadMLP(h[k])
#
# Hidden states `h, c` reset to zero every `lstm_horizon_len = 5` unroll
# steps (paper App. G) to cap BPTT depth.
#
# Loss is cross-entropy against `two_hot(scalar_transform(cumulative_
# reward_so_far))` — i.e. a *prefix sum* of rewards through step k, not
# the per-step reward that vanilla MuZero predicts. Reduces variance,
# better aligned with n-step return targets, kept from EfficientZero v1
# (Ye et al. 2021).
#
# `LSTMCell` lives in `mojo_rl.nn.model.lstm` and is *not* Model-trait
# (explicit (h, c) plumbing for BPTT use cases), so we don't try to
# wrap it in `Sequential[…]`. The head is assembled at call time:
# the agent runs `LSTMCell.step_forward_with_cache` then forwards the
# resulting `h_t` through this MLP.
#
# Wiring into the K-step train loop is deferred (paper-Eq.-3 reward
# target needs to switch from per-step to cumulative, and the agent's
# state struct needs `(h, c)` buffers). This file just provides the MLP
# building block.
comptime RewardPrefixHeadMLP[
    LSTM_HIDDEN: Int,
    MLP_HIDDEN: Int = 64,
    BINS: Int = 51,
] = Sequential[
    LinearReLU[LSTM_HIDDEN, MLP_HIDDEN],
    Linear[MLP_HIDDEN, BINS],
]
