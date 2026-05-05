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


comptime ProjectionMLP[
    HIDDEN: Int,
    PROJ: Int = 1024,
] = Sequential[
    LinearReLU[HIDDEN, PROJ],
    LinearReLU[PROJ, PROJ],
    Linear[PROJ, PROJ],
    LayerNorm[PROJ],
]


# ═════════════════════════════════════════════════════════════════════════
# PredictionMLP — SimSiam predictor (asymmetric bottleneck)
# ═════════════════════════════════════════════════════════════════════════


comptime PredictionMLP[
    PROJ: Int = 1024,
    BOTTLENECK: Int = 512,
] = Sequential[
    LinearReLU[PROJ, BOTTLENECK],
    Linear[BOTTLENECK, PROJ],
]
