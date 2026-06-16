"""REDQ-OFE named presets — Design F config descriptors + factories.

Same shape as `redq/config.mojo` with the OFE-specific extensions:

  1. `REDQOFEConfigT` — trait bundling REDQ comptime knobs PLUS the
     three OFE networks (`SB`, `AB`, `PRED`) and a tuned `DEF_OFE_LR`.
  2. `REDQOFE6Config` (6-layer branches, per_unit=40) and
     `REDQOFE8Config` (8-layer branches, per_unit=30) — zero-field
     comptime tags. Both bake N=2/M=2/UTD=1/POLICY_DELAY=1 as the
     default REDQ knobs, matching `SmallREDQConfig`'s SAC-shape
     regime — the cheapest knobs that demonstrate the OFE
     contribution (the aux feature pre-pass). Tune `N`/`UTD` upward
     via the primitive `REDQOFEAgent[…]` for paper-faithful runs.
  3. `agent_from_config_ofe` + capitalized `REDQOFE6` / `REDQOFE8`
     presets:

         var agent = REDQOFE6["cpu", OBS=11, ACT=3, BATCH=256, CAP=…](ctx=None)

Width math (matching `ofe_nets.mojo`):
  - 6 blocks, per_unit=40 → PHI_S_DIM = OBS + 240,
    PHI_SA_DIM = OBS + ACT + 480
  - 8 blocks, per_unit=30 → PHI_S_DIM = OBS + 240,
    PHI_SA_DIM = OBS + ACT + 480

(Both presets produce φ-width = 240 added features; the 6 vs 8 split
is the branch *depth* trade-off — 8 deeper layers, narrower per-block
growth.)
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.combinators.sequential import Sequential

from ..primitives.stochastic_actor import StochasticActor
from ..training.blocks import SampleBlock, ReplaySampleStep
from ..data.any_replay import AnyReplay

from .agent import REDQOFEAgent
from .ofe_nets import (
    OFEStateBranch6, OFEStateBranch8,
    OFEActionBranch6, OFEActionBranch8,
    OFEPredictorHead,
    state_branch_out_dim, action_branch_out_dim,
)
from ..redq.kernels import REDQ_TARGET_MIN, REDQ_TARGET_AVE


# ──────────────────────────────────────────────────────────────────────
# Net presets — actor/critic shaped against the φ-dim.
# ──────────────────────────────────────────────────────────────────────


comptime REDQOFEActor[
    PHI_S_DIM: Int, ACT: Int, HIDDEN: Int,
] = StochasticActor[
    PHI_S_DIM, ACT,
    LinearReLU[PHI_S_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
"""2-layer fused-MLP trunk + (μ, log σ) heads, consuming φ(s)."""


comptime REDQOFECritic[
    PHI_SA_DIM: Int, HIDDEN: Int,
] = Sequential[
    LinearReLU[PHI_SA_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""2-layer fused-MLP critic, consuming φ(s, a)."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — REDQ knobs + OFE networks + tuned defaults.
# ──────────────────────────────────────────────────────────────────────


trait REDQOFEConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a REDQ-OFE-family algorithm.
    Zero-field conformer convention (never instantiated; only the
    comptime members are read)."""

    comptime TARGET: StaticString
    comptime SAMPLE: SampleBlock
    comptime ACTOR: Module
    comptime CRITIC: Module
    comptime SB: Module
    comptime AB: Module
    comptime PRED: Module
    comptime N: Int
    comptime N_MIN: Int
    comptime UTD: Int
    comptime POLICY_DELAY: Int
    comptime Q_MODE: Int

    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_OFE_LR: Scalar[DT]
    comptime DEF_ALPHA_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_INIT_ALPHA: Scalar[DT]
    comptime DEF_TARGET_ENTROPY: Scalar[DT]
    comptime DEF_LEARNING_STARTS: Int


# ──────────────────────────────────────────────────────────────────────
# 6-block conformer.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct REDQOFE6Config[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 40,
](REDQOFEConfigT):
    """REDQ-OFE with 6 DenseBlocks per branch, per_unit=40.
    Reference HalfCheetah/Hopper/Walker2d shape (`OFENet-main/gins/`).
    Defaults to SAC-shape REDQ knobs (N=2/M=2/UTD=1/POLICY_DELAY=1)
    — the cheapest knobs that exercise the OFE contribution. For
    paper-faithful REDQ knobs (N=10/M=2/UTD=20/POLICY_DELAY=20) use
    `REDQOFETrainer[…]` directly."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime PHI_S_DIM = state_branch_out_dim(Self.OBS, 6, Self.PER_UNIT)
    comptime PHI_SA_DIM = action_branch_out_dim(
        Self.OBS, Self.ACT, 6, Self.PER_UNIT,
    )
    comptime ACTOR = REDQOFEActor[
        Self.PHI_S_DIM, Self.ACT, Self.HIDDEN,
    ]
    comptime CRITIC = REDQOFECritic[Self.PHI_SA_DIM, Self.HIDDEN]
    comptime SB = OFEStateBranch6[Self.OBS, Self.PER_UNIT]
    comptime AB = OFEActionBranch6[
        Self.PHI_S_DIM + Self.ACT, Self.PER_UNIT,
    ]
    comptime PRED = OFEPredictorHead[Self.PHI_SA_DIM, Self.OBS]
    comptime N = 2
    comptime N_MIN = 2
    comptime UTD = 1
    comptime POLICY_DELAY = 1
    comptime Q_MODE = REDQ_TARGET_MIN

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](1e-3)
    comptime DEF_OFE_LR = Scalar[DT](3e-4)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 1_000


# ──────────────────────────────────────────────────────────────────────
# 8-block conformer.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct REDQOFE8Config[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 30,
](REDQOFEConfigT):
    """REDQ-OFE with 8 DenseBlocks per branch, per_unit=30.
    Reference Ant/Humanoid shape (`OFENet-main/gins/`)."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime PHI_S_DIM = state_branch_out_dim(Self.OBS, 8, Self.PER_UNIT)
    comptime PHI_SA_DIM = action_branch_out_dim(
        Self.OBS, Self.ACT, 8, Self.PER_UNIT,
    )
    comptime ACTOR = REDQOFEActor[
        Self.PHI_S_DIM, Self.ACT, Self.HIDDEN,
    ]
    comptime CRITIC = REDQOFECritic[Self.PHI_SA_DIM, Self.HIDDEN]
    comptime SB = OFEStateBranch8[Self.OBS, Self.PER_UNIT]
    comptime AB = OFEActionBranch8[
        Self.PHI_S_DIM + Self.ACT, Self.PER_UNIT,
    ]
    comptime PRED = OFEPredictorHead[Self.PHI_SA_DIM, Self.OBS]
    comptime N = 2
    comptime N_MIN = 2
    comptime UTD = 1
    comptime POLICY_DELAY = 1
    comptime Q_MODE = REDQ_TARGET_MIN

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](1e-3)
    comptime DEF_OFE_LR = Scalar[DT](3e-4)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 1_000


# ──────────────────────────────────────────────────────────────────────
# Paper-faithful conformers — N=10 critics, M=2 subset, UTD=20,
# POLICY_DELAY=20. Use these for Ant / Humanoid scale where the
# OFE feature extractor + REDQ ensemble pays off.
#
# Same OFE architecture as `REDQOFE6Config` / `REDQOFE8Config`; only
# the REDQ critic-loop knobs change. The cost: ~20× per-env-step
# compute relative to UTD=1, but with the same per-step
# (state_branch + action_branch + predictor) cost — the OFE
# feature pre-pass is hoisted out of the inner UTD loop.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct LargeREDQOFE6Config[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 40,
](REDQOFEConfigT):
    """Paper-faithful REDQ knobs (N=10, M=2, UTD=20, POLICY_DELAY=20)
    + 6-block OFE branches. HalfCheetah/Hopper/Walker2d scale."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime PHI_S_DIM = state_branch_out_dim(Self.OBS, 6, Self.PER_UNIT)
    comptime PHI_SA_DIM = action_branch_out_dim(
        Self.OBS, Self.ACT, 6, Self.PER_UNIT,
    )
    comptime ACTOR = REDQOFEActor[
        Self.PHI_S_DIM, Self.ACT, Self.HIDDEN,
    ]
    comptime CRITIC = REDQOFECritic[Self.PHI_SA_DIM, Self.HIDDEN]
    comptime SB = OFEStateBranch6[Self.OBS, Self.PER_UNIT]
    comptime AB = OFEActionBranch6[
        Self.PHI_S_DIM + Self.ACT, Self.PER_UNIT,
    ]
    comptime PRED = OFEPredictorHead[Self.PHI_SA_DIM, Self.OBS]
    comptime N = 10
    comptime N_MIN = 2
    comptime UTD = 20
    comptime POLICY_DELAY = 20
    comptime Q_MODE = REDQ_TARGET_MIN

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](3e-4)
    comptime DEF_OFE_LR = Scalar[DT](3e-4)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 5_000


@fieldwise_init
struct LargeREDQOFE8Config[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 30,
](REDQOFEConfigT):
    """Paper-faithful REDQ knobs + 8-block OFE branches.
    Ant/Humanoid scale."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime PHI_S_DIM = state_branch_out_dim(Self.OBS, 8, Self.PER_UNIT)
    comptime PHI_SA_DIM = action_branch_out_dim(
        Self.OBS, Self.ACT, 8, Self.PER_UNIT,
    )
    comptime ACTOR = REDQOFEActor[
        Self.PHI_S_DIM, Self.ACT, Self.HIDDEN,
    ]
    comptime CRITIC = REDQOFECritic[Self.PHI_SA_DIM, Self.HIDDEN]
    comptime SB = OFEStateBranch8[Self.OBS, Self.PER_UNIT]
    comptime AB = OFEActionBranch8[
        Self.PHI_S_DIM + Self.ACT, Self.PER_UNIT,
    ]
    comptime PRED = OFEPredictorHead[Self.PHI_SA_DIM, Self.OBS]
    comptime N = 10
    comptime N_MIN = 2
    comptime UTD = 20
    comptime POLICY_DELAY = 20
    comptime Q_MODE = REDQ_TARGET_MIN

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](3e-4)
    comptime DEF_OFE_LR = Scalar[DT](3e-4)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 5_000


# ──────────────────────────────────────────────────────────────────────
# Generic factory.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config_ofe[
    CONFIG: REDQOFEConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    ofe_lr: Scalar[DT] = CONFIG.DEF_OFE_LR,
    alpha_lr: Scalar[DT] = CONFIG.DEF_ALPHA_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = CONFIG.DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = CONFIG.DEF_TARGET_ENTROPY,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
) raises -> REDQOFEAgent[
    CONFIG.TARGET,
    CONFIG.SAMPLE,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
    CONFIG.SB,
    CONFIG.AB,
    CONFIG.PRED,
    CONFIG.N,
    CONFIG.N_MIN,
    CONFIG.UTD,
    CONFIG.POLICY_DELAY,
    CONFIG.Q_MODE,
]:
    """Build the primitive `REDQOFEAgent` from any `REDQOFEConfigT`."""
    return REDQOFEAgent[
        CONFIG.TARGET,
        CONFIG.SAMPLE,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
        CONFIG.SB,
        CONFIG.AB,
        CONFIG.PRED,
        CONFIG.N,
        CONFIG.N_MIN,
        CONFIG.UTD,
        CONFIG.POLICY_DELAY,
        CONFIG.Q_MODE,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        ofe_lr=ofe_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha,
        target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized presets — `REDQOFE6` / `REDQOFE8` read like constructors.
# ──────────────────────────────────────────────────────────────────────


def REDQOFE6[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 40,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_CRITIC_LR,
    ofe_lr: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_OFE_LR,
    alpha_lr: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ALPHA_LR,
    gamma: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_GAMMA,
    tau: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TAU,
    action_scale: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TARGET_ENTROPY,
    learning_starts: Int = REDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
) raises -> REDQOFEAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    REDQOFEActor[
        state_branch_out_dim(OBS, 6, PER_UNIT), ACT, HIDDEN,
    ],
    REDQOFECritic[
        action_branch_out_dim(OBS, ACT, 6, PER_UNIT), HIDDEN,
    ],
    OFEStateBranch6[OBS, PER_UNIT],
    OFEActionBranch6[state_branch_out_dim(OBS, 6, PER_UNIT) + ACT, PER_UNIT],
    OFEPredictorHead[
        action_branch_out_dim(OBS, ACT, 6, PER_UNIT), OBS,
    ],
    2, 2, 1, 1, REDQ_TARGET_MIN,
]:
    """6-block REDQ-OFE preset. HalfCheetah/Hopper/Walker2d reference
    architecture per `OFENet-main/gins/`."""
    return agent_from_config_ofe[
        REDQOFE6Config[target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        ofe_lr=ofe_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha, target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
    )


def LargeREDQOFE6[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 40,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_CRITIC_LR,
    ofe_lr: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_OFE_LR,
    alpha_lr: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ALPHA_LR,
    gamma: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_GAMMA,
    tau: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TAU,
    action_scale: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TARGET_ENTROPY,
    learning_starts: Int = LargeREDQOFE6Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
) raises -> REDQOFEAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    REDQOFEActor[
        state_branch_out_dim(OBS, 6, PER_UNIT), ACT, HIDDEN,
    ],
    REDQOFECritic[
        action_branch_out_dim(OBS, ACT, 6, PER_UNIT), HIDDEN,
    ],
    OFEStateBranch6[OBS, PER_UNIT],
    OFEActionBranch6[state_branch_out_dim(OBS, 6, PER_UNIT) + ACT, PER_UNIT],
    OFEPredictorHead[
        action_branch_out_dim(OBS, ACT, 6, PER_UNIT), OBS,
    ],
    10, 2, 20, 20, REDQ_TARGET_MIN,
]:
    """Paper-faithful REDQ knobs (N=10/M=2/UTD=20/POLICY_DELAY=20)
    + 6-block OFE branches. HalfCheetah/Hopper/Walker2d scale."""
    return agent_from_config_ofe[
        LargeREDQOFE6Config[target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        ofe_lr=ofe_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha, target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
    )


def LargeREDQOFE8[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 30,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_CRITIC_LR,
    ofe_lr: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_OFE_LR,
    alpha_lr: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ALPHA_LR,
    gamma: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_GAMMA,
    tau: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TAU,
    action_scale: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TARGET_ENTROPY,
    learning_starts: Int = LargeREDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
) raises -> REDQOFEAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    REDQOFEActor[
        state_branch_out_dim(OBS, 8, PER_UNIT), ACT, HIDDEN,
    ],
    REDQOFECritic[
        action_branch_out_dim(OBS, ACT, 8, PER_UNIT), HIDDEN,
    ],
    OFEStateBranch8[OBS, PER_UNIT],
    OFEActionBranch8[state_branch_out_dim(OBS, 8, PER_UNIT) + ACT, PER_UNIT],
    OFEPredictorHead[
        action_branch_out_dim(OBS, ACT, 8, PER_UNIT), OBS,
    ],
    10, 2, 20, 20, REDQ_TARGET_MIN,
]:
    """Paper-faithful REDQ knobs + 8-block OFE branches.
    Ant/Humanoid scale."""
    return agent_from_config_ofe[
        LargeREDQOFE8Config[target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        ofe_lr=ofe_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha, target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
    )


def REDQOFE8[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
    PER_UNIT: Int = 30,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_CRITIC_LR,
    ofe_lr: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_OFE_LR,
    alpha_lr: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ALPHA_LR,
    gamma: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_GAMMA,
    tau: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TAU,
    action_scale: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_TARGET_ENTROPY,
    learning_starts: Int = REDQOFE8Config[
        target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
) raises -> REDQOFEAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    REDQOFEActor[
        state_branch_out_dim(OBS, 8, PER_UNIT), ACT, HIDDEN,
    ],
    REDQOFECritic[
        action_branch_out_dim(OBS, ACT, 8, PER_UNIT), HIDDEN,
    ],
    OFEStateBranch8[OBS, PER_UNIT],
    OFEActionBranch8[state_branch_out_dim(OBS, 8, PER_UNIT) + ACT, PER_UNIT],
    OFEPredictorHead[
        action_branch_out_dim(OBS, ACT, 8, PER_UNIT), OBS,
    ],
    2, 2, 1, 1, REDQ_TARGET_MIN,
]:
    """8-block REDQ-OFE preset. Ant/Humanoid reference architecture
    per `OFENet-main/gins/`."""
    return agent_from_config_ofe[
        REDQOFE8Config[target, OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        ofe_lr=ofe_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha, target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
    )
