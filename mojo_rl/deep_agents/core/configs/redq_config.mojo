"""REDQ (Randomized Ensembled Double Q-learning) configuration.

REDQ = SAC with:
  (1) N Q-networks instead of 2 (default N = 10)
  (2) TD target uses min over a random subset of M (default M = 2) of the N
  (3) UTD (update-to-data) ratio: default 20 gradient updates per env step
  (4) Policy (and alpha) updated only every POLICY_DELAY critic updates

The policy loss uses the mean over all N online critics (not the subset).
All other SAC machinery (tanh-Gaussian actor, reparameterization, entropy
temperature autotuning) is unchanged.

Reference: Chen et al., "Randomized Ensembled Double Q-Learning: Learning
Fast Without a Model" (ICLR 2021).
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearTanh,
    Sequential,
    Parallel,
    LayerNorm,
    ReLU,
)
from mojo_rl.nn.optimizer import Optimizer, Adam

from .offpolicy_config import OffPolicyConfig
from ..strategies.exploration import Explore, StochasticSample
from ..strategies.update_schedule import Schedule, DelayedActorOnly
from ..strategies.target_value import TargetValue, EntropicTwinQTarget
from ..strategies.target_action import TargetAction, ReparamTarget
from ..strategies.actor_loss import ActorLoss, AutodiffMaxEntLoss


# =============================================================================
# Q-target combination modes
# =============================================================================
# Runtime-integer mode selectors passed to the REDQ ensemble-target kernel.
#   MIN (0): min over a random subset of M of N  (default REDQ)
#   AVE (1): mean over all N critics             (Ensemble Average)
#   REM (2): random ensemble mixture (convex)    (REM)


comptime REDQ_TARGET_MIN: Int = 0
comptime REDQ_TARGET_AVE: Int = 1
comptime REDQ_TARGET_REM: Int = 2


# =============================================================================
# REDQConfig trait — extends OffPolicyConfig with ensemble + UTD + policy delay
# =============================================================================


trait REDQConfig(OffPolicyConfig):
    """Compile-time configuration for REDQ agents.

    Extends OffPolicyConfig with the four REDQ-specific knobs. The
    inherited `NUM_CRITICS` is set to `NUM_ENSEMBLE` so existing
    N-generic infrastructure (CriticGroup, OffPolicyTrainWS) works
    unchanged.
    """

    # Ensemble configuration
    comptime NUM_ENSEMBLE: Int       # Total number of Q-networks (e.g. 10)
    comptime NUM_MIN: Int            # Subset size for target min (e.g. 2)

    # High-UTD training
    comptime UTD_RATIO: Int          # Gradient updates per env step (e.g. 20)
    comptime POLICY_DELAY: Int       # Policy updates every K critic updates

    # Q-target combination mode: REDQ_TARGET_MIN / _AVE / _REM
    comptime Q_TARGET_MODE: Int


# =============================================================================
# DefaultREDQConfig — concrete config with paper-faithful defaults
# =============================================================================


struct DefaultREDQConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 1_000_000,
    BS: Int = 256,
    N_ENS: Int = 10,
    N_MIN: Int = 2,
    UTD: Int = 20,
    POL_DELAY: Int = 20,
    Q_MODE: Int = REDQ_TARGET_MIN,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    action_scale: Float64 = 1.0,
](REDQConfig):
    """Paper-faithful REDQ config: 10 critics, subset-min target (M=2),
    UTD ratio 20, policy update delay 20. Critic is a standard 2-hidden
    MLP (Linear + ReLU). For a LayerNorm-stabilized critic, use
    `DefaultREDQLNConfig` instead.
    """

    comptime NAME: String = "REDQ"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    # SAC actor: Parallel output [mean(ACT), log_std(ACT)]
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],       # mean head
            LinearTanh[Self.HIDDEN, Self.ACT],    # log_std head (tanh-clamped)
        ],
    ]

    # Paper-faithful critic: Linear + ReLU × 2 + Linear head.
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]

    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    # NUM_CRITICS (from OffPolicyConfig) is aliased to NUM_ENSEMBLE so that
    # N-generic infrastructure (CriticGroup, OffPolicyTrainWS) sizes correctly.
    comptime NUM_CRITICS: Int = Self.N_ENS
    comptime HAS_TARGET_ACTOR: Bool = False

    # Strategy fields from OffPolicyConfig — REDQ's custom training loop
    # does NOT dispatch through these, but they must be set for trait
    # conformance. We point them at SAC's strategies so any reuse of
    # OffPolicyAgent sub-routines (e.g. exploration) still works.
    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[action_scale=Self.action_scale]

    # REDQ-specific fields
    comptime NUM_ENSEMBLE: Int = Self.N_ENS
    comptime NUM_MIN: Int = Self.N_MIN
    comptime UTD_RATIO: Int = Self.UTD
    comptime POLICY_DELAY: Int = Self.POL_DELAY
    comptime Q_TARGET_MODE: Int = Self.Q_MODE


# =============================================================================
# DefaultREDQLNConfig — opt-in LayerNorm critic for extra high-UTD stability
# =============================================================================


struct DefaultREDQLNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 1_000_000,
    BS: Int = 256,
    N_ENS: Int = 10,
    N_MIN: Int = 2,
    UTD: Int = 20,
    POL_DELAY: Int = 20,
    Q_MODE: Int = REDQ_TARGET_MIN,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    action_scale: Float64 = 1.0,
](REDQConfig):
    """REDQ with pre-activation LayerNorm on the critic (non-paper-faithful).

    Pattern: Linear → LayerNorm → ReLU × 2 + Linear head. Borrowed from
    MBPO/SR-SAC to bound Q-activation magnitudes under high UTD. REDQ's
    big ensemble already damps overestimation so LayerNorm is a belt-and-
    braces option, not a requirement.
    """

    comptime NAME: String = "REDQ-LN"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            LinearTanh[Self.HIDDEN, Self.ACT],
        ],
    ]

    comptime CriticModel = Sequential[
        Linear[Self.OBS + Self.ACT, Self.HIDDEN],
        LayerNorm[Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, Self.HIDDEN],
        LayerNorm[Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]

    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = Self.N_ENS
    comptime HAS_TARGET_ACTOR: Bool = False

    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[action_scale=Self.action_scale]

    comptime NUM_ENSEMBLE: Int = Self.N_ENS
    comptime NUM_MIN: Int = Self.N_MIN
    comptime UTD_RATIO: Int = Self.UTD
    comptime POLICY_DELAY: Int = Self.POL_DELAY
    comptime Q_TARGET_MODE: Int = Self.Q_MODE
