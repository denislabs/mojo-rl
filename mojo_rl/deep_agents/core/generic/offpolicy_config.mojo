"""Off-policy agent configuration trait and concrete configs.

The OffPolicyConfig trait defines the compile-time interface that every
off-policy algorithm config must provide. The generic OffPolicyAgent is
parameterized by a Config conforming to this trait.

Concrete configs (DDPGConfig, TD3Config, SACConfig) bundle:
  - Network architectures (ActorModel, CriticModel)
  - Optimizer types (ActorOpt, CriticOpt)
  - Algorithm flags (NUM_CRITICS, HAS_TARGET_ACTOR)
  - Strategy types (Explore, Schedule, TargetAction, TargetValue, ActorLoss)
  - Dimension constants (obs_dim, action_dim, batch_size, buffer_capacity)
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearTanh,
    Sequential,
    Parallel,
)
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import Network, NetworkState, NetworkPair
from mojo_rl.nn.initializer import Kaiming, Xavier

from .exploration import Explore, GaussianNoise, StochasticSample
from .update_schedule import Schedule, EveryStep, DelayedAll, DelayedActorOnly
from .target_value import (
    TargetValue,
    SingleQTarget,
    TwinQTarget,
    EntropicTwinQTarget,
)
from .target_action import (
    TargetAction,
    DeterministicTarget,
    SmoothedTarget,
    ReparamTarget,
)
from .actor_loss import ActorLoss, DPGLoss, MaxEntLoss, AutodiffMaxEntLoss


# =============================================================================
# OffPolicyConfig trait
# =============================================================================


trait OffPolicyConfig:
    """Compile-time configuration for off-policy agents.

    Every off-policy algorithm (DDPG, TD3, SAC) implements this trait
    by providing network types, optimizer types, algorithm flags, and
    composable strategy types.
    """

    comptime NAME: String

    # Dimensions
    comptime obs_dim: Int
    comptime action_dim: Int
    comptime batch_size: Int
    comptime buffer_capacity: Int

    # Network architectures and optimizers
    comptime ActorModel: Model
    comptime CriticModel: Model
    comptime ActorOpt: Optimizer
    comptime CriticOpt: Optimizer

    # Algorithm flags
    comptime NUM_CRITICS: Int
    comptime HAS_TARGET_ACTOR: Bool

    # Composable strategies
    comptime Explore: Explore
    comptime Schedule: Schedule
    comptime TargetAction: TargetAction
    comptime TargetValue: TargetValue
    comptime ActorLoss: ActorLoss


# =============================================================================
# DDPGConfig
# =============================================================================


struct DDPGConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 100000,
    BS: Int = 64,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
](OffPolicyConfig):
    """DDPG: single critic, deterministic target actor, every-step updates."""

    comptime NAME: String = "DDPG"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = 1
    comptime HAS_TARGET_ACTOR: Bool = True

    # Strategies
    comptime Explore = GaussianNoise[]
    comptime Schedule = EveryStep
    comptime TargetAction = DeterministicTarget
    comptime TargetValue = SingleQTarget
    comptime ActorLoss = DPGLoss


# =============================================================================
# TD3Config
# =============================================================================


struct TD3Config[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 100000,
    BS: Int = 64,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
](OffPolicyConfig):
    """TD3: twin critics, smoothed target, delayed actor+target updates."""

    comptime NAME: String = "TD3"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = 2
    comptime HAS_TARGET_ACTOR: Bool = True

    # Strategies
    comptime Explore = GaussianNoise[]
    comptime Schedule = DelayedAll
    comptime TargetAction = SmoothedTarget[]
    comptime TargetValue = TwinQTarget
    comptime ActorLoss = DPGLoss


# =============================================================================
# SACConfig
# =============================================================================


struct SACConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 100000,
    BS: Int = 64,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
](OffPolicyConfig):
    """SAC: stochastic actor (Parallel mean+log_std), twin critics, no target actor."""

    comptime NAME: String = "SAC"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    # SAC actor: Parallel output [mean(ACTIONS), log_std(ACTIONS)]
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],      # mean head
            LinearTanh[Self.HIDDEN, Self.ACT],   # log_std head (tanh-clamped)
        ],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = 2
    comptime HAS_TARGET_ACTOR: Bool = False

    # Strategies
    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = MaxEntLoss[]


struct AutodiffSACConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 100000,
    BS: Int = 64,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
](OffPolicyConfig):
    """SAC with autodiff-composed actor loss (no manual backward code).

    Identical to SACConfig but uses AutodiffMaxEntLoss for the actor update.
    The actor loss is expressed as a composed Model graph with automatic
    forward/backward via the autodiff system.

    Usage:
        from mojo_rl.deep_agents.core.generic import (
            GenericOffPolicyAgent, AutodiffSACConfig
        )
        var agent = GenericOffPolicyAgent[AutodiffSACConfig[17, 6]](...)
    """

    comptime NAME: String = "AutodiffSAC"
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
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = 2
    comptime HAS_TARGET_ACTOR: Bool = False

    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[]
