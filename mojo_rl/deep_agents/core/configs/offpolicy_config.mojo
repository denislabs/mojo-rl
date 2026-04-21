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

from ..strategies.exploration import Explore, GaussianNoise, StochasticSample
from ..strategies.update_schedule import Schedule, EveryStep, DelayedAll, DelayedActorOnly
from ..strategies.target_value import (
    TargetValue,
    SingleQTarget,
    TwinQTarget,
    EntropicTwinQTarget,
)
from ..strategies.target_action import (
    TargetAction,
    DeterministicTarget,
    SmoothedTarget,
    ReparamTarget,
)
from ..strategies.actor_loss import ActorLoss, DPGLoss, MaxEntLoss, AutodiffMaxEntLoss, AutodiffDPGLoss, AutodiffTD3Loss


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
    comptime ActorLoss = AutodiffDPGLoss


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
    comptime ActorLoss = AutodiffTD3Loss


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
    action_scale: Float64 = 1.0,
](OffPolicyConfig):
    """SAC: stochastic actor (Parallel mean+log_std), twin critics, no target actor.

    action_scale: Comptime output scale for the actor (a = action_scale * tanh(z)).
      Must match the runtime action_scale passed to agent __init__. Baked into
      the autodiff graph (AutodiffMaxEntLoss → RSample) so the critic sees
      actions on the same scale as the replay buffer.
    """

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
    comptime ActorLoss = AutodiffMaxEntLoss[action_scale=Self.action_scale]


# AutodiffSACConfig is now just an alias for SACConfig (which uses autodiff by default)
comptime AutodiffSACConfig = SACConfig


# AutodiffDDPGConfig is now just an alias for DDPGConfig (which uses autodiff by default)
comptime AutodiffDDPGConfig = DDPGConfig


# AutodiffTD3Config is now just an alias for TD3Config (which uses autodiff by default)
comptime AutodiffTD3Config = TD3Config
