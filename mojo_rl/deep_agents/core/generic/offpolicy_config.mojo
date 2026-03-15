"""Off-policy agent configuration trait and concrete configs.

The OffPolicyConfig trait defines the compile-time interface that every
off-policy algorithm config must provide. The generic OffPolicyAgent is
parameterized by a Config conforming to this trait.

Concrete configs (DDPGConfig, TD3Config, SACConfig) bundle:
  - Network architectures (ActorModel, CriticModel)
  - Optimizer types (ActorOpt, CriticOpt)
  - Algorithm flags (NUM_CRITICS, HAS_TARGET_ACTOR)
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


# =============================================================================
# OffPolicyConfig trait
# =============================================================================


trait OffPolicyConfig:
    """Compile-time configuration for off-policy agents.

    Every off-policy algorithm (DDPG, TD3, SAC) implements this trait
    by providing network types, optimizer types, and algorithm flags.
    The generic OffPolicyAgent uses Self.Config.* to access these.
    """

    comptime obs_dim: Int
    comptime action_dim: Int
    comptime batch_size: Int
    comptime buffer_capacity: Int
    comptime ActorModel: Model
    comptime CriticModel: Model
    comptime ActorOpt: Optimizer
    comptime CriticOpt: Optimizer
    comptime NUM_CRITICS: Int
    comptime HAS_TARGET_ACTOR: Bool


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
