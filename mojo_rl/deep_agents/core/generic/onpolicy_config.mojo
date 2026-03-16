"""On-policy agent configuration trait and concrete configs.

The OnPolicyConfig trait defines the compile-time interface for on-policy
algorithms (PPO, A2C). The generic GenericOnPolicyAgent is parameterized
by a Config conforming to this trait.

Algorithm behavior is controlled by two strategy types:
  - PolicyGrad: policy gradient computation (VanillaPG or ClippedSurrogate)
  - EpochSched: epoch/minibatch schedule (SinglePass or MultiEpochMinibatch)
"""

from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearTanh,
    Sequential,
)
from mojo_rl.nn.optimizer import Optimizer, Adam
from .policy_gradient import PolicyGradient, VanillaPG, ClippedSurrogate
from .epoch_schedule import EpochSchedule, SinglePass, MultiEpochMinibatch


# =============================================================================
# OnPolicyConfig trait
# =============================================================================


trait OnPolicyConfig:
    """Compile-time configuration for on-policy agents."""

    comptime obs_dim: Int
    comptime num_actions: Int
    comptime rollout_len: Int
    comptime ActorModel: Model
    comptime CriticModel: Model
    comptime ActorOpt: Optimizer
    comptime CriticOpt: Optimizer
    comptime PolicyGrad: PolicyGradient
    comptime EpochSched: EpochSchedule


# =============================================================================
# PPOConfig
# =============================================================================


struct PPOConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 64,
    ROLLOUT: Int = 128,
    actor_lr: Float64 = 0.00025,
    critic_lr: Float64 = 0.001,
](OnPolicyConfig):
    """PPO: clipped surrogate, multi-epoch minibatch, per-minibatch normalization."""

    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime rollout_len: Int = Self.ROLLOUT

    comptime ActorModel = Sequential[
        LinearTanh[Self.OBS, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        LinearTanh[Self.OBS, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime PolicyGrad = ClippedSurrogate
    comptime EpochSched = MultiEpochMinibatch


# =============================================================================
# A2CConfig
# =============================================================================


struct A2CConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 128,
    ROLLOUT: Int = 128,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
](OnPolicyConfig):
    """A2C: vanilla PG, single pass, per-rollout normalization."""

    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime rollout_len: Int = Self.ROLLOUT

    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime PolicyGrad = VanillaPG
    comptime EpochSched = SinglePass
