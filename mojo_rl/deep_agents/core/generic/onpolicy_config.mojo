"""On-policy agent configuration trait and concrete configs.

The OnPolicyConfig trait defines the compile-time interface for on-policy
algorithms (PPO, A2C). The generic GenericOnPolicyAgent is parameterized
by a Config conforming to this trait.

Algorithm behavior is controlled by two strategy types:
  - PolicyGrad: policy gradient computation (VanillaPG or ClippedSurrogate)
  - EpochSched: epoch/minibatch schedule (SinglePass or MultiEpochMinibatch)

For continuous action spaces, ContinuousOnPolicyConfig adds action_dim and
uses StochasticActor for unbounded Gaussian policy.
"""

from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearTanh,
    Sequential,
    StochasticActor,
    Conv2DReLU,
    FlattenLayer,
)
from mojo_rl.nn.optimizer import Optimizer, Adam
from .policy_gradient import (
    PolicyGradient,
    VanillaPG,
    ClippedSurrogate,
    AutodiffVanillaPG,
    AutodiffClippedSurrogate,
)
from .epoch_schedule import EpochSchedule, SinglePass, MultiEpochMinibatch


# =============================================================================
# OnPolicyConfig trait
# =============================================================================


trait OnPolicyConfig:
    """Compile-time configuration for on-policy agents."""

    comptime NAME: String
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
    """PPO: clipped surrogate, multi-epoch minibatch, per-minibatch normalization.
    """

    comptime NAME: String = "PPO"
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

    comptime NAME: String = "A2C"
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


# =============================================================================
# ContinuousOnPolicyConfig trait
# =============================================================================


trait ContinuousOnPolicyConfig:
    """Compile-time configuration for continuous-action on-policy agents.

    Same as OnPolicyConfig but adds action_dim (since actor OUT_DIM = 2 * action_dim
    for StochasticActor).
    """

    comptime NAME: String
    comptime obs_dim: Int
    comptime action_dim: Int
    comptime rollout_len: Int
    comptime ActorModel: Model
    comptime CriticModel: Model
    comptime ActorOpt: Optimizer
    comptime CriticOpt: Optimizer
    comptime PolicyGrad: PolicyGradient
    comptime EpochSched: EpochSchedule
    comptime USE_AUTODIFF_GRAD: Bool


# =============================================================================
# ContinuousPPOConfig
# =============================================================================


struct ContinuousPPOConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    ROLLOUT: Int = 128,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
](ContinuousOnPolicyConfig):
    """PPO for continuous actions: unbounded Gaussian policy (CleanRL-style).

    Actor: obs -> Tanh -> Tanh -> StochasticActor (mean + state-independent log_std).
    Critic: obs -> Tanh -> Tanh -> 1 (value).
    """

    comptime NAME: String = "PPO Continuous"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime rollout_len: Int = Self.ROLLOUT

    comptime ActorModel = Sequential[
        LinearTanh[Self.OBS, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.HIDDEN],
        StochasticActor[Self.HIDDEN, Self.ACT],
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
    comptime USE_AUTODIFF_GRAD: Bool = False


# =============================================================================
# PPOCNNConfig (Nature DQN CNN for pixel observations, discrete actions)
# =============================================================================


struct PPOCNNConfig[
    ACT: Int,
    ROLLOUT: Int = 128,
    actor_lr: Float64 = 0.00025,
    critic_lr: Float64 = 0.00025,
](OnPolicyConfig):
    """PPO with Nature CNN for 4x84x84 pixel observations (discrete actions).

    Both actor and critic use the same CNN backbone architecture:
    Conv2DReLU(8,4) -> Conv2DReLU(4,2) -> Conv2DReLU(3,1)
    -> Flatten(3136) -> LinearReLU(512) -> output

    Matches CleanRL's PPO Atari implementation.
    """

    comptime obs_dim: Int = 4 * 84 * 84  # 28224
    comptime num_actions: Int = Self.ACT
    comptime rollout_len: Int = Self.ROLLOUT
    comptime NAME: String = "PPO CNN"

    comptime ActorModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],
        Conv2DReLU[32, 64, 4, 2, 0, 20, 20],
        Conv2DReLU[64, 64, 3, 1, 0, 9, 9],
        FlattenLayer[64 * 7 * 7],
        LinearReLU[64 * 7 * 7, 512],
        Linear[512, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],
        Conv2DReLU[32, 64, 4, 2, 0, 20, 20],
        Conv2DReLU[64, 64, 3, 1, 0, 9, 9],
        FlattenLayer[64 * 7 * 7],
        LinearReLU[64 * 7 * 7, 512],
        Linear[512, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime PolicyGrad = ClippedSurrogate
    comptime EpochSched = MultiEpochMinibatch


# =============================================================================
# AutodiffPPOConfig
# =============================================================================


struct AutodiffPPOConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 64,
    ROLLOUT: Int = 128,
    actor_lr: Float64 = 0.00025,
    critic_lr: Float64 = 0.001,
    clip_eps: Float64 = 0.2,
](OnPolicyConfig):
    """PPO using AutodiffClippedSurrogate: chained DiffOp vjps for actor gradient.

    Identical architecture to PPOConfig but uses AutodiffClippedSurrogate
    instead of ClippedSurrogate. The clip_eps parameter is forwarded to
    AutodiffClippedSurrogate as a compile-time parameter.
    """

    comptime NAME: String = "PPO (Autodiff)"
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

    comptime PolicyGrad = AutodiffClippedSurrogate[Self.clip_eps]
    comptime EpochSched = MultiEpochMinibatch


# =============================================================================
# AutodiffA2CConfig
# =============================================================================


struct AutodiffA2CConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 128,
    ROLLOUT: Int = 128,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
](OnPolicyConfig):
    """A2C using AutodiffVanillaPG: CategoricalLogProbOp vjp for actor gradient.

    Identical architecture to A2CConfig but uses AutodiffVanillaPG
    instead of VanillaPG.
    """

    comptime NAME: String = "A2C (Autodiff)"
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

    comptime PolicyGrad = AutodiffVanillaPG
    comptime EpochSched = SinglePass


# =============================================================================
# AutodiffContinuousPPOConfig
# =============================================================================


struct AutodiffContinuousPPOConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    ROLLOUT: Int = 128,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
    clip_eps: Float64 = 0.2,
](ContinuousOnPolicyConfig):
    """PPO for continuous actions using autodiff-composed policy gradient.

    Identical architecture to ContinuousPPOConfig but uses the autodiff
    continuous actor gradient kernel (GaussianLogProbOp + RatioOp +
    ClipSurrogateOp chained vjps) instead of hand-derived gradients.

    Actor: obs -> Tanh -> Tanh -> StochasticActor (mean + state-independent log_std).
    Critic: obs -> Tanh -> Tanh -> 1 (value).
    """

    comptime NAME: String = "PPO Continuous (Autodiff)"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime rollout_len: Int = Self.ROLLOUT

    comptime ActorModel = Sequential[
        LinearTanh[Self.OBS, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.HIDDEN],
        StochasticActor[Self.HIDDEN, Self.ACT],
    ]
    comptime CriticModel = Sequential[
        LinearTanh[Self.OBS, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime PolicyGrad = AutodiffClippedSurrogate[Self.clip_eps]
    comptime EpochSched = MultiEpochMinibatch
    comptime USE_AUTODIFF_GRAD: Bool = True
