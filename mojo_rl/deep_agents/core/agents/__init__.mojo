"""Generic composable agent infrastructure.

Strategy building blocks (stateless, compile-time parameterized):
  - exploration: GaussianNoise, StochasticSample
  - update_schedule: EveryStep, DelayedAll, DelayedActorOnly
  - target_value: SingleQTarget, TwinQTarget, EntropicTwinQTarget
  - target_action: DeterministicTarget, SmoothedTarget, ReparamTarget
  - actor_loss: DPGLoss, MaxEntLoss, AutodiffDPGLoss, AutodiffTD3Loss, AutodiffMaxEntLoss
  - policy_gradient: VanillaPG, ClippedSurrogate, AutodiffVanillaPG, AutodiffClippedSurrogate
  - epoch_schedule: SinglePass, MultiEpochMinibatch
  - q_target: StandardQTarget, DoubleQTarget
  - q_output: DirectQ, DuelingQ
  - q_gradient: ManualQGradient, AutodiffQGradient

Configs and agents:
  - offpolicy_config: OffPolicyConfig, DDPGConfig, TD3Config, SACConfig
  - offpolicy_agent: GenericOffPolicyAgent (handles DDPG, TD3, SAC)
  - onpolicy_config: OnPolicyConfig, PPOConfig, A2CConfig, ContinuousPPOConfig, AutodiffPPOConfig, AutodiffA2CConfig, AutodiffContinuousPPOConfig
  - onpolicy_agent: GenericOnPolicyAgent (discrete PPO, A2C)
  - onpolicy_continuous_agent: GenericOnPolicyContinuousAgent (continuous PPO)
  - dqn_agent: DiscreteOffPolicyConfig, DQNConfig, DoubleDQNConfig, DuelingDQNConfig, GenericDQNAgent
  - c51_agent: C51Config, GenericC51Agent (categorical distributional DQN)
"""

# Strategy traits and implementations
from ..strategies import (
    Explore, GaussianNoise, StochasticSample,
    Schedule, EveryStep, DelayedAll, DelayedActorOnly,
    TargetValue, SingleQTarget, TwinQTarget, EntropicTwinQTarget,
    TargetAction, DeterministicTarget, SmoothedTarget, ReparamTarget,
    ActorLoss, DPGLoss, MaxEntLoss, AutodiffMaxEntLoss, AutodiffDPGLoss, AutodiffTD3Loss,
    PolicyGradient, VanillaPG, ClippedSurrogate, AutodiffVanillaPG, AutodiffClippedSurrogate,
    EpochSchedule, SinglePass, MultiEpochMinibatch,
    QTarget, StandardQTarget, DoubleQTarget,
    QOutput, DirectQ, DuelingQ,
    QGradient, ManualQGradient, AutodiffQGradient,
)

# Configs
from ..configs import (
    OffPolicyConfig, DDPGConfig, TD3Config, SACConfig, AutodiffSACConfig, AutodiffDDPGConfig, AutodiffTD3Config,
    OnPolicyConfig, PPOConfig, A2CConfig, PPOCNNConfig, ContinuousOnPolicyConfig, ContinuousPPOConfig, AutodiffPPOConfig, AutodiffA2CConfig, AutodiffContinuousPPOConfig,
)

# Agents
from .offpolicy_agent import GenericOffPolicyAgent
from .onpolicy_agent import GenericOnPolicyAgent, PPOGPUStateGeneric
from .onpolicy_continuous_agent import GenericOnPolicyContinuousAgent
from .dqn_agent import DiscreteOffPolicyConfig, DQNConfig, DoubleDQNConfig, DuelingDQNConfig, DQNCNNConfig, DQNPERConfig, AutodiffDQNConfig, HuberDQNConfig, NoisyDQNConfig, GenericDQNAgent, GenericDQNPERAgent, DQNGPUStateGeneric
from .c51_agent import CategoricalDQNConfig, C51Config, GenericC51Agent
from .rainbow_agent import RainbowConfig, GenericRainbowAgent

# Convenience aliases matching old agent names
from .aliases import (
    DQNAgent,
    DQNPERAgent,
    DuelingDQNAgent,
    DQNCNNAgent,
    AutodiffDQNAgent,
    C51Agent,
    NoisyDQNAgent,
    RainbowAgent,
    DeepDDPGAgent,
    DeepTD3Agent,
    DeepSACAgent,
    DeepA2CAgent,
    DeepPPOAgent,
    DeepPPOContinuousAgent,
    DeepPPOCNNAgent,
)
