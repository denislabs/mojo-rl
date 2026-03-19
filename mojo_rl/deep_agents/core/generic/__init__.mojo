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
  - onpolicy_config: OnPolicyConfig, PPOConfig, A2CConfig, ContinuousPPOConfig, AutodiffPPOConfig, AutodiffA2CConfig
  - onpolicy_agent: GenericOnPolicyAgent (discrete PPO, A2C)
  - onpolicy_continuous_agent: GenericOnPolicyContinuousAgent (continuous PPO)
  - dqn_agent: DiscreteOffPolicyConfig, DQNConfig, DoubleDQNConfig, DuelingDQNConfig, GenericDQNAgent
"""

# Strategy traits and implementations
from .exploration import Explore, GaussianNoise, StochasticSample
from .update_schedule import Schedule, EveryStep, DelayedAll, DelayedActorOnly
from .target_value import TargetValue, SingleQTarget, TwinQTarget, EntropicTwinQTarget
from .target_action import TargetAction, DeterministicTarget, SmoothedTarget, ReparamTarget
from .actor_loss import ActorLoss, DPGLoss, MaxEntLoss, AutodiffMaxEntLoss, AutodiffDPGLoss, AutodiffTD3Loss
from .policy_gradient import PolicyGradient, VanillaPG, ClippedSurrogate, AutodiffVanillaPG, AutodiffClippedSurrogate
from .epoch_schedule import EpochSchedule, SinglePass, MultiEpochMinibatch
from .q_target import QTarget, StandardQTarget, DoubleQTarget
from .q_output import QOutput, DirectQ, DuelingQ
from .q_gradient import QGradient, ManualQGradient, AutodiffQGradient

# Configs and agents
from .offpolicy_config import OffPolicyConfig, DDPGConfig, TD3Config, SACConfig, AutodiffSACConfig, AutodiffDDPGConfig, AutodiffTD3Config
from .offpolicy_agent import GenericOffPolicyAgent
from .onpolicy_config import OnPolicyConfig, PPOConfig, A2CConfig, PPOCNNConfig, ContinuousOnPolicyConfig, ContinuousPPOConfig, AutodiffPPOConfig, AutodiffA2CConfig
from .onpolicy_agent import GenericOnPolicyAgent, PPOGPUStateGeneric
from .onpolicy_continuous_agent import GenericOnPolicyContinuousAgent
from .dqn_agent import DiscreteOffPolicyConfig, DQNConfig, DoubleDQNConfig, DuelingDQNConfig, DQNCNNConfig, DQNPERConfig, AutodiffDQNConfig, GenericDQNAgent, GenericDQNPERAgent, DQNGPUStateGeneric

# Convenience aliases matching old agent names
from .aliases import (
    DQNAgent,
    DQNPERAgent,
    DuelingDQNAgent,
    DQNCNNAgent,
    AutodiffDQNAgent,
    DeepDDPGAgent,
    DeepTD3Agent,
    DeepSACAgent,
    DeepA2CAgent,
    DeepPPOAgent,
    DeepPPOContinuousAgent,
    DeepPPOCNNAgent,
)
