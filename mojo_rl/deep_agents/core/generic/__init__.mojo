"""Generic composable agent infrastructure.

Strategy building blocks (stateless, compile-time parameterized):
  - exploration: GaussianNoise, StochasticSample
  - update_schedule: EveryStep, DelayedAll, DelayedActorOnly
  - target_value: SingleQTarget, TwinQTarget, EntropicTwinQTarget
  - target_action: DeterministicTarget, SmoothedTarget, ReparamTarget
  - actor_loss: DPGLoss, MaxEntLoss
  - policy_gradient: VanillaPG, ClippedSurrogate
  - epoch_schedule: SinglePass, MultiEpochMinibatch
  - q_target: StandardQTarget, DoubleQTarget
  - q_output: DirectQ, DuelingQ

Configs and agents:
  - offpolicy_config: OffPolicyConfig, DDPGConfig, TD3Config, SACConfig
  - offpolicy_agent: GenericOffPolicyAgent (handles DDPG, TD3, SAC)
  - onpolicy_config: OnPolicyConfig, PPOConfig, A2CConfig, ContinuousPPOConfig
  - onpolicy_agent: GenericOnPolicyAgent (discrete PPO, A2C)
  - onpolicy_continuous_agent: GenericOnPolicyContinuousAgent (continuous PPO)
  - dqn_agent: DiscreteOffPolicyConfig, DQNConfig, DoubleDQNConfig, DuelingDQNConfig, GenericDQNAgent
"""

# Strategy traits and implementations
from .exploration import Explore, GaussianNoise, StochasticSample
from .update_schedule import Schedule, EveryStep, DelayedAll, DelayedActorOnly
from .target_value import TargetValue, SingleQTarget, TwinQTarget, EntropicTwinQTarget
from .target_action import TargetAction, DeterministicTarget, SmoothedTarget, ReparamTarget
from .actor_loss import ActorLoss, DPGLoss, MaxEntLoss
from .policy_gradient import PolicyGradient, VanillaPG, ClippedSurrogate
from .epoch_schedule import EpochSchedule, SinglePass, MultiEpochMinibatch
from .q_target import QTarget, StandardQTarget, DoubleQTarget
from .q_output import QOutput, DirectQ, DuelingQ

# Configs and agents
from .offpolicy_config import OffPolicyConfig, DDPGConfig, TD3Config, SACConfig
from .offpolicy_agent import GenericOffPolicyAgent
from .onpolicy_config import OnPolicyConfig, PPOConfig, A2CConfig, PPOCNNConfig, ContinuousOnPolicyConfig, ContinuousPPOConfig
from .onpolicy_agent import GenericOnPolicyAgent, PPOGPUStateGeneric
from .onpolicy_continuous_agent import GenericOnPolicyContinuousAgent
from .dqn_agent import DiscreteOffPolicyConfig, DQNConfig, DoubleDQNConfig, DuelingDQNConfig, DQNCNNConfig, GenericDQNAgent, DQNGPUStateGeneric
