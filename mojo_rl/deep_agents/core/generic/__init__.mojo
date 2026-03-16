"""Generic composable agent infrastructure.

Strategy building blocks (stateless, compile-time parameterized):
  - exploration: GaussianNoise, StochasticSample
  - update_schedule: EveryStep, DelayedAll, DelayedActorOnly
  - target_value: SingleQTarget, TwinQTarget, EntropicTwinQTarget
  - target_action: DeterministicTarget, SmoothedTarget, ReparamTarget
  - actor_loss: DPGLoss, MaxEntLoss

Configs and agents:
  - offpolicy_config: OffPolicyConfig, DDPGConfig, TD3Config, SACConfig
  - offpolicy_agent: GenericOffPolicyAgent (handles DDPG, TD3, SAC)
  - onpolicy_config: OnPolicyConfig, PPOConfig, A2CConfig
  - onpolicy_agent: GenericOnPolicyAgent
  - dqn_agent: DiscreteOffPolicyConfig, DQNConfig, GenericDQNAgent
"""

# Strategy traits and implementations
from .exploration import Explore, GaussianNoise, StochasticSample
from .update_schedule import Schedule, EveryStep, DelayedAll, DelayedActorOnly
from .target_value import TargetValue, SingleQTarget, TwinQTarget, EntropicTwinQTarget
from .target_action import TargetAction, DeterministicTarget, SmoothedTarget, ReparamTarget
from .actor_loss import ActorLoss, DPGLoss, MaxEntLoss

# Configs and agents
from .offpolicy_config import OffPolicyConfig, DDPGConfig, TD3Config, SACConfig
from .offpolicy_agent import GenericOffPolicyAgent
from .onpolicy_config import OnPolicyConfig, PPOConfig, A2CConfig
from .onpolicy_agent import GenericOnPolicyAgent, PPOGPUStateGeneric
from .dqn_agent import DiscreteOffPolicyConfig, DQNConfig, GenericDQNAgent
