"""Discrete (categorical) PPO — clipped surrogate, on-policy."""

from .agent import PPODiscreteAgent
from .trainer import PPODiscreteTrainer
from .actor_loss import PPODiscreteActorLoss
from .objective import PPODiscreteObjective
from .config import (
    PPODiscreteConfigT,
    PPODiscreteConfig,
    PPODiscreteActorNet,
    PPODiscreteCriticNet,
    agent_from_config,
    PPODiscrete,
)
