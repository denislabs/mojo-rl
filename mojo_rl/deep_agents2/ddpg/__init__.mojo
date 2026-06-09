"""DDPG — Deep Deterministic Policy Gradient (deterministic actor, single critic)."""

from .agent import DDPGAgent
from .trainer import DDPGTrainer
from .config import (
    DDPGConfigT,
    DDPGConfig,
    DDPGActorNet,
    DDPGCriticNet,
    agent_from_config,
    DDPG,
)
from .metrics import DDPGMetrics
from .target_y_block import DDPGTargetYBlock
from .actor_loss import DDPGActorLoss
