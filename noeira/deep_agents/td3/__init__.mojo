"""TD3 — Twin Delayed DDPG (twin critics, target smoothing, delayed actor)."""

from .agent import TD3Agent
from .trainer import TD3Trainer
from .config import (
    TD3ConfigT,
    TD3Config,
    TD3ActorNet,
    TD3CriticNet,
    agent_from_config,
    TD3,
)
from .metrics import TD3Metrics
from .target_y_block import TD3TargetYBlock
