"""A2C — Advantage Actor-Critic (degenerate single-epoch PPO).

`A2CDiscreteAgent` (categorical) and `A2CAgent` (continuous Gaussian)
are thin facades pinning `N_EPOCHS=1` + `MINIBATCH=ROLLOUT_LEN` over the
discrete / continuous PPO agents — at one full-batch on-policy step the
importance ratio is identically 1, so the PPO clipped surrogate reduces
to the vanilla advantage policy gradient. See `agent.mojo`.
"""

from .agent import A2CDiscreteAgent, A2CAgent
from .config import (
    A2CConfigT,
    A2CConfig,
    A2CDiscreteConfigT,
    A2CDiscreteConfig,
    agent_from_config,
    agent_from_config_discrete,
    A2C,
    A2CDiscrete,
)
