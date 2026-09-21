"""RL-specific shared (agent-agnostic) loss functions.

Agent-specific actor losses (SAC / DDPG / PPO) live under each agent's
own `<agent>/actor_loss.mojo` after the per-agent reorganization.
"""

from .squashed_gaussian import (
    squashed_gaussian_forward,
    squashed_gaussian_backward,
)
from .critic_update_block import CriticUpdateBlock, TwinCriticUpdateBlock
from .loss_block import LossBlock
from .loss_block_bundle import LossBlockBundle
from .seed_grad_inv_batch import seed_grad_inv_batch
