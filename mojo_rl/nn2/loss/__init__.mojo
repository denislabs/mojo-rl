"""Loss functions (not Modules — different (logits, targets) → scalar signature)."""

from .squashed_gaussian import (
    squashed_gaussian_forward,
    squashed_gaussian_backward,
)
from .cross_entropy import CrossEntropyLoss
from .gaussian_nll_loss import GaussianNLLLoss
from .mse import MSELoss
from .sac_actor_loss import SACActorLoss, SACActorLossOut
from .ppo_actor_loss_cg import PPOActorLossCG
from .critic_update_block import CriticUpdateBlock, TwinCriticUpdateBlock
from .ddpg_actor_loss import DDPGActorLoss
from .loss_block import LossBlock
from .loss_block_bundle import LossBlockBundle
from .seed_grad_inv_batch import seed_grad_inv_batch
