"""Loss functions (not Modules — different (logits, targets) → scalar signature)."""

from .squashed_gaussian import (
    squashed_gaussian_forward,
    squashed_gaussian_backward,
)
from .cross_entropy import CrossEntropyLoss
from .mse import MSELoss
from .sac_actor_loss import (
    squashed_gaussian_sample,
    sac_actor_backward,
    sac_actor_loss_value,
)
from .sac_actor_loss_cg import SACActorLossCG, SACActorLossOut
from .critic_update_block import CriticUpdateBlock, TwinCriticUpdateBlock
from .ddpg_actor_loss import DDPGActorLoss
from .loss_block import LossBlock
from .loss_block_bundle import LossBlockBundle
