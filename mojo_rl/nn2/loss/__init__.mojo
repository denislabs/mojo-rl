"""Loss functions (not Modules — different (logits, targets) → scalar signature)."""

from .cross_entropy import CrossEntropyLoss
from .mse import MSELoss
from .sac_actor_loss import (
    squashed_gaussian_sample,
    sac_actor_backward,
    sac_actor_loss_value,
)
from .sac_actor_loss_cg import SACActorLossCG, SACActorLossOut
from .critic_update_block import CriticUpdateBlock, TwinCriticUpdateBlock
