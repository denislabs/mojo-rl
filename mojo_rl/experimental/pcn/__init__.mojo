"""Predictive Coding Networks (Bogacz canonical).

See `docs/PCN_REDESIGN.md` for the design rationale and graduation criteria.
"""

from .predictive_model import (
    PCActivation,
    PCBlockTrait,
    PCIdentity,
    PCReLU,
    PCSwish,
    PCTanh,
)
from .pc_block import PCBlock
from .pc_sequential import PCSequential
from .pc_trainer import PCTrainer, PCTrainResult
from .pc_encoder import PCEncoder
from .pc_utils import clip_grad_norm, spectral_norm_clamp

# NOTE: The PCN-MBPO + SAC-encoder experiment (encoder_wrapped_env,
# pc_dynamics*, sac_encoder_prefix) is a confirmed-failed research line and is
# NOT ported to nn2 — it rides out with legacy `deep_agents`/`nn` in the sunset
# sweep (see docs/NN_DEEP_AGENTS_SUNSET_PLAN.md + docs/PCN_NN2_PORT_PLAN.md §5).
# Its symbols are deliberately removed from the core PCN package surface here so
# the nn2 re-architecture can proceed without the legacy coupling. The remaining
# legacy consumer (deep_agents/mbpo_pcn) imports those source files by direct
# module path until the sweep deletes them.
