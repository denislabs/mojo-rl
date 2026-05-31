"""REDQ package.

Phase R.0 — `CriticEnsemble[CRITIC, N]` container.
Phase R.1 — `redq_ensemble_target_cpu` kernel +
            `EnsembleCriticStep` + `EnsembleTargetYBlock`.
Later phases: EnsembleActorLoss, REDQTrainer, REDQAgent.
"""

from .ensemble import CriticEnsemble
from .kernels import (
    redq_ensemble_target_cpu,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
    REDQ_TARGET_REM,
)
from .ensemble_target_y_block import EnsembleTargetYBlock
from .ensemble_actor_loss import EnsembleActorLoss, EnsembleActorLossResult
from .metrics import REDQMetrics
from .trainer import REDQTrainer
from .agent import REDQAgent
from .config import (
    REDQConfigT,
    REDQConfig,
    SmallREDQConfig,
    REDQActor,
    REDQCritic,
    agent_from_config,
    REDQ,
    SmallREDQ,
)
from .blocks.ensemble_critic_step import EnsembleCriticStep
from .blocks.ensemble_actor_step import EnsembleActorStep
from .blocks.ensemble_polyak_step import EnsemblePolyakStep
