"""REDQ-OFE package.

Phase O.1 — OFE composite (Sequential aliases over `SkipConcat`-based
            DenseBlocks): `OFEDenseBlock`, `OFEStateBranch6/8`,
            `OFEActionBranch6/8`, `OFEPredictorHead`.
Later phases: OFEFeatureStep, OFEAuxLossStep, REDQOFETrainer, REDQOFEAgent.
"""

from .ofe_nets import (
    OFEDenseBlock,
    OFEStateBranch6,
    OFEStateBranch8,
    OFEActionBranch6,
    OFEActionBranch8,
    OFEPredictorHead,
    state_branch_out_dim,
    action_branch_out_dim,
    predictor_in_dim,
)
from .kernels import aux_mse_grad_cpu, aux_mse_loss_cpu
from .aux_loss_step import OFEAuxLossStep
from .feature_step import OFEFeatureStep
from .ensemble_target_y_block_ofe import EnsembleTargetYBlockOFE
from .ensemble_critic_step_ofe import EnsembleCriticStepOFE
from .ensemble_actor_step_ofe import EnsembleActorStepOFE
from .trainer import REDQOFETrainer, REDQOFEStepResult
from .metrics import REDQOFEMetrics
from .agent import REDQOFEAgent
from .config import (
    REDQOFEConfigT,
    REDQOFE6Config,
    REDQOFE8Config,
    LargeREDQOFE6Config,
    LargeREDQOFE8Config,
    REDQOFEActor,
    REDQOFECritic,
    agent_from_config_ofe,
    REDQOFE6,
    REDQOFE8,
    LargeREDQOFE6,
    LargeREDQOFE8,
)
