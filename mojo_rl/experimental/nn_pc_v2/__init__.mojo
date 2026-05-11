"""Predictive Coding Networks (Bogacz canonical, experimental v2).

See `docs/PCN_REDESIGN.md` for the design rationale and graduation criteria.
This package coexists with `experimental/nn_pc/` (Monadillo flavor, archived)
until validated on MNIST + CIFAR.
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
from .encoder_wrapped_env import EncoderWrappedEnv
from .pc_dynamics import PCDynamics
from .pc_dynamics_ensemble import PCDynamicsEnsemble
from .pc_dynamics_ensemble_gpu import PCDynamicsEnsembleGPU
from .pc_dynamics_ensemble_instance import PCDynamicsEnsembleInstanceCPU
from .pc_dynamics_ensemble_instance_gpu import PCDynamicsEnsembleInstanceGPU
