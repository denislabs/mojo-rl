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
    PCTanh,
)
from .pc_block import PCBlock
from .pc_sequential import PCSequential
from .pc_trainer import PCTrainer, PCTrainResult
