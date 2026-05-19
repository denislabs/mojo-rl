"""nn2/core/ — traits + target-tag infrastructure."""

from .param_visitor import ParamVisitor
from .module import Module
from .optimizer import Optimizer
from .loss import Loss
from .initializer import Initializer
from .target_tag import (
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)
