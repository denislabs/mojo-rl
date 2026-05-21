"""Traits + target-tag infrastructure."""

from .param_visitor import ParamVisitor
from .module import Module
from .binary_module import BinaryModule
from .ternary_module import TernaryModule
from .graph_node import GraphNode
from .optimizer import Optimizer
from .loss import Loss
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP, Bf16Compute
from .named_params import NamedParam, named_params
from .map_params import polyak_update, hard_copy_params
from .online_target_pair import OnlineTargetPair
from .target_tag import (
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)
from .param import Param, IsParam
from .walkers import for_each_param_auto, zero_grad_auto
from .amp_matmul import (
    cast_fp32_to_bf16,
    cast_bf16_to_fp32,
    LinearAMPState,
)
