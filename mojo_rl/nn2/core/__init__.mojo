"""Traits + target-tag infrastructure.

Note: target-tag constants (`TARGET_UNINIT/CPU/GPU`, `target_tag_for`)
live in `core/target_tag.mojo` and are no longer re-exported here.
Import them from `..core.target_tag` directly at the use site — keeps
the constants close to their docstring and avoids a stale-re-export
trap if their shape changes."""

from .param_visitor import ParamVisitor
from .module import Module, typed_view, typed_view_mut
from .graph_node import GraphNode
from .optimizer import Optimizer
from .loss import Loss
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP, Bf16Compute
from .named_params import NamedParam, named_params
from .map_params import polyak_update, hard_copy_params
from .online_target_pair import OnlineTargetPair
from .param import Param, IsParam
from .walkers import for_each_param_auto, zero_grad_auto
from .element_op import ElementOp
from .reduce_op import ReduceOp
from .scratch import Scratch, IsScratch
from .scratch_walkers import init_scratch_auto
from .checkpoint import save_params, load_params
from .amp_matmul import (
    cast_fp32_to_bf16,
    cast_bf16_to_fp32,
    LinearAMPState,
)
