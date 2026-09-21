"""EfficientZeroV2 (deep_agents) — MuZero + Gumbel MCTS + SimSiam consistency.

Phase C of the zero-series port. The discrete agent reuses MuZero's learned
model (``nets.MZRepNet/MZDynNet/MZPredNet``) and adds the SimSiam
temporal-consistency objective (``nets.EZProjectorNet/EZPredictorNet`` +
``loss_ops.consistency_loss_and_grad``) plus the Gumbel planner
(``planners/tree_search`` ``GumbelGPUMCTS``). The continuous agent adds a
Gaussian prediction head + the sampled planner. See
``docs/ZERO_SERIES_DEEP_AGENTS_PORT.md`` §4.
"""

from .nets import (
    MZRepNet,
    MZDynNet,
    MZPredNet,
    EZProjectorNet,
    EZPredictorNet,
)
from .nets_continuous import EZContPredNet
from .loss_ops import consistency_loss_and_grad, consistency_loss_grad_k
from .loss_ops_continuous import (
    continuous_policy_loss_and_grad,
    continuous_policy_loss_grad_k,
)
from .unroll_scratch import EZV2UnrollScratch, EZV2UnrollContScratch
from .blocks import ezv2_unroll_train_step_cpu, ezv2_unroll_train_step_gpu
from .blocks_continuous import (
    ezv2_unroll_train_step_continuous_cpu,
    ezv2_unroll_train_step_continuous_gpu,
)
from .selfplay_cpu import run_ezv2_selfplay_cpu
from .selfplay_gpu import run_ezv2_gumbel_selfplay_gpu
from .selfplay_gpu_continuous import run_ezv2_sampled_selfplay_gpu
from .config import EZV2DiscreteMLPConfig
from .config_continuous import EZV2ContinuousMLPConfig
from .agent import EZv2DiscreteAgent
from .agent_continuous import EZv2ContinuousAgent
