"""EfficientZeroV2 (deep_agents2) — MuZero + Gumbel MCTS + SimSiam consistency.

Phase C of the zero-series port. The discrete agent reuses MuZero's learned
model (``nets.MZRepNet/MZDynNet/MZPredNet``) and adds the SimSiam
temporal-consistency objective (``nets.EZProjectorNet/EZPredictorNet`` +
``loss_ops.consistency_loss_and_grad``) plus the Gumbel planner
(``planners/tree_search`` ``GumbelGPUMCTS``). The continuous agent adds a
Gaussian prediction head + the sampled planner. See
``docs/ZERO_SERIES_DEEP_AGENTS2_PORT.md`` §4.
"""

from .nets import (
    MZRepNet,
    MZDynNet,
    MZPredNet,
    EZProjectorNet,
    EZPredictorNet,
)
from .loss_ops import consistency_loss_and_grad
