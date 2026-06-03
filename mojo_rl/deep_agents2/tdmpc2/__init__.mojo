"""TD-MPC2 (deep_agents2 port) — implicit world model + MPPI planning.

Built on nn2 ComputeGraph autodiff (correct BPTT through the latent rollout)
+ the deep_agents2 blocks pattern. See docs/TDMPC2_DEEP_AGENTS2_PORT.md.

Phase 1 (current): world-model graph + BPTT + two-hot losses.
"""

from .losses import MSELossPlain, TDMPC2TwoHotLoss
from .nets import (
    NormedLinear,
    NormedLinearSimNorm,
    TDMPC2Encoder,
    TDMPC2Dynamics,
    TDMPC2Reward,
    TDMPC2QNet,
)
from .wm_graph import TDMPC2WMGraph, NQ
