"""TD-MPC2 (deep_agents2 port) — implicit world model + MPPI planning.

Built on nn2 ComputeGraph autodiff (correct BPTT through the latent rollout)
+ the deep_agents2 blocks pattern. See docs/TDMPC2_DEEP_AGENTS2_PORT.md.

Phases shipped:
  P1 — world-model graph + BPTT + two-hot losses (wm_graph, wm_step).
  P2 — policy update (policy_graph, policy_step), RunningScale,
       TwoHotDecode, TD-target step.
"""

from .losses import (
    MSELossPlain,
    TDMPC2TwoHotLoss,
    TwoHotDecode,
)
from .nets import (
    NormedLinear,
    NormedLinearSimNorm,
    TDMPC2Encoder,
    TDMPC2Dynamics,
    TDMPC2Reward,
    TDMPC2QNet,
    TDMPC2Policy,
)
from .wm_graph import TDMPC2WMGraph, NQ
from .wm_step import WMStep
from .policy_graph import TDMPC2PolicyGraph
from .policy_step import PolicyStep
from .running_scale import RunningScale
from .td_target_step import TDTargetStep
from .callback import TDMPC2RolloutCallbackCPU, TDMPC2RolloutCallbackGPU
from .agent import TDMPC2Agent
