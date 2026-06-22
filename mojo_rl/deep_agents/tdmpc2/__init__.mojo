"""TD-MPC2 (deep_agents port) — implicit world model + MPPI planning.

Built on nn ComputeGraph autodiff (correct BPTT through the latent rollout)
+ the deep_agents blocks pattern. See docs/TDMPC2_DEEP_AGENTS_PORT.md.

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
from .metrics import TDMPC2Metrics
from .agent import TDMPC2Agent
from .config import (
    TDMPC2ConfigT,
    TDMPC2Config,
    agent_from_config,
    TDMPC2,
)

# Multi-task (item C, §14.3) — task-conditioned world model + learned per-task
# embedding table.
from .task_embedding import TaskEmbedding
from .nets_mt import (
    TDMPC2EncoderMT,
    TDMPC2DynamicsMT,
    TDMPC2RewardMT,
    TDMPC2QNetMT,
    TDMPC2TerminationMT,
    TDMPC2PolicyMT,
)
from .wm_graph_mt import TDMPC2WMGraphMT
from .policy_graph_mt import TDMPC2PolicyGraphMT
from .wm_step_mt import WMStepMT
from .policy_step_mt import PolicyStepMT
from .td_target_step_mt import TDTargetStepMT
from .agent_mt import TDMPC2MultiTaskAgent
from .config_mt import TDMPC2MultiTask
