"""`dm_control` `ball_in_cup` — the `catch` task as an env alias.

    from mojo_rl.envs.dm_control.ball_in_cup import DMBallInCupCatch
    var env = DMBallInCupCatch()

GPU-BATCHED as of 2026-08-08 (`DMBallInCupCatchBatched`). `site_xpos` landed
with tranche 2; what this one needed last was MODEL GEOMETRY IN THE RESET
HOOK — its `initialize_episode` is a rejection sampler over the cup's five
capsules, the only reset in the suite that reads geoms rather than joint
ranges. `init_qpos_gpu` now takes `bodies` and `geoms`, which is also what
exposed a CPU defect: the sampler was reading the PREVIOUS episode's cup pose
out of `d.xpos`. See `ball_in_cup_config.custom_reset_cpu`.
"""

from .ball_in_cup_xml import DMBallInCupModel
from .ball_in_cup_config import DMBallInCupConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMBallInCupCatch[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMBallInCupModel, DMBallInCupConfig, DTYPE, False
]


comptime DMBallInCupCatchBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMBallInCupModel, DMBallInCupConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
