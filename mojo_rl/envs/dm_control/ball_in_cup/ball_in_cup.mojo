"""dm_control `ball_in_cup` — the `catch` task as an env alias.

    from mojo_rl.envs.dm_control.ball_in_cup import DMBallInCupCatch
    var env = DMBallInCupCatch()

CPU only, for the same reason as the other dm_control domains whose reward
reads site positions: the batched hook ABI does not carry `site_xpos`.
"""

from .ball_in_cup_xml import DMBallInCupModel
from .ball_in_cup_config import DMBallInCupConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMBallInCupCatch[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMBallInCupModel, DMBallInCupConfig, DTYPE, False
]
