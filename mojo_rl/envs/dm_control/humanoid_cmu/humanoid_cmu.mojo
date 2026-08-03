"""dm_control `humanoid_CMU` — the three registered tasks as env aliases.

    from mojo_rl.envs.dm_control.humanoid_cmu import DMHumanoidCMUStand
    var env = DMHumanoidCMUStand()

All three share one model and one config, parameterized only by the target
speed. Unlike `humanoid` there is no `run_pure_state` variant, so there is only
one model-def alias.

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI carries neither body quaternions nor `xvel` (gap G10). See
docs/DM_CONTROL_PORT.md.
"""

from .humanoid_cmu_xml import DMHumanoidCMUModel
from .humanoid_cmu_config import DMHumanoidCMUConfig, WALK_SPEED, RUN_SPEED
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMHumanoidCMUStand[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHumanoidCMUModel, DMHumanoidCMUConfig[0.0], DTYPE, False
]

comptime DMHumanoidCMUWalk[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHumanoidCMUModel, DMHumanoidCMUConfig[WALK_SPEED], DTYPE, False
]

comptime DMHumanoidCMURun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHumanoidCMUModel, DMHumanoidCMUConfig[RUN_SPEED], DTYPE, False
]
