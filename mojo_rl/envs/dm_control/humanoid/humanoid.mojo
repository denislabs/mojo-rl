"""dm_control `humanoid` — the four registered tasks as env aliases.

    from mojo_rl.envs.dm_control.humanoid import DMHumanoidStand
    var env = DMHumanoidStand()

`run` and `run_pure_state` share a target speed and differ only in the
observation: 67 engineered features vs the raw 55-dim state. That is a
different OBS_DIM, hence a different model-def alias (`DMHumanoidPureModel`),
not just a different config parameter.

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI carries neither body quaternions nor `xvel` (gap G10). See
docs/DM_CONTROL_PORT.md.
"""

from .humanoid_xml import DMHumanoidModel, DMHumanoidPureModel
from .humanoid_config import DMHumanoidConfig, WALK_SPEED, RUN_SPEED
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit. (Note this
# is a real difference from the Gym humanoid, which terminates on an unhealthy
# torso height; dm_control shapes the same signal into the reward instead.)
comptime DMHumanoidStand[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHumanoidModel, DMHumanoidConfig[0.0, False], DTYPE, False
]

comptime DMHumanoidWalk[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHumanoidModel, DMHumanoidConfig[WALK_SPEED, False], DTYPE, False
]

comptime DMHumanoidRun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHumanoidModel, DMHumanoidConfig[RUN_SPEED, False], DTYPE, False
]

comptime DMHumanoidRunPureState[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHumanoidPureModel, DMHumanoidConfig[RUN_SPEED, True], DTYPE, False
]
