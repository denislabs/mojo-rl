"""dm_control `fish` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.fish import DMFishUpright, DMFishSwim
    var env = DMFishSwim()

One model, two tasks; they differ in the observation (swim adds
`mouth_to_target`) and in the reward, not in the physics.

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry the mocap fields yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .fish_xml import DMFishUprightModel, DMFishSwimModel
from .fish_config import DMFishUprightConfig, DMFishSwimConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMFishUpright[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMFishUprightModel, DMFishUprightConfig, DTYPE, False
]

comptime DMFishSwim[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMFishSwimModel, DMFishSwimConfig, DTYPE, False
]
