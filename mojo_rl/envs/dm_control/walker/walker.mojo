"""dm_control `walker` — the three registered tasks as env aliases.

    from mojo_rl.envs.dm_control.walker import DMWalkerWalk
    var env = DMWalkerWalk()

CPU only: the configs' GPU reward/obs hooks are stubs because the batched hook
ABI does not carry body quaternions yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .walker_xml import DMWalkerModel
from .walker_config import DMWalkerConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.

comptime DMWalkerStand[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[0.0], DTYPE, False
]

comptime DMWalkerWalk[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[1.0], DTYPE, False
]

comptime DMWalkerRun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[8.0], DTYPE, False
]
