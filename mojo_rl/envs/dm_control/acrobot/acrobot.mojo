"""dm_control `acrobot` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.acrobot import DMAcrobotSwingup
    var env = DMAcrobotSwingup()

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry body quaternions yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .acrobot_xml import DMAcrobotModel
from .acrobot_config import DMAcrobotConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMAcrobotSwingup[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMAcrobotModel, DMAcrobotConfig[False], DTYPE, False
]

comptime DMAcrobotSwingupSparse[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMAcrobotModel, DMAcrobotConfig[True], DTYPE, False
]
