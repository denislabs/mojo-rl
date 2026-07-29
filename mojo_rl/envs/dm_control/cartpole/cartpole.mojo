"""dm_control `cartpole` — the six registered tasks as env aliases.

    from mojo_rl.envs.dm_control.cartpole import DMCartpoleSwingup
    var env = DMCartpoleSwingup()

CPU only: the configs' GPU reward/obs hooks are stubs because the batched hook
ABI does not carry body quaternions yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .cartpole_xml import (
    DMCartpole1Model,
    DMCartpole2Model,
    DMCartpole3Model,
)
from .cartpole_config import DMCartpoleConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.

comptime DMCartpoleBalance[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCartpole1Model, DMCartpoleConfig[1, False, False], DTYPE, False
]

comptime DMCartpoleBalanceSparse[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCartpole1Model, DMCartpoleConfig[1, False, True], DTYPE, False
]

comptime DMCartpoleSwingup[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCartpole1Model, DMCartpoleConfig[1, True, False], DTYPE, False
]

comptime DMCartpoleSwingupSparse[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCartpole1Model, DMCartpoleConfig[1, True, True], DTYPE, False
]

comptime DMCartpoleTwoPoles[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCartpole2Model, DMCartpoleConfig[2, True, False], DTYPE, False
]

comptime DMCartpoleThreePoles[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCartpole3Model, DMCartpoleConfig[3, True, False], DTYPE, False
]
