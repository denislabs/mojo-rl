"""dm_control `cheetah` — the single registered task as an env alias.

    from mojo_rl.envs.dm_control.cheetah import DMCheetahRun
    var env = DMCheetahRun()

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry body quaternions yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .cheetah_xml import DMCheetahModel
from .cheetah_config import DMCheetahConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMCheetahRun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCheetahModel, DMCheetahConfig, DTYPE, False
]
