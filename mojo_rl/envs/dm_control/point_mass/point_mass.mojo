"""dm_control `point_mass` — the `easy` task as an env alias.

    from mojo_rl.envs.dm_control.point_mass import DMPointMassEasy
    var env = DMPointMassEasy()

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry body quaternions yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .point_mass_xml import DMPointMassModel
from .point_mass_config import DMPointMassConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMPointMassEasy[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMPointMassModel, DMPointMassConfig, DTYPE, False
]
