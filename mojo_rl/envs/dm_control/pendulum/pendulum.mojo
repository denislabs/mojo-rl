"""dm_control `pendulum-swingup` — Phyics3dEnv[DMPendulumModel, DMPendulumConfig].

    from mojo_rl.envs.dm_control.pendulum import DMPendulum
    var env = DMPendulum()

CPU only: the config's GPU reward/obs hooks are stubs because the batched
hook ABI does not carry body quaternions yet (gap G10). See
docs/DM_CONTROL_PORT.md.
"""

from .pendulum_xml import DMPendulumModel
from .pendulum_config import DMPendulumConfig
from ...phyics3d_env import Phyics3dEnv


comptime DMPendulum[
    DTYPE: DType = DType.float64,
] = Phyics3dEnv[
    DMPendulumModel,
    DMPendulumConfig,
    DTYPE,
    # dm_control tasks never terminate early — episodes end on the time limit
    # only. Keep this False so the driver never sees `terminated`.
    False,
]
