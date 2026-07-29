"""dm_control `hopper` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.hopper import DMHopperStand
    var env = DMHopperStand()

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry body quaternions yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .hopper_xml import DMHopperModel
from .hopper_config import DMHopperConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMHopperStand[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHopperModel, DMHopperConfig[False], DTYPE, False
]

comptime DMHopperHop[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHopperModel, DMHopperConfig[True], DTYPE, False
]
