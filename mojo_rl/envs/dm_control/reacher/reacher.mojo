"""dm_control `reacher` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.reacher import DMReacherEasy, DMReacherHard
    var env = DMReacherEasy()

They differ only in the target radius the reward measures against
(`_BIG_TARGET = .05` vs `_SMALL_TARGET = .015`), which is a config comptime
here rather than the reference's per-episode `geom_size` write — the target is
inert (contact is disabled model-wide), so nothing physical depends on it.

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry the mocap fields yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .reacher_xml import DMReacherModel
from .reacher_config import DMReacherConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMReacherEasy[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMReacherModel, DMReacherConfig[0.05], DTYPE, False
]

comptime DMReacherHard[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMReacherModel, DMReacherConfig[0.015], DTYPE, False
]
