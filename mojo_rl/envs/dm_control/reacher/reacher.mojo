"""dm_control `reacher` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.reacher import DMReacherEasy, DMReacherHard
    var env = DMReacherEasy()

They differ only in the target radius (`_BIG_TARGET = .05` vs
`_SMALL_TARGET = .015`). The target is inert (contact is disabled model-wide),
so nothing PHYSICAL depends on it: the reward reads it from a config comptime
rather than the reference's per-episode `geom_size` write.

⚠ THEY DO NOT SHARE A MODEL, though, because the radius is also the drawn size
and the renderer resolves geom sizes at compile time. See
`reacher_xml.DMReacherHardModel`.

CPU only: the config's GPU reward/obs hooks are stubs because the batched hook
ABI does not carry the mocap fields yet (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .reacher_xml import DMReacherModel, DMReacherHardModel
from .reacher_config import DMReacherConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMReacherEasy[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMReacherModel, DMReacherConfig[0.05], DTYPE, False
]

comptime DMReacherHard[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMReacherHardModel, DMReacherConfig[0.015], DTYPE, False
]
"""⚠ A DIFFERENT MODEL FROM `DMReacherEasy`, not just a different config. The
two carry the same physics — the target is inert — but `hard`'s target geom is
`.015` where `easy`'s is `.05`, and the renderer reads that radius at COMPILE
TIME. Sharing the model drew `hard`'s 1.5 cm disc as a 5 cm ball."""
