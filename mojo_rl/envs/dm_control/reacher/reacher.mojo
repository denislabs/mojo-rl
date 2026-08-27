"""`dm_control` `reacher` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.reacher import DMReacherEasy, DMReacherHard
    var env = DMReacherEasy()

They differ only in the target radius (`_BIG_TARGET = .05` vs
`_SMALL_TARGET = .015`). The target is inert (contact is disabled model-wide),
so nothing PHYSICAL depends on it: the reward reads it from a config comptime
rather than the reference's per-episode `geom_size` write.

⚠ THEY DO NOT SHARE A MODEL, though, because the radius is also the drawn size
and the renderer resolves geom sizes at compile time. See
`reacher_xml.DMReacherHardModel`.

**GPU-trainable as of 2026-08-07** — first MOCAP domain on the batched
path (blocker H). See docs/DM_CONTROL_GPU_TRAINING_G10.md.

    from mojo_rl.envs.dm_control.reacher import DMReacherEasyBatched
    var env = DMReacherEasyBatched[N_ENVS=64](ctx)
"""

from .reacher_xml import DMReacherModel, DMReacherHardModel
from .reacher_config import DMReacherConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


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


# ── GPU-batched aliases (Phyics3dBatchedEnv, float32) ─────────────────────
# ⚠ Two MODELS, not one — same reason as the CPU aliases above: `hard`'s target
# geom radius is comptime data.

comptime DMReacherEasyBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMReacherModel, DMReacherConfig[0.05], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMReacherHardBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMReacherHardModel, DMReacherConfig[0.015], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
