"""`dm_control` `hopper` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.hopper import DMHopperStand
    var env = DMHopperStand()

**GPU-trainable as of 2026-08-07** — first domain using the GPU touch
sensor. See docs/DM_CONTROL_GPU_TRAINING_G10.md.

    from mojo_rl.envs.dm_control.hopper import DMHopperHopBatched
    var env = DMHopperHopBatched[N_ENVS=32](ctx)
"""

from .hopper_xml import DMHopperModel
from .hopper_config import DMHopperConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMHopperStand[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHopperModel, DMHopperConfig[False], DTYPE, False
]

comptime DMHopperHop[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMHopperModel, DMHopperConfig[True], DTYPE, False
]


# ── GPU-batched aliases (Phyics3dBatchedEnv, float32) ─────────────────────

comptime DMHopperStandBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMHopperModel, DMHopperConfig[False], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMHopperHopBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMHopperModel, DMHopperConfig[True], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
