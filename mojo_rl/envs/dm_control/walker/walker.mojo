"""dm_control `walker` — the three registered tasks as env aliases.

    from mojo_rl.envs.dm_control.walker import DMWalkerWalk
    var env = DMWalkerWalk()

**GPU-trainable as of 2026-08-06** — see docs/DM_CONTROL_GPU_TRAINING_G10.md.

    from mojo_rl.envs.dm_control.walker import DMWalkerWalkBatched
    var env = DMWalkerWalkBatched[N_ENVS=64](ctx)
"""

from .walker_xml import DMWalkerModel
from .walker_config import DMWalkerConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.

comptime DMWalkerStand[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[0.0], DTYPE, False
]

comptime DMWalkerWalk[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[1.0], DTYPE, False
]

comptime DMWalkerRun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[8.0], DTYPE, False
]


# ── GPU-batched aliases (Phyics3dBatchedEnv, float32) ─────────────────────

comptime DMWalkerStandBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[0.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime DMWalkerWalkBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime DMWalkerRunBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[8.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
