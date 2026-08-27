"""`dm_control` `cartpole` — the six registered tasks as env aliases.

    from mojo_rl.envs.dm_control.cartpole import DMCartpoleSwingup
    var env = DMCartpoleSwingup()

**GPU-trainable as of 2026-08-06** — `DMCartpoleConfig` implements the GPU
obs/reward/reset hooks. Each task has a `*Batched` alias:

    from mojo_rl.envs.dm_control.cartpole import DMCartpoleSwingupBatched
    var env = DMCartpoleSwingupBatched[N_ENVS=64](ctx)

See docs/DM_CONTROL_GPU_TRAINING_G10.md.
"""

from .cartpole_xml import (
    DMCartpole1Model,
    DMCartpole2Model,
    DMCartpole3Model,
)
from .cartpole_config import DMCartpoleConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


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


# ── GPU-batched aliases (Phyics3dBatchedEnv, float32) ─────────────────

comptime DMCartpoleBalanceBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMCartpole1Model, DMCartpoleConfig[1, False, False], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMCartpoleBalanceSparseBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMCartpole1Model, DMCartpoleConfig[1, False, True], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMCartpoleSwingupBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMCartpole1Model, DMCartpoleConfig[1, True, False], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMCartpoleSwingupSparseBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMCartpole1Model, DMCartpoleConfig[1, True, True], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMCartpoleTwoPolesBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMCartpole2Model, DMCartpoleConfig[2, True, False], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMCartpoleThreePolesBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMCartpole3Model, DMCartpoleConfig[3, True, False], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
