"""dm_control `acrobot` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.acrobot import DMAcrobotSwingup
    var env = DMAcrobotSwingup()

**GPU-trainable as of 2026-08-07** — see
docs/DM_CONTROL_GPU_TRAINING_G10.md.

    from mojo_rl.envs.dm_control.acrobot import DMAcrobotSwingupBatched
    var env = DMAcrobotSwingupBatched[N_ENVS=64](ctx)
"""

from .acrobot_xml import DMAcrobotModel
from .acrobot_config import DMAcrobotConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMAcrobotSwingup[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMAcrobotModel, DMAcrobotConfig[False], DTYPE, False
]

comptime DMAcrobotSwingupSparse[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMAcrobotModel, DMAcrobotConfig[True], DTYPE, False
]


# ── GPU-batched aliases (Phyics3dBatchedEnv, float32) ─────────────────────

comptime DMAcrobotSwingupBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMAcrobotModel, DMAcrobotConfig[False], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMAcrobotSwingupSparseBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMAcrobotModel, DMAcrobotConfig[True], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
