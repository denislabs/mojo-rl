"""dm_control `cheetah` — the single registered task as an env alias.

    from mojo_rl.envs.dm_control.cheetah import DMCheetahRun
    var env = DMCheetahRun()

**GPU-trainable as of 2026-08-06** — see docs/DM_CONTROL_GPU_TRAINING_G10.md.

    from mojo_rl.envs.dm_control.cheetah import DMCheetahRunBatched
    var env = DMCheetahRunBatched[N_ENVS=64](ctx)
"""

from .cheetah_xml import DMCheetahModel
from .cheetah_config import DMCheetahConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMCheetahRun[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMCheetahModel, DMCheetahConfig, DTYPE, False
]


comptime DMCheetahRunBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMCheetahModel, DMCheetahConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
