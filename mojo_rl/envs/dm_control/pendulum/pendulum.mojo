"""dm_control `pendulum-swingup` — Phyics3dEnv[DMPendulumModel, DMPendulumConfig].

    from mojo_rl.envs.dm_control.pendulum import DMPendulum, DMPendulumBatched
    var env = DMPendulum()                       # CPU, float64, one env
    var benv = DMPendulumBatched[N_ENVS=64](ctx) # GPU, float32, batched

**GPU-trainable as of 2026-08-06** — the first suite task off the CPU-only
list. `DMPendulumConfig` now implements `custom_extract_obs_gpu`,
`compute_reward_and_done_gpu` and `init_qpos_gpu`, which the widened hook ABI
(`xquat`) and the DTYPE-generic `tolerance` made expressible. See
docs/DM_CONTROL_GPU_TRAINING_G10.md.
"""

from std.gpu.host import DeviceContext

from .pendulum_xml import DMPendulumModel
from .pendulum_config import DMPendulumConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


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


comptime DMPendulumBatched[
    N_ENVS: Int,
] = Phyics3dBatchedEnv[
    DMPendulumModel,
    DMPendulumConfig,
    N_ENVS,
    # As above: truncation at MAX_STEPS is the only episode end.
    TERMINATE_ON_UNHEALTHY=False,
]
