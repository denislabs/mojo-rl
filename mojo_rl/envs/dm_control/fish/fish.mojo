"""dm_control `fish` — the two registered tasks as env aliases.

    from mojo_rl.envs.dm_control.fish import DMFishUpright, DMFishSwim
    var env = DMFishSwim()

One model, two tasks; they differ in the observation (swim adds
`mouth_to_target`) and in the reward, not in the physics.

GPU-BATCHED as of 2026-08-08 (`*Batched` below). Two things had to land
first: `geom_xquat_gpu` — `mouth_to_target` expresses a world vector in the
MOUTH GEOM's frame, and that geom is a `fromto` capsule whose frame the
compiler derived, so a body quaternion is wrong by 90 degrees here — and the
actuator-cadence fix, since fish's servos are POSITION servos whose force
reads `qpos` and so must be recomputed every substep (blocker E).
"""

from .fish_xml import DMFishUprightModel, DMFishSwimModel
from .fish_config import DMFishUprightConfig, DMFishSwimConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMFishUpright[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMFishUprightModel, DMFishUprightConfig, DTYPE, False
]

comptime DMFishSwim[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMFishSwimModel, DMFishSwimConfig, DTYPE, False
]


# ── GPU-batched aliases ────────────────────────────────────────────────
comptime DMFishUprightBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMFishUprightModel, DMFishUprightConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime DMFishSwimBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMFishSwimModel, DMFishSwimConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
