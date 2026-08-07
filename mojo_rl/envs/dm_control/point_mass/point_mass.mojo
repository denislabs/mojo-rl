"""dm_control `point_mass` — the `easy` and `hard` tasks as env aliases.

    from mojo_rl.envs.dm_control.point_mass import DMPointMassEasy
    var env = DMPointMassEasy()

One model serves both, as in the reference; `hard` randomizes the tendon
mixing matrix per episode so each control drives a random linear combination
of the two joints.

point_mass-EASY is **GPU-trainable as of 2026-08-07**:

    from mojo_rl.envs.dm_control.point_mass import DMPointMassEasyBatched
    var env = DMPointMassEasyBatched[N_ENVS=64](ctx)

⚠ `hard` stays CPU-only, and not for want of effort: it mutates
`Model.tendons` per episode (the actuator->joint mixing matrix), and
`fields.Model` is SHARED and UNBATCHED across lanes — every env would get the
last one's draw. That is gap G4. See docs/DM_CONTROL_GPU_TRAINING_G10.md.
"""

from .point_mass_xml import DMPointMassModel
from .point_mass_config import DMPointMassConfig
from .point_mass_hard_config import DMPointMassHardConfig
from ...phyics3d_env import Phyics3dEnv
from ...phyics3d_batched_env import Phyics3dBatchedEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMPointMassEasy[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMPointMassModel, DMPointMassConfig, DTYPE, False
]

comptime DMPointMassHard[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMPointMassModel, DMPointMassHardConfig, DTYPE, False
]


comptime DMPointMassEasyBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMPointMassModel, DMPointMassConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
