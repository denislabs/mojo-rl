"""dm_control `point_mass` — the `easy` and `hard` tasks as env aliases.

    from mojo_rl.envs.dm_control.point_mass import DMPointMassEasy
    var env = DMPointMassEasy()

One model serves both, as in the reference; `hard` randomizes the tendon
mixing matrix per episode so each control drives a random linear combination
of the two joints.

point_mass-EASY is **GPU-trainable as of 2026-08-07**:

    from mojo_rl.envs.dm_control.point_mass import DMPointMassEasyBatched
    var env = DMPointMassEasyBatched[N_ENVS=64](ctx)

point_mass-HARD is GPU-trainable as of 2026-08-08, and G4 is CLOSED FOR IT
WITHOUT BATCHING `Model`. It mutates the actuator->joint mixing per episode,
and `fields.Model` is SHARED and UNBATCHED across lanes by design (the design
batches STATE, not MODEL) — so the four randomized floats live in per-env
state instead, in `d.meta`'s `META_IDX_TASK_PARAM_*` slots, written by
`init_qpos_gpu` and read by `custom_apply_actions_gpu`.

⚠ THAT SHORTCUT IS ONLY VALID BECAUSE NOTHING ELSE READS THESE TENDONS. A
`limited` tendon emits a solver limit row, a spring-loaded one a passive
force, an `<equality><tendon>` a constraint — all built from the SHARED
`Model.tendons`, which these writes do not touch. point_mass's two tendons
carry none of those, and `point_mass_hard_config` asserts it at compile time
rather than trusting it. A domain that randomizes a tendon with any of them
needs real per-env model storage. See docs/DM_CONTROL_GPU_TRAINING_G10.md.
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

comptime DMPointMassHardBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    DMPointMassModel, DMPointMassHardConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
