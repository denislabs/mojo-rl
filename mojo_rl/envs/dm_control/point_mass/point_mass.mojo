"""dm_control `point_mass` — the `easy` and `hard` tasks as env aliases.

    from mojo_rl.envs.dm_control.point_mass import DMPointMassEasy
    var env = DMPointMassEasy()

One model serves both, as in the reference; `hard` randomizes the tendon
mixing matrix per episode so each control drives a random linear combination
of the two joints.

CPU only: the configs' GPU reward/obs hooks are stubs because the batched hook
ABI does not carry body quaternions yet (gap G10). `hard` is CPU-only for a
second, independent reason — its per-episode model randomization writes the
HOST tendon records, which no GPU path re-uploads.
See docs/DM_CONTROL_PORT.md.
"""

from .point_mass_xml import DMPointMassModel
from .point_mass_config import DMPointMassConfig
from .point_mass_hard_config import DMPointMassHardConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMPointMassEasy[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMPointMassModel, DMPointMassConfig, DTYPE, False
]

comptime DMPointMassHard[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMPointMassModel, DMPointMassHardConfig, DTYPE, False
]
