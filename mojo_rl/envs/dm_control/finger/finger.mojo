"""dm_control `finger` — the three registered tasks as env aliases.

    from mojo_rl.envs.dm_control.finger import DMFingerSpin, DMFingerTurnEasy
    var env = DMFingerSpin()

`spin` compiles from its OWN model (`DMFingerSpinModel`): the reference's
`Spin.initialize_episode` lowers `dof_damping['hinge']` from .5 to .03, which
is a dynamics change our shared, unbatched `fields.Model` cannot express as a
per-episode write. The two turn tasks differ only in the target radius the
reward measures against (`_EASY_TARGET_SIZE = .07`, `_HARD_TARGET_SIZE = .03`),
a config comptime here rather than a per-episode `site_size` write.

CPU only, as reacher: the batched hook ABI does not carry the mocap fields
that the turn target needs (gap G10). See docs/DM_CONTROL_PORT.md.
"""

from .finger_xml import DMFingerSpinModel, DMFingerTurnModel
from .finger_config import DMFingerSpinConfig, DMFingerTurnConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — the driver only sees truncation at
# the 1000-step limit.
comptime DMFingerSpin[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMFingerSpinModel, DMFingerSpinConfig, DTYPE, False
]

comptime DMFingerTurnEasy[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMFingerTurnModel, DMFingerTurnConfig[0.07], DTYPE, False
]

comptime DMFingerTurnHard[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMFingerTurnModel, DMFingerTurnConfig[0.03], DTYPE, False
]
