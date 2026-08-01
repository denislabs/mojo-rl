"""dm_control `stacker` env facades — both registered tasks.

    stack_2 = DMStacker2[DTYPE]
    stack_4 = DMStacker4[DTYPE]

Each pairs its own MODEL with a `DMStackerConfig[N_BOXES]`. The models are
separate because `make_model` deletes the boxes each task does not use, which
renumbers the target after them; the config is one struct because `Stack` is
one task class.
"""

from .stacker_xml import DMStacker2Model, DMStacker4Model
from .stacker_config import DMStacker2Config, DMStacker4Config
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False so
# the driver only ever sees truncation at the 1000-step limit.
comptime DMStacker2[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMStacker2Model, DMStacker2Config, DTYPE, False
]

comptime DMStacker4[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMStacker4Model, DMStacker4Config, DTYPE, False
]
