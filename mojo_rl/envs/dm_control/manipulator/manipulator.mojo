"""dm_control `manipulator-bring_ball` env facade.

    bring_ball = DMManipulatorBringBall[DTYPE]

The other three tasks (`bring_peg`, `insert_ball`, `insert_peg`) are SEPARATE
MODELS, not flags on this one — `make_model` deletes the prop bodies each task
does not use, which renumbers everything after the arm. They are not ported.
"""

from .manipulator_xml import DMManipulatorBringBallModel
from .manipulator_config import DMManipulatorConfig
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMManipulatorBringBall[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMManipulatorBringBallModel, DMManipulatorConfig, DTYPE, False
]
