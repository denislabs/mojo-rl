"""`dm_control` `manipulator` env facades — all four registered tasks.

    bring_ball   = DMManipulatorBringBall[DTYPE]
    bring_peg    = DMManipulatorBringPeg[DTYPE]
    insert_ball  = DMManipulatorInsertBall[DTYPE]
    insert_peg   = DMManipulatorInsertPeg[DTYPE]

Each pairs its own MODEL with a `DMManipulatorConfig[USE_PEG, INSERT]`. The
models are separate because `make_model` deletes the prop bodies each task does
not use, which renumbers everything after the arm; the config is one struct
because `Bring` is one task class.
"""

from .manipulator_xml import (
    DMManipulatorBringBallModel,
    DMManipulatorBringPegModel,
    DMManipulatorInsertBallModel,
    DMManipulatorInsertPegModel,
)
from .manipulator_config import (
    DMManipulatorBringBallConfig,
    DMManipulatorBringPegConfig,
    DMManipulatorInsertBallConfig,
    DMManipulatorInsertPegConfig,
)
from ...phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 1000-step limit.
comptime DMManipulatorBringBall[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMManipulatorBringBallModel, DMManipulatorBringBallConfig, DTYPE, False
]

comptime DMManipulatorBringPeg[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMManipulatorBringPegModel, DMManipulatorBringPegConfig, DTYPE, False
]

comptime DMManipulatorInsertBall[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMManipulatorInsertBallModel, DMManipulatorInsertBallConfig, DTYPE, False
]

comptime DMManipulatorInsertPeg[DTYPE: DType = DType.float64] = Phyics3dEnv[
    DMManipulatorInsertPegModel, DMManipulatorInsertPegConfig, DTYPE, False
]
