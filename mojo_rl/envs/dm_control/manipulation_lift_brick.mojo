"""`dm_control` `manipulation/lift_brick_features` env facade.

    lift_brick_features = DMLiftBrick[DTYPE]

`Lift` with a Duplo — the same task class as `lift_large_box_features` and the
same prop as `reach_duplo_features`. See `manipulation_lift_brick_config`.
"""

from .manipulation_lift_brick_def import LiftBrickModel
from .manipulation_lift_brick_config import LiftBrickConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMLiftBrick[DTYPE: DType = DType.float64] = Phyics3dEnv[
    LiftBrickModel, LiftBrickConfig, DTYPE, False
]
