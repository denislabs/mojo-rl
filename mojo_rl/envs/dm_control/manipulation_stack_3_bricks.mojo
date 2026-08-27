"""`dm_control` `manipulation/stack_3_bricks_features` env facade.

    stack_3_bricks_features = DMStack3Bricks[DTYPE]

The fixed-order `Stack` logic is shared — see `manipulation_stack_fixed`.
"""

from .manipulation_stack_3_bricks_def import Stack3BricksModel
from .manipulation_stack_3_bricks_config import Stack3BricksConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMStack3Bricks[DTYPE: DType = DType.float64] = Phyics3dEnv[
    Stack3BricksModel, Stack3BricksConfig, DTYPE, False
]
