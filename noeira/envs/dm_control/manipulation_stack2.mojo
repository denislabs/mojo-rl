"""`dm_control` `manipulation/stack_2_bricks_features` env facade.

    stack_2_bricks_features = DMStack2Bricks[DTYPE]

The smallest of the seven `Stack` / `Reassemble` tasks, and one of the four
whose model does NOT change per episode. See `manipulation_stack2_config`.
"""

from .manipulation_stack2_def import Stack2BricksModel
from .manipulation_stack2_config import Stack2BricksConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMStack2Bricks[DTYPE: DType = DType.float64] = Phyics3dEnv[
    Stack2BricksModel, Stack2BricksConfig, DTYPE, False
]
