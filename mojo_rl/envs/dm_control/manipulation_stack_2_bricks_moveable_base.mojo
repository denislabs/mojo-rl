"""`dm_control` `manipulation/stack_2_bricks_moveable_base_features` env facade.

    stack_2_bricks_moveable_base_features = DMStack2Moveable[DTYPE]

The fixed-order `Stack` logic is shared — see `manipulation_stack_fixed`.
"""

from .manipulation_stack_2_bricks_moveable_base_def import Stack2MoveableModel
from .manipulation_stack_2_bricks_moveable_base_config import Stack2MoveableConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMStack2Moveable[DTYPE: DType = DType.float64] = Phyics3dEnv[
    Stack2MoveableModel, Stack2MoveableConfig, DTYPE, False
]
