"""`dm_control` `manipulation/stack_3_bricks_random_order_features` facade.

    stack_3_bricks_random_order_features = DMStack3Random[DTYPE]

The first task whose reference model changes every episode. See
`manipulation_stack3r_config` for the relabeling that makes one static model
exact, and for the measurement that rules out freezing a free brick instead.
"""

from .manipulation_stack3r_def import Stack3RandomModel
from .manipulation_stack3r_config import Stack3RandomConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMStack3Random[DTYPE: DType = DType.float64] = Phyics3dEnv[
    Stack3RandomModel, Stack3RandomConfig, DTYPE, False
]
