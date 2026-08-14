"""`dm_control` `manipulation/stack_2_of_3_bricks_random_order_features`.

    stack_2_of_3_bricks_random_order_features = DMStack2of3[DTYPE]

`Stack` with three bricks and `target_height=2` — the same model and the same
relabeling as `stack_3_bricks_random_order_features`, with one fewer brick in
the order. See `manipulation_stack_random`.
"""

from .manipulation_stack2of3_def import Stack2of3Model
from .manipulation_stack2of3_config import Stack2of3Config
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMStack2of3[DTYPE: DType = DType.float64] = Phyics3dEnv[
    Stack2of3Model, Stack2of3Config, DTYPE, False
]
