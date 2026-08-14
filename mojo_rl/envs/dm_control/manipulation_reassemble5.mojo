"""`dm_control` `manipulation/reassemble_5_bricks_random_order_features` facade.

    reassemble_5_bricks_random_order_features = DMReassemble5[DTYPE]

The `Reassemble` logic and the relabeling are shared — see
`manipulation_reassemble`.
"""

from .manipulation_reassemble5_def import Reassemble5Model
from .manipulation_reassemble5_config import Reassemble5Config
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMReassemble5[DTYPE: DType = DType.float64] = Phyics3dEnv[
    Reassemble5Model, Reassemble5Config, DTYPE, False
]
