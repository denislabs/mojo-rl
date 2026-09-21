"""`dm_control` `manipulation/reassemble_3_bricks_fixed_order_features` facade.

    reassemble_3_bricks_fixed_order_features = DMReassemble3[DTYPE]

The `Reassemble` logic is shared — see `manipulation_reassemble`.
"""

from .manipulation_reassemble3_def import Reassemble3Model
from .manipulation_reassemble3_config import Reassemble3Config
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMReassemble3[DTYPE: DType = DType.float64] = Phyics3dEnv[
    Reassemble3Model, Reassemble3Config, DTYPE, False
]
