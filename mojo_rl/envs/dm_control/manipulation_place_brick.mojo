"""`dm_control` `manipulation/place_brick_features` env facade.

    place_brick_features = DMPlaceBrick[DTYPE]

See `manipulation_place_common` for what `Place` does; this task and its twin
differ only in what the pedestal holds up.
"""

from .manipulation_place_brick_def import PlaceBrickModel
from .manipulation_place_brick_config import PlaceBrickConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMPlaceBrick[DTYPE: DType = DType.float64] = Phyics3dEnv[
    PlaceBrickModel, PlaceBrickConfig, DTYPE, False
]
