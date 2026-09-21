"""`dm_control` `manipulation/place_cradle_features` env facade.

    place_cradle_features = DMPlaceCradle[DTYPE]

See `manipulation_place_common` for what `Place` does; this task and its twin
differ only in what the pedestal holds up.
"""

from .manipulation_place_cradle_def import PlaceCradleModel
from .manipulation_place_cradle_config import PlaceCradleConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMPlaceCradle[DTYPE: DType = DType.float64] = Phyics3dEnv[
    PlaceCradleModel, PlaceCradleConfig, DTYPE, False
]
