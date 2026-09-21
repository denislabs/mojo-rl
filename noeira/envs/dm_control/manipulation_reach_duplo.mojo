"""`dm_control` `manipulation/reach_duplo_features` env facade.

    reach_duplo_features = DMReachDuplo[DTYPE]

The `Reach` task's prop branch — same task class as `reach_site_features` and
a different task in every way that a port can observe. See
`manipulation_reach_duplo_config`'s header for the four-row table.
"""

from .manipulation_reach_duplo_def import ReachDuploModel
from .manipulation_reach_duplo_config import ReachDuploConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early.
comptime DMReachDuplo[DTYPE: DType = DType.float64] = Phyics3dEnv[
    ReachDuploModel, ReachDuploConfig, DTYPE, False
]
