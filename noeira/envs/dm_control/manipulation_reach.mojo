"""`dm_control` `manipulation/reach_site_features` env facade.

    reach_site_features = DMReachSiteFeatures[DTYPE]

The first of Phase 7's 13 `_features` tasks and the only one with no prop, so
it isolates the Jaco arm + 3-finger hand that all 13 share.

⚠ `reach_duplo_features` — the OTHER `reach` variant — is a different model
(the Duplo brick adds ~40 stud geoms and a freejoint) AND a different reset
(`PropPlacer(settle_physics=True)` instead of the target-site placer). It is
not this config with a flag.
"""

from .manipulation_reach_def import ReachSiteFeaturesModel
from .manipulation_reach_config import ReachSiteFeaturesConfig
from ..phyics3d_env import Phyics3dEnv


# dm_control tasks never terminate early — TERMINATE_ON_UNHEALTHY stays False
# so the driver only ever sees truncation at the 250-step limit.
comptime DMReachSiteFeatures[DTYPE: DType = DType.float64] = Phyics3dEnv[
    ReachSiteFeaturesModel, ReachSiteFeaturesConfig, DTYPE, False
]
