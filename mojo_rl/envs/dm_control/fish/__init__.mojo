"""`dm_control` `fish` domain.

Tasks: upright, swim.
Reference: references/dm_control-main/dm_control/suite/fish.py + .xml
"""

from .fish import (
    DMFishUpright,
    DMFishSwim,
    DMFishUprightBatched,
    DMFishSwimBatched,
)
from .fish_config import DMFishUprightConfig, DMFishSwimConfig, SWIM_RADII
from .fish_xml import (
    DMFishUprightModel,
    DMFishSwimModel,
    TORSO_BODY_IDX,
    TARGET_BODY_IDX,
    MOUTH_GEOM_IDX,
    TARGET_GEOM_IDX,
    N_ROOT_QPOS,
    FREE_QUAT_ADR,
    MOUTH_RADIUS,
    TARGET_RADIUS,
)
