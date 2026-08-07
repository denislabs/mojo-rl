"""dm_control `hopper` domain.

Tasks: stand, hop.
Reference: references/dm_control-main/dm_control/suite/hopper.py + .xml
"""

from .hopper import (
    DMHopperStand,
    DMHopperStandBatched,
    DMHopperHop,
    DMHopperHopBatched,
)
from .hopper_config import DMHopperConfig, STAND_HEIGHT, HOP_SPEED
from .hopper_xml import (
    DMHopperModel,
    dm_hopper_xml,
    TORSO_BODY_IDX,
    FOOT_BODY_IDX,
    TOUCH_TOE_SITE_IDX,
    TOUCH_HEEL_SITE_IDX,
)
