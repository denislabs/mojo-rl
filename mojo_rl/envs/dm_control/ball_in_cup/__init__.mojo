"""dm_control `ball_in_cup` domain.

Task: catch. The first ported domain with a SPATIAL (site-routed) tendon and
the first with a tendon LIMIT — see `ball_in_cup_xml` for what that required.
Reference: references/dm_control-main/dm_control/suite/ball_in_cup.py + .xml
"""

from .ball_in_cup import DMBallInCupCatch
from .ball_in_cup_config import DMBallInCupConfig
from .ball_in_cup_xml import (
    DMBallInCupModel,
    dm_ball_in_cup_xml,
    BALL_BODY_IDX,
    CUP_SITE_IDX,
    TARGET_SITE_IDX,
    BALL_SITE_IDX,
    BALL_GEOM_IDX,
    CUP_GEOM_FIRST,
    CUP_GEOM_LAST,
    TARGET_HALF_X,
    TARGET_HALF_Z,
    BALL_RADIUS,
)
