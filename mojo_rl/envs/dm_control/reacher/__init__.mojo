"""dm_control `reacher` domain.

Tasks: easy, hard.
Reference: references/dm_control-main/dm_control/suite/reacher.py + .xml
"""

from .reacher import DMReacherEasy, DMReacherHard
from .reacher_config import DMReacherConfig
from .reacher_xml import (
    DMReacherModel,
    dm_reacher_xml,
    ARM_BODY_IDX,
    HAND_BODY_IDX,
    FINGER_BODY_IDX,
    TARGET_BODY_IDX,
    FINGER_GEOM_IDX,
    TARGET_GEOM_IDX,
    FINGER_SIZE,
    TARGET_Z,
)
