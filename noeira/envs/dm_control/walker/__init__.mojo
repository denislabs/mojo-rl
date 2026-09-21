"""`dm_control` `walker` domain.

Tasks: stand, walk, run.
Reference: references/dm_control-main/dm_control/suite/walker.py + .xml
"""

from .walker import (
    DMWalkerStand,
    DMWalkerStandBatched,
    DMWalkerWalk,
    DMWalkerWalkBatched,
    DMWalkerRun,
    DMWalkerRunBatched,
)
from .walker_config import DMWalkerConfig, STAND_HEIGHT
from .walker_xml import DMWalkerModel, TORSO_BODY_IDX
