"""`dm_control` `swimmer` domain.

Tasks: swimmer6, swimmer15.
Reference: references/dm_control-main/dm_control/suite/swimmer.py + .xml
"""

from .swimmer import (
    DMSwimmer6,
    DMSwimmer15,
    DMSwimmer6Batched,
    DMSwimmer15Batched,
)
from .swimmer_config import DMSwimmerConfig
from .swimmer_xml import (
    DMSwimmer6Model,
    DMSwimmer15Model,
    dm_swimmer6_xml,
    dm_swimmer15_xml,
    HEAD_BODY_IDX,
    FIRST_SEGMENT_BODY_IDX,
    GROUND_GEOM_IDX,
    HEAD_GEOM_IDX,
    NOSE_GEOM_IDX,
    N_ROOT_DOF,
    TARGET_SIZE,
    TARGET_Z,
)
