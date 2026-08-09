"""`dm_control` `acrobot` domain.

Tasks: swingup, swingup_sparse.
Reference: references/dm_control-main/dm_control/suite/acrobot.py + .xml
"""

from .acrobot import (
    DMAcrobotSwingup,
    DMAcrobotSwingupBatched,
    DMAcrobotSwingupSparse,
    DMAcrobotSwingupSparseBatched,
)
from .acrobot_config import DMAcrobotConfig, TARGET_RADIUS
from .acrobot_xml import (
    DMAcrobotModel,
    dm_acrobot_xml,
    UPPER_ARM_BODY_IDX,
    LOWER_ARM_BODY_IDX,
    TARGET_SITE_IDX,
    TIP_SITE_IDX,
)
