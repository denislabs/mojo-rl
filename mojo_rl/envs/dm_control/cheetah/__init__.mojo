"""dm_control `cheetah` domain.

Task: run.
Reference: references/dm_control-main/dm_control/suite/cheetah.py + .xml
"""

from .cheetah import DMCheetahRun, DMCheetahRunBatched
from .cheetah_config import DMCheetahConfig, RUN_SPEED
from .cheetah_xml import DMCheetahModel, dm_cheetah_xml, TORSO_BODY_IDX
