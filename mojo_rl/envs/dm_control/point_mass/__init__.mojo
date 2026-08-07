"""dm_control `point_mass` domain.

Tasks: easy, hard.
Reference: references/dm_control-main/dm_control/suite/point_mass.py + .xml
"""

from .point_mass import (
    DMPointMassEasy,
    DMPointMassEasyBatched,
    DMPointMassHard,
)
from .point_mass_config import DMPointMassConfig
from .point_mass_hard_config import DMPointMassHardConfig
from .point_mass_xml import (
    DMPointMassModel,
    dm_point_mass_xml,
    POINTMASS_GEOM_IDX,
    TARGET_GEOM_IDX,
    TARGET_SIZE,
    T1_TENDON_IDX,
    T2_TENDON_IDX,
)
