"""dm_control `point_mass` domain.

Task: easy. (`hard` needs per-episode tendon-gain randomization — Tier B.)
Reference: references/dm_control-main/dm_control/suite/point_mass.py + .xml
"""

from .point_mass import DMPointMassEasy
from .point_mass_config import DMPointMassConfig
from .point_mass_xml import (
    DMPointMassModel,
    dm_point_mass_xml,
    POINTMASS_GEOM_IDX,
    TARGET_GEOM_IDX,
    TARGET_SIZE,
)
