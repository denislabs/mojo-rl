"""Hopper Environment - thin wrapper around Phyics3dEnv[HopperModel, HopperConfig].

"""


from .hopper_xml import (
    HopperModel,
)
from .hopper_config import HopperConfig
from ..phyics3d_env import Phyics3dEnv


# =============================================================================
# Hopper Environment
# =============================================================================

comptime Hopper[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    HopperModel,
    HopperConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
