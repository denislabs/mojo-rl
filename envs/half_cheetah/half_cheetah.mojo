"""HalfCheetah Environment - thin wrapper around Phyics3dEnv[HalfCheetahModel, HalfCheetahConfig].

"""


from .half_cheetah_def import (
    HalfCheetahModel,
)
from .half_cheetah_config import HalfCheetahConfig
from ..phyics3d_env import Phyics3dEnv


# =============================================================================
# HalfCheetah Environment
# =============================================================================

comptime HalfCheetah[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    HalfCheetahModel,
    HalfCheetahConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
