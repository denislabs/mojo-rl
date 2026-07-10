"""HalfCheetah Environment - thin wrapper over Phyics3dEnvFields[HalfCheetahModel, HalfCheetahConfig].

"""


from .half_cheetah_xml import (
    HalfCheetahModel,
)
from .half_cheetah_config import HalfCheetahConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


# =============================================================================
# HalfCheetah Environment
# =============================================================================

comptime HalfCheetah[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    HalfCheetahModel,
    HalfCheetahConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
