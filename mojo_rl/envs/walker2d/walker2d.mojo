"""Walker2d Environment - thin wrapper over Phyics3dEnvFields[Walker2dModel, Walker2dConfig]."""

from .walker2d_xml import Walker2dModel
from .walker2d_config import Walker2dConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime Walker2d[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    Walker2dModel,
    Walker2dConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
