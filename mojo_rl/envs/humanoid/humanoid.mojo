"""Humanoid Environment - thin wrapper over Phyics3dEnvFields[HumanoidModel, HumanoidConfig]."""

from .humanoid_xml import HumanoidModel
from .humanoid_config import HumanoidConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime Humanoid[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    HumanoidModel,
    HumanoidConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
