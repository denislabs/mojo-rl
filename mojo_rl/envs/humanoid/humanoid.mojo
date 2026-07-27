"""Humanoid Environment - thin wrapper over Phyics3dEnv[HumanoidModel, HumanoidConfig]."""

from .humanoid_xml import HumanoidModel
from .humanoid_config import HumanoidConfig
from ..phyics3d_env import Phyics3dEnv


comptime Humanoid[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    HumanoidModel,
    HumanoidConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
