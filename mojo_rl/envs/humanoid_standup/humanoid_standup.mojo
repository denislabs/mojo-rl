"""HumanoidStandup Environment - thin wrapper around Phyics3dEnv[HumanoidStandupModel, HumanoidStandupConfig]."""

from .humanoid_standup_xml import HumanoidStandupModel
from .humanoid_standup_config import HumanoidStandupConfig
from ..phyics3d_env import Phyics3dEnv


comptime HumanoidStandup[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    HumanoidStandupModel,
    HumanoidStandupConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
