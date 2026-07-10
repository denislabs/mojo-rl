"""HumanoidStandup Environment - thin wrapper over Phyics3dEnvFields[HumanoidStandupModel, HumanoidStandupConfig]."""

from .humanoid_standup_xml import HumanoidStandupModel
from .humanoid_standup_config import HumanoidStandupConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime HumanoidStandup[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    HumanoidStandupModel,
    HumanoidStandupConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
