"""Reacher Environment - thin wrapper over Phyics3dEnvFields[ReacherModel, ReacherConfig]."""

from .reacher_xml import ReacherModel
from .reacher_config import ReacherConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime Reacher[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    ReacherModel,
    ReacherConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
