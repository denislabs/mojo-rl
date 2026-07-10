"""Reacher Environment - thin wrapper over Phyics3dEnv[ReacherModel, ReacherConfig]."""

from .reacher_xml import ReacherModel
from .reacher_config import ReacherConfig
from ..phyics3d_env import Phyics3dEnv


comptime Reacher[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    ReacherModel,
    ReacherConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
