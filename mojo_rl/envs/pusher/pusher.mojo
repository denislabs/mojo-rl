"""Pusher Environment - thin wrapper over Phyics3dEnvFields[PusherModel, PusherConfig]."""

from .pusher_xml import PusherModel
from .pusher_config import PusherConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime Pusher[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    PusherModel,
    PusherConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
