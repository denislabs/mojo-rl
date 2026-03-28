"""Pusher Environment - thin wrapper around Phyics3dEnv[PusherModel, PusherConfig]."""

from .pusher_xml import PusherModel
from .pusher_config import PusherConfig
from ..phyics3d_env import Phyics3dEnv


comptime Pusher[
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    PusherModel,
    PusherConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
