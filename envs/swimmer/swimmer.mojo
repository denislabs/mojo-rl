"""Swimmer Environment - thin wrapper around Phyics3dEnv[SwimmerModel, SwimmerConfig]."""

from .swimmer_xml import SwimmerModel
from .swimmer_config import SwimmerConfig
from ..phyics3d_env import Phyics3dEnv


comptime Swimmer[
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    SwimmerModel,
    SwimmerConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
