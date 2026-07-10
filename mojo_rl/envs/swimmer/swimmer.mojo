"""Swimmer Environment - thin wrapper over Phyics3dEnvFields[SwimmerModel, SwimmerConfig]."""

from .swimmer_xml import SwimmerModel
from .swimmer_config import SwimmerConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime Swimmer[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    SwimmerModel,
    SwimmerConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
