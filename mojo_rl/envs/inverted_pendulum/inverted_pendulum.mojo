"""InvertedPendulum Environment - thin wrapper over Phyics3dEnvFields[InvertedPendulumModel, InvertedPendulumConfig]."""

from .inverted_pendulum_xml import InvertedPendulumModel
from .inverted_pendulum_config import InvertedPendulumConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime InvertedPendulum[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    InvertedPendulumModel,
    InvertedPendulumConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
