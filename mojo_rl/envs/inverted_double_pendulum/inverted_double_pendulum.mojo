"""InvertedDoublePendulum Environment - thin wrapper over Phyics3dEnvFields[InvertedDoublePendulumModel, InvertedDoublePendulumConfig]."""

from .inverted_double_pendulum_xml import InvertedDoublePendulumModel
from .inverted_double_pendulum_config import InvertedDoublePendulumConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime InvertedDoublePendulum[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnvFields[
    InvertedDoublePendulumModel,
    InvertedDoublePendulumConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
