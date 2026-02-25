"""InvertedDoublePendulum Environment - thin wrapper around Phyics3dEnv[InvertedDoublePendulumModel, InvertedDoublePendulumConfig]."""

from .inverted_double_pendulum_xml import InvertedDoublePendulumModel
from .inverted_double_pendulum_config import InvertedDoublePendulumConfig
from ..phyics3d_env import Phyics3dEnv


comptime InvertedDoublePendulum[
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    InvertedDoublePendulumModel,
    InvertedDoublePendulumConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
