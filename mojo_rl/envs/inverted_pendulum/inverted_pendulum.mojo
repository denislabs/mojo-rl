"""InvertedPendulum Environment - thin wrapper around Phyics3dEnv[InvertedPendulumModel, InvertedPendulumConfig]."""

from .inverted_pendulum_xml import InvertedPendulumModel
from .inverted_pendulum_config import InvertedPendulumConfig
from ..phyics3d_env import Phyics3dEnv


comptime InvertedPendulum[
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    InvertedPendulumModel,
    InvertedPendulumConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
