"""Ant Environment - thin wrapper around Phyics3dEnv[AntModel, AntConfig].

MuJoCo Ant-v5 quadruped with free joint root.
13 bodies, 9 joints (1 free + 8 hinge), 8 actuators.
OBS_DIM=27 (13 qpos + 14 qvel, excluding x,y), ACTION_DIM=8.
"""


from .ant_xml import AntModel
from .ant_config import AntConfig
from ..phyics3d_env import Phyics3dEnv


# =============================================================================
# Ant Environment
# =============================================================================

comptime Ant[
    DTYPE: DType where DTYPE.is_floating_point() = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = True,
] = Phyics3dEnv[
    AntModel,
    AntConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
