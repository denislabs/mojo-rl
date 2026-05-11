"""Sawyer Reach-v3 Environment — Phyics3dEnv[SawyerReachModel, SawyerReachConfig].

MetaWorld Reach task: move the Sawyer arm's end-effector to a goal position.
Uses mocap position control (4D action: delta XYZ + gripper).
"""

from .sawyer_reach_xml import SawyerReachModel
from .sawyer_reach_config import SawyerReachConfig
from ..phyics3d_env import Phyics3dEnv


comptime SawyerReach[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
] = Phyics3dEnv[
    SawyerReachModel,
    SawyerReachConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
