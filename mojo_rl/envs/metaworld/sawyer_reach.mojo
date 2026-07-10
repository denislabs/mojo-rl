"""Sawyer Reach-v3 Environment — Phyics3dEnvFields[SawyerReachModel,
SawyerReachConfig] (per-field tensor engine).

MetaWorld Reach task: move the Sawyer arm's end-effector to a goal position.
Uses mocap position control (4D action: delta XYZ + gripper). The mocap target
is welded to the hand; the fields facade presets the mocap body pose and skips
it in FK so the weld-equality solve (SOLVER=newton) tracks it — see
`phyics3d_env_fields._sync_mocap_to_fields` and `test_sawyer_fields_parity`.
"""

from .sawyer_reach_xml import SawyerReachModel
from .sawyer_reach_config import SawyerReachConfig
from ..phyics3d_env_fields import Phyics3dEnvFields


comptime SawyerReach[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
] = Phyics3dEnvFields[
    SawyerReachModel,
    SawyerReachConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
