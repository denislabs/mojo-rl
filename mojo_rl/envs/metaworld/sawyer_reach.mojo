"""Sawyer Reach-v3 Environment — Phyics3dEnv[SawyerReachModel,
SawyerReachConfig] (per-field tensor engine).

MetaWorld Reach task: move the Sawyer arm's end-effector to a goal position.
Uses mocap position control (4D action: delta XYZ + gripper). The mocap target
is welded to the hand; the fields facade presets the mocap body pose and skips
it in FK so the weld-equality solve (SOLVER=newton) tracks it — see
`phyics3d_env._sync_mocap_to_fields`. Gated by
`test_sawyer_settle_vs_mujoco`'s WELDED-BODY assertion (`HAND_TOL`), which
compares `xpos[eq_obj2id]` against MuJoCo's after an 800-step rollout — 0.91 mm.
⚠ It used to cite `test_sawyer_fields_parity`, WHICH HAS NEVER EXISTED, and
defect 28 (a 77.6 mm sag on this exact path) hid behind that citation.
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
