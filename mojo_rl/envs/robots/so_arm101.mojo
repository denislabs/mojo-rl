"""SO-ARM101 reach — `Phyics3dEnv[SoArm101Model, SoArm101ReachConfig]`.

    from mojo_rl.envs.robots import SoArm101Reach
    var env = SoArm101Reach[DType.float64]()

Same task as SO-ARM100, deliberately: the two envs differ in their ROBOT, not
in what is asked of it, which is what makes a cross-model policy comparison
mean anything. See `so_arm101_xml.mojo` for the `fullinertia` bake and the 10x
collision-mesh cost, and `docs/SO_ARM101_PORT_ASSESSMENT.md` §5 for why SO-100
is the better first target even though this is the arm on the desk.

⚠ THIS IS THE MODEL TO TRAIN ON IF THE POLICY IS GOING ON REAL HARDWARE.
SO-100 and SO-101 are 25% apart on moving mass and 68% on the jaw. The
assessment's §5 says to decide before Phase 5 rather than after; this facade
existing is not that decision.
"""

from .so_arm101_xml import (
    SoArm101Model,
    SO_ARM101_NMESH_VERTS,
    MOVING_JAW_BODY_IDX,
    TARGET_BODY_IDX,
)
from .so_arm_reach_config import SoArmReachConfig
from ..phyics3d_env import Phyics3dEnv


# ⚠ NO KEYFRAME EXISTS — measured, `nkey = 0`. Unlike SO-100 this is not a
# gap we are papering over: upstream ships no reset pose, and `new_calib`
# already places every joint's zero at the MIDDLE of its range, so all-zeros
# is a sensible folded pose rather than the fully-extended one SO-100's
# `qpos0` gives. Left explicit so the asymmetry with `so_arm100.mojo` reads as
# a measured difference rather than an oversight.
comptime SO_ARM101_HOME_0: Float64 = 0.0
comptime SO_ARM101_HOME_1: Float64 = 0.0
comptime SO_ARM101_HOME_2: Float64 = 0.0
comptime SO_ARM101_HOME_3: Float64 = 0.0
comptime SO_ARM101_HOME_4: Float64 = 0.0
comptime SO_ARM101_HOME_5: Float64 = 0.0

# ⚠ THE ARM EXTENDS ALONG +X AT qpos 0 — measured, the joint anchors walk from
# (0.0388, 0, 0.0624) to (0.317, 0.018, 0.255). Centre 0, where SO-100's is
# -pi/2. The two are NOT interchangeable; see `so_arm100.mojo`'s note.
comptime SO_ARM101_AZ_CENTER: Float64 = 0.0

comptime SoArm101ReachConfig = SoArmReachConfig[
    NMESHV=SO_ARM101_NMESH_VERTS,
    EE_BODY=MOVING_JAW_BODY_IDX,
    TARGET_BODY=TARGET_BODY_IDX,
    TIMESTEP=0.002,
    AZ_CENTER=SO_ARM101_AZ_CENTER,
    HOME_0=SO_ARM101_HOME_0,
    HOME_1=SO_ARM101_HOME_1,
    HOME_2=SO_ARM101_HOME_2,
    HOME_3=SO_ARM101_HOME_3,
    HOME_4=SO_ARM101_HOME_4,
    HOME_5=SO_ARM101_HOME_5,
]

comptime SoArm101Reach[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
] = Phyics3dEnv[
    SoArm101Model,
    SoArm101ReachConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
