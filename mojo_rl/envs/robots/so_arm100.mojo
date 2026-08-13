"""SO-ARM100 reach — `Phyics3dEnv[SoArm100Model, SoArm100ReachConfig]`.

    from mojo_rl.envs.robots import SoArm100Reach
    var env = SoArm100Reach[DType.float64]()

The model is `so_arm100_xml.mojo` (layer-1 gated against Menagerie at
tolerance 0.0 by `tests/robots/so_arm_ref.py`); the task is the shared
`SoArmReachConfig`, which SO-ARM101 also instantiates. Read
`so_arm_reach_config.mojo`'s header for the action-space warning — actions are
JOINT POSITION TARGETS IN RADIANS with per-actuator bounds, not [-1, 1].
"""

from .so_arm100_xml import (
    SoArm100Model,
    SO_ARM100_NMESH_VERTS,
    MOVING_JAW_BODY_IDX,
    TARGET_BODY_IDX,
)
from .so_arm_reach_config import SoArmReachConfig
from ..phyics3d_env import Phyics3dEnv

from std.math import pi


# The `home` keyframe, verbatim from `trs_so_arm100/so_arm100.xml`:
#
#     <key name="home" qpos="0 -1.57 1.57 1.57 -1.57 0" ctrl="..."/>
#
# ⚠ BAKED BECAUSE `<keyframe>` IS UNPARSED (`docs/TODDLERBOT_PORT_PLAN.md`
# §4.6). `qpos0` is all zeros — the arm fully extended along -y — which is a
# perfectly stable pose and a completely different starting distribution. This
# is the reset infidelity the assessment's §8 risk 4 names, made explicit
# rather than left implicit.
comptime SO_ARM100_HOME_0: Float64 = 0.0
comptime SO_ARM100_HOME_1: Float64 = -1.57
comptime SO_ARM100_HOME_2: Float64 = 1.57
comptime SO_ARM100_HOME_3: Float64 = 1.57
comptime SO_ARM100_HOME_4: Float64 = -1.57
comptime SO_ARM100_HOME_5: Float64 = 0.0

# The `rest` keyframe, kept for the viewer and for anyone wanting the folded
# pose. Not used by reset.
comptime SO_ARM100_REST_1: Float64 = -3.32
comptime SO_ARM100_REST_2: Float64 = 3.11
comptime SO_ARM100_REST_3: Float64 = 1.18
comptime SO_ARM100_REST_5: Float64 = -0.174

# ⚠ THE ARM EXTENDS ALONG -Y AT qpos 0 — measured, the joint anchors walk from
# (0, -0.0452, 0.0165) to (0, -0.408, 0.116). So the reachable azimuth cone is
# centred on -pi/2, NOT 0. SO-101's is centred on 0, and getting these two
# swapped would put every target behind the arm and read as "the task is
# unlearnable".
comptime SO_ARM100_AZ_CENTER: Float64 = -0.5 * pi

comptime SoArm100ReachConfig = SoArmReachConfig[
    NMESHV=SO_ARM100_NMESH_VERTS,
    EE_BODY=MOVING_JAW_BODY_IDX,
    TARGET_BODY=TARGET_BODY_IDX,
    TIMESTEP=0.002,
    AZ_CENTER=SO_ARM100_AZ_CENTER,
    HOME_0=SO_ARM100_HOME_0,
    HOME_1=SO_ARM100_HOME_1,
    HOME_2=SO_ARM100_HOME_2,
    HOME_3=SO_ARM100_HOME_3,
    HOME_4=SO_ARM100_HOME_4,
    HOME_5=SO_ARM100_HOME_5,
]

comptime SoArm100Reach[
    DTYPE: DType = DType.float64,
    TERMINATE_ON_UNHEALTHY: Bool = False,
] = Phyics3dEnv[
    SoArm100Model,
    SoArm100ReachConfig,
    DTYPE,
    TERMINATE_ON_UNHEALTHY,
]
