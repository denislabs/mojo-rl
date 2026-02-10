"""HalfCheetah as a compile-time robot definition.

Defines all 8 bodies and 10 joints as type aliases using BodySpec/JointSpec,
composed into HalfCheetahRobot via RobotDef. Validates that compile-time
dimensions match the existing environment (NQ=10, NV=10, NBODY=8, NJOINT=10).

Body/joint values match MuJoCo half_cheetah.xml and the existing
envs/half_cheetah_gc/ implementation.
"""

from physics3d.robot.body_spec import CapsuleBody
from physics3d.robot.joint_spec import HingeJoint, SlideJoint
from physics3d.robot.robot_def import Bodies, Joints, RobotDef


# =============================================================================
# Body Type Aliases
# =============================================================================

# Shared radius for all capsules
comptime _R: Float64 = 0.046

# Quaternion constants for body orientations
comptime _Q90Y_Y: Float64 = 0.70710678  # sin(45deg) for 90deg Y rotation
comptime _Q90Y_W: Float64 = 0.70710678  # cos(45deg) for 90deg Y rotation

# Body 0: Torso — horizontal capsule (90deg Y rotation)
# Root of kinematic tree, parent = -1
comptime Torso = CapsuleBody[
    parent= -1,
    mass=6.25,
    radius=_R,
    half_length=0.5,
    quat_y=_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,  # No self-collision (MuJoCo XML)
]

# Body 1: Back Thigh — vertical capsule at back of torso
# body_pos: (bthigh_half, 0, -torso_half) = (0.145, 0, -0.5)
# body_quat: -90deg Y to counter-rotate torso
comptime BThigh = CapsuleBody[
    parent=0,
    mass=1.54,
    radius=_R,
    half_length=0.145,
    pos_x=0.145,
    pos_z= -0.5,
    quat_y= -_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
]

# Body 2: Back Shin — vertical capsule below bthigh
# body_pos: (0, 0, -(bthigh_half + bshin_half)) = (0, 0, -0.295)
comptime BShin = CapsuleBody[
    parent=1,
    mass=1.58,
    radius=_R,
    half_length=0.15,
    pos_z= -0.295,  # -(0.145 + 0.15)
    conaffinity=0,
]

# Body 3: Back Foot — horizontal capsule (90deg Y rotation)
# body_pos: (0, 0, -bshin_half) = (0, 0, -0.15)
comptime BFoot = CapsuleBody[
    parent=2,
    mass=1.10,
    radius=_R,
    half_length=0.094,
    pos_z= -0.15,
    quat_y=_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
]

# Body 4: Front Thigh — vertical capsule at front of torso
# body_pos: (fthigh_half, 0, +torso_half) = (0.133, 0, 0.5)
comptime FThigh = CapsuleBody[
    parent=0,
    mass=1.43,
    radius=_R,
    half_length=0.133,
    pos_x=0.133,
    pos_z=0.5,
    quat_y= -_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
]

# Body 5: Front Shin — vertical capsule below fthigh
# body_pos: (0, 0, -(fthigh_half + fshin_half)) = (0, 0, -0.239)
comptime FShin = CapsuleBody[
    parent=4,
    mass=1.17,
    radius=_R,
    half_length=0.106,
    pos_z= -0.239,  # -(0.133 + 0.106)
    conaffinity=0,
]

# Body 6: Front Foot — horizontal capsule (90deg Y rotation)
# body_pos: (0, 0, -fshin_half) = (0, 0, -0.106)
comptime FFoot = CapsuleBody[
    parent=5,
    mass=0.93,
    radius=_R,
    half_length=0.07,
    pos_z= -0.106,
    quat_y=_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
]

# Body 7: Head — tilted capsule at front of torso
# MuJoCo XML: pos=".6 0 .1" axisangle="0 1 0 .87"
# In torso local frame: px=-0.1, py=0, pz=0.6
# Relative rotation: (0.87 - pi/2) rad Y ~ -0.7 rad Y
# quat for -0.7 rad Y: (0, sin(-0.35), 0, cos(-0.35)) ~ (0, -0.3429, 0, 0.9394)
comptime _HEAD_SIN_HALF: Float64 = -0.34290  # sin((0.87 - pi/2) / 2)
comptime _HEAD_COS_HALF: Float64 = 0.93937  # cos((0.87 - pi/2) / 2)

comptime Head = CapsuleBody[
    parent=0,
    mass=0.90,
    radius=_R,
    half_length=0.15,
    pos_x= -0.1,  # -HEAD_POS_Z (world z -> -local x)
    pos_z=0.6,  # HEAD_POS_X (world x -> local z)
    quat_y=_HEAD_SIN_HALF,
    quat_w=_HEAD_COS_HALF,
    conaffinity=0,
]


# =============================================================================
# Joint Type Aliases
# =============================================================================

# Joint 0: rootx — Slide along X (body 0, unactuated)
comptime RootX = SlideJoint[
    body_idx=0,
    axis_x=1.0,
    axis_y=0.0,
    axis_z=0.0,
]

# Joint 1: rootz — Slide along Z (body 0, unactuated)
comptime RootZ = SlideJoint[
    body_idx=0,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
    init_qpos=0.7,  # INIT_HEIGHT (MuJoCo qpos0)
]

# Joint 2: rooty — Hinge around Y (body 0, unactuated)
comptime RootY = HingeJoint[
    body_idx=0,
    tau_limit=0.0,
    armature=0.0,
]

# Joint 3: bthigh — Back thigh hinge (body 1)
# Joint pos in torso frame: (0, 0, -torso_half) = (0, 0, -0.5)
comptime BThighJ = HingeJoint[
    body_idx=1,
    pos_z= -0.5,
    tau_limit=120.0,
    range_min= -0.52,
    range_max=1.05,
    damping=6.0,
    stiffness=240.0,
]

# Joint 4: bshin — Back shin hinge (body 2)
# Joint pos in bthigh frame: (0, 0, -bthigh_half) = (0, 0, -0.145)
comptime BShinJ = HingeJoint[
    body_idx=2,
    pos_z= -0.145,
    tau_limit=90.0,
    range_min= -0.785,
    range_max=0.785,
    damping=4.5,
    stiffness=180.0,
]

# Joint 5: bfoot — Back foot hinge (body 3)
# Joint pos in bshin frame: (0, 0, -bshin_half) = (0, 0, -0.15)
comptime BFootJ = HingeJoint[
    body_idx=3,
    pos_z= -0.15,
    tau_limit=60.0,
    range_min= -0.4,
    range_max=0.785,
    damping=3.0,
    stiffness=120.0,
]

# Joint 6: fthigh — Front thigh hinge (body 4)
# Joint pos in torso frame: (0, 0, +torso_half) = (0, 0, 0.5)
comptime FThighJ = HingeJoint[
    body_idx=4,
    pos_z=0.5,
    tau_limit=120.0,
    range_min= -1.0,
    range_max=0.7,
    damping=4.5,
    stiffness=180.0,
]

# Joint 7: fshin — Front shin hinge (body 5)
# Joint pos in fthigh frame: (0, 0, -fthigh_half) = (0, 0, -0.133)
comptime FShinJ = HingeJoint[
    body_idx=5,
    pos_z= -0.133,
    tau_limit=60.0,
    range_min= -1.2,
    range_max=0.87,
    damping=3.0,
    stiffness=120.0,
]

# Joint 8: ffoot — Front foot hinge (body 6)
# Joint pos in fshin frame: (0, 0, -fshin_half) = (0, 0, -0.106)
comptime FFootJ = HingeJoint[
    body_idx=6,
    pos_z= -0.106,
    tau_limit=30.0,
    range_min= -0.5,
    range_max=0.5,
    damping=1.5,
    stiffness=60.0,
]

# Joint 9: head — Head hinge (body 7, fixed with zero range)
# Joint pos in torso frame: (-0.1, 0, 0.6)
comptime HeadJ = HingeJoint[
    body_idx=7,
    pos_x= -0.1,
    pos_z=0.6,
    tau_limit=0.0,
    range_min=0.0,
    range_max=0.0,
    damping=0.01,
    stiffness=8.0,
]


# =============================================================================
# HalfCheetahRobot — Full Robot Definition
# =============================================================================

comptime HalfCheetahBodies = Bodies[
    Torso, BThigh, BShin, BFoot, FThigh, FShin, FFoot, Head
]

comptime HalfCheetahJoints = Joints[
    RootX, RootZ, RootY, BThighJ, BShinJ, BFootJ, FThighJ, FShinJ, FFootJ, HeadJ
]

comptime HalfCheetahRobot = RobotDef[
    HalfCheetahBodies.N,
    HalfCheetahJoints.N,
    HalfCheetahJoints._sum_nq(),
    HalfCheetahJoints._sum_nv(),
]


# =============================================================================
# Static Assertions — verify dimensions match existing environment
# =============================================================================


fn _static_assertions():
    constrained[HalfCheetahRobot.NQ == 10, "HalfCheetah NQ must be 10"]()
    constrained[HalfCheetahRobot.NV == 10, "HalfCheetah NV must be 10"]()
    constrained[HalfCheetahRobot.NBODY == 8, "HalfCheetah NBODY must be 8"]()
    constrained[
        HalfCheetahRobot.NJOINT == 10, "HalfCheetah NJOINT must be 10"
    ]()
