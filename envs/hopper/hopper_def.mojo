"""Hopper body and joint definitions as compile-time BodySpec/JointSpec types.

Defines 4 bodies and 6 joints matching MuJoCo hopper.xml. Used by
RobotRenderer for automated rendering. Values match existing
HopperConstants and hopper.mojo body setup code.
"""

from physics3d.robot.body_spec import CapsuleBody
from physics3d.robot.joint_spec import HingeJoint, SlideJoint
from physics3d.robot.robot_def import Bodies, Joints, RobotDef
from render3d import Color3D


# =============================================================================
# Body Type Aliases
# =============================================================================

# Quaternion constants for 90deg Y rotation (foot orientation)
comptime _Q90Y_Y: Float64 = 0.70710678
comptime _Q90Y_W: Float64 = 0.70710678

# Body 0: Torso — vertical capsule, root of kinematic tree
comptime HopperTorso = CapsuleBody[
    parent= -1,
    mass=3.53429174,
    radius=0.05,
    half_length=0.2,
    color=Color3D(60, 120, 200),
]

# Body 1: Thigh — vertical capsule below torso
# pos_z = -(torso_half + thigh_half) = -(0.2 + 0.225) = -0.425
comptime HopperThigh = CapsuleBody[
    parent=0,
    mass=3.92699082,
    radius=0.05,
    half_length=0.225,
    pos_z= -0.425,
    color=Color3D(80, 200, 80),
]

# Body 2: Leg — vertical capsule below thigh
# pos_z = -(thigh_half + leg_half) = -(0.225 + 0.25) = -0.475
comptime HopperLeg = CapsuleBody[
    parent=1,
    mass=2.71433605,
    radius=0.04,
    half_length=0.25,
    pos_z= -0.475,
    color=Color3D(220, 140, 60),
]

# Body 3: Foot — horizontal capsule (90deg Y rotation), below leg
# pos_z = -leg_half = -0.25
comptime HopperFoot = CapsuleBody[
    parent=2,
    mass=5.0893801,
    radius=0.06,
    half_length=0.195,
    pos_z= -0.25,
    quat_y=_Q90Y_Y,
    quat_w=_Q90Y_W,
    color=Color3D(220, 80, 80),
]


# =============================================================================
# Joint Type Aliases
# =============================================================================

# Joint 0: rootx — Slide along X (body 0, unactuated)
comptime HopperRootX = SlideJoint[
    body_idx=0,
    axis_x=1.0,
    axis_y=0.0,
    axis_z=0.0,
    exclude_obs_qpos=True,
]

# Joint 1: rootz — Slide along Z (body 0, unactuated)
# init_qpos = INITIAL_Z (computed from body geometry stack)
comptime HopperRootZ = SlideJoint[
    body_idx=0,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
    init_qpos=1.25,  # FOOT_R + LEG_R + 2*LEG_HL + 2*THIGH_HL + TORSO_HL
]

# Joint 2: rooty — Hinge around Y (body 0, unactuated)
comptime HopperRootY = HingeJoint[
    body_idx=0,
    tau_limit=0.0,
    armature=0.0,
    is_actuated=False,
    has_limits=False,
]

# Joint 3: thigh — Hinge around Y (body 1)
# Joint at bottom of torso: (0, 0, -torso_half)
comptime HopperThighJ = HingeJoint[
    body_idx=1,
    pos_z= -0.2,
    tau_limit=200.0,
    range_min= -2.618,
    range_max=0.0,
    armature=1.0,
    damping=1.0,
]

# Joint 4: leg — Hinge around Y (body 2)
# Joint at bottom of thigh: (0, 0, -thigh_half)
comptime HopperLegJ = HingeJoint[
    body_idx=2,
    pos_z= -0.225,
    tau_limit=200.0,
    range_min= -2.618,
    range_max=0.0,
    armature=1.0,
    damping=1.0,
]

# Joint 5: foot — Hinge around Y (body 3)
# Joint at bottom of leg: (0, 0, -leg_half)
comptime HopperFootJ = HingeJoint[
    body_idx=3,
    pos_z= -0.25,
    tau_limit=200.0,
    range_min= -0.785,
    range_max=0.785,
    armature=1.0,
    damping=1.0,
]


# =============================================================================
# Composed Robot Definition
# =============================================================================

comptime HopperBodies = Bodies[
    HopperTorso, HopperThigh, HopperLeg, HopperFoot
]

comptime HopperJoints = Joints[
    HopperRootX, HopperRootZ, HopperRootY,
    HopperThighJ, HopperLegJ, HopperFootJ,
]

comptime HopperRobot = RobotDef[
    HopperBodies.N,
    HopperJoints.N,
    HopperJoints._sum_nq(),
    HopperJoints._sum_nv(),
]


# =============================================================================
# Body/Joint Index Constants
# =============================================================================

comptime BODY_TORSO: Int = 0
comptime BODY_THIGH: Int = 1
comptime BODY_LEG: Int = 2
comptime BODY_FOOT: Int = 3

comptime NUM_BODIES: Int = 4
