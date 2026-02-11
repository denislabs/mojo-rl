"""HalfCheetah as a compile-time robot definition.

Defines all 8 bodies and 10 joints as type aliases using BodySpec/JointSpec,
composed into HalfCheetahRobot via RobotDef. Validates that compile-time
dimensions match the existing environment (NQ=10, NV=10, NBODY=8, NJOINT=10).

Body/joint values match MuJoCo half_cheetah.xml and the existing
envs/half_cheetah/ implementation.

Also defines HalfCheetahParams — the environment-specific parameters
(physics, reward, termination, curriculum) that are NOT derivable from the
robot definition. Replaces the former constants.mojo.
"""

from physics3d.robot.body_spec import CapsuleBody
from physics3d.robot.joint_spec import HingeJoint, SlideJoint
from physics3d.robot.robot_def import Bodies, Joints, RobotDef

from physics3d.gpu.constants import (
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    contacts_offset,
    metadata_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    model_curriculum_offset,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    CURRICULUM_IDX_MIN_HEIGHT,
    CURRICULUM_IDX_MAX_PITCH,
)


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
# Joint Type Aliases (with observation/actuation flags)
# =============================================================================

# Joint 0: rootx — Slide along X (body 0, unactuated)
# exclude_obs_qpos=True: rootx excluded from observation for translation invariance
comptime RootX = SlideJoint[
    body_idx=0,
    axis_x=1.0,
    axis_y=0.0,
    axis_z=0.0,
    exclude_obs_qpos=True,  # rootx excluded from obs (translation invariance)
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
    is_actuated=False,
    has_limits=False,
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
# Excluded from obs and actions, no meaningful limits
comptime HeadJ = HingeJoint[
    body_idx=7,
    pos_x= -0.1,
    pos_z=0.6,
    tau_limit=0.0,
    range_min=0.0,
    range_max=0.0,
    damping=0.01,
    stiffness=8.0,
    exclude_obs_qpos=True,
    exclude_obs_qvel=True,
    is_actuated=False,
    has_limits=False,
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
# HalfCheetahParams — Environment-Specific Parameters
# =============================================================================


struct HalfCheetahParams[DTYPE: DType = DType.float64]:
    """Environment-specific parameters not derivable from the robot definition.

    Replaces the former HalfCheetahConstants struct. Everything about body
    geometry, joint limits, gear ratios, damping, stiffness, and indices is
    now in the robot definition (BodySpec/JointSpec).

    Type Parameters:
        DTYPE: The floating point type for physics constants.
    """

    # Physics
    comptime DT: Scalar[Self.DTYPE] = 0.002  # Physics timestep (500 Hz)
    comptime FRAME_SKIP: Int = 5  # Number of physics steps per env step
    comptime GRAVITY_Z: Scalar[Self.DTYPE] = -9.81
    comptime FRICTION: Scalar[Self.DTYPE] = 0.9
    comptime MAX_CONTACTS: Int = 20

    # Solref/solimp (from half_cheetah.xml)
    comptime SOLREF_CONTACT_0: Scalar[Self.DTYPE] = 0.02  # timeconst
    comptime SOLREF_CONTACT_1: Scalar[Self.DTYPE] = 1.0  # dampratio
    comptime SOLIMP_CONTACT_0: Scalar[Self.DTYPE] = 0.0  # dmin
    comptime SOLIMP_CONTACT_1: Scalar[Self.DTYPE] = 0.8  # dmax
    comptime SOLIMP_CONTACT_2: Scalar[Self.DTYPE] = 0.01  # width
    comptime SOLREF_LIMIT_0: Scalar[Self.DTYPE] = 0.02
    comptime SOLREF_LIMIT_1: Scalar[Self.DTYPE] = 1.0
    comptime SOLIMP_LIMIT_0: Scalar[Self.DTYPE] = 0.0
    comptime SOLIMP_LIMIT_1: Scalar[Self.DTYPE] = 0.8
    comptime SOLIMP_LIMIT_2: Scalar[Self.DTYPE] = 0.03

    # Reward
    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.1
    comptime ANGLE_PENALTY_WEIGHT: Scalar[Self.DTYPE] = 0.5

    # Termination
    comptime MAX_PITCH: Scalar[Self.DTYPE] = 1.0  # ~57 deg
    comptime MAX_STEPS: Int = 1000

    # Curriculum
    comptime CURRICULUM_INITIAL_MAX_PITCH: Scalar[Self.DTYPE] = 3.0
    comptime CURRICULUM_FINAL_MAX_PITCH: Scalar[Self.DTYPE] = 1.0

    # Reset
    comptime RESET_NOISE_SCALE: Scalar[Self.DTYPE] = 0.1
    comptime MIN_ROOTZ: Scalar[Self.DTYPE] = -0.3

    # Dimensions (derived from robot definition, for convenience)
    comptime NQ: Int = HalfCheetahRobot.NQ
    comptime NV: Int = HalfCheetahRobot.NV
    comptime NUM_BODIES: Int = HalfCheetahRobot.NBODY
    comptime NUM_JOINTS: Int = HalfCheetahRobot.NJOINT
    comptime OBS_DIM: Int = 17
    comptime ACTION_DIM: Int = 6

    # Initial height (rootz init_qpos)
    comptime INITIAL_Z: Scalar[Self.DTYPE] = 0.7

    # GPU layout sizes
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()
    comptime MODEL_SIZE: Int = model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()

    # GPU layout helper methods
    @staticmethod
    @always_inline
    fn get_qpos_offset() -> Int:
        return qpos_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qvel_offset() -> Int:
        return qvel_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qacc_offset() -> Int:
        return qacc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qfrc_offset() -> Int:
        return qfrc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_xpos_offset() -> Int:
        return xpos_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xquat_offset() -> Int:
        return xquat_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_metadata_offset() -> Int:
        return metadata_offset[
            Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
        ]()

    @staticmethod
    @always_inline
    fn get_model_body_offset(body_idx: Int) -> Int:
        return model_body_offset(body_idx)

    @staticmethod
    @always_inline
    fn get_model_joint_offset(joint_idx: Int) -> Int:
        return model_joint_offset[Self.NUM_BODIES](joint_idx)

    @staticmethod
    @always_inline
    fn get_model_metadata_offset() -> Int:
        return model_metadata_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()

    @staticmethod
    @always_inline
    fn get_model_curriculum_offset() -> Int:
        return model_curriculum_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()


# Convenience type aliases
comptime HalfCheetahParamsCPU = HalfCheetahParams[DType.float64]
comptime HalfCheetahParamsGPU = HalfCheetahParams[DType.float32]

# Backward-compatibility aliases (old name → new name)
comptime HalfCheetahConstants = HalfCheetahParams
comptime HalfCheetahConstantsCPU = HalfCheetahParamsCPU
comptime HalfCheetahConstantsGPU = HalfCheetahParamsGPU


# =============================================================================
# Body/Joint Index Constants (for backward compatibility with external consumers)
# =============================================================================

comptime BODY_TORSO: Int = 0
comptime BODY_BTHIGH: Int = 1
comptime BODY_BSHIN: Int = 2
comptime BODY_BFOOT: Int = 3
comptime BODY_FTHIGH: Int = 4
comptime BODY_FSHIN: Int = 5
comptime BODY_FFOOT: Int = 6
comptime BODY_HEAD: Int = 7

comptime JOINT_ROOTX: Int = 0
comptime JOINT_ROOTZ: Int = 1
comptime JOINT_ROOTY: Int = 2
comptime JOINT_BTHIGH: Int = 3
comptime JOINT_BSHIN: Int = 4
comptime JOINT_BFOOT: Int = 5
comptime JOINT_FTHIGH: Int = 6
comptime JOINT_FSHIN: Int = 7
comptime JOINT_FFOOT: Int = 8
comptime JOINT_HEAD: Int = 9

# Geometry constants for renderer
comptime CAPSULE_RADIUS: Float64 = _R
comptime TORSO_HALF_LENGTH: Float64 = 0.5
comptime HEAD_HALF_LENGTH: Float64 = 0.15
comptime HEAD_POS_X: Float64 = 0.6
comptime HEAD_POS_Y: Float64 = 0.0
comptime HEAD_POS_Z: Float64 = 0.1
comptime HEAD_AXIS_ANGLE: Float64 = 0.87
comptime BTHIGH_HALF_LENGTH: Float64 = 0.145
comptime BSHIN_HALF_LENGTH: Float64 = 0.15
comptime BFOOT_HALF_LENGTH: Float64 = 0.094
comptime FTHIGH_HALF_LENGTH: Float64 = 0.133
comptime FSHIN_HALF_LENGTH: Float64 = 0.106
comptime FFOOT_HALF_LENGTH: Float64 = 0.07

# Body mass constants for backward compatibility
comptime TORSO_MASS: Float64 = 6.25
comptime BTHIGH_MASS: Float64 = 1.54
comptime BSHIN_MASS: Float64 = 1.58
comptime BFOOT_MASS: Float64 = 1.10
comptime FTHIGH_MASS: Float64 = 1.43
comptime FSHIN_MASS: Float64 = 1.17
comptime FFOOT_MASS: Float64 = 0.93

# Dimension constants for backward compatibility
comptime NQ: Int = HalfCheetahRobot.NQ
comptime NV: Int = HalfCheetahRobot.NV
comptime NBODY: Int = HalfCheetahRobot.NBODY
comptime NJOINT: Int = HalfCheetahRobot.NJOINT
comptime MAX_CONTACTS: Int = 20
comptime OBS_DIM: Int = 17
comptime ACTION_DIM: Int = 6

# Physics constants for backward compatibility
comptime DT: Float64 = 0.002
comptime FRAME_SKIP: Int = 5
comptime EFFECTIVE_DT: Float64 = DT * FRAME_SKIP
comptime GRAVITY_Z: Float64 = -9.81
comptime GROUND_Z: Float64 = 0.0
comptime MAX_STEPS: Int = 1000
comptime INIT_HEIGHT: Float64 = 0.7
comptime FRICTION: Float64 = 0.9
comptime RESTITUTION: Float64 = 0.0
comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0
comptime CTRL_COST_WEIGHT: Float64 = 0.1
comptime RESET_NOISE_SCALE: Float64 = 0.1

# Gear ratios for backward compatibility
comptime BTHIGH_GEAR: Float64 = 120.0
comptime BSHIN_GEAR: Float64 = 90.0
comptime BFOOT_GEAR: Float64 = 60.0
comptime FTHIGH_GEAR: Float64 = 120.0
comptime FSHIN_GEAR: Float64 = 60.0
comptime FFOOT_GEAR: Float64 = 30.0

# Joint limits for backward compatibility
comptime BTHIGH_LOWER: Float64 = -0.52
comptime BTHIGH_UPPER: Float64 = 1.05
comptime BSHIN_LOWER: Float64 = -0.785
comptime BSHIN_UPPER: Float64 = 0.785
comptime BFOOT_LOWER: Float64 = -0.4
comptime BFOOT_UPPER: Float64 = 0.785
comptime FTHIGH_LOWER: Float64 = -1.0
comptime FTHIGH_UPPER: Float64 = 0.7
comptime FSHIN_LOWER: Float64 = -1.2
comptime FSHIN_UPPER: Float64 = 0.87
comptime FFOOT_LOWER: Float64 = -0.5
comptime FFOOT_UPPER: Float64 = 0.5


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
    constrained[
        HalfCheetahJoints._obs_dim() == 17, "HalfCheetah OBS_DIM must be 17"
    ]()
    constrained[
        HalfCheetahJoints._action_dim() == 6,
        "HalfCheetah ACTION_DIM must be 6",
    ]()
