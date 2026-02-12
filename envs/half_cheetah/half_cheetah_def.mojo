"""HalfCheetah as a compile-time model definition.

Defines all 8 bodies and 10 joints as type aliases using BodySpec/JointSpec,
composed into HalfCheetahModel via ModelDef. Validates that compile-time
dimensions match the existing environment (NQ=10, NV=10, NBODY=8, NJOINT=10).

Body/joint values match MuJoCo half_cheetah.xml and the existing
envs/half_cheetah/ implementation.

Also defines HalfCheetahParams — the environment-specific parameters
(physics, reward, termination, curriculum) that are NOT derivable from the
robot definition. Replaces the former constants.mojo.
"""

from physics3d.model.body_spec import CapsuleBody
from physics3d.model.joint_spec import HingeJoint, SlideJoint
from physics3d.model.model_def import Bodies, Joints, Geoms, ModelDef
from physics3d.model.geom_spec import PlaneGeom, BodyCapsuleGeom
from render3d import Color3D
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
    name="torso",
    radius=_R,
    half_length=0.5,
    quat_y=_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,  # No self-collision (MuJoCo XML)
    color = Color3D(204, 153, 102),
]

# Body 1: Back Thigh — vertical capsule at back of torso
# body_pos: (bthigh_half, 0, -torso_half) = (0.145, 0, -0.5)
# body_quat: -90deg Y to counter-rotate torso
comptime BThigh = CapsuleBody[
    parent=0,
    mass=1.54,
    name="bthigh",
    radius=_R,
    half_length=0.145,
    pos_x=0.145,
    pos_z= -0.5,
    quat_y= -_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]

# Body 2: Back Shin — vertical capsule below bthigh
# body_pos: (0, 0, -(bthigh_half + bshin_half)) = (0, 0, -0.295)
comptime BShin = CapsuleBody[
    parent=1,
    mass=1.58,
    name="bshin",
    radius=_R,
    half_length=0.15,
    pos_z= -0.295,  # -(0.145 + 0.15)
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Body 3: Back Foot — horizontal capsule (90deg Y rotation)
# body_pos: (0, 0, -bshin_half) = (0, 0, -0.15)
comptime BFoot = CapsuleBody[
    parent=2,
    mass=1.10,
    name="bfoot",
    radius=_R,
    half_length=0.094,
    pos_z= -0.15,
    quat_y=_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Body 4: Front Thigh — vertical capsule at front of torso
# body_pos: (fthigh_half, 0, +torso_half) = (0.133, 0, 0.5)
comptime FThigh = CapsuleBody[
    parent=0,
    mass=1.43,
    name="fthigh",
    radius=_R,
    half_length=0.133,
    pos_x=0.133,
    pos_z=0.5,
    quat_y= -_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]

# Body 5: Front Shin — vertical capsule below fthigh
# body_pos: (0, 0, -(fthigh_half + fshin_half)) = (0, 0, -0.239)
comptime FShin = CapsuleBody[
    parent=4,
    mass=1.17,
    name="fshin",
    radius=_R,
    half_length=0.106,
    pos_z= -0.239,  # -(0.133 + 0.106)
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Body 6: Front Foot — horizontal capsule (90deg Y rotation)
# body_pos: (0, 0, -fshin_half) = (0, 0, -0.106)
comptime FFoot = CapsuleBody[
    parent=5,
    mass=0.93,
    name="ffoot",
    radius=_R,
    half_length=0.07,
    pos_z= -0.106,
    quat_y=_Q90Y_Y,
    quat_w=_Q90Y_W,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Body 7 (Head) removed — now a geom attached to torso (body 0)
# See HeadGeom below in the Geoms section.

# Head geometry constants (used by HeadGeom)
comptime _HEAD_SIN_HALF: Float64 = -0.34290  # sin((0.87 - pi/2) / 2)
comptime _HEAD_COS_HALF: Float64 = 0.93937  # cos((0.87 - pi/2) / 2)


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

# Joint 9 (HeadJ) removed — head is now a geom on the torso, no separate joint


# =============================================================================
# HalfCheetahModel — Full Model Definition
# =============================================================================

comptime HalfCheetahBodies = Bodies[
    Torso, BThigh, BShin, BFoot, FThigh, FShin, FFoot
]

comptime HalfCheetahJoints = Joints[
    RootX, RootZ, RootY, BThighJ, BShinJ, BFootJ, FThighJ, FShinJ, FFootJ
]


# =============================================================================
# Geoms — Unified geometry (replaces WorldBody + body-attached shapes)
# =============================================================================

# Ground plane
comptime GroundGeom = PlaneGeom[
    z=0.0, friction=0.4, conaffinity=1, size_x=40.0, size_y=40.0
]

# Body-centered capsule geoms (identity local offset — body FK provides world transform)
comptime TorsoGeom = BodyCapsuleGeom[
    body_idx=0,
    radius=_R,
    half_length=0.5,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]
comptime BThighGeom = BodyCapsuleGeom[
    body_idx=1,
    radius=_R,
    half_length=0.145,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]
comptime BShinGeom = BodyCapsuleGeom[
    body_idx=2,
    radius=_R,
    half_length=0.15,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]
comptime BFootGeom = BodyCapsuleGeom[
    body_idx=3,
    radius=_R,
    half_length=0.094,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]
comptime FThighGeom = BodyCapsuleGeom[
    body_idx=4,
    radius=_R,
    half_length=0.133,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]
comptime FShinGeom = BodyCapsuleGeom[
    body_idx=5,
    radius=_R,
    half_length=0.106,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]
comptime FFootGeom = BodyCapsuleGeom[
    body_idx=6,
    radius=_R,
    half_length=0.07,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Head geom — attached to torso (body 0) with local offset
# Position in torso frame: (-0.1, 0, 0.6), orientation: tilted ~0.87 rad about Y
comptime HeadGeom = BodyCapsuleGeom[
    body_idx=0,
    radius=_R,
    half_length=0.15,
    pos_x= -0.1,
    pos_z=0.6,
    quat_y=_HEAD_SIN_HALF,
    quat_w=_HEAD_COS_HALF,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]

comptime HalfCheetahGeoms = Geoms[
    GroundGeom,
    TorsoGeom,
    HeadGeom,
    BThighGeom,
    BShinGeom,
    BFootGeom,
    FThighGeom,
    FShinGeom,
    FFootGeom,
]


comptime HalfCheetahModel = ModelDef[
    HalfCheetahBodies.N,
    HalfCheetahJoints.N,
    HalfCheetahJoints._sum_nq(),
    HalfCheetahJoints._sum_nv(),
    HalfCheetahGeoms.N,
]


# =============================================================================
# HalfCheetahParams — Environment-Specific Parameters
# =============================================================================


struct HalfCheetahParams[DTYPE: DType = DType.float64]:
    """Environment-specific parameters not derivable from the model definition.

    Everything about body geometry, joint limits, gear ratios, damping, stiffness,
    and indices is now in the model definition (BodySpec/JointSpec).

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

    # Dimensions (derived from model definition, for convenience)
    comptime NQ: Int = HalfCheetahModel.NQ
    comptime NV: Int = HalfCheetahModel.NV
    comptime NUM_BODIES: Int = HalfCheetahModel.NBODY
    comptime NUM_JOINTS: Int = HalfCheetahModel.NJOINT
    comptime NGEOM: Int = HalfCheetahModel.NGEOM
    comptime OBS_DIM: Int = 17
    comptime ACTION_DIM: Int = 6

    # Initial height (rootz init_qpos)
    comptime INITIAL_Z: Scalar[Self.DTYPE] = 0.7

    # GPU layout sizes
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()
    comptime MODEL_SIZE: Int = model_size[
        Self.NUM_BODIES, Self.NUM_JOINTS, Self.NGEOM
    ]()

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

comptime JOINT_ROOTX: Int = 0
comptime JOINT_ROOTZ: Int = 1
comptime JOINT_ROOTY: Int = 2
comptime JOINT_BTHIGH: Int = 3
comptime JOINT_BSHIN: Int = 4
comptime JOINT_BFOOT: Int = 5
comptime JOINT_FTHIGH: Int = 6
comptime JOINT_FSHIN: Int = 7
comptime JOINT_FFOOT: Int = 8
