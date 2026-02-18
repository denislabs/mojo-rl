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
from physics3d.types import ConeType
from physics3d.model.model_def import (
    Bodies,
    Joints,
    Geoms,
    Actuators,
    ModelDef,
    ModelDefaults,
)
from physics3d.model.actuator_spec import MotorActuator
from physics3d.model.geom_spec import Plane, Capsule, FromToCapsule
from render import Color3D
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
# Body Type Aliases (MuJoCo-matching body frame convention)
# =============================================================================
#
# body_pos = body origin in parent frame (joint position)
# body_quat = identity for all bodies (no capsule rotation in kinematic chain)
# body_ipos = CoM offset from body origin in body frame
# body_iquat = inertia frame orientation in body frame
# Inertia = MuJoCo values (from inertiafromgeom + settotalmass="14")
#
# MuJoCo quaternion order: (w,x,y,z). Our engine: (x,y,z,w).
# Conversion: mujoco (w,x,y,z) → our (x,y,z,w)

# Shared radius for all capsules
comptime _R: Float64 = 0.046

# Body 1: Torso — root body, parent = 0 (worldbody)
# MuJoCo body_pos=(0,0,0.7) — body origin offset in parent (world) frame
comptime Torso = CapsuleBody[
    parent=0,
    mass=6.250209,
    name="torso",
    radius=_R,
    half_length=0.5,
    pos_z=0.7,  # MuJoCo body_pos z-offset
    # body_quat = identity (default)
    ipos_x=0.152390,
    ipos_y=0.0,
    ipos_z=0.025398,
    # MuJoCo iquat (w,x,y,z) = (0.52243, 0.476516, 0.476516, 0.52243)
    # Our (x,y,z,w) = (0.476516, 0.476516, 0.52243, 0.52243)
    iquat_x=0.476516,
    iquat_y=0.476516,
    iquat_z=0.522430,
    iquat_w=0.522430,
    ixx_override=0.897118,
    iyy_override=0.885655,
    izz_override=0.017961,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]

# Body 2: Back Thigh
# MuJoCo body_pos = (-0.5, 0, 0) relative to torso
comptime BThigh = CapsuleBody[
    parent=1,
    mass=1.543515,
    name="bthigh",
    radius=_R,
    half_length=0.145,
    pos_x= -0.5,
    # body_quat = identity (default)
    ipos_x=0.1,
    ipos_z= -0.13,
    # MuJoCo iquat (w,x,y,z) = (-0.32329, 0, -0.9463, 0)
    # Our (x,y,z,w) = (0, -0.9463, 0, -0.32329)
    iquat_y= -0.946300,
    iquat_w= -0.323290,
    ixx_override=0.016844,
    iyy_override=0.016844,
    izz_override=0.001576,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]

# Body 3: Back Shin
# MuJoCo body_pos = (0.16, 0, -0.25) relative to bthigh
comptime BShin = CapsuleBody[
    parent=2,
    mass=1.587448,
    name="bshin",
    radius=_R,
    half_length=0.15,
    pos_x=0.16,
    pos_z= -0.25,
    ipos_x= -0.14,
    ipos_z= -0.07,
    # MuJoCo iquat (w,x,y,z) = (0.52762, 0, -0.849481, 0)
    # Our (x,y,z,w) = (0, -0.849481, 0, 0.52762)
    iquat_y= -0.849481,
    iquat_w=0.527620,
    ixx_override=0.018267,
    iyy_override=0.018267,
    izz_override=0.001623,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Body 4: Back Foot
# MuJoCo body_pos = (-0.28, 0, -0.14) relative to bshin
comptime BFoot = CapsuleBody[
    parent=3,
    mass=1.095397,
    name="bfoot",
    radius=_R,
    half_length=0.094,
    pos_x= -0.28,
    pos_z= -0.14,
    ipos_x=0.03,
    ipos_z= -0.097,
    # MuJoCo iquat (w,x,y,z) = (0.990901, 0, -0.13459, 0)
    # Our (x,y,z,w) = (0, -0.13459, 0, 0.990901)
    iquat_y= -0.134590,
    iquat_w=0.990901,
    ixx_override=0.006352,
    iyy_override=0.006352,
    izz_override=0.001102,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Body 5: Front Thigh
# MuJoCo body_pos = (0.5, 0, 0) relative to torso
comptime FThigh = CapsuleBody[
    parent=1,
    mass=1.438075,
    name="fthigh",
    radius=_R,
    half_length=0.133,
    pos_x=0.5,
    # body_quat = identity (default)
    ipos_x= -0.07,
    ipos_z= -0.12,
    # MuJoCo iquat (w,x,y,z) = (0.96639, 0, 0.257081, 0)
    # Our (x,y,z,w) = (0, 0.257081, 0, 0.96639)
    iquat_y=0.257081,
    iquat_w=0.966390,
    ixx_override=0.013740,
    iyy_override=0.013740,
    izz_override=0.001464,
    conaffinity=0,
    color = Color3D(204, 153, 102),
]

# Body 6: Front Shin
# MuJoCo body_pos = (-0.14, 0, -0.24) relative to fthigh
comptime FShin = CapsuleBody[
    parent=5,
    mass=1.200837,
    name="fshin",
    radius=_R,
    half_length=0.106,
    pos_x= -0.14,
    pos_z= -0.24,
    ipos_x=0.065,
    ipos_z= -0.09,
    # MuJoCo iquat (w,x,y,z) = (0.955336, 0, -0.29552, 0)
    # Our (x,y,z,w) = (0, -0.29552, 0, 0.955336)
    iquat_y= -0.295520,
    iquat_w=0.955336,
    ixx_override=0.008222,
    iyy_override=0.008222,
    izz_override=0.001213,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]

# Body 7: Front Foot
# MuJoCo body_pos = (0.13, 0, -0.18) relative to fshin
comptime FFoot = CapsuleBody[
    parent=6,
    mass=0.884519,
    name="ffoot",
    radius=_R,
    half_length=0.07,
    pos_x=0.13,
    pos_z= -0.18,
    ipos_x=0.045,
    ipos_z= -0.07,
    # MuJoCo iquat (w,x,y,z) = (0.955336, 0, -0.29552, 0)
    # Our (x,y,z,w) = (0, -0.29552, 0, 0.955336)
    iquat_y= -0.295520,
    iquat_w=0.955336,
    ixx_override=0.003529,
    iyy_override=0.003529,
    izz_override=0.000879,
    conaffinity=0,
    color = Color3D(230, 153, 153),
]


# =============================================================================
# Joint Type Aliases (with observation/actuation flags)
# =============================================================================

# Joint 0: rootx — Slide along X (body 1/torso, unactuated)
# exclude_obs_qpos=True: rootx excluded from observation for translation invariance
comptime RootX = SlideJoint[
    body_idx=1,
    axis_x=1.0,
    axis_y=0.0,
    axis_z=0.0,
    exclude_obs_qpos=True,  # rootx excluded from obs (translation invariance)
]

# Joint 1: rootz — Slide along Z (body 1/torso, unactuated)
# Height comes from body_pos_z=0.7, NOT from init_qpos (MuJoCo qpos0 is all zeros)
comptime RootZ = SlideJoint[
    body_idx=1,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
]

# Joint 2: rooty — Hinge around Y (body 1/torso, unactuated)
comptime RootY = HingeJoint[
    body_idx=1,
    tau_limit=0.0,
    armature=0.0,
    is_actuated=False,
    has_limits=False,
]

# Joint 3: bthigh — Back thigh hinge (body 2)
# MuJoCo joint pos = (0, 0, 0) — joint at body origin
comptime BThighJ = HingeJoint[
    body_idx=2,
    # pos = (0,0,0) default — joint at body origin
    tau_limit=120.0,
    range_min= -0.52,
    range_max=1.05,
    damping=6.0,
    stiffness=240.0,
]

# Joint 4: bshin — Back shin hinge (body 3)
comptime BShinJ = HingeJoint[
    body_idx=3,
    tau_limit=90.0,
    range_min= -0.785,
    range_max=0.785,
    damping=4.5,
    stiffness=180.0,
]

# Joint 5: bfoot — Back foot hinge (body 4)
comptime BFootJ = HingeJoint[
    body_idx=4,
    tau_limit=60.0,
    range_min= -0.4,
    range_max=0.785,
    damping=3.0,
    stiffness=120.0,
]

# Joint 6: fthigh — Front thigh hinge (body 5)
comptime FThighJ = HingeJoint[
    body_idx=5,
    tau_limit=120.0,
    range_min= -1.0,
    range_max=0.7,
    damping=4.5,
    stiffness=180.0,
]

# Joint 7: fshin — Front shin hinge (body 6)
comptime FShinJ = HingeJoint[
    body_idx=6,
    tau_limit=60.0,
    range_min= -1.2,
    range_max=0.87,
    damping=3.0,
    stiffness=120.0,
]

# Joint 8: ffoot — Front foot hinge (body 7)
comptime FFootJ = HingeJoint[
    body_idx=7,
    tau_limit=30.0,
    range_min= -0.5,
    range_max=0.5,
    damping=1.5,
    stiffness=60.0,
]

# Joint 9 (HeadJ) removed — head is now a geom on the torso, no separate joint


# =============================================================================
# Actuator Type Aliases (MuJoCo-style motor actuators)
# =============================================================================
# All joints are 1-DOF hinges. Root joints (0-2) are unactuated.
# Actuators map action[i] -> gear * clamp(ctrl, -1, 1) -> qfrc[dof_adr].
# dof_adr = qpos_adr = joint_idx for this model (all joints have NQ=NV=1).

comptime BThighMotor = MotorActuator[joint_idx=3, dof_adr=3, gear=120.0]
comptime BShinMotor = MotorActuator[joint_idx=4, dof_adr=4, gear=90.0]
comptime BFootMotor = MotorActuator[joint_idx=5, dof_adr=5, gear=60.0]
comptime FThighMotor = MotorActuator[joint_idx=6, dof_adr=6, gear=120.0]
comptime FShinMotor = MotorActuator[joint_idx=7, dof_adr=7, gear=60.0]
comptime FFootMotor = MotorActuator[joint_idx=8, dof_adr=8, gear=30.0]


# =============================================================================
# HalfCheetahModel — Full Model Definition
# =============================================================================

comptime HalfCheetahBodies = Bodies[
    Torso, BThigh, BShin, BFoot, FThigh, FShin, FFoot
]

comptime HalfCheetahJoints = Joints[
    RootX, RootZ, RootY, BThighJ, BShinJ, BFootJ, FThighJ, FShinJ, FFootJ
]

comptime HalfCheetahActuators = Actuators[
    BThighMotor, BShinMotor, BFootMotor, FThighMotor, FShinMotor, FFootMotor
]


# =============================================================================
# Geoms — Unified geometry (replaces WorldBody + body-attached shapes)
# =============================================================================

# Ground plane
comptime GroundGeom = Plane[
    z=0.0, friction=0.4, conaffinity=1, size_x=40.0, size_y=40.0
]

# Body capsule geoms with local pos/quat from MuJoCo
# (body frames are now identity-oriented, so geoms need their own transforms)
# MuJoCo geom quat is (w,x,y,z), our engine uses (x,y,z,w)
# Body capsule geoms — friction/friction_spin/friction_roll/conaffinity
# inherited from HalfCheetahDefaults (friction=0.4, conaffinity=0)
comptime TorsoGeom = FromToCapsule[
    body_idx=1,
    radius=_R,
    # MuJoCo: fromto="-.5 0 0 .5 0 0"
    from_x=-0.5, to_x=0.5,
    color = Color3D(204, 153, 102),
]
comptime BThighGeom = Capsule[
    body_idx=2,
    radius=_R,
    half_length=0.145,
    pos_x=0.1,
    pos_z= -0.13,
    # MuJoCo quat (w,x,y,z) = (-0.32329, 0, -0.9463, 0)
    quat_y= -0.946300,
    quat_w= -0.323290,
    color = Color3D(204, 153, 102),
]
comptime BShinGeom = Capsule[
    body_idx=3,
    radius=_R,
    half_length=0.15,
    pos_x= -0.14,
    pos_z= -0.07,
    # MuJoCo quat (w,x,y,z) = (0.52762, 0, -0.849481, 0)
    quat_y= -0.849481,
    quat_w=0.527620,
    color = Color3D(230, 153, 153),
]
comptime BFootGeom = Capsule[
    body_idx=4,
    radius=_R,
    half_length=0.094,
    pos_x=0.03,
    pos_z= -0.097,
    # MuJoCo quat (w,x,y,z) = (0.990901, 0, -0.13459, 0)
    quat_y= -0.134590,
    quat_w=0.990901,
    color = Color3D(230, 153, 153),
]
comptime FThighGeom = Capsule[
    body_idx=5,
    radius=_R,
    half_length=0.133,
    pos_x= -0.07,
    pos_z= -0.12,
    # MuJoCo quat (w,x,y,z) = (0.96639, 0, 0.257081, 0)
    quat_y=0.257081,
    quat_w=0.966390,
    color = Color3D(204, 153, 102),
]
comptime FShinGeom = Capsule[
    body_idx=6,
    radius=_R,
    half_length=0.106,
    pos_x=0.065,
    pos_z= -0.09,
    # MuJoCo quat (w,x,y,z) = (0.955336, 0, -0.29552, 0)
    quat_y= -0.295520,
    quat_w=0.955336,
    color = Color3D(230, 153, 153),
]
comptime FFootGeom = Capsule[
    body_idx=7,
    radius=_R,
    half_length=0.07,
    pos_x=0.045,
    pos_z= -0.07,
    # MuJoCo quat (w,x,y,z) = (0.955336, 0, -0.29552, 0)
    quat_y= -0.295520,
    quat_w=0.955336,
    color = Color3D(230, 153, 153),
]

# Head geom — attached to torso (body 0) with local offset
# MuJoCo: pos=(0.6, 0, 0.1), quat=(0.90687, 0, 0.42141, 0)
comptime HeadGeom = Capsule[
    body_idx=1,
    radius=_R,
    half_length=0.15,
    pos_x=0.6,
    pos_z=0.1,
    # MuJoCo quat (w,x,y,z) = (0.90687, 0, 0.42141, 0) → our (0, 0.42141, 0, 0.90687)
    quat_y=0.421410,
    quat_w=0.906870,
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


# =============================================================================
# HalfCheetahDefaults — MuJoCo-style model defaults
# =============================================================================
# Geom defaults match half_cheetah.xml: friction=0.4, conaffinity=0 for body
# geoms (ground plane overrides conaffinity=1 explicitly).
# Solver params: solref/solimp for contacts and limits.

comptime HalfCheetahDefaults = ModelDefaults[
    geom_friction=0.4,
    geom_friction_spin=0.1,
    geom_friction_roll=0.1,
    geom_conaffinity=0,
    geom_solref_0=0.02,
    geom_solref_1=1.0,
    geom_solimp_0=0.0,
    geom_solimp_1=0.8,
    geom_solimp_2=0.01,
    joint_solref_limit_0=0.02,
    joint_solref_limit_1=1.0,
    joint_solimp_limit_0=0.0,
    joint_solimp_limit_1=0.8,
    joint_solimp_limit_2=0.03,
    gravity_z= -9.81,
    timestep=0.01,
    settotalmass=14.0,
]


comptime HalfCheetahModel = ModelDef[
    HalfCheetahBodies.N + 1,  # +1 for worldbody at index 0
    HalfCheetahJoints.N,
    HalfCheetahJoints._sum_nq(),
    HalfCheetahJoints._sum_nv(),
    HalfCheetahGeoms.N,
    0,
    ConeType.PYRAMIDAL,
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
    comptime FRAME_SKIP: Int = 5  # Number of physics steps per env step (matching MuJoCo)
    comptime MAX_CONTACTS: Int = 20

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

    # Initial torso height (from body_pos_z; qpos[rootz] starts at 0)
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

comptime BODY_WORLDBODY: Int = 0
comptime BODY_TORSO: Int = 1
comptime BODY_BTHIGH: Int = 2
comptime BODY_BSHIN: Int = 3
comptime BODY_BFOOT: Int = 4
comptime BODY_FTHIGH: Int = 5
comptime BODY_FSHIN: Int = 6
comptime BODY_FFOOT: Int = 7

comptime JOINT_ROOTX: Int = 0
comptime JOINT_ROOTZ: Int = 1
comptime JOINT_ROOTY: Int = 2
comptime JOINT_BTHIGH: Int = 3
comptime JOINT_BSHIN: Int = 4
comptime JOINT_BFOOT: Int = 5
comptime JOINT_FTHIGH: Int = 6
comptime JOINT_FSHIN: Int = 7
comptime JOINT_FFOOT: Int = 8
