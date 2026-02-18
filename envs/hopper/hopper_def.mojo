"""Hopper as a compile-time model definition.

Defines all 4 bodies and 6 joints as type aliases using BodySpec/JointSpec,
composed into HopperModel via ModelDef. Validates that compile-time
dimensions match the existing environment (NQ=6, NV=6, NBODY=4, NJOINT=6).

Body/joint values match MuJoCo hopper.xml and the existing
envs/hopper/ implementation.

Also defines HopperParams — the environment-specific parameters
(physics, reward, termination, curriculum) that are NOT derivable from the
model definition. Replaces the former constants.mojo.
"""

from physics3d.model.body_spec import CapsuleBody
from physics3d.model.joint_spec import HingeJoint, SlideJoint
from physics3d.model.model_def import (
    Bodies,
    Joints,
    Geoms,
    Actuators,
    ModelDef,
    ModelDefaults,
)
from physics3d.model.actuator_spec import MotorActuator
from physics3d.model.geom_spec import Plane, Capsule
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
#
# body_pos = MuJoCo body_pos (joint position relative to parent body origin)
# body_quat = identity for ALL bodies (no capsule rotation in kinematic chain)
# body_ipos = CoM offset from body origin in body frame
# body_iquat = inertia frame orientation in body frame
# Inertia overrides from MuJoCo (inertiafromgeom + settotalmass)
# =============================================================================

# Body 1: Torso — vertical capsule, root of kinematic tree
# MuJoCo: body_pos=(0,0,1.25) — body origin offset in parent (world) frame
comptime HopperTorso = CapsuleBody[
    parent=0,
    mass=3.665191,
    name="torso",
    radius=0.05,
    half_length=0.2,
    pos_z=1.25,  # MuJoCo body_pos z-offset
    # ipos = (0, 0, 0), iquat = identity (defaults)
    ixx_override=0.069246,
    iyy_override=0.069246,
    izz_override=0.004451,
    color = Color3D(60, 120, 200),
]

# Body 2: Thigh — vertical capsule below torso
# MuJoCo: body_pos=(0, 0, -0.2) relative to torso
comptime HopperThigh = CapsuleBody[
    parent=1,
    mass=4.057891,
    name="thigh",
    radius=0.05,
    half_length=0.225,
    pos_z= -0.2,
    # CoM is 0.225 below body origin (geom center)
    ipos_z= -0.225,
    ixx_override=0.093299,
    iyy_override=0.093299,
    izz_override=0.004941,
    color = Color3D(80, 200, 80),
]

# Body 3: Leg — vertical capsule below thigh
# MuJoCo: body_pos=(0, 0, -0.7) relative to thigh
comptime HopperLeg = CapsuleBody[
    parent=2,
    mass=2.781357,
    name="leg",
    radius=0.04,
    half_length=0.25,
    pos_z= -0.7,
    # ipos = (0, 0, 0), iquat = identity (defaults)
    ixx_override=0.072303,
    iyy_override=0.072303,
    izz_override=0.002182,
    color = Color3D(220, 140, 60),
]

# Body 4: Foot — horizontal capsule, below leg
# MuJoCo: body_pos=(0.13, 0, -0.35) relative to leg
# body_quat = identity (capsule rotation is in geom, not body frame)
comptime HopperFoot = CapsuleBody[
    parent=3,
    mass=5.315575,
    name="foot",
    radius=0.06,
    half_length=0.195,
    pos_x=0.13,
    pos_z= -0.35,
    # CoM offset from body origin
    ipos_x= -0.065,
    ipos_z=0.1,
    # Inertia frame orientation (90deg Y rotation for horizontal capsule)
    iquat_y= -0.707107,
    iquat_w=0.707107,
    ixx_override=0.103523,
    iyy_override=0.103523,
    izz_override=0.009242,
    color = Color3D(220, 80, 80),
]


# =============================================================================
# Joint Type Aliases
# =============================================================================

# Joint 0: rootx — Slide along X (body 1/torso, unactuated)
# MuJoCo: armature=0, damping=0, stiffness=0
comptime HopperRootX = SlideJoint[
    body_idx=1,
    axis_x=1.0,
    axis_y=0.0,
    axis_z=0.0,
    armature=0.0,
    damping=0.0,
    stiffness=0.0,
    exclude_obs_qpos=True,
]

# Joint 1: rootz — Slide along Z (body 1/torso, unactuated)
# MuJoCo: armature=0, damping=0, stiffness=0
# Height comes from body_pos_z=1.25, NOT from init_qpos (MuJoCo qpos0 is all zeros)
comptime HopperRootZ = SlideJoint[
    body_idx=1,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
    armature=0.0,
    damping=0.0,
    stiffness=0.0,
]

# Joint 2: rooty — Hinge around Y (body 1/torso, unactuated)
# MuJoCo: armature=0, damping=0, stiffness=0
comptime HopperRootY = HingeJoint[
    body_idx=1,
    tau_limit=0.0,
    armature=0.0,
    damping=0.0,
    stiffness=0.0,
    is_actuated=False,
    has_limits=False,
]

# Joint 3: thigh — Hinge around Y (body 2)
# MuJoCo: joint pos=(0,0,0) relative to body (joint at body origin)
comptime HopperThighJ = HingeJoint[
    body_idx=2,
    tau_limit=200.0,
    range_min= -2.618,
    range_max=0.0,
    armature=1.0,
    damping=1.0,
]

# Joint 4: leg — Hinge around Y (body 3)
# MuJoCo: joint pos=(0, 0, 0.25) relative to body
comptime HopperLegJ = HingeJoint[
    body_idx=3,
    pos_z=0.25,
    tau_limit=200.0,
    range_min= -2.618,
    range_max=0.0,
    armature=1.0,
    damping=1.0,
]

# Joint 5: foot — Hinge around Y (body 4)
# MuJoCo: joint pos=(-0.13, 0, 0.1) relative to body
comptime HopperFootJ = HingeJoint[
    body_idx=4,
    pos_x= -0.13,
    pos_z=0.1,
    tau_limit=200.0,
    range_min= -0.785,
    range_max=0.785,
    armature=1.0,
    damping=1.0,
]


# =============================================================================
# Geom Definitions (unified collision geometry)
# =============================================================================

# Geom 0: Ground plane (overrides: friction=1.0, conaffinity=1, condim=3)
comptime HopperGroundGeom = Plane[
    z=0.0, friction=1.0, conaffinity=1, condim=3, size_x=20.0, size_y=20.0
]

# Geom 1: Torso capsule (body 1) — friction/condim from HopperDefaults
comptime HopperTorsoGeom = Capsule[
    body_idx=1,
    radius=0.05,
    half_length=0.2,
    color = Color3D(60, 120, 200),
]

# Geom 2: Thigh capsule (body 2) — MuJoCo geom_pos=(0, 0, -0.225)
comptime HopperThighGeom = Capsule[
    body_idx=2,
    radius=0.05,
    half_length=0.225,
    pos_z= -0.225,
    color = Color3D(80, 200, 80),
]

# Geom 3: Leg capsule (body 3) — at body origin
comptime HopperLegGeom = Capsule[
    body_idx=3,
    radius=0.04,
    half_length=0.25,
    color = Color3D(220, 140, 60),
]

# Geom 4: Foot capsule (body 4) — friction=2.0 overrides default 0.9
comptime HopperFootGeom = Capsule[
    body_idx=4,
    radius=0.06,
    half_length=0.195,
    pos_x= -0.065,
    pos_z=0.1,
    quat_y= -0.707107,
    quat_w=0.707107,
    friction=2.0,
    color = Color3D(220, 80, 80),
]

comptime HopperGeoms = Geoms[
    HopperGroundGeom,
    HopperTorsoGeom,
    HopperThighGeom,
    HopperLegGeom,
    HopperFootGeom,
]


# =============================================================================
# Composed Model Definition
# =============================================================================

comptime HopperBodies = Bodies[HopperTorso, HopperThigh, HopperLeg, HopperFoot]

comptime HopperJoints = Joints[
    HopperRootX,
    HopperRootZ,
    HopperRootY,
    HopperThighJ,
    HopperLegJ,
    HopperFootJ,
]

# =============================================================================
# Actuator Type Aliases (MuJoCo-style motor actuators)
# =============================================================================
# All actuated joints are 1-DOF hinges at indices 3-5 (dof_adr = joint_idx).
# Hopper MuJoCo: all 3 actuators have gear=200 (ctrllimited, ctrlrange=[-1,1]).

comptime HopperThighMotor = MotorActuator[joint_idx=3, dof_adr=3, gear=200.0]
comptime HopperLegMotor = MotorActuator[joint_idx=4, dof_adr=4, gear=200.0]
comptime HopperFootMotor = MotorActuator[joint_idx=5, dof_adr=5, gear=200.0]

comptime HopperActuators = Actuators[
    HopperThighMotor, HopperLegMotor, HopperFootMotor
]


# =============================================================================
# HopperDefaults — MuJoCo-style model defaults
# =============================================================================
# Geom defaults match hopper.xml: friction=0.9, condim=1 for body geoms.
# Ground plane and foot override friction explicitly.

comptime HopperDefaults = ModelDefaults[
    geom_friction=0.9,
    geom_friction_spin=0.005,
    geom_friction_roll=0.0001,
    geom_condim=1,
    geom_margin=0.001,
    geom_solref_0=0.02,
    geom_solref_1=1.0,
    geom_solimp_0=0.8,  # MuJoCo XML default: solimp=".8 .8 .01"
    geom_solimp_1=0.8,
    geom_solimp_2=0.01,
    joint_solref_limit_0=0.02,
    joint_solref_limit_1=1.0,
    joint_solimp_limit_0=0.0,
    joint_solimp_limit_1=0.8,
    joint_solimp_limit_2=0.03,
    gravity_z= -9.81,
    timestep=0.002,
]


comptime HopperModel = ModelDef[
    HopperBodies.N + 1,  # +1 for worldbody at index 0
    HopperJoints.N,
    HopperJoints._sum_nq(),
    HopperJoints._sum_nv(),
    HopperGeoms.N,
]


# =============================================================================
# HopperParams — Environment-Specific Parameters
# =============================================================================


struct HopperParams[DTYPE: DType = DType.float64]:
    """Environment-specific parameters not derivable from the model definition.

    Replaces the former HopperConstants struct. Everything about body
    geometry, joint limits, gear ratios, damping, and indices is
    now in the model definition (BodySpec/JointSpec).

    Type Parameters:
        DTYPE: The floating point type for physics constants.
    """

    # Physics
    comptime DT: Scalar[Self.DTYPE] = 0.002  # Physics timestep (500 Hz)
    comptime FRAME_SKIP: Int = 4  # Number of physics steps per env step
    comptime GRAVITY_Z: Scalar[Self.DTYPE] = -9.81
    comptime MAX_CONTACTS: Int = 20

    # Reward
    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.001
    comptime HEALTHY_REWARD: Scalar[Self.DTYPE] = 1.0

    # Termination
    comptime MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime MAX_PITCH: Scalar[Self.DTYPE] = 0.2  # ~11 deg
    comptime MAX_STEPS: Int = 1000

    # Curriculum
    comptime CURRICULUM_INITIAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.3
    comptime CURRICULUM_INITIAL_MAX_PITCH: Scalar[Self.DTYPE] = 1.0
    comptime CURRICULUM_FINAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime CURRICULUM_FINAL_MAX_PITCH: Scalar[Self.DTYPE] = 0.2

    # Reset
    comptime RESET_NOISE_SCALE: Scalar[Self.DTYPE] = 0.005

    # Dimensions (derived from model definition, for convenience)
    comptime NQ: Int = HopperModel.NQ
    comptime NV: Int = HopperModel.NV
    comptime NUM_BODIES: Int = HopperModel.NBODY
    comptime NUM_JOINTS: Int = HopperModel.NJOINT
    comptime NGEOM: Int = HopperModel.NGEOM
    comptime OBS_DIM: Int = 11
    comptime ACTION_DIM: Int = 3

    # Initial torso height (from body_pos_z; qpos[rootz] starts at 0)
    comptime INITIAL_Z: Scalar[Self.DTYPE] = 1.25

    # Motor
    comptime TORQUE_LIMIT: Scalar[Self.DTYPE] = 200.0

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
comptime HopperParamsCPU = HopperParams[DType.float64]
comptime HopperParamsGPU = HopperParams[DType.float32]

# Backward-compatibility aliases (old name -> new name)
comptime HopperConstants = HopperParams
comptime HopperConstantsCPU = HopperParamsCPU
comptime HopperConstantsGPU = HopperParamsGPU


# =============================================================================
# Body/Joint Index Constants (for backward compatibility)
# =============================================================================

comptime BODY_WORLDBODY: Int = 0
comptime BODY_TORSO: Int = 1
comptime BODY_THIGH: Int = 2
comptime BODY_LEG: Int = 3
comptime BODY_FOOT: Int = 4

comptime JOINT_ROOTX: Int = 0
comptime JOINT_ROOTZ: Int = 1
comptime JOINT_ROOTY: Int = 2
comptime JOINT_THIGH: Int = 3
comptime JOINT_LEG: Int = 4
comptime JOINT_FOOT: Int = 5

# Body geometry constants for renderer
comptime TORSO_RADIUS: Float64 = 0.05
comptime TORSO_HALF_LENGTH: Float64 = 0.2
comptime THIGH_RADIUS: Float64 = 0.05
comptime THIGH_HALF_LENGTH: Float64 = 0.225
comptime LEG_RADIUS: Float64 = 0.04
comptime LEG_HALF_LENGTH: Float64 = 0.25
comptime FOOT_RADIUS: Float64 = 0.06
comptime FOOT_HALF_LENGTH: Float64 = 0.195

# Body mass constants for backward compatibility
comptime TORSO_MASS: Float64 = 3.665191
comptime THIGH_MASS: Float64 = 4.057891
comptime LEG_MASS: Float64 = 2.781357
comptime FOOT_MASS: Float64 = 5.315575

# Dimension constants for backward compatibility
comptime NQ: Int = HopperModel.NQ
comptime NV: Int = HopperModel.NV
comptime NBODY: Int = HopperModel.NBODY
comptime NJOINT: Int = HopperModel.NJOINT
comptime MAX_CONTACTS: Int = 20
comptime NGEOM: Int = HopperModel.NGEOM
comptime OBS_DIM: Int = 11
comptime ACTION_DIM: Int = 3
comptime NUM_BODIES: Int = 5

# Physics constants for backward compatibility (derived from HopperDefaults)
comptime DT: Float64 = HopperDefaults.TIMESTEP
comptime FRAME_SKIP: Int = 4
comptime EFFECTIVE_DT: Float64 = DT * FRAME_SKIP
comptime MAX_STEPS: Int = 1000
comptime INITIAL_Z: Float64 = 1.25
comptime FRICTION: Float64 = 1.0
comptime RESTITUTION: Float64 = 0.0
comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0
comptime CTRL_COST_WEIGHT: Float64 = 0.001
comptime HEALTHY_REWARD: Float64 = 1.0
comptime RESET_NOISE_SCALE: Float64 = 0.005
comptime TORQUE_LIMIT: Float64 = 200.0

# Joint limits for backward compatibility
comptime THIGH_JOINT_MIN: Float64 = -2.618
comptime THIGH_JOINT_MAX: Float64 = 0.0
comptime LEG_JOINT_MIN: Float64 = -2.618
comptime LEG_JOINT_MAX: Float64 = 0.0
comptime FOOT_JOINT_MIN: Float64 = -0.785
comptime FOOT_JOINT_MAX: Float64 = 0.785
