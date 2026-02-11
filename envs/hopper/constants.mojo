"""Constants for HopperGC environment using Generalized Coordinates engine.

MuJoCo-style Hopper with joint-space dynamics.
Uses DefaultIntegrator for physics.
"""

from physics3d.gpu.constants import (
    state_size,
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
    model_size,
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


struct HopperGCConstants[DTYPE: DType = DType.float64]:
    """Constants for HopperGC environment using Generalized Coordinates.

    MuJoCo Hopper configuration:
    - 4 bodies: torso, thigh, leg, foot
    - 6 joints: rootx (slide), rootz (slide), rooty (hinge),
                thigh (hinge), knee (hinge), ankle (hinge)

    Joint space dimensions:
    - NQ = 6 (2 slide positions + 4 hinge angles)
    - NV = 6 (2 slide velocities + 4 hinge angular velocities)

    Type Parameters:
        DTYPE: The floating point type for physics constants.
    """

    # ==========================================================================
    # Physics Parameters
    # ==========================================================================

    comptime DT: Scalar[Self.DTYPE] = 0.002  # MuJoCo uses 0.002
    comptime FRAME_SKIP: Int = 4  # Match Gymnasium Hopper v5 (frame_skip=4)
    comptime GRAVITY_Z: Scalar[Self.DTYPE] = -9.81

    # Contact physics
    comptime FRICTION: Scalar[Self.DTYPE] = 1.0

    # MuJoCo solref/solimp (from hopper.xml)
    comptime SOLREF_CONTACT_0: Scalar[Self.DTYPE] = 0.02  # timeconst
    comptime SOLREF_CONTACT_1: Scalar[Self.DTYPE] = 1.0   # dampratio
    comptime SOLIMP_CONTACT_0: Scalar[Self.DTYPE] = 0.0   # dmin
    comptime SOLIMP_CONTACT_1: Scalar[Self.DTYPE] = 0.8   # dmax
    comptime SOLIMP_CONTACT_2: Scalar[Self.DTYPE] = 0.01  # width
    comptime SOLREF_LIMIT_0: Scalar[Self.DTYPE] = 0.02    # timeconst
    comptime SOLREF_LIMIT_1: Scalar[Self.DTYPE] = 1.0     # dampratio
    comptime SOLIMP_LIMIT_0: Scalar[Self.DTYPE] = 0.0     # dmin
    comptime SOLIMP_LIMIT_1: Scalar[Self.DTYPE] = 0.8     # dmax
    comptime SOLIMP_LIMIT_2: Scalar[Self.DTYPE] = 0.03    # width

    # ==========================================================================
    # Body Geometry (from MuJoCo Hopper XML)
    # ==========================================================================

    # Torso (vertical capsule)
    comptime TORSO_MASS: Scalar[Self.DTYPE] = 3.53429174
    comptime TORSO_RADIUS: Scalar[Self.DTYPE] = 0.05
    comptime TORSO_HALF_LENGTH: Scalar[Self.DTYPE] = 0.2

    # Thigh (vertical capsule)
    comptime THIGH_MASS: Scalar[Self.DTYPE] = 3.92699082
    comptime THIGH_RADIUS: Scalar[Self.DTYPE] = 0.05
    comptime THIGH_HALF_LENGTH: Scalar[Self.DTYPE] = 0.225

    # Leg (vertical capsule)
    comptime LEG_MASS: Scalar[Self.DTYPE] = 2.71433605
    comptime LEG_RADIUS: Scalar[Self.DTYPE] = 0.04
    comptime LEG_HALF_LENGTH: Scalar[Self.DTYPE] = 0.25

    # Foot (horizontal capsule - rotated 90° around Y)
    comptime FOOT_MASS: Scalar[Self.DTYPE] = 5.0893801
    comptime FOOT_RADIUS: Scalar[Self.DTYPE] = 0.06
    comptime FOOT_HALF_LENGTH: Scalar[Self.DTYPE] = 0.195

    # ==========================================================================
    # Body Indices
    # ==========================================================================

    comptime BODY_TORSO: Int = 0
    comptime BODY_THIGH: Int = 1
    comptime BODY_LEG: Int = 2
    comptime BODY_FOOT: Int = 3

    # ==========================================================================
    # Joint Configuration (MuJoCo style)
    # ==========================================================================

    # Joint indices (in order they appear in qpos/qvel)
    comptime JOINT_ROOTX: Int = 0  # Slide along X (body 0)
    comptime JOINT_ROOTZ: Int = 1  # Slide along Z (body 0)
    comptime JOINT_ROOTY: Int = 2  # Hinge around Y (body 0)
    comptime JOINT_THIGH: Int = 3  # Hinge around Y (body 1)
    comptime JOINT_LEG: Int = 4  # Hinge around Y (body 2)
    comptime JOINT_FOOT: Int = 5  # Hinge around Y (body 3)

    # ==========================================================================
    # Joint Space Dimensions
    # ==========================================================================

    # Total dimensions (all joints have qpos_size=1 and qvel_size=1)
    comptime NQ: Int = 6  # qpos dimension
    comptime NV: Int = 6  # qvel dimension

    # ==========================================================================
    # Motor Parameters
    # ==========================================================================

    comptime TORQUE_LIMIT: Scalar[Self.DTYPE] = 200.0  # MuJoCo gear=200

    # ==========================================================================
    # Joint Limits (from MuJoCo hopper.xml, converted to radians)
    # ==========================================================================

    # thigh_joint: range="-150 0" degrees
    comptime THIGH_JOINT_MIN: Scalar[Self.DTYPE] = -2.618  # -150 degrees
    comptime THIGH_JOINT_MAX: Scalar[Self.DTYPE] = 0.0  # 0 degrees

    # leg_joint: range="-150 0" degrees
    comptime LEG_JOINT_MIN: Scalar[Self.DTYPE] = -2.618  # -150 degrees
    comptime LEG_JOINT_MAX: Scalar[Self.DTYPE] = 0.0  # 0 degrees

    # foot_joint: range="-45 45" degrees
    comptime FOOT_JOINT_MIN: Scalar[Self.DTYPE] = -0.785  # -45 degrees
    comptime FOOT_JOINT_MAX: Scalar[Self.DTYPE] = 0.785  # 45 degrees

    # ==========================================================================
    # Termination Parameters (defaults - can be overridden by curriculum)
    # ==========================================================================

    comptime MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime MAX_PITCH: Scalar[Self.DTYPE] = 0.2  # ~11 degrees

    # ==========================================================================
    # Curriculum Parameters
    # ==========================================================================

    # Number of curriculum params used by Hopper (min_height, max_pitch)
    comptime NUM_CURRICULUM_PARAMS: Int = 2

    # Initial (lenient) values for curriculum
    comptime CURRICULUM_INITIAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.3
    comptime CURRICULUM_INITIAL_MAX_PITCH: Scalar[
        Self.DTYPE
    ] = 1.0  # ~57 degrees

    # Final (strict) values for curriculum (same as MuJoCo defaults)
    comptime CURRICULUM_FINAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime CURRICULUM_FINAL_MAX_PITCH: Scalar[Self.DTYPE] = 0.2  # ~11 degrees

    # ==========================================================================
    # Episode Parameters
    # ==========================================================================

    comptime MAX_STEPS: Int = 1000

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[
        Self.DTYPE
    ] = 0.001  # MuJoCo default (uses normalized actions [-1,1])
    comptime HEALTHY_REWARD: Scalar[Self.DTYPE] = 1.0

    # ==========================================================================
    # Layout Constants
    # ==========================================================================

    comptime NUM_BODIES: Int = 4
    comptime NUM_JOINTS: Int = 6  # 2 slide + 4 hinge
    comptime MAX_CONTACTS: Int = 10

    # ==========================================================================
    # Initial State Constants
    # ==========================================================================

    # Initial torso Z position (computed from body geometry)
    # foot_z = FOOT_RADIUS
    # leg_z = foot_z + LEG_RADIUS + LEG_HALF_LENGTH
    # thigh_z = leg_z + LEG_HALF_LENGTH + THIGH_HALF_LENGTH
    # torso_z = thigh_z + THIGH_HALF_LENGTH + TORSO_HALF_LENGTH
    comptime INITIAL_Z: Scalar[Self.DTYPE] = (
        Self.FOOT_RADIUS
        + Self.LEG_RADIUS
        + Self.LEG_HALF_LENGTH
        + Self.LEG_HALF_LENGTH
        + Self.THIGH_HALF_LENGTH
        + Self.THIGH_HALF_LENGTH
        + Self.TORSO_HALF_LENGTH
    )

    # ==========================================================================
    # Observation/Action Dimensions
    # ==========================================================================

    comptime OBS_DIM: Int = 11  # Same as MuJoCo Hopper
    comptime ACTION_DIM: Int = 3  # thigh, leg, foot (not root joints)

    # ==========================================================================
    # GPU Layout Constants (using physics3d GC layout)
    # ==========================================================================

    # Compute state size for GPU buffer
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

    # Model buffer size
    comptime MODEL_SIZE: Int = model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()

    # ==========================================================================
    # GPU Layout Helper Methods
    # ==========================================================================

    @staticmethod
    @always_inline
    fn get_qpos_offset() -> Int:
        """Get offset to qpos array in state buffer."""
        return qpos_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qvel_offset() -> Int:
        """Get offset to qvel array in state buffer."""
        return qvel_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qacc_offset() -> Int:
        """Get offset to qacc array in state buffer."""
        return qacc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qfrc_offset() -> Int:
        """Get offset to qfrc array in state buffer."""
        return qfrc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_xpos_offset() -> Int:
        """Get offset to xpos array in state buffer."""
        return xpos_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xquat_offset() -> Int:
        """Get offset to xquat array in state buffer."""
        return xquat_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xvel_offset() -> Int:
        """Get offset to xvel array in state buffer."""
        return xvel_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xangvel_offset() -> Int:
        """Get offset to xangvel array in state buffer."""
        return xangvel_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_contacts_offset() -> Int:
        """Get offset to contacts array in state buffer."""
        return contacts_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_metadata_offset() -> Int:
        """Get offset to metadata in state buffer."""
        return metadata_offset[
            Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
        ]()

    @staticmethod
    @always_inline
    fn get_model_body_offset(body_idx: Int) -> Int:
        """Get offset to body data in model buffer."""
        return model_body_offset(body_idx)

    @staticmethod
    @always_inline
    fn get_model_joint_offset(joint_idx: Int) -> Int:
        """Get offset to joint data in model buffer."""
        return model_joint_offset[Self.NUM_BODIES](joint_idx)

    @staticmethod
    @always_inline
    fn get_model_metadata_offset() -> Int:
        """Get offset to metadata in model buffer."""
        return model_metadata_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()

    @staticmethod
    @always_inline
    fn get_model_curriculum_offset() -> Int:
        """Get offset to curriculum parameters in model buffer."""
        return model_curriculum_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()


# Type aliases for convenience
comptime HopperGCConstantsCPU = HopperGCConstants[DType.float64]
comptime HopperGCConstantsGPU = HopperGCConstants[DType.float32]
