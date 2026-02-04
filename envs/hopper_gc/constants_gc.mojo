"""Constants for HopperGC environment using Generalized Coordinates engine.

MuJoCo-style Hopper with joint-space dynamics.
Uses SemiImplicitEulerIntegrator for physics.
"""

from physics3d_v2.gpu.constants import (
    gc_state_size,
    gc_qpos_offset,
    gc_qvel_offset,
    gc_qacc_offset,
    gc_qfrc_offset,
    gc_xpos_offset,
    gc_xquat_offset,
    gc_xvel_offset,
    gc_xangvel_offset,
    gc_contacts_offset,
    gc_metadata_offset,
    gc_model_size,
    gc_model_body_offset,
    gc_model_joint_offset,
    gc_model_metadata_offset,
    GC_MODEL_BODY_SIZE,
    GC_MODEL_JOINT_SIZE,
    GC_MODEL_META_SIZE,
    GC_CONTACT_SIZE,
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
    comptime FRAME_SKIP: Int = 5  # 5 substeps per action
    comptime GRAVITY_Z: Scalar[Self.DTYPE] = -9.81

    # Contact physics
    comptime FRICTION: Scalar[Self.DTYPE] = 0.5

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
    comptime JOINT_LEG: Int = 4    # Hinge around Y (body 2)
    comptime JOINT_FOOT: Int = 5   # Hinge around Y (body 3)

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
    # Termination Parameters
    # ==========================================================================

    comptime MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime MAX_PITCH: Scalar[Self.DTYPE] = 0.2  # ~11 degrees

    # ==========================================================================
    # Episode Parameters
    # ==========================================================================

    comptime MAX_STEPS: Int = 1000

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.05  # Increased from 0.001 to penalize sliding
    comptime HEALTHY_REWARD: Scalar[Self.DTYPE] = 1.0

    # ==========================================================================
    # Layout Constants
    # ==========================================================================

    comptime NUM_BODIES: Int = 4
    comptime NUM_JOINTS: Int = 6  # 2 slide + 4 hinge
    comptime MAX_CONTACTS: Int = 10

    # ==========================================================================
    # Observation/Action Dimensions
    # ==========================================================================

    comptime OBS_DIM: Int = 11  # Same as MuJoCo Hopper
    comptime ACTION_DIM: Int = 3  # thigh, leg, foot (not root joints)

    # ==========================================================================
    # GPU Layout Constants (using physics3d_v2 GC layout)
    # ==========================================================================

    # Compute state size for GPU buffer
    comptime STATE_SIZE: Int = gc_state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

    # Model buffer size
    comptime MODEL_SIZE: Int = gc_model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()

    # ==========================================================================
    # GPU Layout Helper Methods
    # ==========================================================================

    @staticmethod
    @always_inline
    fn get_qpos_offset() -> Int:
        """Get offset to qpos array in state buffer."""
        return gc_qpos_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qvel_offset() -> Int:
        """Get offset to qvel array in state buffer."""
        return gc_qvel_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qacc_offset() -> Int:
        """Get offset to qacc array in state buffer."""
        return gc_qacc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qfrc_offset() -> Int:
        """Get offset to qfrc array in state buffer."""
        return gc_qfrc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_xpos_offset() -> Int:
        """Get offset to xpos array in state buffer."""
        return gc_xpos_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xquat_offset() -> Int:
        """Get offset to xquat array in state buffer."""
        return gc_xquat_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xvel_offset() -> Int:
        """Get offset to xvel array in state buffer."""
        return gc_xvel_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xangvel_offset() -> Int:
        """Get offset to xangvel array in state buffer."""
        return gc_xangvel_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_contacts_offset() -> Int:
        """Get offset to contacts array in state buffer."""
        return gc_contacts_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_metadata_offset() -> Int:
        """Get offset to metadata in state buffer."""
        return gc_metadata_offset[Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS]()

    @staticmethod
    @always_inline
    fn get_model_body_offset(body_idx: Int) -> Int:
        """Get offset to body data in model buffer."""
        return gc_model_body_offset(body_idx)

    @staticmethod
    @always_inline
    fn get_model_joint_offset(joint_idx: Int) -> Int:
        """Get offset to joint data in model buffer."""
        return gc_model_joint_offset[Self.NUM_BODIES](joint_idx)

    @staticmethod
    @always_inline
    fn get_model_metadata_offset() -> Int:
        """Get offset to metadata in model buffer."""
        return gc_model_metadata_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()


# Type aliases for convenience
comptime HopperGCConstantsCPU = HopperGCConstants[DType.float64]
comptime HopperGCConstantsGPU = HopperGCConstants[DType.float32]
