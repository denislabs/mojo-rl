"""Half Cheetah GC environment constants.

Physics constants and body/joint parameters matching MuJoCo Half Cheetah.
Based on Gymnasium's half_cheetah.xml.

The Half Cheetah is a 2D planar robot (movement in XZ plane, rotation around Y axis)
with 7 bodies and 9 joints (3 root DOFs + 6 actuated joints).
"""


# ============================================================================
# Physics Parameters
# ============================================================================

comptime DT: Float64 = 0.002  # Physics timestep (500 Hz)
comptime FRAME_SKIP: Int = 5  # Number of physics steps per environment step
comptime EFFECTIVE_DT: Float64 = DT * FRAME_SKIP  # 0.01s per env step

comptime GRAVITY_Z: Float64 = -9.81  # Gravity acceleration (m/s²)
comptime GROUND_Z: Float64 = 0.0  # Ground plane height

comptime MAX_STEPS: Int = 1000  # Maximum episode length
comptime INIT_HEIGHT: Float64 = 0.7  # Initial torso z-position (matching MuJoCo)

# Contact physics
comptime FRICTION: Float64 = 0.9
comptime RESTITUTION: Float64 = 0.0


# ============================================================================
# Generalized Coordinates Dimensions
# ============================================================================

# Joint configuration:
#   rootx (slide), rootz (slide), rooty (hinge),
#   bthigh (hinge), bshin (hinge), bfoot (hinge),
#   fthigh (hinge), fshin (hinge), ffoot (hinge),
#   head (hinge - fixed with zero range)

comptime NQ: Int = 10  # Position DOFs (all joints have 1 qpos each)
comptime NV: Int = 10  # Velocity DOFs (all joints have 1 qvel each)
comptime NBODY: Int = 8  # Number of rigid bodies (including head)
comptime NJOINT: Int = 10  # Number of joints (including head joint)
comptime MAX_CONTACTS: Int = 20  # Maximum contact points


# ============================================================================
# Body Indices
# ============================================================================

comptime BODY_TORSO: Int = 0
comptime BODY_BTHIGH: Int = 1  # Back thigh
comptime BODY_BSHIN: Int = 2  # Back shin
comptime BODY_BFOOT: Int = 3  # Back foot
comptime BODY_FTHIGH: Int = 4  # Front thigh
comptime BODY_FSHIN: Int = 5  # Front shin
comptime BODY_FFOOT: Int = 6  # Front foot
comptime BODY_HEAD: Int = 7  # Head (rigidly attached to torso)


# ============================================================================
# Joint Indices (qpos/qvel addresses)
# ============================================================================

comptime JOINT_ROOTX: Int = 0  # Root x slide (unactuated)
comptime JOINT_ROOTZ: Int = 1  # Root z slide (unactuated)
comptime JOINT_ROOTY: Int = 2  # Root y hinge (unactuated)
comptime JOINT_BTHIGH: Int = 3  # Back thigh hinge (actuated)
comptime JOINT_BSHIN: Int = 4  # Back shin hinge (actuated)
comptime JOINT_BFOOT: Int = 5  # Back foot hinge (actuated)
comptime JOINT_FTHIGH: Int = 6  # Front thigh hinge (actuated)
comptime JOINT_FSHIN: Int = 7  # Front shin hinge (actuated)
comptime JOINT_FFOOT: Int = 8  # Front foot hinge (actuated)
comptime JOINT_HEAD: Int = 9  # Head hinge (fixed with zero range)


# ============================================================================
# Body Geometry (Capsules)
# All bodies are capsules with radius 0.046m (matching MuJoCo)
# Values taken directly from MuJoCo XML size attributes (radius, half_length)
# ============================================================================

comptime CAPSULE_RADIUS: Float64 = 0.046

# Half-lengths for each body capsule (from MuJoCo XML size attribute)
comptime TORSO_HALF_LENGTH: Float64 = 0.5  # Torso extends from -0.5 to 0.5 along X
comptime HEAD_HALF_LENGTH: Float64 = 0.15  # Head capsule half-length
comptime BTHIGH_HALF_LENGTH: Float64 = 0.145  # Back thigh half-length
comptime BSHIN_HALF_LENGTH: Float64 = 0.15  # Back shin half-length
comptime BFOOT_HALF_LENGTH: Float64 = 0.094  # Back foot half-length
comptime FTHIGH_HALF_LENGTH: Float64 = 0.133  # Front thigh half-length
comptime FSHIN_HALF_LENGTH: Float64 = 0.106  # Front shin half-length
comptime FFOOT_HALF_LENGTH: Float64 = 0.07  # Front foot half-length

# Head position relative to torso (from MuJoCo XML: pos=".6 0 .1")
comptime HEAD_POS_X: Float64 = 0.6  # Forward of torso center
comptime HEAD_POS_Y: Float64 = 0.0
comptime HEAD_POS_Z: Float64 = 0.1  # Slightly above torso

# Head orientation (from MuJoCo XML: axisangle="0 1 0 .87")
# Tilted upward by 0.87 radians (~50 degrees) around Y axis
comptime HEAD_AXIS_ANGLE: Float64 = 0.87


# ============================================================================
# Body Masses
# Total mass is 14 kg, distributed across bodies
# Based on MuJoCo density calculations for capsules
# ============================================================================

comptime TORSO_MASS: Float64 = 6.25  # Heavier torso
comptime HEAD_MASS: Float64 = 0.90  # Head capsule mass
comptime BTHIGH_MASS: Float64 = 1.54
comptime BSHIN_MASS: Float64 = 1.58
comptime BFOOT_MASS: Float64 = 1.10
comptime FTHIGH_MASS: Float64 = 1.43
comptime FSHIN_MASS: Float64 = 1.17
comptime FFOOT_MASS: Float64 = 0.93


# ============================================================================
# Joint Limits (radians)
# From MuJoCo XML joint range specifications
# ============================================================================

# Back thigh joint limits
comptime BTHIGH_LOWER: Float64 = -0.52
comptime BTHIGH_UPPER: Float64 = 1.05

# Back shin joint limits
comptime BSHIN_LOWER: Float64 = -0.785
comptime BSHIN_UPPER: Float64 = 0.785

# Back foot joint limits
comptime BFOOT_LOWER: Float64 = -0.4
comptime BFOOT_UPPER: Float64 = 0.785

# Front thigh joint limits
comptime FTHIGH_LOWER: Float64 = -1.0
comptime FTHIGH_UPPER: Float64 = 0.7

# Front shin joint limits
comptime FSHIN_LOWER: Float64 = -1.2
comptime FSHIN_UPPER: Float64 = 0.87

# Front foot joint limits
comptime FFOOT_LOWER: Float64 = -0.5
comptime FFOOT_UPPER: Float64 = 0.5

# Head joint limits (fixed - zero range makes it rigid)
comptime HEAD_LOWER: Float64 = 0.0
comptime HEAD_UPPER: Float64 = 0.0


# ============================================================================
# Joint Gear Ratios (for torque scaling)
# From MuJoCo XML actuator gear specifications
# ============================================================================

comptime BTHIGH_GEAR: Float64 = 120.0
comptime BSHIN_GEAR: Float64 = 90.0
comptime BFOOT_GEAR: Float64 = 60.0
comptime FTHIGH_GEAR: Float64 = 120.0
comptime FSHIN_GEAR: Float64 = 60.0
comptime FFOOT_GEAR: Float64 = 30.0


# ============================================================================
# Joint Stiffness and Damping
# From MuJoCo XML joint specifications
# ============================================================================

comptime BTHIGH_STIFFNESS: Float64 = 240.0
comptime BTHIGH_DAMPING: Float64 = 6.0

comptime BSHIN_STIFFNESS: Float64 = 180.0
comptime BSHIN_DAMPING: Float64 = 4.5

comptime BFOOT_STIFFNESS: Float64 = 120.0
comptime BFOOT_DAMPING: Float64 = 3.0

comptime FTHIGH_STIFFNESS: Float64 = 180.0
comptime FTHIGH_DAMPING: Float64 = 4.5

comptime FSHIN_STIFFNESS: Float64 = 120.0
comptime FSHIN_DAMPING: Float64 = 3.0

comptime FFOOT_STIFFNESS: Float64 = 60.0
comptime FFOOT_DAMPING: Float64 = 1.5


# ============================================================================
# Joint Armature (rotor inertia for stability)
# ============================================================================

comptime JOINT_ARMATURE: Float64 = 0.1


# ============================================================================
# Reward Parameters
# ============================================================================

comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0
comptime CTRL_COST_WEIGHT: Float64 = 0.1


# ============================================================================
# Reset Noise Scale
# ============================================================================

comptime RESET_NOISE_SCALE: Float64 = 0.1


# ============================================================================
# Observation and Action Dimensions
# ============================================================================

comptime OBS_DIM: Int = 17  # 8 qpos (excluding rootx) + 9 qvel
comptime ACTION_DIM: Int = 6  # 6 actuated joints


# ============================================================================
# GPU Layout Constants Import
# ============================================================================

from physics3d.gpu.constants import (
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
    gc_model_curriculum_offset,
    GC_MODEL_BODY_SIZE,
    GC_MODEL_JOINT_SIZE,
    GC_MODEL_META_SIZE,
    GC_MODEL_CURRICULUM_SIZE,
    GC_CONTACT_SIZE,
    GC_CURRICULUM_IDX_MIN_HEIGHT,
    GC_CURRICULUM_IDX_MAX_PITCH,
)


# ============================================================================
# HalfCheetahGCConstants Struct for GPU
# ============================================================================


struct HalfCheetahGCConstants[DTYPE: DType = DType.float64]:
    """Constants for HalfCheetahGC environment using Generalized Coordinates.

    MuJoCo Half Cheetah configuration:
    - 8 bodies: torso, bthigh, bshin, bfoot, fthigh, fshin, ffoot, head
    - 10 joints: rootx (slide), rootz (slide), rooty (hinge),
                 6 leg hinges, head (hinge, fixed)

    Joint space dimensions:
    - NQ = 10 (2 slide positions + 8 hinge angles)
    - NV = 10 (2 slide velocities + 8 hinge angular velocities)

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
    comptime FRICTION: Scalar[Self.DTYPE] = 0.9

    # ==========================================================================
    # Body Geometry (from MuJoCo Half Cheetah XML)
    # ==========================================================================

    comptime CAPSULE_RADIUS: Scalar[Self.DTYPE] = 0.046

    # Torso (horizontal capsule, rotated 90° around Y)
    comptime TORSO_MASS: Scalar[Self.DTYPE] = 6.25
    comptime TORSO_HALF_LENGTH: Scalar[Self.DTYPE] = 0.5

    # Head (tilted capsule)
    comptime HEAD_MASS: Scalar[Self.DTYPE] = 0.90
    comptime HEAD_HALF_LENGTH: Scalar[Self.DTYPE] = 0.15
    comptime HEAD_POS_X: Scalar[Self.DTYPE] = 0.6
    comptime HEAD_POS_Y: Scalar[Self.DTYPE] = 0.0
    comptime HEAD_POS_Z: Scalar[Self.DTYPE] = 0.1
    comptime HEAD_AXIS_ANGLE: Scalar[Self.DTYPE] = 0.87

    # Back leg
    comptime BTHIGH_MASS: Scalar[Self.DTYPE] = 1.54
    comptime BTHIGH_HALF_LENGTH: Scalar[Self.DTYPE] = 0.145
    comptime BSHIN_MASS: Scalar[Self.DTYPE] = 1.58
    comptime BSHIN_HALF_LENGTH: Scalar[Self.DTYPE] = 0.15
    comptime BFOOT_MASS: Scalar[Self.DTYPE] = 1.10
    comptime BFOOT_HALF_LENGTH: Scalar[Self.DTYPE] = 0.094

    # Front leg
    comptime FTHIGH_MASS: Scalar[Self.DTYPE] = 1.43
    comptime FTHIGH_HALF_LENGTH: Scalar[Self.DTYPE] = 0.133
    comptime FSHIN_MASS: Scalar[Self.DTYPE] = 1.17
    comptime FSHIN_HALF_LENGTH: Scalar[Self.DTYPE] = 0.106
    comptime FFOOT_MASS: Scalar[Self.DTYPE] = 0.93
    comptime FFOOT_HALF_LENGTH: Scalar[Self.DTYPE] = 0.07

    # ==========================================================================
    # Body Indices
    # ==========================================================================

    comptime BODY_TORSO: Int = 0
    comptime BODY_BTHIGH: Int = 1
    comptime BODY_BSHIN: Int = 2
    comptime BODY_BFOOT: Int = 3
    comptime BODY_FTHIGH: Int = 4
    comptime BODY_FSHIN: Int = 5
    comptime BODY_FFOOT: Int = 6
    comptime BODY_HEAD: Int = 7

    # ==========================================================================
    # Joint Configuration (MuJoCo style)
    # ==========================================================================

    comptime JOINT_ROOTX: Int = 0  # Slide along X (body 0)
    comptime JOINT_ROOTZ: Int = 1  # Slide along Z (body 0)
    comptime JOINT_ROOTY: Int = 2  # Hinge around Y (body 0)
    comptime JOINT_BTHIGH: Int = 3  # Hinge around Y (body 1)
    comptime JOINT_BSHIN: Int = 4  # Hinge around Y (body 2)
    comptime JOINT_BFOOT: Int = 5  # Hinge around Y (body 3)
    comptime JOINT_FTHIGH: Int = 6  # Hinge around Y (body 4)
    comptime JOINT_FSHIN: Int = 7  # Hinge around Y (body 5)
    comptime JOINT_FFOOT: Int = 8  # Hinge around Y (body 6)
    comptime JOINT_HEAD: Int = 9  # Hinge around Y (body 7, fixed)

    # ==========================================================================
    # Joint Space Dimensions
    # ==========================================================================

    comptime NQ: Int = 10  # qpos dimension
    comptime NV: Int = 10  # qvel dimension

    # ==========================================================================
    # Motor Parameters (Gear Ratios)
    # ==========================================================================

    comptime BTHIGH_GEAR: Scalar[Self.DTYPE] = 120.0
    comptime BSHIN_GEAR: Scalar[Self.DTYPE] = 90.0
    comptime BFOOT_GEAR: Scalar[Self.DTYPE] = 60.0
    comptime FTHIGH_GEAR: Scalar[Self.DTYPE] = 120.0
    comptime FSHIN_GEAR: Scalar[Self.DTYPE] = 60.0
    comptime FFOOT_GEAR: Scalar[Self.DTYPE] = 30.0

    # ==========================================================================
    # Joint Limits (from MuJoCo half_cheetah.xml)
    # ==========================================================================

    comptime BTHIGH_JOINT_MIN: Scalar[Self.DTYPE] = -0.52
    comptime BTHIGH_JOINT_MAX: Scalar[Self.DTYPE] = 1.05

    comptime BSHIN_JOINT_MIN: Scalar[Self.DTYPE] = -0.785
    comptime BSHIN_JOINT_MAX: Scalar[Self.DTYPE] = 0.785

    comptime BFOOT_JOINT_MIN: Scalar[Self.DTYPE] = -0.4
    comptime BFOOT_JOINT_MAX: Scalar[Self.DTYPE] = 0.785

    comptime FTHIGH_JOINT_MIN: Scalar[Self.DTYPE] = -1.0
    comptime FTHIGH_JOINT_MAX: Scalar[Self.DTYPE] = 0.7

    comptime FSHIN_JOINT_MIN: Scalar[Self.DTYPE] = -1.2
    comptime FSHIN_JOINT_MAX: Scalar[Self.DTYPE] = 0.87

    comptime FFOOT_JOINT_MIN: Scalar[Self.DTYPE] = -0.5
    comptime FFOOT_JOINT_MAX: Scalar[Self.DTYPE] = 0.5

    comptime HEAD_JOINT_MIN: Scalar[Self.DTYPE] = 0.0
    comptime HEAD_JOINT_MAX: Scalar[Self.DTYPE] = 0.0

    # ==========================================================================
    # Episode Parameters
    # ==========================================================================

    comptime MAX_STEPS: Int = 1000

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.1
    comptime ANGLE_PENALTY_WEIGHT: Scalar[
        Self.DTYPE
    ] = 0.5  # Penalty for |y_angle|

    # ==========================================================================
    # Health Parameters (for TERMINATE_ON_UNHEALTHY mode)
    # ==========================================================================

    comptime MAX_PITCH: Scalar[
        Self.DTYPE
    ] = 1.0  # ~57 deg, terminate when |y_angle| > max_pitch

    # ==========================================================================
    # Curriculum Parameters
    # ==========================================================================

    # Initial (lenient) values for curriculum
    comptime CURRICULUM_INITIAL_MAX_PITCH: Scalar[
        Self.DTYPE
    ] = 3.0  # Very lenient (~172 deg)

    # Final (strict) values for curriculum
    comptime CURRICULUM_FINAL_MAX_PITCH: Scalar[Self.DTYPE] = 1.0  # ~57 deg

    # ==========================================================================
    # Layout Constants
    # ==========================================================================

    comptime NUM_BODIES: Int = 8
    comptime NUM_JOINTS: Int = 10  # 2 slide + 8 hinge
    comptime MAX_CONTACTS: Int = 20

    # ==========================================================================
    # Initial State Constants
    # ==========================================================================

    comptime INITIAL_Z: Scalar[Self.DTYPE] = 0.7  # Initial torso height

    # ==========================================================================
    # Observation/Action Dimensions
    # ==========================================================================

    comptime OBS_DIM: Int = 17  # 8 qpos (excluding rootx and head) + 9 qvel (excluding head)
    comptime ACTION_DIM: Int = 6  # 6 actuated joints (not root or head)

    # ==========================================================================
    # GPU Layout Constants (using physics3d GC layout)
    # ==========================================================================

    comptime STATE_SIZE: Int = gc_state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

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
        return gc_metadata_offset[
            Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
        ]()

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

    @staticmethod
    @always_inline
    fn get_model_curriculum_offset() -> Int:
        """Get offset to curriculum parameters in model buffer."""
        return gc_model_curriculum_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()


# Type aliases for convenience
comptime HalfCheetahGCConstantsCPU = HalfCheetahGCConstants[DType.float64]
comptime HalfCheetahGCConstantsGPU = HalfCheetahGCConstants[DType.float32]
