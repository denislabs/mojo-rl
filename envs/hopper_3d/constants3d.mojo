"""Constants for Hopper3D environment.

Based on MuJoCo Hopper with physics3d engine.
Supports both CPU (Float64) and GPU (Float32) execution.
"""

from physics3d import Model, Data
from physics3d.gpu.constants import (
    compute_state_size,
    body_offset,
    joint_offset,
    slide_joint_offset,
    metadata_offset,
    BODY_STATE_SIZE,
    JOINT_STATE_SIZE,
    SLIDE_JOINT_STATE_SIZE,
    CONTACT_STATE_SIZE,
    METADATA_SIZE,
    MODEL_BODY_SIZE,
)


struct Hopper3DConstants[DTYPE: DType = DType.float64]:
    """Constants for Hopper3D environment.

    Based on MuJoCo Hopper configuration.
    All joints rotate around the Y-axis (lateral rotation).
    Gravity is along negative Z-axis.

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
    comptime RESTITUTION: Scalar[Self.DTYPE] = 0.0

    # ==========================================================================
    # Body Geometry (from MuJoCo Hopper XML)
    # ==========================================================================

    # Torso (vertical capsule)
    comptime TORSO_MASS: Scalar[Self.DTYPE] = 1.0
    comptime TORSO_RADIUS: Scalar[Self.DTYPE] = 0.05
    comptime TORSO_HALF_LENGTH: Scalar[Self.DTYPE] = 0.2

    # Thigh (vertical capsule)
    comptime THIGH_MASS: Scalar[Self.DTYPE] = 0.5
    comptime THIGH_RADIUS: Scalar[Self.DTYPE] = 0.05
    comptime THIGH_HALF_LENGTH: Scalar[Self.DTYPE] = 0.225

    # Leg (vertical capsule)
    comptime LEG_MASS: Scalar[Self.DTYPE] = 0.3
    comptime LEG_RADIUS: Scalar[Self.DTYPE] = 0.04
    comptime LEG_HALF_LENGTH: Scalar[Self.DTYPE] = 0.25

    # Foot (horizontal capsule - rotated 90° around Y)
    comptime FOOT_MASS: Scalar[Self.DTYPE] = 0.2
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
    # Joint Indices
    # ==========================================================================

    # Hinge joints
    comptime JOINT_ROOTY: Int = 0  # Root pitch (world -> torso)
    comptime JOINT_HIP: Int = 1  # Hip (torso -> thigh)
    comptime JOINT_KNEE: Int = 2  # Knee (thigh -> leg)
    comptime JOINT_ANKLE: Int = 3  # Ankle (leg -> foot)

    # Slide joints
    comptime SLIDE_ROOTX: Int = 0  # Root X translation
    comptime SLIDE_ROOTZ: Int = 1  # Root Z translation

    # ==========================================================================
    # Motor Parameters
    # ==========================================================================

    comptime TORQUE_LIMIT: Scalar[Self.DTYPE] = 200.0  # MuJoCo gear=200

    # ==========================================================================
    # Termination Parameters
    # ==========================================================================

    comptime MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime MAX_PITCH: Scalar[Self.DTYPE] = 1.0  # ~57 degrees

    # ==========================================================================
    # Episode Parameters
    # ==========================================================================

    comptime MAX_STEPS: Int = 1000

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.001
    comptime ALIVE_BONUS: Scalar[Self.DTYPE] = 1.0

    # ==========================================================================
    # Layout Constants
    # ==========================================================================

    comptime NUM_BODIES: Int = 4
    comptime NUM_HINGE_JOINTS: Int = 4  # RootY, Hip, Knee, Ankle
    comptime NUM_SLIDE_JOINTS: Int = 2  # RootX, RootZ
    comptime NUM_ACTUATED_JOINTS: Int = 3  # Hip, Knee, Ankle (not RootY)
    comptime MAX_CONTACTS: Int = 20

    # ==========================================================================
    # Observation/Action Dimensions
    # ==========================================================================

    comptime OBS_DIM: Int = 11
    comptime ACTION_DIM: Int = 3

    # ==========================================================================
    # GPU Layout Constants (using physics3d layout)
    # ==========================================================================

    # Compute state size for GPU buffer using physics3d's layout formula
    comptime STATE_SIZE: Int = compute_state_size[
        Self.NUM_BODIES,
        Self.MAX_CONTACTS,
        Self.NUM_HINGE_JOINTS,
        Self.NUM_SLIDE_JOINTS,
    ]()

    # ==========================================================================
    # GPU Layout Helper Methods
    # ==========================================================================

    @staticmethod
    @always_inline
    fn get_body_offset(body_idx: Int) -> Int:
        """Get offset to start of body state within environment state."""
        return body_offset[
            Self.NUM_BODIES,
            Self.MAX_CONTACTS,
            Self.NUM_HINGE_JOINTS,
            Self.NUM_SLIDE_JOINTS,
        ](body_idx)

    @staticmethod
    @always_inline
    fn get_joint_offset(joint_idx: Int) -> Int:
        """Get offset to start of hinge joint state within environment state."""
        return joint_offset[
            Self.NUM_BODIES,
            Self.MAX_CONTACTS,
            Self.NUM_HINGE_JOINTS,
            Self.NUM_SLIDE_JOINTS,
        ](joint_idx)

    @staticmethod
    @always_inline
    fn get_slide_joint_offset(slide_joint_idx: Int) -> Int:
        """Get offset to start of slide joint state within environment state."""
        return slide_joint_offset[
            Self.NUM_BODIES,
            Self.MAX_CONTACTS,
            Self.NUM_HINGE_JOINTS,
            Self.NUM_SLIDE_JOINTS,
        ](slide_joint_idx)

    @staticmethod
    @always_inline
    fn get_metadata_offset() -> Int:
        """Get offset to metadata within environment state."""
        return metadata_offset[
            Self.NUM_BODIES,
            Self.MAX_CONTACTS,
            Self.NUM_HINGE_JOINTS,
            Self.NUM_SLIDE_JOINTS,
        ]()


# Type aliases for convenience
comptime Hopper3DConstantsCPU = Hopper3DConstants[DType.float64]
comptime Hopper3DConstantsGPU = Hopper3DConstants[DType.float32]
