"""Constants for HalfCheetah3D GPU environment.

Uses HalfCheetahLayout3D from physics3d for compile-time layout computation.
Extends 2D HalfCheetah constants to 3D with Y-axis joint rotation.

NOTE: HC3DConstants is parametrized with DTYPE to support both:
- Float64 for CPU code (default, matches Vec3/Quat defaults)
- Float32 for GPU code (matches physics3d dtype)
"""

from physics3d import HalfCheetahLayout3D


struct HC3DConstants[DTYPE: DType = DType.float64]:
    """Constants for HalfCheetah3D environment.

    Based on MuJoCo HalfCheetah with physics3d layout for GPU support.
    All joints rotate around the Y-axis (lateral rotation like 2D).
    Gravity is along negative Z-axis.

    Type Parameters:
        DTYPE: The floating point type for physics constants.
               Use Float64 (default) for CPU, Float32 for GPU.
    """

    # Layout type alias
    comptime Layout = HalfCheetahLayout3D

    # ==========================================================================
    # Physics Parameters
    # ==========================================================================

    comptime DT: Scalar[Self.DTYPE] = 0.002  # Balanced timestep (was 0.01, too unstable; 0.001 too slow)
    comptime FRAME_SKIP: Int = 25  # Balanced substeps (total sim time = 0.05s per step)
    comptime GRAVITY_X: Scalar[Self.DTYPE] = 0.0
    comptime GRAVITY_Y: Scalar[Self.DTYPE] = 0.0
    comptime GRAVITY_Z: Scalar[Self.DTYPE] = -9.81

    # Physics solver iterations (reduced - projection handles rest)
    comptime VELOCITY_ITERATIONS: Int = 10
    comptime POSITION_ITERATIONS: Int = 5

    # Contact physics
    comptime FRICTION: Scalar[Self.DTYPE] = 0.9
    comptime RESTITUTION: Scalar[Self.DTYPE] = 0.0
    comptime BAUMGARTE: Scalar[Self.DTYPE] = 0.8  # High value for tighter joint constraints
    comptime SLOP: Scalar[Self.DTYPE] = 0.001  # Reduced slop for tighter joints

    # ==========================================================================
    # Body Geometry (from MuJoCo HalfCheetah XML)
    # ==========================================================================

    # Torso (longer horizontal body along X-axis)
    comptime TORSO_LENGTH: Scalar[Self.DTYPE] = 1.0
    comptime TORSO_RADIUS: Scalar[Self.DTYPE] = 0.046
    comptime TORSO_MASS: Scalar[Self.DTYPE] = 6.25

    # Back thigh
    comptime BTHIGH_LENGTH: Scalar[Self.DTYPE] = 0.145
    comptime BTHIGH_RADIUS: Scalar[Self.DTYPE] = 0.046
    comptime BTHIGH_MASS: Scalar[Self.DTYPE] = 1.0

    # Back shin
    comptime BSHIN_LENGTH: Scalar[Self.DTYPE] = 0.15
    comptime BSHIN_RADIUS: Scalar[Self.DTYPE] = 0.046
    comptime BSHIN_MASS: Scalar[Self.DTYPE] = 1.0

    # Back foot
    comptime BFOOT_LENGTH: Scalar[Self.DTYPE] = 0.094
    comptime BFOOT_RADIUS: Scalar[Self.DTYPE] = 0.046
    comptime BFOOT_MASS: Scalar[Self.DTYPE] = 1.0

    # Front thigh
    comptime FTHIGH_LENGTH: Scalar[Self.DTYPE] = 0.133
    comptime FTHIGH_RADIUS: Scalar[Self.DTYPE] = 0.046
    comptime FTHIGH_MASS: Scalar[Self.DTYPE] = 1.0

    # Front shin
    comptime FSHIN_LENGTH: Scalar[Self.DTYPE] = 0.106
    comptime FSHIN_RADIUS: Scalar[Self.DTYPE] = 0.046
    comptime FSHIN_MASS: Scalar[Self.DTYPE] = 1.0

    # Front foot
    comptime FFOOT_LENGTH: Scalar[Self.DTYPE] = 0.07
    comptime FFOOT_RADIUS: Scalar[Self.DTYPE] = 0.046
    comptime FFOOT_MASS: Scalar[Self.DTYPE] = 1.0

    # Initial height (torso center Z coordinate)
    comptime INIT_HEIGHT: Scalar[Self.DTYPE] = 0.7

    # ==========================================================================
    # Body Indices
    # ==========================================================================

    comptime BODY_TORSO: Int = 0
    comptime BODY_BTHIGH: Int = 1  # Back thigh
    comptime BODY_BSHIN: Int = 2  # Back shin
    comptime BODY_BFOOT: Int = 3  # Back foot
    comptime BODY_FTHIGH: Int = 4  # Front thigh
    comptime BODY_FSHIN: Int = 5  # Front shin
    comptime BODY_FFOOT: Int = 6  # Front foot

    # ==========================================================================
    # Joint Indices
    # ==========================================================================

    comptime JOINT_BTHIGH: Int = 0  # Back hip
    comptime JOINT_BSHIN: Int = 1  # Back knee
    comptime JOINT_BFOOT: Int = 2  # Back ankle
    comptime JOINT_FTHIGH: Int = 3  # Front hip
    comptime JOINT_FSHIN: Int = 4  # Front knee
    comptime JOINT_FFOOT: Int = 5  # Front ankle

    # ==========================================================================
    # Joint Axis (all joints rotate around Y-axis for lateral rotation)
    # ==========================================================================

    comptime JOINT_AXIS_X: Scalar[Self.DTYPE] = 0.0
    comptime JOINT_AXIS_Y: Scalar[Self.DTYPE] = 1.0
    comptime JOINT_AXIS_Z: Scalar[Self.DTYPE] = 0.0

    # ==========================================================================
    # Joint Limits (from MuJoCo XML)
    # ==========================================================================

    comptime BTHIGH_LIMIT_LOW: Scalar[Self.DTYPE] = -0.52
    comptime BTHIGH_LIMIT_HIGH: Scalar[Self.DTYPE] = 1.05

    comptime BSHIN_LIMIT_LOW: Scalar[Self.DTYPE] = -0.785
    comptime BSHIN_LIMIT_HIGH: Scalar[Self.DTYPE] = 0.785

    comptime BFOOT_LIMIT_LOW: Scalar[Self.DTYPE] = -0.4
    comptime BFOOT_LIMIT_HIGH: Scalar[Self.DTYPE] = 0.785

    comptime FTHIGH_LIMIT_LOW: Scalar[Self.DTYPE] = -1.0
    comptime FTHIGH_LIMIT_HIGH: Scalar[Self.DTYPE] = 0.7

    comptime FSHIN_LIMIT_LOW: Scalar[Self.DTYPE] = -1.2
    comptime FSHIN_LIMIT_HIGH: Scalar[Self.DTYPE] = 0.87

    comptime FFOOT_LIMIT_LOW: Scalar[Self.DTYPE] = -0.5
    comptime FFOOT_LIMIT_HIGH: Scalar[Self.DTYPE] = 0.5

    # ==========================================================================
    # Motor Parameters - Per-Joint Torque Limits
    # ==========================================================================
    # With proper joint stiffness/damping and soft constraint compliance,
    # we can now use realistic MuJoCo-like torque values.
    # The passive joint dynamics help stabilize the system.

    # Per-joint max torque (Nm) - scaled for explicit integration
    # MuJoCo uses 120 Nm with implicit solver. For explicit integration,
    # we need much lower values to maintain stability (roughly 10% of MuJoCo).
    comptime BTHIGH_MAX_TORQUE: Scalar[Self.DTYPE] = 15.0  # Hip - strongest (was 120)
    comptime BSHIN_MAX_TORQUE: Scalar[Self.DTYPE] = 12.0   # Knee (was 90)
    comptime BFOOT_MAX_TORQUE: Scalar[Self.DTYPE] = 8.0    # Ankle (was 60)
    comptime FTHIGH_MAX_TORQUE: Scalar[Self.DTYPE] = 15.0  # Hip - strongest (was 120)
    comptime FSHIN_MAX_TORQUE: Scalar[Self.DTYPE] = 8.0    # Knee (was 60)
    comptime FFOOT_MAX_TORQUE: Scalar[Self.DTYPE] = 4.0    # Ankle - weakest (was 30)

    comptime GEAR_RATIO: Scalar[Self.DTYPE] = 120.0  # MuJoCo gear scaling (for reference only)
    comptime MOTOR_KP: Scalar[Self.DTYPE] = 100.0  # PD controller P gain (for reference)
    comptime MOTOR_KD: Scalar[Self.DTYPE] = 10.0  # PD controller D gain (for reference)

    # ==========================================================================
    # Joint Passive Dynamics (MuJoCo-style spring-damper)
    # ==========================================================================
    # IMPORTANT: MuJoCo uses an implicit solver that can handle high stiffness.
    # Our explicit semi-implicit Euler integrator CANNOT handle MuJoCo's values.
    # High stiffness (240 Nm/rad) causes numerical instability with explicit integration.
    # Set to 0 to disable passive forces - rely on constraint solver instead.
    #
    # If needed, use values 100x smaller than MuJoCo (e.g., 2.4 instead of 240).

    # Per-joint stiffness (Nm/rad) - spring constant
    # DISABLED: Incompatible with explicit integration at MuJoCo values
    comptime BTHIGH_STIFFNESS: Scalar[Self.DTYPE] = 0.0  # Was 240.0
    comptime BSHIN_STIFFNESS: Scalar[Self.DTYPE] = 0.0   # Was 180.0
    comptime BFOOT_STIFFNESS: Scalar[Self.DTYPE] = 0.0   # Was 120.0
    comptime FTHIGH_STIFFNESS: Scalar[Self.DTYPE] = 0.0  # Was 180.0
    comptime FSHIN_STIFFNESS: Scalar[Self.DTYPE] = 0.0   # Was 120.0
    comptime FFOOT_STIFFNESS: Scalar[Self.DTYPE] = 0.0   # Was 60.0

    # Per-joint damping (Nm·s/rad) - velocity damping coefficient
    # DISABLED: Incompatible with explicit integration at MuJoCo values
    comptime BTHIGH_DAMPING: Scalar[Self.DTYPE] = 0.0   # Was 6.0
    comptime BSHIN_DAMPING: Scalar[Self.DTYPE] = 0.0    # Was 4.5
    comptime BFOOT_DAMPING: Scalar[Self.DTYPE] = 0.0    # Was 3.0
    comptime FTHIGH_DAMPING: Scalar[Self.DTYPE] = 0.0   # Was 4.5
    comptime FSHIN_DAMPING: Scalar[Self.DTYPE] = 0.0    # Was 3.0
    comptime FFOOT_DAMPING: Scalar[Self.DTYPE] = 0.0    # Was 1.5

    # Per-joint armature (kg·m²) - rotor inertia for effective mass
    # Adds stability and makes the system less sensitive to torques
    comptime BTHIGH_ARMATURE: Scalar[Self.DTYPE] = 0.1   # Hip
    comptime BSHIN_ARMATURE: Scalar[Self.DTYPE] = 0.08   # Knee
    comptime BFOOT_ARMATURE: Scalar[Self.DTYPE] = 0.05   # Ankle
    comptime FTHIGH_ARMATURE: Scalar[Self.DTYPE] = 0.1   # Hip
    comptime FSHIN_ARMATURE: Scalar[Self.DTYPE] = 0.08   # Knee
    comptime FFOOT_ARMATURE: Scalar[Self.DTYPE] = 0.05   # Ankle

    # ==========================================================================
    # Soft Constraint Parameters (MuJoCo solref/solimp style)
    # ==========================================================================
    # These soften the constraint response, preventing numerical instability
    # when high torques would otherwise cause joint separation.

    comptime SOFT_TIMECONST: Scalar[Self.DTYPE] = 0.02   # Time constant (s)
    comptime SOFT_DAMPRATIO: Scalar[Self.DTYPE] = 1.0    # Damping ratio (critical damping)

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.1
    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0

    # ==========================================================================
    # Episode Parameters
    # ==========================================================================

    comptime MAX_STEPS: Int = 1000

    # ==========================================================================
    # Termination Parameters (healthy bounds)
    # ==========================================================================

    # HalfCheetah doesn't terminate on falling by default (like MuJoCo)
    comptime TERMINATE_WHEN_UNHEALTHY: Bool = False
    comptime HEALTHY_Z_MIN: Scalar[Self.DTYPE] = -0.5
    comptime HEALTHY_Z_MAX: Scalar[Self.DTYPE] = 2.0
    comptime HEALTHY_ANGLE_MAX: Scalar[Self.DTYPE] = 2.0  # ~115 degrees - generous for cheetah

    # ==========================================================================
    # Layout Constants (from HalfCheetahLayout3D)
    # ==========================================================================

    # Counts
    comptime NUM_BODIES: Int = Self.Layout.NUM_BODIES  # 7
    comptime NUM_SHAPES: Int = Self.Layout.NUM_SHAPES  # 7
    comptime MAX_CONTACTS: Int = Self.Layout.MAX_CONTACTS  # 12
    comptime MAX_JOINTS: Int = Self.Layout.MAX_JOINTS  # 6
    comptime NUM_JOINTS: Int = 6  # Actual number of hinge joints

    # Dimensions
    comptime OBS_DIM: Int = Self.Layout.OBS_DIM  # 17
    comptime ACTION_DIM: Int = 6  # 6 joint torques

    # Offsets within each environment's state
    comptime OBS_OFFSET: Int = Self.Layout.OBS_OFFSET
    comptime BODIES_OFFSET: Int = Self.Layout.BODIES_OFFSET
    comptime JOINTS_OFFSET: Int = Self.Layout.JOINTS_OFFSET
    comptime JOINT_COUNT_OFFSET: Int = Self.Layout.JOINT_COUNT_OFFSET
    comptime CONTACTS_OFFSET: Int = Self.Layout.CONTACTS_OFFSET
    comptime CONTACT_COUNT_OFFSET: Int = Self.Layout.CONTACT_COUNT_OFFSET
    comptime METADATA_OFFSET: Int = Self.Layout.METADATA_OFFSET

    # ==========================================================================
    # Metadata Field Indices
    # ==========================================================================

    comptime META_STEP_COUNT: Int = 0
    comptime META_PREV_X: Int = 1  # Previous x position for velocity reward
    comptime META_PREV_SHAPING: Int = 2
    comptime META_DONE: Int = 3
    comptime META_TOTAL_REWARD: Int = 4
    comptime META_UNUSED1: Int = 5
    comptime META_UNUSED2: Int = 6
    comptime META_UNUSED3: Int = 7

    # ==========================================================================
    # Total State Size
    # ==========================================================================

    comptime STATE_SIZE: Int = Self.Layout.STATE_SIZE

    # ==========================================================================
    # Observation Layout (17D total)
    # ==========================================================================
    # [0]: z position of torso (height)
    # [1]: pitch angle of torso (rotation around Y-axis, from quaternion)
    # [2]: angle of back thigh
    # [3]: angle of back shin
    # [4]: angle of back foot
    # [5]: angle of front thigh
    # [6]: angle of front shin
    # [7]: angle of front foot
    # [8]: velocity of x (forward)
    # [9]: velocity of z (vertical)
    # [10]: angular velocity of torso (Y component - pitch rate)
    # [11]: angular velocity of back thigh
    # [12]: angular velocity of back shin
    # [13]: angular velocity of back foot
    # [14]: angular velocity of front thigh
    # [15]: angular velocity of front shin
    # [16]: angular velocity of front foot


# Type aliases for convenience
comptime HC3DConstantsCPU = HC3DConstants[DType.float64]
comptime HC3DConstantsGPU = HC3DConstants[DType.float32]
