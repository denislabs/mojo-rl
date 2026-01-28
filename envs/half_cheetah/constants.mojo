"""Constants for HalfCheetahPlanar GPU environment.

Uses HalfCheetahLayout from physics2d for compile-time layout computation.
"""

from physics2d import HalfCheetahLayout


struct HCConstants:
    """Constants for HalfCheetahPlanar environment.

    Based on MuJoCo HalfCheetah with physics2d layout for GPU support.
    """

    # Layout type alias
    comptime HC = HalfCheetahLayout

    # ==========================================================================
    # Physics Parameters
    # ==========================================================================

    comptime DT: Float64 = 0.01
    comptime FRAME_SKIP: Int = 5
    comptime GRAVITY_X: Float64 = 0.0
    comptime GRAVITY_Y: Float64 = -9.81

    # Physics solver iterations
    comptime VELOCITY_ITERATIONS: Int = 4
    comptime POSITION_ITERATIONS: Int = 2

    # Contact physics
    comptime FRICTION: Float64 = 0.9
    comptime RESTITUTION: Float64 = 0.0
    comptime BAUMGARTE: Float64 = 0.2
    comptime SLOP: Float64 = 0.005

    # ==========================================================================
    # Body Geometry (from MuJoCo HalfCheetah XML)
    # ==========================================================================

    # Torso (longer horizontal body)
    comptime TORSO_LENGTH: Float64 = 1.0
    comptime TORSO_RADIUS: Float64 = 0.046
    comptime TORSO_MASS: Float64 = 6.25

    # Back thigh
    comptime BTHIGH_LENGTH: Float64 = 0.145
    comptime BTHIGH_RADIUS: Float64 = 0.046
    comptime BTHIGH_MASS: Float64 = 1.0

    # Back shin
    comptime BSHIN_LENGTH: Float64 = 0.15
    comptime BSHIN_RADIUS: Float64 = 0.046
    comptime BSHIN_MASS: Float64 = 1.0

    # Back foot
    comptime BFOOT_LENGTH: Float64 = 0.094
    comptime BFOOT_RADIUS: Float64 = 0.046
    comptime BFOOT_MASS: Float64 = 1.0

    # Front thigh
    comptime FTHIGH_LENGTH: Float64 = 0.133
    comptime FTHIGH_RADIUS: Float64 = 0.046
    comptime FTHIGH_MASS: Float64 = 1.0

    # Front shin
    comptime FSHIN_LENGTH: Float64 = 0.106
    comptime FSHIN_RADIUS: Float64 = 0.046
    comptime FSHIN_MASS: Float64 = 1.0

    # Front foot
    comptime FFOOT_LENGTH: Float64 = 0.07
    comptime FFOOT_RADIUS: Float64 = 0.046
    comptime FFOOT_MASS: Float64 = 1.0

    # Initial height
    comptime INIT_HEIGHT: Float64 = 0.7

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
    # Joint Limits (from MuJoCo XML)
    # ==========================================================================

    comptime BTHIGH_LIMIT_LOW: Float64 = -0.52
    comptime BTHIGH_LIMIT_HIGH: Float64 = 1.05

    comptime BSHIN_LIMIT_LOW: Float64 = -0.785
    comptime BSHIN_LIMIT_HIGH: Float64 = 0.785

    comptime BFOOT_LIMIT_LOW: Float64 = -0.4
    comptime BFOOT_LIMIT_HIGH: Float64 = 0.785

    comptime FTHIGH_LIMIT_LOW: Float64 = -1.0
    comptime FTHIGH_LIMIT_HIGH: Float64 = 0.7

    comptime FSHIN_LIMIT_LOW: Float64 = -1.2
    comptime FSHIN_LIMIT_HIGH: Float64 = 0.87

    comptime FFOOT_LIMIT_LOW: Float64 = -0.5
    comptime FFOOT_LIMIT_HIGH: Float64 = 0.5

    # ==========================================================================
    # Motor Parameters
    # ==========================================================================
    # NOTE: MAX_TORQUE must be scaled for the small limb inertias.
    # With I = 0.00175 kg·m² (bthigh), torque of 1 Nm gives Δω ≈ 5.7 rad/s per substep.
    # MuJoCo's gear=120 scales the action, not the actual torque magnitude.

    comptime MAX_TORQUE: Float64 = 1.0  # Nm - scaled for physics2d limb inertias
    comptime GEAR_RATIO: Float64 = 120.0  # Used for MuJoCo compatibility (not torque)

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime CTRL_COST_WEIGHT: Float64 = 0.1
    comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0

    # Height bonus to discourage crawling behavior
    # Rewards keeping torso elevated above ground
    comptime HEIGHT_BONUS_WEIGHT: Float64 = 1.0  # Reward per unit height (increased)
    comptime TARGET_HEIGHT: Float64 = 0.5  # Height at which bonus saturates
    comptime MIN_HEIGHT_FOR_BONUS: Float64 = 0.1  # Below this, no bonus

    # Ground contact penalty - penalize non-foot body parts touching ground
    # Bodies that SHOULD NOT touch ground: torso(0), bthigh(1), bshin(2), fthigh(4), fshin(5)
    # Bodies that CAN touch ground: bfoot(3), ffoot(6)
    comptime GROUND_CONTACT_PENALTY: Float64 = 0.5  # Penalty per illegal ground contact

    # ==========================================================================
    # Episode Parameters
    # ==========================================================================

    comptime MAX_STEPS: Int = 1000

    # ==========================================================================
    # Termination Parameters (healthy bounds)
    # ==========================================================================

    # Terminate if torso height drops below this (fallen over)
    # NOTE: MuJoCo HalfCheetah doesn't terminate on falling by default.
    # Set to False for standard training (recommended), True for faster iteration.
    comptime TERMINATE_WHEN_UNHEALTHY: Bool = False
    # Lenient thresholds - HalfCheetah is meant to run horizontally, not stay upright
    # Torso radius is 0.046, so center at 0.0 means touching ground
    comptime HEALTHY_Z_MIN: Float64 = -0.5  # Allow torso below ground (sliding/recovering)
    comptime HEALTHY_Z_MAX: Float64 = 2.0  # Reasonable ceiling
    comptime HEALTHY_ANGLE_MAX: Float64 = 2.0  # ~115 degrees - very generous for cheetah

    # ==========================================================================
    # Layout Constants (from HalfCheetahLayout)
    # ==========================================================================

    # Counts
    comptime NUM_BODIES: Int = Self.HC.NUM_BODIES  # 7
    comptime NUM_SHAPES: Int = Self.HC.NUM_SHAPES  # 7
    comptime MAX_CONTACTS: Int = Self.HC.MAX_CONTACTS  # 14
    comptime MAX_JOINTS: Int = Self.HC.MAX_JOINTS  # 6
    comptime MAX_TERRAIN_EDGES: Int = Self.HC.MAX_TERRAIN_EDGES  # 2

    # Dimensions
    comptime OBS_DIM_VAL: Int = Self.HC.OBS_DIM  # 17
    comptime ACTION_DIM_VAL: Int = 6  # 6 joint torques

    # Buffer sizes
    comptime BODIES_SIZE: Int = Self.HC.BODIES_SIZE
    comptime FORCES_SIZE: Int = Self.HC.FORCES_SIZE
    comptime JOINTS_SIZE: Int = Self.HC.JOINTS_SIZE
    comptime EDGES_SIZE: Int = Self.HC.EDGES_SIZE
    comptime METADATA_SIZE: Int = Self.HC.METADATA_SIZE

    # Offsets within each environment's state
    comptime OBS_OFFSET: Int = Self.HC.OBS_OFFSET
    comptime BODIES_OFFSET: Int = Self.HC.BODIES_OFFSET
    comptime FORCES_OFFSET: Int = Self.HC.FORCES_OFFSET
    comptime JOINTS_OFFSET: Int = Self.HC.JOINTS_OFFSET
    comptime JOINT_COUNT_OFFSET: Int = Self.HC.JOINT_COUNT_OFFSET
    comptime EDGES_OFFSET: Int = Self.HC.EDGES_OFFSET
    comptime EDGE_COUNT_OFFSET: Int = Self.HC.EDGE_COUNT_OFFSET
    comptime METADATA_OFFSET: Int = Self.HC.METADATA_OFFSET

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

    comptime STATE_SIZE_VAL: Int = Self.HC.STATE_SIZE

    # ==========================================================================
    # Observation Layout (17D total)
    # ==========================================================================
    # [0]: z position of torso (height)
    # [1]: angle of torso
    # [2]: angle of back thigh
    # [3]: angle of back shin
    # [4]: angle of back foot
    # [5]: angle of front thigh
    # [6]: angle of front shin
    # [7]: angle of front foot
    # [8]: velocity of x
    # [9]: velocity of z (vertical)
    # [10]: angular velocity of torso
    # [11]: angular velocity of back thigh
    # [12]: angular velocity of back shin
    # [13]: angular velocity of back foot
    # [14]: angular velocity of front thigh
    # [15]: angular velocity of front shin
    # [16]: angular velocity of front foot
