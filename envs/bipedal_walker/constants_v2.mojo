"""Constants for BipedalWalker v2 GPU environment.

Uses BipedalWalkerLayout from physics2d for compile-time layout computation.
"""

from physics2d import BipedalWalkerLayout, PhysicsState


struct BWConstants:
    """Constants for BipedalWalker v2 environment.

    Based on Gymnasium BipedalWalker-v3 with physics2d layout.
    """

    # Layout type alias
    comptime BW = BipedalWalkerLayout

    # ==========================================================================
    # Physics Parameters
    # ==========================================================================

    comptime FPS: Float64 = 50.0
    comptime DT: Float64 = 1.0 / Self.FPS
    comptime SCALE: Float64 = 30.0
    comptime GRAVITY_X: Float64 = 0.0
    comptime GRAVITY_Y: Float64 = -10.0

    # Physics solver iterations
    comptime VELOCITY_ITERATIONS: Int = 6
    comptime POSITION_ITERATIONS: Int = 2

    # Contact physics
    comptime FRICTION: Float64 = 2.5
    comptime RESTITUTION: Float64 = 0.0
    comptime BAUMGARTE: Float64 = 0.2
    comptime SLOP: Float64 = 0.005

    # ==========================================================================
    # Motor Parameters
    # ==========================================================================

    comptime MOTORS_TORQUE: Float64 = 80.0
    comptime SPEED_HIP: Float64 = 4.0
    comptime SPEED_KNEE: Float64 = 6.0

    # ==========================================================================
    # Lidar Parameters
    # ==========================================================================

    comptime LIDAR_RANGE: Float64 = 160.0 / Self.SCALE  # ~5.33 meters
    comptime NUM_LIDAR: Int = 10

    # ==========================================================================
    # Body Geometry
    # ==========================================================================

    # Hull dimensions (pentagon shape)
    comptime HULL_POLY: InlineArray[Float64, 10] = InlineArray[Float64, 10](
        -30.0 / Self.SCALE,
        0.0 / Self.SCALE,  # bottom left
        -6.0 / Self.SCALE,
        30.0 / Self.SCALE,  # top left
        6.0 / Self.SCALE,
        30.0 / Self.SCALE,  # top right
        30.0 / Self.SCALE,
        0.0 / Self.SCALE,  # bottom right
        0.0 / Self.SCALE,
        -12.0 / Self.SCALE,  # bottom center
    )

    # Leg dimensions
    comptime LEG_DOWN: Float64 = -8.0 / Self.SCALE
    comptime LEG_W: Float64 = 8.0 / Self.SCALE
    comptime LEG_H: Float64 = 34.0 / Self.SCALE

    # Upper leg dimensions
    comptime UPPER_LEG_W: Float64 = 6.4 / Self.SCALE  # LEG_W * 0.8
    comptime UPPER_LEG_H: Float64 = 34.0 / Self.SCALE

    # Lower leg dimensions
    comptime LOWER_LEG_W: Float64 = 5.12 / Self.SCALE  # LEG_W * 0.64
    comptime LOWER_LEG_H: Float64 = 34.0 / Self.SCALE

    # Hull mass/inertia
    comptime HULL_MASS: Float64 = 5.0
    comptime HULL_INERTIA: Float64 = 2.0

    # Leg mass/inertia
    comptime LEG_MASS: Float64 = 0.8  # Upper leg
    comptime LEG_INERTIA: Float64 = 0.1
    comptime LOWER_LEG_MASS: Float64 = 0.5  # Lower leg
    comptime LOWER_LEG_INERTIA: Float64 = 0.05

    # ==========================================================================
    # Terrain Parameters
    # ==========================================================================

    comptime TERRAIN_STEP: Float64 = 14.0 / Self.SCALE
    comptime TERRAIN_LENGTH: Int = 200
    comptime TERRAIN_HEIGHT: Float64 = 400.0 / Self.SCALE / 4.0  # ~3.33
    comptime TERRAIN_GRASS: Int = 10
    comptime TERRAIN_STARTPAD: Int = 20

    # Viewport
    comptime VIEWPORT_W: Float64 = 600.0
    comptime VIEWPORT_H: Float64 = 400.0
    comptime W_UNITS: Float64 = Self.VIEWPORT_W / Self.SCALE
    comptime H_UNITS: Float64 = Self.VIEWPORT_H / Self.SCALE

    # ==========================================================================
    # Body Indices
    # ==========================================================================

    comptime BODY_HULL: Int = 0
    comptime BODY_UPPER_LEG_L: Int = 1
    comptime BODY_LOWER_LEG_L: Int = 2
    comptime BODY_UPPER_LEG_R: Int = 3
    comptime BODY_LOWER_LEG_R: Int = 4

    # ==========================================================================
    # Joint Indices
    # ==========================================================================

    comptime JOINT_HIP_L: Int = 0
    comptime JOINT_KNEE_L: Int = 1
    comptime JOINT_HIP_R: Int = 2
    comptime JOINT_KNEE_R: Int = 3

    # ==========================================================================
    # Joint Limits (in radians)
    # ==========================================================================

    comptime HIP_LIMIT_LOW: Float64 = -0.8
    comptime HIP_LIMIT_HIGH: Float64 = 1.1
    comptime KNEE_LIMIT_LOW: Float64 = -1.6
    comptime KNEE_LIMIT_HIGH: Float64 = -0.1

    # ==========================================================================
    # Layout Constants (from BipedalWalkerLayout)
    # ==========================================================================

    # Counts
    comptime NUM_BODIES: Int = Self.BW.NUM_BODIES  # 5
    comptime NUM_SHAPES: Int = Self.BW.NUM_SHAPES  # 5
    comptime MAX_CONTACTS: Int = Self.BW.MAX_CONTACTS  # 20
    comptime MAX_JOINTS: Int = Self.BW.MAX_JOINTS  # 4
    comptime MAX_TERRAIN_EDGES: Int = Self.BW.MAX_TERRAIN_EDGES  # 128

    # Dimensions
    comptime OBS_DIM_VAL: Int = Self.BW.OBS_DIM  # 24
    comptime ACTION_DIM_VAL: Int = 4  # 4 joint torques

    # Buffer sizes
    comptime BODIES_SIZE: Int = Self.BW.BODIES_SIZE
    comptime FORCES_SIZE: Int = Self.BW.FORCES_SIZE
    comptime JOINTS_SIZE: Int = Self.BW.JOINTS_SIZE
    comptime EDGES_SIZE: Int = Self.BW.EDGES_SIZE
    comptime METADATA_SIZE: Int = Self.BW.METADATA_SIZE

    # Offsets within each environment's state
    comptime OBS_OFFSET: Int = Self.BW.OBS_OFFSET
    comptime BODIES_OFFSET: Int = Self.BW.BODIES_OFFSET
    comptime FORCES_OFFSET: Int = Self.BW.FORCES_OFFSET
    comptime JOINTS_OFFSET: Int = Self.BW.JOINTS_OFFSET
    comptime JOINT_COUNT_OFFSET: Int = Self.BW.JOINT_COUNT_OFFSET
    comptime EDGES_OFFSET: Int = Self.BW.EDGES_OFFSET
    comptime EDGE_COUNT_OFFSET: Int = Self.BW.EDGE_COUNT_OFFSET
    comptime METADATA_OFFSET: Int = Self.BW.METADATA_OFFSET

    # ==========================================================================
    # Metadata Field Indices
    # ==========================================================================

    comptime META_STEP_COUNT: Int = 0
    comptime META_TOTAL_REWARD: Int = 1
    comptime META_PREV_SHAPING: Int = 2
    comptime META_DONE: Int = 3
    comptime META_SCROLL: Int = 4
    comptime META_LEFT_CONTACT: Int = 5
    comptime META_RIGHT_CONTACT: Int = 6
    comptime META_GAME_OVER: Int = 7

    # ==========================================================================
    # Total State Size
    # ==========================================================================

    comptime STATE_SIZE_VAL: Int = Self.BW.STATE_SIZE

    # ==========================================================================
    # Observation Layout (24D total)
    # ==========================================================================
    # [0]: hull_angle
    # [1]: hull_angular_velocity
    # [2]: vel_x (horizontal speed)
    # [3]: vel_y (vertical speed)
    # [4]: hip1_angle
    # [5]: hip1_speed
    # [6]: knee1_angle
    # [7]: knee1_speed
    # [8]: leg1_contact (0.0 or 1.0)
    # [9]: hip2_angle
    # [10]: hip2_speed
    # [11]: knee2_angle
    # [12]: knee2_speed
    # [13]: leg2_contact (0.0 or 1.0)
    # [14-23]: lidar readings (10 values)

    comptime LIDAR_START_IDX: Int = 14

    # ==========================================================================
    # Reward Shaping
    # ==========================================================================

    comptime CRASH_PENALTY: Float64 = -100.0
    comptime SUCCESS_BONUS: Float64 = 0.0  # BipedalWalker has no landing bonus
