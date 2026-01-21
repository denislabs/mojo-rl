"""CarRacing V2 environment constants.

These constants define the environment configuration, using physics constants
from physics_gpu/car/ for consistency with the GPU physics implementation.
"""

from physics_gpu.car import (
    CarRacingLayout,
    SIZE,
    ENGINE_POWER,
    BRAKE_FORCE,
    WHEEL_MOMENT_OF_INERTIA,
    WHEEL_RADIUS,
    FRICTION_LIMIT,
    FRICTION_COEF,
    ROAD_FRICTION,
    GRASS_FRICTION,
    STEERING_LIMIT,
    STEERING_MOTOR_SPEED,
    CAR_DT,
)


struct CRConstants:
    """CarRacing environment constants."""

    # =========================================================================
    # Physics Constants (from physics_gpu/car/)
    # =========================================================================

    comptime DT: Float64 = CAR_DT  # 0.02 = 50 FPS

    # Scale factor
    comptime SIZE: Float64 = SIZE  # 0.02
    comptime SCALE: Float64 = 6.0

    # Engine and brake
    comptime ENGINE_POWER: Float64 = ENGINE_POWER  # 40000
    comptime BRAKE_FORCE: Float64 = BRAKE_FORCE  # 15.0

    # Wheel properties
    comptime WHEEL_MOMENT: Float64 = WHEEL_MOMENT_OF_INERTIA  # 1.6
    comptime WHEEL_RAD: Float64 = WHEEL_RADIUS  # 0.54
    comptime WHEEL_R_VISUAL: Float64 = 27.0  # Visual radius in SIZE units
    comptime WHEEL_W_VISUAL: Float64 = 14.0  # Visual width in SIZE units

    # Friction
    comptime FRICTION_LIMIT: Float64 = FRICTION_LIMIT  # 400
    comptime FRICTION_COEF: Float64 = FRICTION_COEF  # 82
    comptime ROAD_FRICTION: Float64 = ROAD_FRICTION  # 1.0
    comptime GRASS_FRICTION: Float64 = GRASS_FRICTION  # 0.6

    # Steering
    comptime STEER_LIMIT: Float64 = STEERING_LIMIT  # 0.4 rad
    comptime STEER_SPEED: Float64 = STEERING_MOTOR_SPEED  # 3.0 rad/s

    # =========================================================================
    # Track Constants
    # =========================================================================

    comptime TRACK_RAD: Float64 = 900.0 / Self.SCALE  # Base track radius
    comptime PLAYFIELD: Float64 = 2000.0 / Self.SCALE  # Game over boundary
    comptime TRACK_DETAIL_STEP: Float64 = 21.0 / Self.SCALE
    comptime TRACK_TURN_RATE: Float64 = 0.31
    comptime TRACK_WIDTH: Float64 = 40.0 / Self.SCALE
    comptime BORDER: Float64 = 8.0 / Self.SCALE
    comptime GRASS_DIM: Float64 = Self.PLAYFIELD / 20.0

    comptime NUM_CHECKPOINTS: Int = 12
    comptime MAX_TRACK_TILES: Int = 300

    # =========================================================================
    # Rendering Constants
    # =========================================================================

    comptime FPS: Int = 50
    comptime STATE_W: Int = 96
    comptime STATE_H: Int = 96
    comptime VIDEO_W: Int = 600
    comptime VIDEO_H: Int = 400
    comptime WINDOW_W: Int = 1000
    comptime WINDOW_H: Int = 800
    comptime ZOOM: Float64 = 2.7

    # =========================================================================
    # Episode Constants
    # =========================================================================

    comptime MAX_STEPS: Int = 500
    comptime LAP_COMPLETE_PERCENT: Float64 = 0.95

    # =========================================================================
    # State Layout (from CarRacingLayout)
    # =========================================================================

    comptime Layout = CarRacingLayout[OBS_DIM=13, METADATA_SIZE=5]

    # Observation dimension
    comptime OBS_DIM: Int = Self.Layout.OBS_DIM  # 13

    # Action dimension (continuous)
    comptime ACTION_DIM: Int = 3  # [steering, gas, brake]

    # State size per environment
    comptime STATE_SIZE: Int = Self.Layout.STATE_SIZE  # 39

    # Offsets
    comptime OBS_OFFSET: Int = Self.Layout.OBS_OFFSET
    comptime HULL_OFFSET: Int = Self.Layout.HULL_OFFSET
    comptime WHEELS_OFFSET: Int = Self.Layout.WHEELS_OFFSET
    comptime CONTROLS_OFFSET: Int = Self.Layout.CONTROLS_OFFSET
    comptime METADATA_OFFSET: Int = Self.Layout.METADATA_OFFSET

    # Metadata fields
    comptime META_STEP_COUNT: Int = 0
    comptime META_TOTAL_REWARD: Int = 1
    comptime META_DONE: Int = 2
    comptime META_TRUNCATED: Int = 3
    comptime META_TILES_VISITED: Int = 4

    # =========================================================================
    # Wheel Constants
    # =========================================================================

    comptime NUM_WHEELS: Int = 4
    comptime WHEEL_FL: Int = 0  # Front-left
    comptime WHEEL_FR: Int = 1  # Front-right
    comptime WHEEL_RL: Int = 2  # Rear-left
    comptime WHEEL_RR: Int = 3  # Rear-right

    # =========================================================================
    # Hull Constants
    # =========================================================================

    comptime HULL_MASS: Float64 = 10.0
    comptime HULL_INERTIA: Float64 = 10.0

    # Wheel positions (in SIZE units, from physics_gpu/car/constants.mojo)
    comptime WHEEL_POS_FL_X: Float64 = -55.0 * Self.SIZE
    comptime WHEEL_POS_FL_Y: Float64 = 80.0 * Self.SIZE
    comptime WHEEL_POS_FR_X: Float64 = 55.0 * Self.SIZE
    comptime WHEEL_POS_FR_Y: Float64 = 80.0 * Self.SIZE
    comptime WHEEL_POS_RL_X: Float64 = -55.0 * Self.SIZE
    comptime WHEEL_POS_RL_Y: Float64 = -82.0 * Self.SIZE
    comptime WHEEL_POS_RR_X: Float64 = 55.0 * Self.SIZE
    comptime WHEEL_POS_RR_Y: Float64 = -82.0 * Self.SIZE

    # Hull polygon vertices (for rendering)
    # Hull polygon 1 (front spoiler)
    comptime HULL_POLY1_X0: Float64 = -60.0 * Self.SIZE
    comptime HULL_POLY1_Y0: Float64 = 130.0 * Self.SIZE
    comptime HULL_POLY1_X1: Float64 = 60.0 * Self.SIZE
    comptime HULL_POLY1_Y1: Float64 = 130.0 * Self.SIZE
    comptime HULL_POLY1_X2: Float64 = 60.0 * Self.SIZE
    comptime HULL_POLY1_Y2: Float64 = 110.0 * Self.SIZE
    comptime HULL_POLY1_X3: Float64 = -60.0 * Self.SIZE
    comptime HULL_POLY1_Y3: Float64 = 110.0 * Self.SIZE
