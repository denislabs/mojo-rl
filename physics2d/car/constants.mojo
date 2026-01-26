"""CarRacing physics constants.

These constants are extracted from Gymnasium's car_dynamics.py and define
the physical properties of the car, wheels, and friction model.

Reference: gymnasium/envs/box2d/car_dynamics.py
"""

from math import pi

# =============================================================================
# Scale Factor
# =============================================================================

# Base scale factor for all dimensions (from Gymnasium)
comptime SIZE: Float64 = 0.02

# =============================================================================
# Engine and Brake
# =============================================================================

# Engine power (torque at wheel)
# Original: 100000000 * SIZE * SIZE = 40000
comptime ENGINE_POWER: Float64 = 100000000.0 * SIZE * SIZE

# Brake force (angular deceleration rate in rad/s)
comptime BRAKE_FORCE: Float64 = 15.0

# =============================================================================
# Wheel Properties
# =============================================================================

# Wheel moment of inertia
# Original: 4000 * SIZE * SIZE = 1.6
comptime WHEEL_MOMENT_OF_INERTIA: Float64 = 4000.0 * SIZE * SIZE

# Wheel visual radius (used in physics for slip calculation)
# Original: 27 * SIZE = 0.54
comptime WHEEL_RADIUS: Float64 = 27.0 * SIZE

# Wheel visual width
comptime WHEEL_WIDTH: Float64 = 14.0 * SIZE

# =============================================================================
# Friction Model
# =============================================================================

# Maximum friction force limit
# Original: 1000000 * SIZE * SIZE = 400
comptime FRICTION_LIMIT: Float64 = 1000000.0 * SIZE * SIZE

# Friction coefficient (damping for slip velocity)
# Original: 205000 * SIZE * SIZE = 82
comptime FRICTION_COEF: Float64 = 205000.0 * SIZE * SIZE

# Surface friction multipliers
comptime ROAD_FRICTION: Float64 = 1.0
comptime GRASS_FRICTION: Float64 = 0.6

# =============================================================================
# Steering
# =============================================================================

# Maximum steering angle (radians)
comptime STEERING_LIMIT: Float64 = 0.4

# Maximum steering motor speed (rad/s)
comptime STEERING_MOTOR_SPEED: Float64 = 3.0

# Steering proportional gain (clamped to motor speed)
comptime STEERING_GAIN: Float64 = 50.0

# =============================================================================
# Wheel Positions (Local Coordinates)
# =============================================================================
# Relative to hull center, in SIZE units
# Front-left, Front-right, Rear-left, Rear-right

# Front-left wheel
comptime WHEEL_POS_FL_X: Float64 = -55.0 * SIZE  # -1.1
comptime WHEEL_POS_FL_Y: Float64 = 80.0 * SIZE   # 1.6

# Front-right wheel
comptime WHEEL_POS_FR_X: Float64 = 55.0 * SIZE   # 1.1
comptime WHEEL_POS_FR_Y: Float64 = 80.0 * SIZE   # 1.6

# Rear-left wheel
comptime WHEEL_POS_RL_X: Float64 = -55.0 * SIZE  # -1.1
comptime WHEEL_POS_RL_Y: Float64 = -82.0 * SIZE  # -1.64

# Rear-right wheel
comptime WHEEL_POS_RR_X: Float64 = 55.0 * SIZE   # 1.1
comptime WHEEL_POS_RR_Y: Float64 = -82.0 * SIZE  # -1.64

# =============================================================================
# Hull Properties
# =============================================================================

# Hull mass (estimated from Box2D density + area)
comptime HULL_MASS: Float64 = 10.0

# Hull moment of inertia (estimated)
comptime HULL_INERTIA: Float64 = 10.0

# Hull inverse mass (precomputed)
comptime HULL_INV_MASS: Float64 = 1.0 / HULL_MASS

# Hull inverse inertia (precomputed)
comptime HULL_INV_INERTIA: Float64 = 1.0 / HULL_INERTIA

# =============================================================================
# Time Step
# =============================================================================

# Physics timestep (matches Gymnasium's 50 FPS)
comptime CAR_DT: Float64 = 0.02

# =============================================================================
# Track Tile Constants
# =============================================================================

# Data size per tile (9 floats)
# [v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y, friction]
comptime TILE_DATA_SIZE: Int = 9

# Maximum number of track tiles (can handle most procedural tracks)
comptime MAX_TRACK_TILES: Int = 300

# Tile vertex indices
comptime TILE_V0_X: Int = 0
comptime TILE_V0_Y: Int = 1
comptime TILE_V1_X: Int = 2
comptime TILE_V1_Y: Int = 3
comptime TILE_V2_X: Int = 4
comptime TILE_V2_Y: Int = 5
comptime TILE_V3_X: Int = 6
comptime TILE_V3_Y: Int = 7
comptime TILE_FRICTION: Int = 8

# =============================================================================
# State Indices
# =============================================================================

# Hull state indices (within hull block)
comptime HULL_X: Int = 0
comptime HULL_Y: Int = 1
comptime HULL_ANGLE: Int = 2
comptime HULL_VX: Int = 3
comptime HULL_VY: Int = 4
comptime HULL_OMEGA: Int = 5
comptime HULL_STATE_SIZE: Int = 6

# Wheel state indices (per wheel)
comptime WHEEL_OMEGA: Int = 0      # Angular velocity of wheel
comptime WHEEL_JOINT_ANGLE: Int = 1  # Steering joint angle (front wheels only)
comptime WHEEL_PHASE: Int = 2       # Cumulative rotation (for rendering)
comptime WHEEL_STATE_SIZE: Int = 3

# Control indices
comptime CTRL_STEERING: Int = 0
comptime CTRL_GAS: Int = 1
comptime CTRL_BRAKE: Int = 2
comptime CONTROL_SIZE: Int = 3

# =============================================================================
# Wheel Indices
# =============================================================================

comptime WHEEL_FL: Int = 0  # Front-left
comptime WHEEL_FR: Int = 1  # Front-right
comptime WHEEL_RL: Int = 2  # Rear-left
comptime WHEEL_RR: Int = 3  # Rear-right
comptime NUM_WHEELS: Int = 4

# =============================================================================
# Mathematical Constants
# =============================================================================

comptime CAR_PI: Float64 = pi
comptime CAR_TWO_PI: Float64 = 2.0 * pi
