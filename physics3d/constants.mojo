"""Physics3D engine constants and type definitions.

This module defines the core constants for the 3D physics engine,
extending the patterns from physics2d/constants.mojo to 3D.
"""

from math import pi

# =============================================================================
# Type Constants
# =============================================================================

# Default data type for physics computations
comptime dtype = DType.float32

# GPU kernel configuration
comptime TILE: Int = 16  # Tile size for tiled operations
comptime TPB: Int = 256  # Threads per block for elementwise ops

# =============================================================================
# 3D Body State Layout
# =============================================================================
# Bodies are stored as flat arrays with this layout per body (26 floats):
# [position(3), orientation(4), linear_vel(3), angular_vel(3),
#  force(3), torque(3), mass(2), inertia_diag(3), shape_idx(1), body_type(1)]
#
# Total: 3 + 4 + 3 + 3 + 3 + 3 + 2 + 3 + 1 + 1 = 26 floats

comptime BODY_STATE_SIZE_3D: Int = 26

# Position (world coordinates)
comptime IDX_PX: Int = 0
comptime IDX_PY: Int = 1
comptime IDX_PZ: Int = 2

# Orientation quaternion (w, x, y, z)
comptime IDX_QW: Int = 3
comptime IDX_QX: Int = 4
comptime IDX_QY: Int = 5
comptime IDX_QZ: Int = 6

# Linear velocity
comptime IDX_VX: Int = 7
comptime IDX_VY: Int = 8
comptime IDX_VZ: Int = 9

# Angular velocity (in world frame)
comptime IDX_WX: Int = 10
comptime IDX_WY: Int = 11
comptime IDX_WZ: Int = 12

# Force accumulator (cleared after each step)
comptime IDX_FX: Int = 13
comptime IDX_FY: Int = 14
comptime IDX_FZ: Int = 15

# Torque accumulator (cleared after each step)
comptime IDX_TX: Int = 16
comptime IDX_TY: Int = 17
comptime IDX_TZ: Int = 18

# Mass properties
comptime IDX_MASS: Int = 19
comptime IDX_INV_MASS: Int = 20

# Diagonal inertia tensor (local frame)
comptime IDX_IXX: Int = 21
comptime IDX_IYY: Int = 22
comptime IDX_IZZ: Int = 23

# Shape reference
comptime IDX_SHAPE_3D: Int = 24

# Body type (0=dynamic, 1=kinematic, 2=static)
comptime IDX_BODY_TYPE: Int = 25

# Body type values
comptime BODY_DYNAMIC: Int = 0
comptime BODY_KINEMATIC: Int = 1
comptime BODY_STATIC: Int = 2

# =============================================================================
# 3D Shape Layout
# =============================================================================
# Shapes are stored with a type discriminator followed by type-specific data

comptime SHAPE_MAX_SIZE_3D: Int = 16  # Max floats per shape

# Shape types
comptime SHAPE_BOX: Int = 0  # Box with half-extents
comptime SHAPE_SPHERE: Int = 1  # Sphere with radius
comptime SHAPE_CAPSULE: Int = 2  # Capsule with radius and half-height
comptime SHAPE_PLANE: Int = 3  # Infinite plane (for ground)

# Box layout: [type, half_x, half_y, half_z]
# Sphere layout: [type, radius]
# Capsule layout: [type, radius, half_height, axis(0=X, 1=Y, 2=Z)]
# Plane layout: [type, nx, ny, nz, d] (plane equation: n·p + d = 0)

# =============================================================================
# 3D Contact Layout
# =============================================================================
# Contacts store collision information for constraint solving

comptime CONTACT_DATA_SIZE_3D: Int = 16

comptime CONTACT_BODY_A_3D: Int = 0
comptime CONTACT_BODY_B_3D: Int = 1  # -1 for static/ground
comptime CONTACT_POINT_X: Int = 2  # World-space contact point
comptime CONTACT_POINT_Y: Int = 3
comptime CONTACT_POINT_Z: Int = 4
comptime CONTACT_NORMAL_X: Int = 5  # Contact normal (from A to B)
comptime CONTACT_NORMAL_Y: Int = 6
comptime CONTACT_NORMAL_Z: Int = 7
comptime CONTACT_DEPTH_3D: Int = 8  # Penetration depth
comptime CONTACT_IMPULSE_N: Int = 9  # Normal impulse (for warm starting)
comptime CONTACT_IMPULSE_T1: Int = 10  # Tangent impulse 1
comptime CONTACT_IMPULSE_T2: Int = 11  # Tangent impulse 2
# Tangent basis
comptime CONTACT_TANGENT1_X: Int = 12
comptime CONTACT_TANGENT1_Y: Int = 13
comptime CONTACT_TANGENT1_Z: Int = 14
comptime CONTACT_FLAGS: Int = 15  # Contact flags

# =============================================================================
# 3D Joint Layout
# =============================================================================
# Joints connect two bodies and constrain their relative motion
# Layout is joint-type specific, but all start with common header

comptime JOINT_DATA_SIZE_3D: Int = 32

# Common joint header
comptime JOINT3D_TYPE: Int = 0
comptime JOINT3D_BODY_A: Int = 1
comptime JOINT3D_BODY_B: Int = 2

# Local anchors
comptime JOINT3D_ANCHOR_AX: Int = 3
comptime JOINT3D_ANCHOR_AY: Int = 4
comptime JOINT3D_ANCHOR_AZ: Int = 5
comptime JOINT3D_ANCHOR_BX: Int = 6
comptime JOINT3D_ANCHOR_BY: Int = 7
comptime JOINT3D_ANCHOR_BZ: Int = 8

# Joint axis (for hinge joints)
comptime JOINT3D_AXIS_X: Int = 9
comptime JOINT3D_AXIS_Y: Int = 10
comptime JOINT3D_AXIS_Z: Int = 11

# Joint state
comptime JOINT3D_POSITION: Int = 12  # Joint position (angle or distance)
comptime JOINT3D_VELOCITY: Int = 13  # Joint velocity

# Motor parameters
comptime JOINT3D_MOTOR_TARGET: Int = 14
comptime JOINT3D_MOTOR_KP: Int = 15  # P gain for position control
comptime JOINT3D_MOTOR_KD: Int = 16  # D gain for velocity damping
comptime JOINT3D_MAX_FORCE: Int = 17

# Limits
comptime JOINT3D_LOWER_LIMIT: Int = 18
comptime JOINT3D_UPPER_LIMIT: Int = 19
comptime JOINT3D_FLAGS: Int = 20

# Accumulated impulses for warm starting
comptime JOINT3D_IMPULSE_X: Int = 21
comptime JOINT3D_IMPULSE_Y: Int = 22
comptime JOINT3D_IMPULSE_Z: Int = 23
comptime JOINT3D_MOTOR_IMPULSE: Int = 24

# Reserved for additional data
# 25-31 available

# Joint types
comptime JOINT_HINGE: Int = 0  # 1-DOF revolute joint
comptime JOINT_BALL: Int = 1  # 3-DOF spherical joint
comptime JOINT_FREE: Int = 2  # 6-DOF free joint (floating base)
comptime JOINT_FIXED: Int = 3  # 0-DOF fixed joint

# Joint flags
comptime JOINT3D_FLAG_LIMIT_ENABLED: Int = 1
comptime JOINT3D_FLAG_MOTOR_ENABLED: Int = 2

# Maximum joints per environment
comptime MAX_JOINTS_PER_ENV_3D: Int = 20

# =============================================================================
# Default Physics Constants
# =============================================================================

comptime DEFAULT_GRAVITY_X_3D: Float64 = 0.0
comptime DEFAULT_GRAVITY_Y_3D: Float64 = 0.0
comptime DEFAULT_GRAVITY_Z_3D: Float64 = -9.81
comptime DEFAULT_DT_3D: Float64 = 0.02  # 50 FPS

# Solver defaults
comptime DEFAULT_VELOCITY_ITERATIONS_3D: Int = 10
comptime DEFAULT_POSITION_ITERATIONS_3D: Int = 5

# Contact physics defaults
comptime DEFAULT_FRICTION_3D: Float64 = 0.5
comptime DEFAULT_RESTITUTION_3D: Float64 = 0.0
comptime DEFAULT_BAUMGARTE_3D: Float64 = 0.2
comptime DEFAULT_SLOP_3D: Float64 = 0.001

# =============================================================================
# Mathematical Constants
# =============================================================================

comptime PI: Float64 = pi
comptime TWO_PI: Float64 = 2.0 * pi
comptime HALF_PI: Float64 = pi / 2.0
comptime DEG_TO_RAD: Float64 = pi / 180.0
comptime RAD_TO_DEG: Float64 = 180.0 / pi
