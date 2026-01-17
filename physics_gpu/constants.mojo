"""Physics engine constants and type definitions.

This module defines the core constants used throughout the GPU physics engine,
following the pattern established in deep_rl/constants.mojo.
"""

# =============================================================================
# Type Constants
# =============================================================================

# Default data type for physics computations
comptime dtype = DType.float32

# GPU kernel configuration
comptime TILE: Int = 16  # Tile size for tiled operations
comptime TPB: Int = 256  # Threads per block for elementwise ops

# =============================================================================
# Body State Layout
# =============================================================================
# Bodies are stored as flat arrays with this layout per body:
# [x, y, angle, vx, vy, omega, fx, fy, tau, mass, inv_mass, inv_inertia, shape_idx]

comptime BODY_STATE_SIZE: Int = 13

# Position and orientation
comptime IDX_X: Int = 0
comptime IDX_Y: Int = 1
comptime IDX_ANGLE: Int = 2

# Linear and angular velocity
comptime IDX_VX: Int = 3
comptime IDX_VY: Int = 4
comptime IDX_OMEGA: Int = 5

# Accumulated forces and torque (cleared after each step)
comptime IDX_FX: Int = 6
comptime IDX_FY: Int = 7
comptime IDX_TAU: Int = 8

# Mass properties
comptime IDX_MASS: Int = 9
comptime IDX_INV_MASS: Int = 10
comptime IDX_INV_INERTIA: Int = 11

# Shape reference
comptime IDX_SHAPE: Int = 12

# =============================================================================
# Shape Layout
# =============================================================================
# Shapes are stored with a type discriminator followed by type-specific data

comptime SHAPE_MAX_SIZE: Int = 20  # Max floats per shape (polygon with 8 vertices)

# Shape types
comptime SHAPE_POLYGON: Int = 0
comptime SHAPE_CIRCLE: Int = 1
comptime SHAPE_EDGE: Int = 2

# Polygon layout: [type, n_verts, v0x, v0y, v1x, v1y, ..., v7x, v7y]
comptime MAX_POLYGON_VERTS: Int = 8

# Circle layout: [type, radius, center_x, center_y]
# Edge layout: [type, v0x, v0y, v1x, v1y, normal_x, normal_y]

# =============================================================================
# Contact Layout
# =============================================================================
# Contacts store collision information for constraint solving

comptime CONTACT_DATA_SIZE: Int = 9

comptime CONTACT_BODY_A: Int = 0
comptime CONTACT_BODY_B: Int = 1  # -1 for static/ground
comptime CONTACT_POINT_X: Int = 2
comptime CONTACT_POINT_Y: Int = 3
comptime CONTACT_NORMAL_X: Int = 4
comptime CONTACT_NORMAL_Y: Int = 5
comptime CONTACT_DEPTH: Int = 6
comptime CONTACT_NORMAL_IMPULSE: Int = 7  # For warm starting
comptime CONTACT_TANGENT_IMPULSE: Int = 8

# =============================================================================
# Default Physics Constants
# =============================================================================

comptime DEFAULT_GRAVITY_X: Float64 = 0.0
comptime DEFAULT_GRAVITY_Y: Float64 = -10.0
comptime DEFAULT_DT: Float64 = 0.02  # 50 FPS

# Solver defaults
comptime DEFAULT_VELOCITY_ITERATIONS: Int = 6
comptime DEFAULT_POSITION_ITERATIONS: Int = 2

# Contact physics defaults
comptime DEFAULT_FRICTION: Float64 = 0.3
comptime DEFAULT_RESTITUTION: Float64 = 0.0
comptime DEFAULT_BAUMGARTE: Float64 = 0.2  # Position correction factor
comptime DEFAULT_SLOP: Float64 = 0.005  # Penetration allowance

# =============================================================================
# Mathematical Constants
# =============================================================================

from math import pi

comptime PI: Float64 = pi
comptime TWO_PI: Float64 = 2.0 * pi
