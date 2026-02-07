"""Physics3D v2 constants - minimal rebuild.

This module defines compile-time constants for the physics engine.
Following MuJoCo's approach with parameterized dtype for GPU compatibility.
"""


# GPU kernel configuration (same as deep_rl)
comptime TILE: Int = 16  # Optimal for Apple Silicon
comptime TPB: Int = 256  # Threads per block


struct PhysicsConstants[DTYPE: DType]:
    # Physics defaults
    comptime DEFAULT_GRAVITY_Z: Scalar[Self.DTYPE] = -9.81
    comptime DEFAULT_TIMESTEP: Scalar[Self.DTYPE] = 0.01


# Geometry types (Phase 2)
comptime GEOM_PLANE: Int = 0
comptime GEOM_SPHERE: Int = 1
comptime GEOM_CAPSULE: Int = 2
comptime GEOM_BOX: Int = 3

# ==============================================================================
# Phase 3: Multi-body layout constants
# ==============================================================================

# Body state layout (26 floats per body, matches physics3d)
comptime BODY_STATE_SIZE: Int = 26

# Field indices within body state
comptime IDX_PX: Int = 0  # Position x
comptime IDX_PY: Int = 1  # Position y
comptime IDX_PZ: Int = 2  # Position z
comptime IDX_QW: Int = 3  # Quaternion w (scalar)
comptime IDX_QX: Int = 4  # Quaternion x
comptime IDX_QY: Int = 5  # Quaternion y
comptime IDX_QZ: Int = 6  # Quaternion z
comptime IDX_VX: Int = 7  # Velocity x
comptime IDX_VY: Int = 8  # Velocity y
comptime IDX_VZ: Int = 9  # Velocity z
comptime IDX_WX: Int = 10  # Angular velocity x
comptime IDX_WY: Int = 11  # Angular velocity y
comptime IDX_WZ: Int = 12  # Angular velocity z
comptime IDX_MASS: Int = 13
comptime IDX_INV_MASS: Int = 14
comptime IDX_IXX: Int = 15  # Inertia diagonal
comptime IDX_IYY: Int = 16
comptime IDX_IZZ: Int = 17
comptime IDX_RADIUS: Int = 18  # Sphere radius
comptime IDX_GEOM_TYPE: Int = 19

# Contact layout (12 floats per contact, MuJoCo-style)
comptime CONTACT_SIZE: Int = 12
comptime CIDX_BODY_A: Int = 0
comptime CIDX_BODY_B: Int = 1  # -1 for ground
comptime CIDX_POS_X: Int = 2
comptime CIDX_POS_Y: Int = 3
comptime CIDX_POS_Z: Int = 4
comptime CIDX_NORMAL_X: Int = 5
comptime CIDX_NORMAL_Y: Int = 6
comptime CIDX_NORMAL_Z: Int = 7
comptime CIDX_DIST: Int = 8  # Signed distance (negative = penetration)
comptime CIDX_IMPULSE_N: Int = 9  # Normal impulse (warm start)
comptime CIDX_IMPULSE_T1: Int = 10  # Tangent impulse 1
comptime CIDX_IMPULSE_T2: Int = 11  # Tangent impulse 2
