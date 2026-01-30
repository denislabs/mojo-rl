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
