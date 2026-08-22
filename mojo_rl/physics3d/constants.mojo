"""Physics3D constants.

This module defines compile-time constants for the physics engine.
"""


# GPU kernel configuration
comptime TILE: Int = 16  # Optimal for Apple Silicon
comptime TPB: Int = 256  # Threads per block


struct PhysicsConstants[DTYPE: DType]:
    # Physics defaults
    comptime DEFAULT_GRAVITY_Z: Scalar[Self.DTYPE] = -9.81
    comptime DEFAULT_TIMESTEP: Scalar[Self.DTYPE] = 0.01


# Geometry types
comptime GEOM_PLANE: Int = 0
comptime GEOM_SPHERE: Int = 1
comptime GEOM_CAPSULE: Int = 2
comptime GEOM_BOX: Int = 3
comptime GEOM_CYLINDER: Int = 4
comptime GEOM_MESH: Int = 5
# `ellipsoid` used to fall through to GEOM_SPHERE SILENTLY (no `ellipsoid`
# case in `_geom_type_from_str`, whose default is sphere). Harmless while every
# ellipsoid in the repo carried `mass="0"` with contacts disabled — swimmer's
# head, finger's touch SITES — and load-bearing the moment fish arrived, whose
# tail and fins ARE ellipsoids with density-derived mass: a sphere of radius
# size[0] gave tail1 1/128th of its mass and each fin 26x too much.
#
# INERTIA ONLY. There is no ellipsoid narrow phase; `init_fields` raises if an
# ellipsoid geom can actually collide, rather than silently colliding it as a
# sphere. See `geom_volume` / `geom_inertia`.
comptime GEOM_ELLIPSOID: Int = 6

# `<geom type="hfield">` — a HEIGHTFIELD, and until it existed this fell
# through `_geom_type_from_str`'s `return _GEOM_SPHERE  # default` and collided
# as a BALL of radius `size[0]`. Measured on
# `google_barkour_vb/scene_hfield_mjx`: MuJoCo emitted 8 contacts and we
# emitted 4, on 6 different body pairs, 2.219e-01 apart in depth and 81.1 deg
# apart in normal.
#
# ⚠ THE NUMBER IS OURS, NOT MuJoCo'S. `mjtGeom` puts HFIELD at 1 and PLANE at
# 0; this enum has never matched it (SPHERE is 1 here and 2 there), so every
# comparison against `m.geom_type` goes through the parser's mapping. Appending
# keeps every stored model file readable.
comptime GEOM_HFIELD: Int = 7

# `kAngleTol` in `mjCMesh::MakePolygons` (`user_mesh.cc:2905`): the bucket width,
# in radians, used to decide that two hull triangles are coplanar and belong to
# the same polygon. Faces whose normals differ by less than this merge.
comptime MESH_POLY_ANGLE_TOL: Float64 = 0.01
