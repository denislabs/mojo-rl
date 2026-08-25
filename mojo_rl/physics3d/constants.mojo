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


@always_inline
def mj_geom_type_rank(t: Int) -> Int:
    """`mjtGeom`'s ordinal for one of OUR `GEOM_*` ids — the pair sort key.

    ⚠⚠ `pushPairArena` SORTS BY MuJoCo's TYPE ID, AND OURS IS NOT MuJoCo's.
    `engine_collision_driver.c:489` canonicalises every candidate pair with

        if (m->geom_type[g1] > m->geom_type[g2]) { swap }

    against `mjtGeom` — PLANE 0, HFIELD 1, SPHERE 2, CAPSULE 3, ELLIPSOID 4,
    CYLINDER 5, BOX 6, MESH 7. The enum above is PLANE 0, SPHERE 1, CAPSULE 2,
    BOX 3, CYLINDER 4, MESH 5, ELLIPSOID 6, HFIELD 7, and the note on
    `GEOM_HFIELD` already says it "has never matched it".

    THE TWO ARE NOT A MONOTONE REMAPPING OF EACH OTHER, so comparing our raw
    ids orders **10 of the 28 unordered type pairs the OPPOSITE way** from the
    reference: box/cylinder, ellipsoid/{box, cylinder, mesh} and hfield/{sphere,
    capsule, box, cylinder, ellipsoid, mesh}. Reachable in up to 28 Menagerie
    scenes (box/cylinder alone).

    ⚠ THAT IS NOT COSMETIC. `mjc_ccd`'s multi-contact is NOT symmetric in its
    two objects: it takes the REFERENCE face from obj1 and clips obj2's face
    against it, so running a pair in the other order returns the same
    PENETRATION DEPTH and different WITNESS POSITIONS.

    ⚠ AN IF-CHAIN, NOT A TABLE. A per-thread `InlineArray` indexed by a RUNTIME
    value reads back the wrong value on Metal, with no crash — four instances
    in this engine already (`87960e10`, `836a65ff`). This is called from inside
    the collision kernels with a runtime type.
    """
    if t == GEOM_PLANE:
        return 0
    if t == GEOM_HFIELD:
        return 1
    if t == GEOM_SPHERE:
        return 2
    if t == GEOM_CAPSULE:
        return 3
    if t == GEOM_ELLIPSOID:
        return 4
    if t == GEOM_CYLINDER:
        return 5
    if t == GEOM_BOX:
        return 6
    return 7  # GEOM_MESH

# `kAngleTol` in `mjCMesh::MakePolygons` (`user_mesh.cc:2905`): the bucket width,
# in radians, used to decide that two hull triangles are coplanar and belong to
# the same polygon. Faces whose normals differ by less than this merge.
comptime MESH_POLY_ANGLE_TOL: Float64 = 0.01
