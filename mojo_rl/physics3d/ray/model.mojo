"""`mj_ray` — the whole model against one ray, and `ray_eliminate`'s filter.

`engine_ray.c:1308` and `:68`. This is the piece that turns the per-geom
routines into a query: loop every geom, skip the ones the filter excludes,
dispatch on type, keep the nearest, and report which geom it was.

⚠⚠ THIS MODULE IS THE ONE THAT TOUCHES `Model` AND `Data`, and it is separate
from the rest of `ray/` for exactly that reason. `geom.mojo`, `triangle.mojo`,
`hfield.mojo` and `mesh.mojo` take scalars and know nothing about record
layouts, which is what lets a GPU kernel call them with its own storage. Only
this file binds the CPU-side tables, so a batched tracer replaces THIS and
reuses everything underneath.

WHAT `ray_eliminate` ACTUALLY DECIDES, because three of its four rules are
easy to get subtly wrong:

  1. **body exclusion** — one body id, not a subtree. A rangefinder on a
     quadruped's torso excludes the TORSO and still sees its own legs. That
     is MuJoCo's behaviour and it is not obviously the intended one; it is
     reproduced rather than improved.
  2. **invisible** — precomputed into `GEOM_IDX_RAY_VISIBLE` at build time,
     because both of MuJoCo's spellings (no material with `rgba[3] == 0`, or a
     material with `rgba[3] == 0`) are fixed once the model exists.
  3. **static** — `flg_static == 0` drops every geom welded to the world.
     ⚠ `body_weldid == 0`, NOT `body == 0`: a body rigidly welded to the world
     through a chain of weld constraints is static too, and testing the body
     id alone would let the floor through on a model that welds to it.
  4. **group mask** — only when a mask is given. The index is CLAMPED into
     `[0, mjNGROUP-1]` rather than rejected, so a `<geom group="9">` reads
     the last mask slot instead of falling off the end.

⚠ A RANGEFINDER USES NONE OF 3 AND 4. `engine_sensor.c:616` calls
`mj_ray(..., geomgroup=NULL, flg_static=1, bodyexclude=site_bodyid, ...)`, so
statics are INCLUDED and no group is filtered. Rules 1 and 2 are the whole
filter for the sensor this was built for — which is why 2 is not optional and
why it is precomputed rather than skipped.
"""

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_MESH,
    GEOM_HFIELD,
)
from ..gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_POS_Y,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_MESH_ID,
    GEOM_IDX_HFIELD_ID,
    GEOM_IDX_RAY_VISIBLE,
    GEOM_IDX_GROUP,
    MODEL_BODY_SIZE,
    BODY_IDX_WELDID,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_TRIADR,
    MESH_META_IDX_TRINUM,
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    HFIELD_META_IDX_SIZE_Y,
    HFIELD_META_IDX_SIZE_Z,
    HFIELD_META_IDX_SIZE_BASE,
)
from .geom import RAY_NO_HIT, ray_geom
from .hfield import ray_hfield
from .mesh import ray_mesh

comptime RAY_NGROUP: Int = 6
"""`mjNGROUP`. The group mask is this wide and the geom's group is clamped
into it, never rejected."""


@fieldwise_init
struct RayHit[DTYPE: DType](Copyable, Movable):
    """`mj_ray`'s three outputs. `geom` is -1 exactly when `t` is negative."""

    var t: Scalar[Self.DTYPE]
    var geom: Int
    var normal: Vec3Generic[Self.DTYPE]


def ray_model[
    DTYPE: DType,
](
    ref geoms: List[Scalar[DTYPE]],
    ngeom: Int,
    ref bodies: List[Scalar[DTYPE]],
    geom_xpos: List[Vec3Generic[DTYPE]],
    geom_xquat: List[QuatGeneric[DTYPE]],
    ref mesh_meta: List[Scalar[DTYPE]],
    ref mesh_tris: List[Scalar[DTYPE]],
    ref hfield_meta: List[Scalar[DTYPE]],
    ref hfield_data: List[Scalar[DTYPE]],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
    bodyexclude: Int = -1,
    flg_static: Bool = True,
    use_group: Bool = False,
    # ⚠ A BITMASK, NOT AN ARRAY. `geomgroup` is six booleans and the geom's
    # group indexes them at RUNTIME — which is the per-thread-array read that
    # is silently wrong on Metal (`87960e10`, the fourth instance in this
    # engine). Six bits in an `Int` is the same information with no thread
    # storage: bit `g` set means group `g` is visible.
    group_mask: Int = 0x3F,
) -> RayHit[DTYPE] where DTYPE.is_floating_point():
    """Nearest intersection of `pnt + x*vec` with the model.

    `geom_xpos`/`geom_xquat` are the geoms' WORLD poses, composed by the
    caller — this module does not run forward kinematics, so a caller stepping
    a batch passes the env it means.

    ⚠ `vec` NEED NOT BE NORMALISED and `t` is in units of `|vec|`, the same
    contract every routine underneath keeps. The reference RAISES on a
    zero-length `vec`; here that case simply hits nothing, because a ray query
    inside a sensor loop is not a place to abort a simulation from.
    """
    var best = Scalar[DTYPE](RAY_NO_HIT)
    var best_geom = -1
    var best_normal = Vec3Generic[DTYPE](0, 0, 0)

    for g in range(ngeom):
        var base = g * MODEL_GEOM_SIZE

        # ── ray_eliminate ────────────────────────────────────────────────
        var body = Int(geoms[base + GEOM_IDX_BODY])
        if body == bodyexclude:
            continue
        if geoms[base + GEOM_IDX_RAY_VISIBLE] == 0:
            continue
        if not flg_static:
            # ⚠ weldid, not the body id — see the module docstring.
            var wb = body
            if body >= 0:
                wb = Int(
                    bodies[body * MODEL_BODY_SIZE + BODY_IDX_WELDID]
                )
            if wb == 0:
                continue
        if use_group:
            var gid = Int(geoms[base + GEOM_IDX_GROUP])
            if gid < 0:
                gid = 0
            if gid > RAY_NGROUP - 1:
                gid = RAY_NGROUP - 1
            if (group_mask >> gid) & 1 == 0:
                continue

        var gtype = Int(geoms[base + GEOM_IDX_TYPE])
        var pos = geom_xpos[g]
        var quat = geom_xquat[g]
        var t = Scalar[DTYPE](RAY_NO_HIT)
        var n = Vec3Generic[DTYPE](0, 0, 0)

        if gtype == GEOM_MESH:
            var mid = Int(
                geoms[base + GEOM_IDX_MESH_ID]
            )
            if mid >= 0:
                var mb = mid * MODEL_MESH_META_SIZE
                var r = ray_mesh[DTYPE](
                    pos,
                    quat,
                    Vec3Generic[DTYPE](
                        geoms[base + GEOM_IDX_HALF_X],
                        geoms[base + GEOM_IDX_HALF_Y],
                        geoms[base + GEOM_IDX_HALF_Z],
                    ),
                    mesh_tris,
                    Int(
                        mesh_meta[mb + MESH_META_IDX_TRIADR]
                    ),
                    Int(
                        mesh_meta[mb + MESH_META_IDX_TRINUM]
                    ),
                    pnt,
                    vec,
                )
                t = r[0]
                n = r[1]
        elif gtype == GEOM_HFIELD:
            var hid = Int(
                geoms[base + GEOM_IDX_HFIELD_ID]
            )
            if hid >= 0:
                var hb = hid * MODEL_HFIELD_META_SIZE
                var r = ray_hfield[DTYPE](
                    pos,
                    quat,
                    Int(
                        hfield_meta[hb + HFIELD_META_IDX_NROW]
                    ),
                    Int(
                        hfield_meta[hb + HFIELD_META_IDX_NCOL]
                    ),
                    hfield_meta[hb + HFIELD_META_IDX_SIZE_X],
                    hfield_meta[hb + HFIELD_META_IDX_SIZE_Y],
                    hfield_meta[hb + HFIELD_META_IDX_SIZE_Z],
                    hfield_meta[hb + HFIELD_META_IDX_SIZE_BASE],
                    hfield_data,
                    Int(
                        hfield_meta[hb + HFIELD_META_IDX_ADR]
                    ),
                    pnt,
                    vec,
                )
                t = r[0]
                n = r[1]
        else:
            # ⚠ THE PURE GEOMS TAKE `size` IN THEIR OWN SPELLING, and the
            # record keeps each type's numbers in its own slots rather than a
            # shared `size[3]`. A sphere's radius is `RADIUS`, a capsule's
            # half-length is `HALF_LENGTH`, a box's extents are `HALF_*` —
            # feeding `HALF_*` to a capsule would silently make it a different
            # capsule, which is why this is spelled per type and not hoisted.
            var hx = geoms[base + GEOM_IDX_HALF_X]
            var hy = geoms[base + GEOM_IDX_HALF_Y]
            var hz = geoms[base + GEOM_IDX_HALF_Z]
            var rad = geoms[base + GEOM_IDX_RADIUS]
            var hl = geoms[base + GEOM_IDX_HALF_LENGTH]
            # Read from `full_parser`'s per-type assignment, not guessed:
            #   SPHERE     radius = s0, and half_* = s0 as well
            #   CAPSULE    radius = s0, half_length = s1
            #   CYLINDER   radius = s0, half_length = s1
            #   BOX        half_* = s0,s1,s2   (radius is the DIAGONAL)
            #   ELLIPSOID  half_* = s0,s1,s2   (radius keeps s0 for the AABB)
            #   PLANE      half_x/y = s0,s1;  half_z is the RENDER GRID SPACING
            # ⚠ `radius` is NOT `size[0]` for a box or an ellipsoid, so a
            # single `(radius, half_length, half_z)` for every type would give
            # `ray_ellipsoid` a sphere of the x semi-axis — the same mistake
            # `_aabb_half_extents` made on flybody's labrum.
            # ⚠ A PLANE'S `half_z` MUST NOT REACH `ray_plane`: it is the grid
            # spacing, and `ray_plane` only reads x and y, so it is passed as
            # zero to keep a future reader from finding a meaningful-looking
            # number there.
            var size = Vec3Generic[DTYPE](hx, hy, hz)
            if gtype == GEOM_SPHERE:
                size = Vec3Generic[DTYPE](rad, rad, rad)
            elif gtype == GEOM_CAPSULE or gtype == GEOM_CYLINDER:
                size = Vec3Generic[DTYPE](rad, hl, Scalar[DTYPE](0))
            elif gtype == GEOM_PLANE:
                size = Vec3Generic[DTYPE](hx, hy, Scalar[DTYPE](0))
            var r = ray_geom[DTYPE](pos, quat, size, pnt, vec, gtype)
            t = r[0]
            n = r[1]

        if t >= 0 and (best < 0 or t < best):
            best = t
            best_geom = g
            best_normal = n

    return RayHit[DTYPE](best, best_geom, best_normal)


def geom_world_poses[
    DTYPE: DType
](
    ref geoms: List[Scalar[DTYPE]],
    ngeom: Int,
    ref xpos: List[Scalar[DTYPE]],
    ref xquat: List[Scalar[DTYPE]],
) -> Tuple[List[Vec3Generic[DTYPE]], List[QuatGeneric[DTYPE]]]:
    """Every geom's WORLD pose, composed once.

    The same composition `contact_detection._geom_world_pos` makes, hoisted so
    a caller with N rangefinders pays one pass over the geom table instead of
    N. ⚠ `Data.xquat` is packed (x, y, z, w) and `math3d.Quat` takes
    (w, x, y, z) — the one place the two conventions meet.

    ⚠ `body <= 0` RETURNS THE LOCAL POSE UNCHANGED, which is right for the
    worldbody (identity transform) and is also what the record holds for a
    static geom marked -1. Composing against `xpos[0]`/`xquat[0]` would give
    the same answer; skipping it keeps a static geom bit-identical to the
    number the parser wrote rather than putting it through a rotation.
    """
    var ps = List[Vec3Generic[DTYPE]](capacity=ngeom)
    var qs = List[QuatGeneric[DTYPE]](capacity=ngeom)
    for g in range(ngeom):
        var o = g * MODEL_GEOM_SIZE
        var body = Int(geoms[o + GEOM_IDX_BODY])
        var lp = Vec3Generic[DTYPE](
            geoms[o + GEOM_IDX_POS_X],
            geoms[o + GEOM_IDX_POS_Y],
            geoms[o + GEOM_IDX_POS_Z],
        )
        var lq = QuatGeneric[DTYPE](
            geoms[o + GEOM_IDX_QUAT_W],
            geoms[o + GEOM_IDX_QUAT_X],
            geoms[o + GEOM_IDX_QUAT_Y],
            geoms[o + GEOM_IDX_QUAT_Z],
        )
        if body <= 0:
            ps.append(lp)
            qs.append(lq)
            continue
        var bq = QuatGeneric[DTYPE](
            xquat[body * 4 + 3],
            xquat[body * 4 + 0],
            xquat[body * 4 + 1],
            xquat[body * 4 + 2],
        )
        var bp = Vec3Generic[DTYPE](
            xpos[body * 3 + 0],
            xpos[body * 3 + 1],
            xpos[body * 3 + 2],
        )
        ps.append(bp + bq.rotate_vec(lp))
        qs.append(bq * lq)
    return (ps^, qs^)
