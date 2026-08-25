"""`mju_rayGeom` — ray against one pure geom, distance and surface normal.

A transcription of `engine_ray.c` (MuJoCo 3.10.0, the runtime this tree gates
against), lines 38-560 and 972. Same routines, same order of operations, same
early-outs. `mujoco_warp/_src/ray.py` is the second reference and agrees; where
the two differ in spelling this follows the C.

WHAT THIS UNLOCKS, AND WHY IT IS ONE MODULE AND NOT THREE
=========================================================
Three separate pieces of work were queued against three different-looking
problems, and all three are this function:

  · the `rangefinder` sensor, which is the last thing standing between the
    port and `quadruped escape` — the 49th and final suite task;
  · studio PICKING, which today has its own approximate raycaster
    (`studio/pick.mojo`: no plane, no hfield, mesh and ellipsoid as bounding
    SPHERES, capsule `t` "roughly the surface" by its own admission);
  · BATCHED CAMERA OBSERVATIONS. MuJoCo Warp's renderer is not a rasteriser —
    `mujoco_warp/_src/render.py` imports nothing but `ray_box`, `ray_capsule`,
    `ray_cylinder`, `ray_ellipsoid`, `ray_mesh`, `ray_plane`, `ray_sphere` and
    casts one primary ray per (world, pixel). A ray library IS the renderer.

So this is deliberately dependency-free: scalars, `Vec3` and `Quat`, no
`Model`, no `Data`, no `LayoutTensor`. Each consumer binds its own storage.

⚠⚠ TWO DELIBERATE DEVIATIONS FROM THE REFERENCE SIGNATURE
=========================================================
1. **Orientation is a QUATERNION, not `mat[9]`.** Every caller in this tree
   holds quaternions (`Data.xquat` composed with the geom's local `quat`);
   materialising a 3x3 to hand it over and transposing it inside would be
   work in both directions. `ray_map`'s `mat' * v` becomes
   `q.rotate_vec_inverse(v)` and the normal's `mat * n` becomes
   `q.rotate_vec(n)`, which are the same map. ⚠ They are NOT the same
   FLOATING-POINT operation — expect a difference against MuJoCo at the last
   bits rather than exact zero, and see the gate for what it actually costs.
2. **`geomtype` is OUR enum, not `mjtGeom`.** They have never matched:
   `mjtGeom` is PLANE 0, HFIELD 1, SPHERE 2, ... while `constants.mojo` is
   PLANE 0, SPHERE 1, CAPSULE 2, BOX 3, CYLINDER 4, MESH 5, ELLIPSOID 6,
   HFIELD 7 (that file says so, and says why: appending keeps stored models
   readable). Anything comparing against `m.geom_type` must translate, and a
   gate that forgets is the shape of
   `feedback_a_gates_classification_goes_stale_when_the_dispatch_moves`.

CONVENTIONS, taken from the reference and not negotiable:
  · the ray is `pnt + x*vec`, `x >= 0`; **`vec` need not be normalised**, and
    `x` is in units of `|vec|`. Callers wanting metres pass a unit `vec`.
  · **-1.0 means NO HIT**, and it is a legitimate return, not an error code.
  · a returned normal is a UNIT vector in the WORLD frame, and is zeroed
    whenever the distance is -1.
  · `size` is MuJoCo's `geom_size` for the type, unused entries ignored.

NOT HERE YET: mesh (needs `ray_triangle` over the hull), hfield (a grid walk),
sdf, flex, skin, the `mj_ray` model traversal and its `ray_eliminate` filter.
Those are the next slices; this one is the piece all of them call.
"""

from std.math import sqrt

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
)


comptime RAY_MINVAL: Float64 = 1e-15
"""`mjMINVAL` (`mjtype.h:27`) — "minimum value in any denominator".

⚠ NOT a tolerance to be tuned. It appears as `a < mjMINVAL` in `ray_quad` and
`|lvec[i]| > mjMINVAL` in the flat-face loops, and both are guards against
dividing by a component of a ray that is parallel to the surface. Changing it
changes which grazing rays are reported as hits.
"""

comptime RAY_NO_HIT: Float64 = -1.0
"""What every routine here returns when the ray misses. A LEGITIMATE value."""


@always_inline
def ray_map[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Vec3Generic[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_map` — take the ray into the geom's local frame.

    The reference is `lpnt = mat' * (pnt - pos)`, `lvec = mat' * vec`; `mat'`
    is the inverse rotation, so with a quaternion it is `rotate_vec_inverse`.
    ⚠ `vec` is a DIRECTION and gets no translation — the classic way to write
    this wrong is to subtract `pos` from both.
    """
    var dif = pnt - pos
    return (quat.rotate_vec_inverse(dif), quat.rotate_vec_inverse(vec))


@always_inline
def ray_quad[
    DTYPE: DType
](
    a: Scalar[DTYPE], b: Scalar[DTYPE], c: Scalar[DTYPE]
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]] where DTYPE.is_floating_point():
    """`ray_quad` — solve `a*x^2 + 2*b*x + c = 0`, smallest non-negative root.

    Returns `(best, x0, x1)`. ⚠ `b` IS THE HALF-COEFFICIENT: the reference
    solves `a*x^2 + 2*b*x + c`, so the determinant is `b*b - a*c` and not
    `b*b - 4*a*c`. Every caller passes a `b` that is already halved, and the
    capsule needs BOTH roots, not just the best one — which is why they come
    back rather than being discarded.

    `x0 <= x1` is guaranteed. When there is no real solution, or the ray is
    parallel to the surface (`a < mjMINVAL`), all three are -1.
    """
    var det = b * b - a * c
    if det < 0 or a < Scalar[DTYPE](RAY_MINVAL):
        var m = Scalar[DTYPE](RAY_NO_HIT)
        return (m, m, m)

    var sd = sqrt(det)
    var x0 = (-b - sd) / a
    var x1 = (-b + sd) / a

    var best = Scalar[DTYPE](RAY_NO_HIT)
    if x0 >= 0:
        best = x0
    elif x1 >= 0:
        best = x1
    return (best, x0, x1)


@always_inline
def ray_plane[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    size: Vec3Generic[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_plane`. ⚠ ONE-SIDED and ⚠ `size <= 0` means INFINITE on that axis.

    The front-face test `lvec[2] > -mjMINVAL` rejects a ray travelling along or
    away from +z, so a plane is invisible from below — that is MuJoCo's
    behaviour and it is what makes a ground plane not occlude a camera placed
    under the floor by mistake.

    `size[0]`/`size[1]` are the RENDERED half-extents. A `<geom type="plane"
    size="0 0 0.05"/>` — which is every dm_control floor — is unbounded in x
    and y, and treating `0` as a zero-size rectangle would make the floor
    invisible to every ray. The reference spells that `size[0] <= 0 || ...`.
    """
    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]
    var zero = Vec3Generic[DTYPE](0, 0, 0)

    if lvec.z > Scalar[DTYPE](-RAY_MINVAL):
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var x = -lpnt.z / lvec.z
    if x < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var p0 = lpnt.x + x * lvec.x
    var p1 = lpnt.y + x * lvec.y
    if (size.x <= 0 or abs(p0) <= size.x) and (size.y <= 0 or abs(p1) <= size.y):
        # `mat[2], mat[5], mat[8]` is the third COLUMN — the local +z axis
        # taken to world, which is what rotating the unit vector gives.
        return (x, quat.rotate_vec(Vec3Generic[DTYPE](0, 0, 1)))
    return (Scalar[DTYPE](RAY_NO_HIT), zero)


@always_inline
def ray_sphere[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    dist_sqr: Scalar[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_sphere`. ⚠ Takes the radius SQUARED, and no orientation.

    The squared radius is not a micro-optimisation — `ray_capsule`,
    `ray_cylinder` and `ray_box` all call this as a bounding-sphere reject with
    a radius that is already a sum of squares (`size[0]^2 + size[1]^2` and so
    on), and taking a root just to square it again would change their answers
    at the last bits. A sphere is orientation-free, so the reference passes
    `NULL` for `mat` on those calls; there is nothing to pass here.
    """
    var dif = pnt - pos
    var a = vec.dot(vec)
    var b = vec.dot(dif)
    var c = dif.dot(dif) - dist_sqr

    var q = ray_quad[DTYPE](a, b, c)
    var x = q[0]
    if x < 0:
        return (x, Vec3Generic[DTYPE](0, 0, 0))
    # Normal in the GLOBAL frame directly — a sphere needs no rotation back.
    var s = pnt + vec * x
    return (x, (s - pos).normalized())


@always_inline
def ray_capsule[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    size: Vec3Generic[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_capsule` — round side plus two hemispherical caps.

    ⚠⚠ THE CAPS USE BOTH ROOTS, NOT THE BEST ONE. Each cap solves a full
    sphere and then keeps only the roots on its own half (`lz >= size[1]` for
    the top). `ray_quad`'s "smallest non-negative" answer can be the root on
    the WRONG half — the ray entering the cylinder body below the cap — in
    which case taking it alone loses the hit entirely. This is why `ray_quad`
    returns `x0` and `x1`, and it is the bug
    `studio/pick.mojo::_hit_capsule` documents in itself from the other
    direction ("a ray straight down a capsule's axis reported the FAR cap").

    `type` (-1 bottom, 0 cylinder, +1 top) selects the normal: on the round
    side z is dropped, on a cap the sphere centre `+-size[1]` is subtracted.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)

    # Bounding sphere reject, exactly as the reference orders it.
    var ssz = size.x + size.y
    var bs = ray_sphere[DTYPE](pos, ssz * ssz, pnt, vec)
    if bs[0] < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]

    var x = Scalar[DTYPE](RAY_NO_HIT)
    var typ: Int = 0

    # Round side: only x and y enter, so a ray parallel to the axis gives
    # a == 0 and `ray_quad` rejects it on `a < mjMINVAL`.
    var a = lvec.x * lvec.x + lvec.y * lvec.y
    var b = lvec.x * lpnt.x + lvec.y * lpnt.y
    var c = lpnt.x * lpnt.x + lpnt.y * lpnt.y - size.x * size.x
    var q = ray_quad[DTYPE](a, b, c)
    var sol = q[0]
    if sol >= 0 and abs(lpnt.z + sol * lvec.z) <= size.y:
        if x < 0 or sol < x:
            x = sol
            typ = 0

    # Top cap.
    var da = lvec.dot(lvec)
    var dtop = Vec3Generic[DTYPE](lpnt.x, lpnt.y, lpnt.z - size.y)
    var qt = ray_quad[DTYPE](da, lvec.dot(dtop), dtop.dot(dtop) - size.x * size.x)
    for i in range(2):
        var xi = qt[1] if i == 0 else qt[2]
        if xi >= 0 and lpnt.z + xi * lvec.z >= size.y:
            if x < 0 or xi < x:
                x = xi
                typ = 1

    # Bottom cap.
    var dbot = Vec3Generic[DTYPE](lpnt.x, lpnt.y, lpnt.z + size.y)
    var qb = ray_quad[DTYPE](da, lvec.dot(dbot), dbot.dot(dbot) - size.x * size.x)
    for i in range(2):
        var xi = qb[1] if i == 0 else qb[2]
        if xi >= 0 and lpnt.z + xi * lvec.z <= -size.y:
            if x < 0 or xi < x:
                x = xi
                typ = -1

    if x < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var n = Vec3Generic[DTYPE](
        lpnt.x + lvec.x * x,
        lpnt.y + lvec.y * x,
        Scalar[DTYPE](0) if typ == 0
        else lpnt.z + lvec.z * x - size.y * Scalar[DTYPE](typ),
    )
    return (x, quat.rotate_vec(n.normalized()))


@always_inline
def ray_ellipsoid[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    size: Vec3Generic[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_ellipsoid` — one quadratic in the metric `diag(1/size^2)`.

    ⚠ NO bounding-sphere reject here, unlike capsule/cylinder/box. That is the
    reference's choice, not an omission to be tidied: the quadratic is the
    whole cost, so a reject would be pure overhead.

    The normal is the GRADIENT of the implicit function, `diag(1/size^2) * l`,
    which is NOT the direction from the centre — that mistake is invisible on a
    sphere and wrong everywhere else, and it is the same class of error as
    `feedback_a_geom_type_absent_from_three_fallbacks`' ellipsoid support
    function returning the CENTRE.
    """
    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]

    var s = Vec3Generic[DTYPE](
        Scalar[DTYPE](1) / (size.x * size.x),
        Scalar[DTYPE](1) / (size.y * size.y),
        Scalar[DTYPE](1) / (size.z * size.z),
    )
    var a = s.x * lvec.x * lvec.x + s.y * lvec.y * lvec.y + s.z * lvec.z * lvec.z
    var b = s.x * lvec.x * lpnt.x + s.y * lvec.y * lpnt.y + s.z * lvec.z * lpnt.z
    var c = (
        s.x * lpnt.x * lpnt.x
        + s.y * lpnt.y * lpnt.y
        + s.z * lpnt.z * lpnt.z
        - Scalar[DTYPE](1)
    )

    var q = ray_quad[DTYPE](a, b, c)
    var x = q[0]
    if x < 0:
        return (x, Vec3Generic[DTYPE](0, 0, 0))

    var l = lpnt + lvec * x
    var n = Vec3Generic[DTYPE](s.x * l.x, s.y * l.y, s.z * l.z)
    return (x, quat.rotate_vec(n.normalized()))


@always_inline
def ray_cylinder[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    size: Vec3Generic[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_cylinder` — two flat caps plus the round side.

    ⚠ A cylinder is NOT a capsule, and `studio/pick.mojo` treats it as one.
    The caps here are FLAT discs, so a ray hitting the rim lands on a disc
    normal `(0, 0, +-1)` rather than a rounded one — see
    `ee3977ad`, "a cylinder is not a capsule — three dispatch copies".
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)

    # ⚠ The bound is `size[0]^2 + size[1]^2` and is passed ALREADY SQUARED —
    # it is the squared half-diagonal, not a radius that needs squaring.
    var ssz = size.x * size.x + size.y * size.y
    var bs = ray_sphere[DTYPE](pos, ssz, pnt, vec)
    if bs[0] < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]

    var x = Scalar[DTYPE](RAY_NO_HIT)
    var typ: Int = 0

    if abs(lvec.z) > Scalar[DTYPE](RAY_MINVAL):
        for k in range(2):
            var side: Int = -1 if k == 0 else 1
            var sol = (Scalar[DTYPE](side) * size.y - lpnt.z) / lvec.z
            if sol >= 0:
                var p0 = lpnt.x + sol * lvec.x
                var p1 = lpnt.y + sol * lvec.y
                if p0 * p0 + p1 * p1 <= size.x * size.x:
                    if x < 0 or sol < x:
                        x = sol
                        typ = side

    var a = lvec.x * lvec.x + lvec.y * lvec.y
    var b = lvec.x * lpnt.x + lvec.y * lpnt.y
    var c = lpnt.x * lpnt.x + lpnt.y * lpnt.y - size.x * size.x
    var q = ray_quad[DTYPE](a, b, c)
    var sol2 = q[0]
    if sol2 >= 0 and abs(lpnt.z + sol2 * lvec.z) <= size.y:
        if x < 0 or sol2 < x:
            x = sol2
            typ = 0

    if x < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var n: Vec3Generic[DTYPE]
    if typ == 0:
        n = Vec3Generic[DTYPE](
            lpnt.x + lvec.x * x, lpnt.y + lvec.y * x, Scalar[DTYPE](0)
        ).normalized()
    else:
        # ⚠ NOT normalised in the reference — it is already a unit vector, and
        # normalising it would be a different rounding.
        n = Vec3Generic[DTYPE](0, 0, Scalar[DTYPE](typ))
    return (x, quat.rotate_vec(n))


@fieldwise_init
struct RayBoxHit[DTYPE: DType](Copyable, Movable):
    """`ray_box`'s full answer, including the reference's optional `all[6]`.

    `all[2*i + (side+1)/2]` is the distance at which the ray crosses the face
    on axis `i`, side `-1`/`+1`, or -1 where that face is not hit. It exists
    for `mj_rayHfield`, which uses it twice: to clip the ray to the segment
    inside the field's top box, and to test the four vertical SIDES of that box
    against the elevation profile at the crossing.

    ⚠ A face can be recorded in `all` and still not be the winner — `all` is
    every accepted face, `t` is the nearest.
    """

    var t: Scalar[Self.DTYPE]
    var normal: Vec3Generic[Self.DTYPE]
    var all: InlineArray[Scalar[Self.DTYPE], 6]


@always_inline
def ray_box_all[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    size: Vec3Generic[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> RayBoxHit[DTYPE] where DTYPE.is_floating_point():
    """`ray_box` — six slabs, nearest accepted face, plus the per-face table.

    ⚠ THE FACE NORMAL IS `+-1` ON THE WINNING AXIS IN THE LOCAL FRAME, so the
    axis and the side must both survive out of the loop. Keeping only the
    distance and recomputing the face afterwards from the hit point is the
    tempting simplification and it is ambiguous exactly on an edge.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)
    var none = Scalar[DTYPE](RAY_NO_HIT)
    var all = InlineArray[Scalar[DTYPE], 6](fill=none)

    var ssz = size.x * size.x + size.y * size.y + size.z * size.z
    var bs = ray_sphere[DTYPE](pos, ssz, pnt, vec)
    if bs[0] < 0:
        return RayBoxHit[DTYPE](none, zero, all.copy())

    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]

    var x = none
    var face_axis: Int = -1
    var face_side: Int = 0

    for i in range(3):
        var li = lvec.x if i == 0 else (lvec.y if i == 1 else lvec.z)
        if abs(li) <= Scalar[DTYPE](RAY_MINVAL):
            continue
        var pi = lpnt.x if i == 0 else (lpnt.y if i == 1 else lpnt.z)
        var si = size.x if i == 0 else (size.y if i == 1 else size.z)
        # `iface` in the reference: the two axes that are NOT i.
        var j0 = 1 if i == 0 else 0
        var j1 = 2 if i < 2 else 1
        var lj0 = lvec.x if j0 == 0 else (lvec.y if j0 == 1 else lvec.z)
        var lj1 = lvec.x if j1 == 0 else (lvec.y if j1 == 1 else lvec.z)
        var pj0 = lpnt.x if j0 == 0 else (lpnt.y if j0 == 1 else lpnt.z)
        var pj1 = lpnt.x if j1 == 0 else (lpnt.y if j1 == 1 else lpnt.z)
        var sj0 = size.x if j0 == 0 else (size.y if j0 == 1 else size.z)
        var sj1 = size.x if j1 == 0 else (size.y if j1 == 1 else size.z)

        for k in range(2):
            var side: Int = -1 if k == 0 else 1
            var sol = (Scalar[DTYPE](side) * si - pi) / li
            if sol < 0:
                continue
            var p0 = pj0 + sol * lj0
            var p1 = pj1 + sol * lj1
            if abs(p0) <= sj0 and abs(p1) <= sj1:
                if x < 0 or sol < x:
                    x = sol
                    face_axis = i
                    face_side = side
                all[2 * i + (side + 1) // 2] = sol

    if x < 0:
        return RayBoxHit[DTYPE](none, zero, all.copy())

    var n = Vec3Generic[DTYPE](
        Scalar[DTYPE](face_side) if face_axis == 0 else Scalar[DTYPE](0),
        Scalar[DTYPE](face_side) if face_axis == 1 else Scalar[DTYPE](0),
        Scalar[DTYPE](face_side) if face_axis == 2 else Scalar[DTYPE](0),
    )
    return RayBoxHit[DTYPE](x, quat.rotate_vec(n), all.copy())


@always_inline
def ray_box[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    size: Vec3Generic[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_box` without the per-face table — the `mju_rayGeom` entry point.

    A wrapper rather than a second copy of the slab loop: the `all` writes do
    not participate in `x`, so discarding them cannot change the answer, and
    two hand-maintained copies of this loop is precisely how a fix crosses to
    one of them. The `mju_rayGeom` sweep re-run after this refactor is what
    says the rounding did not move.
    """
    var h = ray_box_all[DTYPE](pos, quat, size, pnt, vec)
    return (h.t, h.normal)


@always_inline
def ray_geom[
    DTYPE: DType
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    size: Vec3Generic[DTYPE],
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
    geomtype: Int,
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`mju_rayGeom` — dispatch over the PURE geoms.

    ⚠⚠ `geomtype` IS THIS TREE'S ENUM (`physics3d/constants.mojo`), NOT
    `mjtGeom`. See the module docstring; they have never agreed.

    ⚠ MESH and HFIELD are NOT pure geoms and are absent from the reference's
    switch too — it calls `mjERROR` on them, because they need model data
    (vertices, an elevation grid) that a "pure geom" signature has no room for.
    Here they return NO HIT rather than raising, so a scene containing one
    renders the rest instead of aborting, and a caller that must not miss them
    checks the type ITSELF. ⚠ That is a silent-miss risk of exactly the shape
    `feedback_a_cap_that_returns_the_fallback_code` warns about, so it is
    stated here and asserted in the gate rather than left to be discovered.
    """
    if geomtype == GEOM_PLANE:
        return ray_plane[DTYPE](pos, quat, size, pnt, vec)
    if geomtype == GEOM_SPHERE:
        return ray_sphere[DTYPE](pos, size.x * size.x, pnt, vec)
    if geomtype == GEOM_CAPSULE:
        return ray_capsule[DTYPE](pos, quat, size, pnt, vec)
    if geomtype == GEOM_ELLIPSOID:
        return ray_ellipsoid[DTYPE](pos, quat, size, pnt, vec)
    if geomtype == GEOM_CYLINDER:
        return ray_cylinder[DTYPE](pos, quat, size, pnt, vec)
    if geomtype == GEOM_BOX:
        return ray_box[DTYPE](pos, quat, size, pnt, vec)
    return (Scalar[DTYPE](RAY_NO_HIT), Vec3Generic[DTYPE](0, 0, 0))
