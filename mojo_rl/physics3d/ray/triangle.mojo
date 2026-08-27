"""`ray_triangle` and the normal-plane basis it needs.

`engine_ray.c:132` plus the basis construction that appears — identically —
inside both `mju_rayTree` (`:797`) and `mj_rayHfield` (`:614`). The reference
repeats it in two places; here it is one function, because a ray library that
spells its own basis twice is the shape
[[feedback_a_rule_written_inline_twice_drifts]] warns about.

HOW THE TEST WORKS, since it is not the textbook Moller-Trumbore. The ray
becomes the ORIGIN of a 2D coordinate system spanned by `b0`/`b1`, two unit
vectors orthogonal to `lvec`. Each triangle vertex is projected into that
plane, and the question "does the ray pass through the triangle" becomes "does
the 2D origin lie inside the projected triangle" — solved by one 2x2 system in
barycentric coordinates. Only then is the distance computed, by intersecting
the ray with the triangle's plane.

⚠ THE BASIS IS BUILT ONCE PER RAY, NOT PER TRIANGLE. That is the whole point
of the split: `mj_rayHfield` walks up to a few hundred triangles and
`mju_rayTree` a few thousand, all sharing one `lvec`. Rebuilding it inside the
triangle test would be correct and roughly double the cost.

⚠ THE NORMAL IS NOT FLIPPED TOWARD THE RAY. `cross(v0-v2, v1-v2)` normalised,
whatever side the ray came from — so it depends on the triangle's WINDING.
`mj_rayHfield` relies on that: its comment "swap v1 and v2 for consistent CCW
winding (normals point up)" exists because the grid's two triangles would
otherwise disagree in sign.
"""

from std.math import sqrt

from mojo_rl.math3d import Vec3 as Vec3Generic

from .geom import RAY_MINVAL, RAY_NO_HIT


@always_inline
def ray_basis[
    DTYPE: DType
](
    lvec: Vec3Generic[DTYPE]
) -> Tuple[Vec3Generic[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """Two orthonormal vectors spanning the plane normal to `lvec`.

    ⚠ THE SEED IS `(1, 1, 1)` WITH THE LARGEST COMPONENT OF `lvec` ZEROED, and
    the choice is not cosmetic: zeroing the largest component guarantees the
    seed is not parallel to `lvec`, so the Gram-Schmidt step below cannot
    produce a zero vector. Seeding with a fixed axis instead — the obvious
    simplification — degenerates for any ray travelling along that axis, which
    is exactly the ray a downward rangefinder casts at a heightfield.

    Returns `(b0, b1)` in the reference's final order: `b1` is the
    Gram-Schmidt result, `b0` is `b1 x lvec`.
    """
    var b0: Vec3Generic[DTYPE]
    var ax = abs(lvec.x)
    var ay = abs(lvec.y)
    var az = abs(lvec.z)
    if ax >= ay and ax >= az:
        b0 = Vec3Generic[DTYPE](0, 1, 1)
    elif ay >= az:
        b0 = Vec3Generic[DTYPE](1, 0, 1)
    else:
        b0 = Vec3Generic[DTYPE](1, 1, 0)

    # b1 = b0 - lvec * (lvec.b0 / lvec.lvec)   -- the reference's mju_addScl3
    var b1 = b0 + lvec * (-lvec.dot(b0) / lvec.dot(lvec))
    b1 = b1.normalized()
    b0 = b1.cross(lvec).normalized()
    return (b0, b1)


@always_inline
def ray_triangle[
    DTYPE: DType
](
    v0: Vec3Generic[DTYPE],
    v1: Vec3Generic[DTYPE],
    v2: Vec3Generic[DTYPE],
    lpnt: Vec3Generic[DTYPE],
    lvec: Vec3Generic[DTYPE],
    b0: Vec3Generic[DTYPE],
    b1: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """`ray_triangle` — distance along `lvec`, and the triangle's plane normal.

    Everything is in the geom's LOCAL frame, including the returned normal; the
    caller rotates it out once, after the winning triangle is known, rather
    than per candidate.

    ⚠ Returns a distance that may be NEGATIVE — the plane intersection is not
    clamped. The reference does not clamp it either, and every caller filters
    on `sol >= 0`. A routine that clamped here would report a hit BEHIND the
    ray origin as a hit at zero.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)
    var d0 = v0 - lpnt
    var d1 = v1 - lpnt
    var d2 = v2 - lpnt

    # Project into the normal plane.
    var p00 = b0.dot(d0)
    var p01 = b1.dot(d0)
    var p10 = b0.dot(d1)
    var p11 = b1.dot(d1)
    var p20 = b0.dot(d2)
    var p21 = b1.dot(d2)

    # Cheap reject: all three vertices strictly on one side of either axis.
    if (
        (p00 > 0 and p10 > 0 and p20 > 0)
        or (p00 < 0 and p10 < 0 and p20 < 0)
        or (p01 > 0 and p11 > 0 and p21 > 0)
        or (p01 < 0 and p11 < 0 and p21 < 0)
    ):
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    # Is the 2D origin inside the projected triangle?
    # A = (p0-p2, p1-p2), b = -p2, solve A*t = b.
    var a0 = p00 - p20
    var a1 = p10 - p20
    var a2 = p01 - p21
    var a3 = p11 - p21
    var det = a0 * a3 - a1 * a2
    if abs(det) < Scalar[DTYPE](RAY_MINVAL):
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    var bb0 = -p20
    var bb1 = -p21
    var t0 = (a3 * bb0 - a1 * bb1) / det
    var t1 = (-a2 * bb0 + a0 * bb1) / det
    if t0 < 0 or t1 < 0 or t0 + t1 > 1:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    # Intersect the ray with the triangle's plane.
    var e0 = v0 - v2
    var e1 = v1 - v2
    var e2 = lpnt - v2
    var nrm = e0.cross(e1)
    var denom = lvec.dot(nrm)
    if abs(denom) < Scalar[DTYPE](RAY_MINVAL):
        return (Scalar[DTYPE](RAY_NO_HIT), zero)

    return (-e2.dot(nrm) / denom, nrm.normalized())
