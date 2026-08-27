"""Tendon wrapping around a sphere or cylinder — MuJoCo's `mju_wrap`.

Port of `references/mujoco-3.10.0/src/engine/engine_util_misc.c:281-417` and
the three static helpers it needs (`is_intersect`, `length_circle`,
`wrap_circle`, `wrap_inside`, lines 36-232).

WHAT IT COMPUTES
================
Given a segment `x0 -> x1` and a wrap object (sphere or cylinder) at
`xpos`/`xmat` with `radius`, decide whether the straight segment passes
THROUGH the object; if it does, replace it with `x0 -> w0 ~arc~ w1 -> x1` and
return the arc length. Returning `-1` means "no wrap" and the caller keeps its
straight segment.

⚠ THE WHOLE THING IS 2D. Both endpoints are mapped into the object's local
frame and then projected onto a plane — for a SPHERE the plane through the
origin containing both points, for a CYLINDER the object's own xy-plane. The
tangent solve is a circle problem there, and the cylinder's z is recovered
afterwards by interpolating along the unrolled path length.

⚠⚠ NO DERIVATIVE, AND THAT IS NOT AN OMISSION. MuJoCo's own velocity path
refuses geom wrapping outright — `mj_tendonDot` (`engine_core_smooth.c:1173`)
is `mjERROR("geom wrapping not supported")` behind a
`TODO(tassa) ... requires derivatives of mju_wrap`. It is only reached when
`tendon_armature != 0`, which no model in the tree sets. Matching MuJoCo here
means NOT having one either.

WHY A SIDESITE
==============
A tangent solve has two answers — the tendon can pass either side of the
object — and the two are equally short only by accident. `sidesite` names the
side the modeller meant: the candidate whose midpoint direction agrees with it
wins. Without one the SHORTER path wins, which is right for a pulley the
tendon merely clears and wrong for one it is meant to hook around.

⚠ A SIDESITE INSIDE THE OBJECT SELECTS A DIFFERENT ALGORITHM. `wrap_inside`
solves the tendon running INSIDE the circle and touching it, which is the
"loop through a ring" case; `wrap_circle` solves the outside tangent. MuJoCo
switches on `|s| < radius`, so the sidesite's DISTANCE, not just its
direction, is load-bearing.
"""

from std.collections import InlineArray
from std.math import sqrt, acos, asin, cos, sin


from ..gpu.constants import (
    WRAP_NONE, WRAP_SITE, WRAP_SPHERE, WRAP_CYLINDER, WRAP_PULLEY,
)


def _mjminval[DTYPE: DType]() -> Scalar[DTYPE]:
    return Scalar[DTYPE](1e-15)


def _pi[DTYPE: DType]() -> Scalar[DTYPE]:
    return Scalar[DTYPE](3.14159265358979323846)


def _abs[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    return -x if x < 0 else x


def _acos_safe[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`acos` with the argument clamped to [-1, 1].

    ⚠ NOT COSMETIC. `mju_acos` clamps, and the arguments here are dot products
    of vectors normalised in float — `1 + 1e-8` happens, and an unclamped
    `acos` returns NaN, which then travels silently into the tendon length.
    """
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    if x <= Scalar[DTYPE](-1):
        return _pi[DTYPE]()
    if x >= Scalar[DTYPE](1):
        return Scalar[DTYPE](0)
    return acos(x)


def _asin_safe[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    if x <= Scalar[DTYPE](-1):
        return -_pi[DTYPE]() / 2
    if x >= Scalar[DTYPE](1):
        return _pi[DTYPE]() / 2
    return asin(x)


struct WrapOut[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """`wlen < 0` => no wrap, and `p*` are then meaningless."""

    var wlen: Scalar[Self.DTYPE]
    var p0x: Scalar[Self.DTYPE]
    var p0y: Scalar[Self.DTYPE]
    var p0z: Scalar[Self.DTYPE]
    var p1x: Scalar[Self.DTYPE]
    var p1y: Scalar[Self.DTYPE]
    var p1z: Scalar[Self.DTYPE]

    def __init__(out self):
        self.wlen = Scalar[Self.DTYPE](-1)
        self.p0x = Scalar[Self.DTYPE](0)
        self.p0y = Scalar[Self.DTYPE](0)
        self.p0z = Scalar[Self.DTYPE](0)
        self.p1x = Scalar[Self.DTYPE](0)
        self.p1y = Scalar[Self.DTYPE](0)
        self.p1z = Scalar[Self.DTYPE](0)


struct _Pnt4[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """The 2D tangent pair, plus the arc length or -1 for "no wrap"."""

    var len: Scalar[Self.DTYPE]
    var ax: Scalar[Self.DTYPE]
    var ay: Scalar[Self.DTYPE]
    var bx: Scalar[Self.DTYPE]
    var by: Scalar[Self.DTYPE]

    def __init__(out self, wl: Scalar[Self.DTYPE]):
        self.len = wl
        self.ax = Scalar[Self.DTYPE](0)
        self.ay = Scalar[Self.DTYPE](0)
        self.bx = Scalar[Self.DTYPE](0)
        self.by = Scalar[Self.DTYPE](0)


def _is_intersect[
    DTYPE: DType
](
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE],
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE],
    p3x: Scalar[DTYPE], p3y: Scalar[DTYPE],
    p4x: Scalar[DTYPE], p4y: Scalar[DTYPE],
) -> Bool:
    """Do the 2D segments p1-p2 and p3-p4 cross? (`is_intersect`)."""
    var det = (p4y - p3y) * (p2x - p1x) - (p4x - p3x) * (p2y - p1y)
    if _abs[DTYPE](det) < _mjminval[DTYPE]():
        return False
    var a = ((p4x - p3x) * (p1y - p3y) - (p4y - p3y) * (p1x - p3x)) / det
    var b = ((p2x - p1x) * (p1y - p3y) - (p2y - p1y) * (p1x - p3x)) / det
    return (
        a >= Scalar[DTYPE](0)
        and a <= Scalar[DTYPE](1)
        and b >= Scalar[DTYPE](0)
        and b <= Scalar[DTYPE](1)
    )


def _length_circle[
    DTYPE: DType
](
    p0x: Scalar[DTYPE], p0y: Scalar[DTYPE],
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE],
    ind: Int,
    radius: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Arc length between the two tangent points (`length_circle`).

    ⚠ `ind` IS THE SOLUTION BRANCH, and it decides whether the arc is the
    minor or the MAJOR one: `acos` only ever returns [0, pi], so the sign of
    the 2D cross product against the branch index is what says "go the long
    way round". Dropping it gives an arc that is short by `2*pi*r - 2*theta*r`
    on exactly half the wraps — a plausible number, never a NaN.
    """
    var n0 = sqrt(p0x * p0x + p0y * p0y)
    var n1 = sqrt(p1x * p1x + p1y * p1y)
    if n0 < _mjminval[DTYPE]() or n1 < _mjminval[DTYPE]():
        return Scalar[DTYPE](0)
    var a0x = p0x / n0
    var a0y = p0y / n0
    var a1x = p1x / n1
    var a1y = p1y / n1
    var angle = _acos_safe[DTYPE](a0x * a1x + a0y * a1y)

    var cross = p0y * p1x - p0x * p1y
    if (cross > 0 and ind != 0) or (cross < 0 and ind == 0):
        angle = 2 * _pi[DTYPE]() - angle

    return radius * angle


def _wrap_circle[
    DTYPE: DType
](
    e0x: Scalar[DTYPE], e0y: Scalar[DTYPE],
    e1x: Scalar[DTYPE], e1y: Scalar[DTYPE],
    has_side: Bool,
    sx: Scalar[DTYPE], sy: Scalar[DTYPE],
    radius: Scalar[DTYPE],
) -> _Pnt4[DTYPE]:
    """The OUTSIDE tangent solve (`wrap_circle`). `-1` length => no wrap."""
    var sqlen0 = e0x * e0x + e0y * e0y
    var sqlen1 = e1x * e1x + e1y * e1y
    var sqrad = radius * radius

    # An endpoint inside the circle has no tangent; a degenerate circle has no
    # arc.
    if sqlen0 < sqrad or sqlen1 < sqrad or radius < _mjminval[DTYPE]():
        return _Pnt4[DTYPE](Scalar[DTYPE](-1))

    var dx = e1x - e0x
    var dy = e1y - e0y
    var dd = dx * dx + dy * dy
    if dd < _mjminval[DTYPE]():
        return _Pnt4[DTYPE](Scalar[DTYPE](-1))

    # Nearest point on the segment to the origin, clamped to the segment.
    var a = -(dx * e0x + dy * e0y) / dd
    if a < 0:
        a = Scalar[DTYPE](0)
    elif a > 1:
        a = Scalar[DTYPE](1)

    # ⚠ THE NO-WRAP TEST, AND THE SIDESITE PARTICIPATES IN IT. The segment
    # clearing the circle is not enough: with a sidesite, a segment that
    # clears it on the WRONG side still wraps, because the modeller said the
    # tendon goes round the other way.
    var tx = a * dx + e0x
    var ty = a * dy + e0y
    if tx * tx + ty * ty > sqrad and (
        not has_side or (sx * tx + sy * ty) >= 0
    ):
        return _Pnt4[DTYPE](Scalar[DTYPE](-1))

    var sqrt0 = sqrt(sqlen0 - sqrad)
    var sqrt1 = sqrt(sqlen1 - sqrad)

    # The two tangent solutions, scored. `good` is agreement with the sidesite
    # when there is one, and NEGATIVE chord length otherwise (i.e. shortest).
    #
    # ⚠ THE TIE GOES TO SOLUTION 1. MuJoCo writes `i = (good[0] > good[1] ? 0
    # : 1)`, so equal scores pick the SECOND branch — and equal scores are not
    # exotic: a segment symmetric about the object gives two mirror tangents
    # with identical chord lengths. `>` in a running-best loop picks 0 there,
    # which is the opposite side of the pulley for a whole class of poses.
    var ax = InlineArray[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))
    var ay = InlineArray[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))
    var bx = InlineArray[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))
    var by = InlineArray[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))
    var good = InlineArray[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))

    for i in range(2):
        var sgn = Scalar[DTYPE](1) if i == 0 else Scalar[DTYPE](-1)
        ax[i] = (e0x * sqrad + sgn * radius * e0y * sqrt0) / sqlen0
        ay[i] = (e0y * sqrad - sgn * radius * e0x * sqrt0) / sqlen0
        bx[i] = (e1x * sqrad - sgn * radius * e1y * sqrt1) / sqlen1
        by[i] = (e1y * sqrad + sgn * radius * e1x * sqrt1) / sqlen1

        if has_side:
            var mx = ax[i] + bx[i]
            var my = ay[i] + by[i]
            var mn = sqrt(mx * mx + my * my)
            if mn < _mjminval[DTYPE]():
                good[i] = Scalar[DTYPE](0)
            else:
                good[i] = (mx / mn) * sx + (my / mn) * sy
        else:
            var cx = ax[i] - bx[i]
            var cy = ay[i] - by[i]
            good[i] = -(cx * cx + cy * cy)

        if _is_intersect[DTYPE](
            e0x, e0y, ax[i], ay[i], e1x, e1y, bx[i], by[i]
        ):
            good[i] = Scalar[DTYPE](-10000)

    var best = 0 if good[0] > good[1] else 1
    var b_ax = ax[best]
    var b_ay = ay[best]
    var b_bx = bx[best]
    var b_by = by[best]

    if _is_intersect[DTYPE](e0x, e0y, b_ax, b_ay, e1x, e1y, b_bx, b_by):
        return _Pnt4[DTYPE](Scalar[DTYPE](-1))

    var out = _Pnt4[DTYPE](
        _length_circle[DTYPE](b_ax, b_ay, b_bx, b_by, best, radius)
    )
    out.ax = b_ax
    out.ay = b_ay
    out.bx = b_bx
    out.by = b_by
    return out^


def _wrap_inside[
    DTYPE: DType
](
    e0x: Scalar[DTYPE], e0y: Scalar[DTYPE],
    e1x: Scalar[DTYPE], e1y: Scalar[DTYPE],
    radius: Scalar[DTYPE],
) -> _Pnt4[DTYPE]:
    """The INSIDE wrap (`wrap_inside`): the tendon touches the circle from within.

    Both wrap points coincide — the contact is a single point, so the arc
    length is 0 and only the position matters. Newton on
    `asin(A z) + asin(B z) - 2 asin(z) + G = 0`.
    """
    # `sin`/`cos` need the proof; `sqrt`/`asin`/`acos` do not, which is why
    # this is the only function that carries it.
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    comptime MAXITER = 20
    var zinit = Scalar[DTYPE](1) - Scalar[DTYPE](1e-7)
    var tolerance = Scalar[DTYPE](1e-6)

    var len0 = sqrt(e0x * e0x + e0y * e0y)
    var len1 = sqrt(e1x * e1x + e1y * e1y)
    var dx = e1x - e0x
    var dy = e1y - e0y
    var dd = dx * dx + dy * dy

    if (
        len0 <= radius
        or len1 <= radius
        or radius < _mjminval[DTYPE]()
        or len0 < _mjminval[DTYPE]()
        or len1 < _mjminval[DTYPE]()
    ):
        return _Pnt4[DTYPE](Scalar[DTYPE](-1))

    # The straight segment already cuts the circle: nothing to wrap around.
    if dd > _mjminval[DTYPE]():
        var a = -(dx * e0x + dy * e0y) / dd
        if a > 0 and a < 1:
            var tx = e0x + a * dx
            var ty = e0y + a * dy
            if sqrt(tx * tx + ty * ty) <= radius:
                return _Pnt4[DTYPE](Scalar[DTYPE](-1))

    # The fallback if Newton fails: the bisector, scaled to the radius.
    var fx = Scalar[DTYPE](0.5) * (e0x + e1x)
    var fy = Scalar[DTYPE](0.5) * (e0y + e1y)
    var fn_ = sqrt(fx * fx + fy * fy)
    if fn_ > _mjminval[DTYPE]():
        fx = fx / fn_ * radius
        fy = fy / fn_ * radius
    var out = _Pnt4[DTYPE](Scalar[DTYPE](0))
    out.ax = fx
    out.ay = fy
    out.bx = fx
    out.by = fy

    var A = radius / len0
    var B = radius / len1
    var cosG = (len0 * len0 + len1 * len1 - dd) / (2 * len0 * len1)
    if cosG < Scalar[DTYPE](-1) + _mjminval[DTYPE]():
        return _Pnt4[DTYPE](Scalar[DTYPE](-1))
    elif cosG > Scalar[DTYPE](1) - _mjminval[DTYPE]():
        return out^
    var G = _acos_safe[DTYPE](cosG)

    var z = zinit
    var f = (
        _asin_safe[DTYPE](A * z)
        + _asin_safe[DTYPE](B * z)
        - 2 * _asin_safe[DTYPE](z)
        + G
    )
    if f > 0:
        return out^

    var it = 0
    while it < MAXITER and _abs[DTYPE](f) > tolerance:
        var d0 = sqrt(Scalar[DTYPE](1) - z * z * A * A)
        var d1 = sqrt(Scalar[DTYPE](1) - z * z * B * B)
        var d2 = sqrt(Scalar[DTYPE](1) - z * z)
        var df = (
            A / (d0 if d0 > _mjminval[DTYPE]() else _mjminval[DTYPE]())
            + B / (d1 if d1 > _mjminval[DTYPE]() else _mjminval[DTYPE]())
            - 2 / (d2 if d2 > _mjminval[DTYPE]() else _mjminval[DTYPE]())
        )
        if df > -_mjminval[DTYPE]():
            return out^
        var z1 = z - f / df
        if z1 > z:
            return out^
        z = z1
        f = (
            _asin_safe[DTYPE](A * z)
            + _asin_safe[DTYPE](B * z)
            - 2 * _asin_safe[DTYPE](z)
            + G
        )
        if f > tolerance:
            return out^
        it += 1

    if it >= MAXITER:
        return out^

    var vx: Scalar[DTYPE]
    var vy: Scalar[DTYPE]
    var ang: Scalar[DTYPE]
    if e0x * e1y - e0y * e1x > 0:
        vx = e0x
        vy = e0y
        ang = _asin_safe[DTYPE](z) - _asin_safe[DTYPE](A * z)
    else:
        vx = e1x
        vy = e1y
        ang = _asin_safe[DTYPE](z) - _asin_safe[DTYPE](B * z)
    var vn = sqrt(vx * vx + vy * vy)
    if vn > _mjminval[DTYPE]():
        vx /= vn
        vy /= vn
    out.ax = radius * (cos(ang) * vx - sin(ang) * vy)
    out.ay = radius * (sin(ang) * vx + cos(ang) * vy)
    out.bx = out.ax
    out.by = out.ay
    return out^


def mju_wrap[
    DTYPE: DType
](
    x0x: Scalar[DTYPE], x0y: Scalar[DTYPE], x0z: Scalar[DTYPE],
    x1x: Scalar[DTYPE], x1y: Scalar[DTYPE], x1z: Scalar[DTYPE],
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    m00: Scalar[DTYPE], m01: Scalar[DTYPE], m02: Scalar[DTYPE],
    m10: Scalar[DTYPE], m11: Scalar[DTYPE], m12: Scalar[DTYPE],
    m20: Scalar[DTYPE], m21: Scalar[DTYPE], m22: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    wtype: Int,
    has_side: Bool,
    sidex: Scalar[DTYPE], sidey: Scalar[DTYPE], sidez: Scalar[DTYPE],
) -> WrapOut[DTYPE]:
    """MuJoCo's `mju_wrap`. `wlen < 0` means the segment does not wrap.

    `m??` is the geom's world rotation in ROW-major order, matching MuJoCo's
    `geom_xmat`; the mapping into the local frame is `mat^T * (x - xpos)`.
    """
    var none = WrapOut[DTYPE]()
    if wtype != WRAP_SPHERE and wtype != WRAP_CYLINDER:
        return none^

    # x -> local frame (mat^T applied to the offset).
    var t0x = x0x - gx
    var t0y = x0y - gy
    var t0z = x0z - gz
    var p0x = m00 * t0x + m10 * t0y + m20 * t0z
    var p0y = m01 * t0x + m11 * t0y + m21 * t0z
    var p0z = m02 * t0x + m12 * t0y + m22 * t0z

    var t1x = x1x - gx
    var t1y = x1y - gy
    var t1z = x1z - gz
    var p1x = m00 * t1x + m10 * t1y + m20 * t1z
    var p1y = m01 * t1x + m11 * t1y + m21 * t1z
    var p1z = m02 * t1x + m12 * t1y + m22 * t1z

    var n0 = sqrt(p0x * p0x + p0y * p0y + p0z * p0z)
    var n1 = sqrt(p1x * p1x + p1y * p1y + p1z * p1z)
    if n0 < _mjminval[DTYPE]() or n1 < _mjminval[DTYPE]():
        return none^

    # ── the 2D frame ─────────────────────────────────────────────────────
    var a0x: Scalar[DTYPE]
    var a0y: Scalar[DTYPE]
    var a0z: Scalar[DTYPE]
    var a1x: Scalar[DTYPE]
    var a1y: Scalar[DTYPE]
    var a1z: Scalar[DTYPE]

    if wtype == WRAP_SPHERE:
        # A sphere has no preferred axis, so the plane is the one through the
        # origin and both endpoints.
        a0x = p0x / n0
        a0y = p0y / n0
        a0z = p0z / n0

        var nx = p0y * p1z - p0z * p1y
        var ny = p0z * p1x - p0x * p1z
        var nz = p0x * p1y - p0y * p1x
        var nn = sqrt(nx * nx + ny * ny + nz * nz)

        if nn < _mjminval[DTYPE]():
            # ⚠ COLLINEAR ENDPOINTS HAVE NO PLANE. MuJoCo picks a second axis
            # by zeroing the LARGEST component of axis0 and taking ones
            # elsewhere, which is guaranteed non-parallel to it.
            var i = 0
            if _abs[DTYPE](a0y) > _abs[DTYPE](a0x) and _abs[DTYPE](a0y) > _abs[
                DTYPE
            ](a0z):
                i = 1
            if _abs[DTYPE](a0z) > _abs[DTYPE](a0x) and _abs[DTYPE](a0z) > _abs[
                DTYPE
            ](a0y):
                i = 2
            var bx = Scalar[DTYPE](0) if i == 0 else Scalar[DTYPE](1)
            var by = Scalar[DTYPE](0) if i == 1 else Scalar[DTYPE](1)
            var bz = Scalar[DTYPE](0) if i == 2 else Scalar[DTYPE](1)
            nx = a0y * bz - a0z * by
            ny = a0z * bx - a0x * bz
            nz = a0x * by - a0y * bx
            nn = sqrt(nx * nx + ny * ny + nz * nz)

        if nn > _mjminval[DTYPE]():
            nx /= nn
            ny /= nn
            nz /= nn

        a1x = ny * a0z - nz * a0y
        a1y = nz * a0x - nx * a0z
        a1z = nx * a0y - ny * a0x
        var an = sqrt(a1x * a1x + a1y * a1y + a1z * a1z)
        if an > _mjminval[DTYPE]():
            a1x /= an
            a1y /= an
            a1z /= an
    else:
        # A cylinder wraps in its OWN xy-plane; z is recovered afterwards.
        a0x = Scalar[DTYPE](1)
        a0y = Scalar[DTYPE](0)
        a0z = Scalar[DTYPE](0)
        a1x = Scalar[DTYPE](0)
        a1y = Scalar[DTYPE](1)
        a1z = Scalar[DTYPE](0)

    var d0 = p0x * a0x + p0y * a0y + p0z * a0z
    var d1 = p0x * a1x + p0y * a1y + p0z * a1z
    var d2 = p1x * a0x + p1y * a0y + p1z * a0z
    var d3 = p1x * a1x + p1y * a1y + p1z * a1z

    # ── the sidesite, in the same frame ──────────────────────────────────
    var sdx = Scalar[DTYPE](0)
    var sdy = Scalar[DTYPE](0)
    var s_norm = Scalar[DTYPE](0)
    if has_side:
        var ux = sidex - gx
        var uy = sidey - gy
        var uz = sidez - gz
        var slx = m00 * ux + m10 * uy + m20 * uz
        var sly = m01 * ux + m11 * uy + m21 * uz
        var slz = m02 * ux + m12 * uy + m22 * uz
        s_norm = sqrt(slx * slx + sly * sly + slz * slz)
        sdx = slx * a0x + sly * a0y + slz * a0z
        sdy = slx * a1x + sly * a1y + slz * a1z
        var sn = sqrt(sdx * sdx + sdy * sdy)
        if sn > _mjminval[DTYPE]():
            sdx = sdx / sn * radius
            sdy = sdy / sn * radius

    var sol: _Pnt4[DTYPE]
    if has_side and s_norm < radius:
        sol = _wrap_inside[DTYPE](d0, d1, d2, d3, radius)
    else:
        sol = _wrap_circle[DTYPE](d0, d1, d2, d3, has_side, sdx, sdy, radius)

    var wlen = sol.len
    if wlen < 0:
        return none^

    # ── back to 3D in the local frame ────────────────────────────────────
    var r0x = a0x * sol.ax + a1x * sol.ay
    var r0y = a0y * sol.ax + a1y * sol.ay
    var r0z = a0z * sol.ax + a1z * sol.ay
    var r1x = a0x * sol.bx + a1x * sol.by
    var r1y = a0y * sol.bx + a1y * sol.by
    var r1z = a0z * sol.bx + a1z * sol.by

    if wtype == WRAP_CYLINDER:
        # ⚠ THE ARC WAS SOLVED FLAT; the tendon is a helix. z is interpolated
        # by UNROLLED path length (chord + arc + chord), then the arc length
        # is corrected by the height it climbed.
        var l0 = sqrt(
            (p0x - r0x) * (p0x - r0x) + (p0y - r0y) * (p0y - r0y)
        )
        var l1 = sqrt(
            (p1x - r1x) * (p1x - r1x) + (p1y - r1y) * (p1y - r1y)
        )
        var tot = l0 + wlen + l1
        if tot > _mjminval[DTYPE]():
            r0z = p0z + (p1z - p0z) * l0 / tot
            r1z = p0z + (p1z - p0z) * (l0 + wlen) / tot
        else:
            r0z = p0z
            r1z = p0z
        var height = _abs[DTYPE](r1z - r0z)
        wlen = sqrt(wlen * wlen + height * height)

    # ── back to the world ────────────────────────────────────────────────
    var out = WrapOut[DTYPE]()
    out.wlen = wlen
    out.p0x = m00 * r0x + m01 * r0y + m02 * r0z + gx
    out.p0y = m10 * r0x + m11 * r0y + m12 * r0z + gy
    out.p0z = m20 * r0x + m21 * r0y + m22 * r0z + gz
    out.p1x = m00 * r1x + m01 * r1y + m02 * r1z + gx
    out.p1y = m10 * r1x + m11 * r1y + m12 * r1z + gy
    out.p1z = m20 * r1x + m21 * r1y + m22 * r1z + gz
    return out^
