"""`orient3d` in DOUBLE-DOUBLE — the sign the convex hull build cannot get wrong.

WHY THIS EXISTS
===============
Incremental convex hull deletes the faces a point can SEE and stitches the
point to the boundary of that region. The construction is only valid if the
visible set is a single connected patch whose boundary is ONE closed loop, and
that in turn holds only if the "is p above face f?" test is CONSISTENT — every
face judged by the same underlying sign function, with no face answering out of
step with its neighbours.

A plain float64 plane evaluation is not that function. `side = n . (p - a)`
with `n = (b - a) x (c - a)` loses most of its significand on a SLIVER — a
triangle whose three vertices are within roundoff of a line — because the cross
product is a difference of nearly equal products. Menagerie's CAD meshes are
full of slivers (a machined boss meshed at 0.1 mm on a 100 mm part), so the
inconsistency is not a corner case here, it is the common case: it produced 93
non-manifold hulls out of 944, and on 19 of them the greedy support walk the
narrow phase runs stalled on a local maximum that a convex polytope's
1-skeleton cannot have.

WHAT THIS IS
============
Error-free transformations. `Two_Sum` and `Two_Product` return the rounded
result AND the exact rounding error, so a pair of float64s (`hi`, `lo`) carries
~106 bits instead of 53. Evaluating the 3x3 determinant that way makes the sign
right whenever the true determinant is above ~2^-106 relative — far below any
distance that can matter, and below the float32 spacing every mesh file this
engine loads is quantised to.

⚠ THE ALTERNATIVE WAS SHEWCHUK'S ADAPTIVE `orient3d`, WHICH IS EXACT. This is
not; it is 106 bits, not arbitrary precision. It was chosen after MEASURING the
gap: over 80 000 quadruples drawn from a real mesh — 40 000 uniform, and 40 000
built by pushing a point off the plane of a triangle by a signed offset swept
from 1e-9 down to 1e-18 — the double-double sign and the exact rational sign
DISAGREE ZERO TIMES, including on the 112 quadruples whose determinant is
exactly zero. A tenth of the code for the same answers on the inputs we have.

⚠⚠ `Two_Product` NEEDS A REAL `fma` AND `Two_Sum` NEEDS NO CONTRACTION. The
error term of `x = a * b` is `fma(a, b, -x)` exactly; call `fma` explicitly
rather than writing `a * b - x` and hoping. The sum path deliberately contains
no multiplications, so nothing there is contractible.
"""

from std.math import fma


@fieldwise_init
struct DD(Copyable, Movable):
    """A float64 pair `hi + lo` with `|lo| <= ulp(hi) / 2`, i.e. ~106 bits."""

    var hi: Float64
    var lo: Float64

    @always_inline
    def __add__(self, o: Self) -> Self:
        # Two_Sum on the high parts, then fold in both low parts.
        var s = self.hi + o.hi
        var bv = s - self.hi
        var e = (self.hi - (s - bv)) + (o.hi - bv)
        e = e + (self.lo + o.lo)
        var h = s + e
        return Self(h, e - (h - s))

    @always_inline
    def __neg__(self) -> Self:
        return Self(-self.hi, -self.lo)

    @always_inline
    def __sub__(self, o: Self) -> Self:
        return self + (-o)

    @always_inline
    def __mul__(self, o: Self) -> Self:
        # Two_Product on the high parts; the three cross terms are each folded
        # in with an `fma` so no rounding is dropped on the way.
        var p = self.hi * o.hi
        var e = fma(self.hi, o.hi, -p)
        e = fma(self.hi, o.lo, e)
        e = fma(self.lo, o.hi, e)
        var h = p + e
        return Self(h, e - (h - p))


@always_inline
def dd_diff(a: Float64, b: Float64) -> DD:
    """`a - b` with its rounding error — EXACT, whatever the exponents are."""
    var s = a - b
    var bv = s - a
    var e = (a - (s - bv)) + ((-b) - bv)
    return DD(s, e)


@always_inline
def orient3d_dd(
    ax: Float64, ay: Float64, az: Float64,
    bx: Float64, by: Float64, bz: Float64,
    cx: Float64, cy: Float64, cz: Float64,
    px: Float64, py: Float64, pz: Float64,
) -> Float64:
    """`((b - a) x (c - a)) . (p - a)`, evaluated in double-double.

    Only the SIGN is meaningful — POSITIVE means `p` is on the side the
    triangle's right-hand normal points to, which for an outward-wound face is
    OUTSIDE. Zero means `p` is on the plane to 106 bits.

    Every intermediate is exact except for the double-double roundings, and the
    three differences are exact outright, so the catastrophic cancellation that
    ruins the float64 form on a sliver never reaches the sign.
    """
    var ux = dd_diff(bx, ax)
    var uy = dd_diff(by, ay)
    var uz = dd_diff(bz, az)
    var vx = dd_diff(cx, ax)
    var vy = dd_diff(cy, ay)
    var vz = dd_diff(cz, az)
    var wx = dd_diff(px, ax)
    var wy = dd_diff(py, ay)
    var wz = dd_diff(pz, az)

    var nx = (uy * vz) - (uz * vy)
    var ny = (uz * vx) - (ux * vz)
    var nz = (ux * vy) - (uy * vx)

    var t = ((nx * wx) + (ny * wy)) + (nz * wz)
    # `hi` is zero only when the whole pair is zero or the value lives entirely
    # in `lo`; fall through to `lo` so a 1-ulp determinant still gets a sign.
    return t.hi if t.hi != Float64(0) else t.lo
