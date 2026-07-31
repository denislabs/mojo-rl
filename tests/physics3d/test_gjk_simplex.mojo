"""Unit gate: GJK simplex handling — `_closest_point_on_simplex` + a flat-face
end-to-end query.

Two defects lived in the nsimplex==4 branch, both silent and both reachable
from ordinary geometry:

  1. FLAT SIMPLEX READ AS AN ENCLOSURE. The outside test was the bare product
     `dot_opp * dot_origin < 0`. For a COPLANAR quadruple every face has
     dot_opp == 0 exactly, so no face is ever flagged, and the routine falls
     through to "origin is inside the tetrahedron" — v = 0, which `gjk_epa`
     reads as penetration. GJK converges onto a planar facet whenever the
     closest feature is one (a hull face parallel to a box or cylinder cap), so
     this is the common case, not a tie: it invented contacts between geoms
     centimetres apart. It cost the SawyerReach mesh gate its meaning — that
     gate's "obj teleported into the gripper hull" pose is 15.1 mm CLEAR of the
     hull, and the contact it asserted was fabricated.

  2. ALIASED REDUCTION. Reducing to the winning face copied slot-by-slot in
     place. Face ADB is (i0,i1,i2) = (0,27,9): slot 1 is overwritten with D and
     then read back as "B", retaining {A,D,D}. A degenerate simplex stalls the
     next iteration.

The tetrahedra below are hand-built so each verdict is checkable by hand.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_gjk_simplex.mojo
"""

from std.math import abs, sqrt
from layout import Layout, LayoutTensor
from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.collision.gjk import gjk_epa
from mojo_rl.physics3d.collision.gjk_support import _closest_point_on_simplex
from mojo_rl.physics3d.constants import GEOM_CYLINDER, GEOM_MESH

comptime DT = DType.float64
comptime NMV = 8
comptime L_MV = Layout.row_major(NMV, 3)


def _put(
    mut s: InlineArray[Scalar[DT], 36],
    slot: Int,
    x: Float64,
    y: Float64,
    z: Float64,
    tag: Float64,
) -> None:
    """Write one simplex vertex; the 6 witness slots carry `tag` so that a
    partial or aliased copy is visible, not just a wrong xyz."""
    s[slot * 9 + 0] = Scalar[DT](x)
    s[slot * 9 + 1] = Scalar[DT](y)
    s[slot * 9 + 2] = Scalar[DT](z)
    for k in range(3, 9):
        s[slot * 9 + k] = Scalar[DT](tag)


def _xyz(s: InlineArray[Scalar[DT], 36], slot: Int) -> Tuple[
    Float64, Float64, Float64
]:
    return (
        Float64(s[slot * 9 + 0]),
        Float64(s[slot * 9 + 1]),
        Float64(s[slot * 9 + 2]),
    )


def test_flat_simplex_is_not_an_enclosure() raises:
    """Four COPLANAR points at z = 0.5 cannot enclose the origin."""
    print("=== flat simplex is not an enclosure ===")
    var s = InlineArray[Scalar[DT], 36](fill=Scalar[DT](0))
    _put(s, 0, -1.0, -1.0, 0.5, 100.0)
    _put(s, 1, 1.0, -1.0, 0.5, 200.0)
    _put(s, 2, 1.0, 1.0, 0.5, 300.0)
    _put(s, 3, -1.0, 1.0, 0.5, 400.0)

    var cp = _closest_point_on_simplex[DT](s, 4)
    var n = Int(cp[3])
    print("  n =", n, " v =", Float64(cp[0]), Float64(cp[1]), Float64(cp[2]))
    if n == 4:
        raise Error(
            "flat simplex reported as enclosing the origin — gjk_epa will read"
            " this as penetration"
        )
    # Closest point is the origin's projection onto the plane: (0, 0, 0.5).
    if (
        abs(Float64(cp[0])) > 1e-12
        or abs(Float64(cp[1])) > 1e-12
        or abs(Float64(cp[2]) - 0.5) > 1e-12
    ):
        raise Error("closest point on the z=0.5 plane should be (0,0,0.5)")
    print("  PASS")


def test_true_enclosure_still_detected() raises:
    """A tetrahedron that genuinely contains the origin must still report it —
    the flat-simplex fix must not swing the other way."""
    print("=== true enclosure still detected ===")
    var s = InlineArray[Scalar[DT], 36](fill=Scalar[DT](0))
    # Regular-ish tetra straddling the origin.
    _put(s, 0, 1.0, 1.0, 1.0, 100.0)
    _put(s, 1, 1.0, -1.0, -1.0, 200.0)
    _put(s, 2, -1.0, 1.0, -1.0, 300.0)
    _put(s, 3, -1.0, -1.0, 1.0, 400.0)

    var cp = _closest_point_on_simplex[DT](s, 4)
    print("  n =", Int(cp[3]))
    if Int(cp[3]) != 4:
        raise Error("origin IS inside this tetrahedron — enclosure missed")
    print("  PASS")


def test_face_adb_reduction_is_not_aliased() raises:
    """Origin outside face ADB only; the reduction must keep {A, D, B}.

    A=(1,0,0) D=(0,1,0) B=(0,0,1) span x+y+z=1; the origin is at 0 and the
    opposite vertex C=(1,1,1) at 3, so ADB is the one face the origin is
    outside of. The reduced simplex must be those three vertices — the in-place
    copy used to yield {A, D, D}.
    """
    print("=== face-ADB reduction is not aliased ===")
    var s = InlineArray[Scalar[DT], 36](fill=Scalar[DT](0))
    _put(s, 0, 1.0, 0.0, 0.0, 100.0)  # A
    _put(s, 1, 0.0, 0.0, 1.0, 200.0)  # B
    _put(s, 2, 1.0, 1.0, 1.0, 300.0)  # C
    _put(s, 3, 0.0, 1.0, 0.0, 400.0)  # D

    var cp = _closest_point_on_simplex[DT](s, 4)
    if Int(cp[3]) != 3:
        raise Error("expected a reduction to the ADB face (n == 3)")
    # v = projection of the origin onto x+y+z=1.
    for k in range(3):
        if abs(Float64(cp[k]) - 1.0 / 3.0) > 1e-12:
            raise Error("closest point on ADB should be (1/3, 1/3, 1/3)")

    var v0 = _xyz(s, 0)
    var v1 = _xyz(s, 1)
    var v2 = _xyz(s, 2)
    print("  kept:", v0[0], v0[1], v0[2], "|", v1[0], v1[1], v1[2],
          "|", v2[0], v2[1], v2[2])
    if v0[0] != 1.0 or v1[1] != 1.0 or v2[2] != 1.0:
        raise Error("reduced simplex is not {A, D, B} — aliased copy")
    # Witness tags must travel with their vertex.
    if (
        Float64(s[0 * 9 + 3]) != 100.0
        or Float64(s[1 * 9 + 3]) != 400.0
        or Float64(s[2 * 9 + 3]) != 200.0
    ):
        raise Error("witness points did not follow their vertices")
    print("  PASS")


def _plate(z: Float64) raises -> TensorImpl[DT]:
    """A flat 4-corner plate at height `z` — 4 coplanar mesh verts."""
    var t = TensorImpl[DT].alloc(NMV * 3)
    var xs = [-0.5, 0.5, 0.5, -0.5]
    var ys = [-0.5, -0.5, 0.5, 0.5]
    for i in range(4):
        t.data[i * 3 + 0] = xs[i]
        t.data[i * 3 + 1] = ys[i]
        t.data[i * 3 + 2] = z
    return t^


def test_flat_face_query_reports_exact_gap() raises:
    """End-to-end: a cylinder cap coaxial over a flat plate, gap exactly 0.7.

    Parallel flat features are what drive GJK onto a coplanar simplex, and the
    cap-over-plate arrangement is the synthetic twin of the SawyerReach query
    that exposed this (cylinder obj vs a gripper-hull facet). The gap is
    analytic — cylinder centre 0.72, half-length 0.02, plate at 0 — so a wrong
    simplex shows up as a wrong NUMBER, not just a wrong sign: the old code
    returned 0.69339 here.
    """
    print("=== cylinder cap over plate reports the exact gap ===")
    var mv = _plate(0.0)
    var result = gjk_epa[DT, NMV](
        GEOM_CYLINDER,
        0.0, 0.0, 0.72,
        0.0, 0.0, 0.0, 1.0,
        0.2, 0.02,  # radius, half-length
        0.0, 0.0, 0.0,
        mv.lt["cpu", L_MV](), 0, 0,
        GEOM_MESH,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0, 0.0,
        0, 4,
    )
    var dist = Float64(result[0])
    print("  dist =", dist, "(expected 0.7)")
    if dist < 0:
        raise Error(
            "separated cylinder/plate reported as penetrating — flat simplex"
            " read as an enclosure"
        )
    if abs(dist - 0.7) > 1e-9:
        raise Error("expected a 0.7 gap, got " + String(dist))
    print("  PASS")


def main() raises:
    test_flat_simplex_is_not_an_enclosure()
    test_true_enclosure_still_detected()
    test_face_adb_reduction_is_not_aliased()
    test_flat_face_query_reports_exact_gap()
    print("test_gjk_simplex: ALL PASS")
