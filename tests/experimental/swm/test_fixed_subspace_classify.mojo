"""G22 — SWM Phase 7: the verdict carries `dim ker(H - I)`.

6b measured that `det H` under-reports above two dimensions — an O(3)
reflection and O(3) `-I` are both `det = -1` and fix a plane and nothing
respectively — but `classify` consumed only the determinant, so the finding
never reached a verdict. `classify_cycle` takes the holonomy itself and returns
the class together with the fixed-subspace dimension.

Gated: the three `det = -1` cases classify IDENTICALLY through the old path
(that is the under-report, shown rather than assumed) and DIFFERENTLY through
the new one; an O(3) rotation is UNDECIDED with a one-dimensional fixed axis;
the identity is NOMINAL with the full space fixed; and an outlier residual
still dominates every reading.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_fixed_subspace_classify.mojo
"""

from std.math import cos, sin
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat, householder
from mojo_rl.experimental.swm.observables import (
    classify,
    classify_cycle,
    CLASS_NOMINAL,
    CLASS_ABERRANT,
    CLASS_OBSTRUCTION,
    CLASS_UNDECIDED,
)

comptime DT = DType.float64
comptime TOL = 0.2


def rot3z(t: Float64) -> SqMat[3, DT]:
    var m = SqMat[3, DT].identity()
    m[0, 0] = Scalar[DT](cos(t))
    m[0, 1] = Scalar[DT](-sin(t))
    m[1, 0] = Scalar[DT](sin(t))
    m[1, 1] = Scalar[DT](cos(t))
    return m^


def main() raises:
    var checks = 0
    var refl2 = SqMat[2, DT].identity()
    refl2[1, 1] = Scalar[DT](-1)
    var refl3 = SqMat[3, DT].identity()
    refl3[2, 2] = Scalar[DT](-1)
    var neg3 = SqMat[3, DT].identity().scaled(Scalar[DT](-1))
    var eye3 = SqMat[3, DT].identity()
    var rz = rot3z(0.5)

    var v2 = classify_cycle[2, DT](0.01, 0.01, refl2, TOL, False)
    var v3r = classify_cycle[3, DT](0.01, 0.01, refl3, TOL, False)
    var v3n = classify_cycle[3, DT](0.01, 0.01, neg3, TOL, False)
    var v3i = classify_cycle[3, DT](0.01, 0.01, eye3, TOL, False)
    var v3z = classify_cycle[3, DT](0.01, 0.01, rz, TOL, False)
    var v3o = classify_cycle[3, DT](1.0, 0.01, neg3, TOL, False)
    print("O(2) reflection :", v2.describe())
    print("O(3) reflection :", v3r.describe())
    print("O(3) -I         :", v3n.describe())
    print("O(3) identity   :", v3i.describe())
    print("O(3) rot z 0.5  :", v3z.describe())
    print("O(3) -I, outlier:", v3o.describe())

    # The under-report, shown: the old path cannot tell the three apart.
    var c2 = classify(0.01, 0.01, Float64(refl2.det()), Float64(refl2.dist_to_identity()), TOL, False)
    var c3r = classify(0.01, 0.01, Float64(refl3.det()), Float64(refl3.dist_to_identity()), TOL, False)
    var c3n = classify(0.01, 0.01, Float64(neg3.det()), Float64(neg3.dist_to_identity()), TOL, False)
    checks += 8
    assert_true(
        c2 == CLASS_OBSTRUCTION and c3r == CLASS_OBSTRUCTION and c3n == CLASS_OBSTRUCTION,
        "CONTROL: det-only classification files all three det = -1 cases as "
        + "the same OBSTRUCTION — that is the under-report",
    )
    assert_true(
        v2.cls == CLASS_OBSTRUCTION and v3r.cls == CLASS_OBSTRUCTION
        and v3n.cls == CLASS_OBSTRUCTION,
        "the class itself must not change",
    )
    assert_true(v2.fixed_dim == 1, "an O(2) reflection fixes a LINE")
    assert_true(v3r.fixed_dim == 2, "an O(3) reflection fixes a PLANE")
    assert_true(v3n.fixed_dim == 0, "O(3) -I fixes NOTHING — the doc's own example")
    assert_true(
        v3i.cls == CLASS_NOMINAL and v3i.fixed_dim == 3,
        "the identity is NOMINAL with the whole space fixed",
    )
    assert_true(
        v3z.cls == CLASS_UNDECIDED and v3z.fixed_dim == 1,
        "a rotation is UNDECIDED by one cycle and fixes its AXIS: "
        + v3z.describe(),
    )
    assert_true(
        v3o.cls == CLASS_ABERRANT and v3o.fixed_dim == 0,
        "an outlier residual dominates the class; the fixed subspace is still "
        + "reported",
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G22 the verdict carries dim ker(H - I)")
