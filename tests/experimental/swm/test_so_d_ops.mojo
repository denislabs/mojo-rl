"""G1 — SWM Phase 1 gate: O(D) algebra and the polar Procrustes factor.

Everything downstream reads `det H in {+1, -1}` off products of these matrices,
so if the generators silently drift off the manifold, or the polar factor
quietly returns something that is not orthogonal, every later holonomy reading
is meaningless. This gate pins the generators before anything is built on them.

Validates:
  - `expm_skew(skew(t)) == rot(t)` and `cayley(skew(t)) == rot(-2 atan t)`
    in 2D, to 1e-12 — exact closed forms, the only legs that pin the VALUE
    rather than just membership in O(D)
  - `exp(S) == exp(S/2)^2` for a large generator (scaling-and-squaring)
  - `cayley` and `expm_skew` land in SO(D): `R^T R = I` and `det = +1`
  - `householder` lands in the other component: orthogonal with `det = -1`
  - `polar_orthogonal_factor` recovers a planted `O(D)` matrix from `M = R S`
    (S symmetric positive definite), in BOTH components
  - NEGATIVE CONTROL: a planted non-orthogonal matrix must be REJECTED by
    `is_orthogonal`, and its polar factor must differ from it. Without this
    leg the tolerance could be vacuous and every assertion above would pass
    on a broken implementation.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_so_d_ops.mojo
"""

from std.collections import InlineArray
from std.math import abs, sqrt, cos, sin, atan
from std.random import seed, random_float64
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import (
    SqMat,
    skew_from_vector,
    cayley,
    expm_skew,
    householder,
)
from mojo_rl.experimental.swm.procrustes import polar_orthogonal_factor

comptime DT = DType.float64
comptime ORTHO_TOL = 1e-12
comptime DET_TOL = 1e-10
comptime RECOVER_TOL = 1e-10


def random_skew[D: Int](scale: Float64) raises -> SqMat[D, DT]:
    comptime P = D * (D - 1) // 2
    var v = InlineArray[Scalar[DT], P](fill=0)
    for i in range(P):
        v[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * scale)
    return skew_from_vector[D, DT](Span(v))


def random_spd[D: Int]() -> SqMat[D, DT]:
    """`B^T B + I` — symmetric positive definite, so `M = R S` is nonsingular."""
    var b = SqMat[D, DT]()
    for i in range(D):
        for j in range(D):
            b[i, j] = Scalar[DT](random_float64() * 2.0 - 1.0)
    return b.transpose() * b + SqMat[D, DT].identity()


def check_generators[D: Int](mut checks: Int) raises:
    """Cayley / expm land in SO(D); householder lands in det = -1."""
    var s = random_skew[D](0.7)

    var r_cay = cayley[D, DT](s)
    checks += 1
    assert_true(
        Float64(r_cay.orthogonality_error()) <= ORTHO_TOL,
        "cayley: R^T R != I at D=" + String(D),
    )
    checks += 1
    assert_true(
        abs(Float64(r_cay.det()) - 1.0) <= DET_TOL,
        "cayley: det != +1 at D=" + String(D),
    )

    var r_exp = expm_skew[D, DT](s)
    checks += 1
    assert_true(
        Float64(r_exp.orthogonality_error()) <= ORTHO_TOL,
        "expm_skew: R^T R != I at D=" + String(D),
    )
    checks += 1
    assert_true(
        abs(Float64(r_exp.det()) - 1.0) <= DET_TOL,
        "expm_skew: det != +1 at D=" + String(D),
    )

    # Scaling-and-squaring correctness. NOTE: an orthogonality check CANNOT
    # see this bug — dropping the squaring returns exp(S/2^k), which is still
    # perfectly orthogonal, just the wrong rotation. (Measured: a mutant that
    # skipped the squaring passed an orthogonality-only version of this leg.)
    # The semigroup property exp(S) = exp(S/2)^2 is what actually pins it.
    var big = random_skew[D](12.0)
    var e_full = expm_skew[D, DT](big)
    var e_half = expm_skew[D, DT](big.scaled(Scalar[DT](0.5)))
    checks += 1
    assert_true(
        Float64(e_full.max_abs_diff(e_half * e_half)) <= 1e-11,
        "expm_skew: exp(S) != exp(S/2)^2 at D=" + String(D),
    )
    checks += 1
    assert_true(
        Float64(e_full.orthogonality_error()) <= ORTHO_TOL,
        "expm_skew: drifted off O(D) for a large generator at D=" + String(D),
    )

    var v = InlineArray[Scalar[DT], D](fill=0)
    for i in range(D):
        v[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    v[0] = v[0] + 1.5  # keep it away from the zero vector
    var q = householder[D, DT](Span(v))
    checks += 1
    assert_true(
        Float64(q.orthogonality_error()) <= ORTHO_TOL,
        "householder: Q^T Q != I at D=" + String(D),
    )
    checks += 1
    assert_true(
        abs(Float64(q.det()) + 1.0) <= DET_TOL,
        "householder: det != -1 at D=" + String(D),
    )

    # The two components must not be reachable from each other: this is exactly
    # why the transport carries a discrete orientation bit (v2 §4.2).
    checks += 1
    assert_true(
        Float64((q * r_cay).det()) < 0.0,
        "Q R must sit in the det = -1 component at D=" + String(D),
    )


def check_polar_recovery[D: Int](mut checks: Int) raises:
    """Polar factor of `M = R S` must return `R`, in both components."""
    var vh = InlineArray[Scalar[DT], D](fill=0)
    vh[0] = 1
    vh[1] = Scalar[DT](0.37)
    var q = householder[D, DT](Span(vh))

    for branch in range(2):
        var r_true = expm_skew[D, DT](random_skew[D](0.9))
        if branch == 1:
            r_true = q * r_true
        var expected_det = 1.0 if branch == 0 else -1.0

        var m = r_true * random_spd[D]()
        var r_fit = polar_orthogonal_factor[D, DT](m)

        checks += 1
        assert_true(
            Float64(r_fit.max_abs_diff(r_true)) <= RECOVER_TOL,
            "polar factor did not recover the planted R at D="
            + String(D)
            + " branch="
            + String(branch),
        )
        checks += 1
        assert_true(
            abs(Float64(r_fit.det()) - expected_det) <= DET_TOL,
            "polar factor lost the reflection at D="
            + String(D)
            + " branch="
            + String(branch),
        )


def check_negative_control[D: Int](mut checks: Int) raises:
    """The tolerance must have teeth: a non-orthogonal matrix must be rejected.

    Without this leg, `is_orthogonal` returning True unconditionally would pass
    every assertion above.
    """
    var m = expm_skew[D, DT](random_skew[D](0.6))
    # Stretch one axis by 1.05 — still well-conditioned, plainly not in O(D).
    for j in range(D):
        m[0, j] = m[0, j] * Scalar[DT](1.05)

    checks += 1
    assert_true(
        not m.is_orthogonal(ORTHO_TOL),
        "NEGATIVE CONTROL FAILED: a stretched matrix passed is_orthogonal at D="
        + String(D),
    )
    checks += 1
    assert_true(
        Float64(m.orthogonality_error()) > 1e-3,
        "NEGATIVE CONTROL FAILED: stretch was invisible at D=" + String(D),
    )

    # ...and its polar factor must actually move it back onto the manifold.
    var r = polar_orthogonal_factor[D, DT](m)
    checks += 1
    assert_true(
        Float64(r.orthogonality_error()) <= ORTHO_TOL,
        "polar factor of a non-orthogonal matrix is not orthogonal at D="
        + String(D),
    )
    checks += 1
    assert_true(
        Float64(r.max_abs_diff(m)) > 1e-3,
        "NEGATIVE CONTROL FAILED: polar factor was a no-op at D=" + String(D),
    )


def skew2(t: Float64) raises -> SqMat[2, DT]:
    """`[[0, -t], [t, 0]]` — the single generator of so(2)."""
    comptime P = 2 * (2 - 1) // 2
    var v = InlineArray[Scalar[DT], P](fill=Scalar[DT](t))
    return skew_from_vector[2, DT](Span(v))


def check_closed_form_2d(mut checks: Int) raises:
    """Exact analytic targets in 2D — the only legs that pin the VALUE.

    Every other assertion in this file checks that a result sits on the
    manifold. Orthogonality is preserved by a whole family of wrong answers
    (any rotation is orthogonal), so without these two the generators could
    return the wrong angle and nothing would notice.

      skew(t) = [[0, -t], [t, 0]]
      exp(skew(t))    = rot(t)
      cayley(skew(t)) = rot(-2 atan t)      with rot(a) = [[cos a, -sin a],
                                                           [sin a,  cos a]]
    """
    var angles: List[Float64] = [0.05, 0.7, 2.9, 11.3]
    for idx in range(len(angles)):
        var t = angles[idx]
        var s = skew2(t)

        var want_exp = SqMat[2, DT]()
        want_exp[0, 0] = Scalar[DT](cos(t))
        want_exp[0, 1] = Scalar[DT](-sin(t))
        want_exp[1, 0] = Scalar[DT](sin(t))
        want_exp[1, 1] = Scalar[DT](cos(t))
        checks += 1
        assert_true(
            Float64(expm_skew[2, DT](s).max_abs_diff(want_exp)) <= 1e-12,
            "expm_skew(skew(t)) != rot(t) at t=" + String(t),
        )

        var a = -2.0 * atan(t)
        var want_cay = SqMat[2, DT]()
        want_cay[0, 0] = Scalar[DT](cos(a))
        want_cay[0, 1] = Scalar[DT](-sin(a))
        want_cay[1, 0] = Scalar[DT](sin(a))
        want_cay[1, 1] = Scalar[DT](cos(a))
        checks += 1
        assert_true(
            Float64(cayley[2, DT](s).max_abs_diff(want_cay)) <= 1e-12,
            "cayley(skew(t)) != rot(-2 atan t) at t=" + String(t),
        )


def main() raises:
    seed(20260904)
    var checks = 0

    check_closed_form_2d(checks)

    check_generators[2](checks)
    check_generators[3](checks)
    check_generators[8](checks)
    check_generators[32](checks)

    check_polar_recovery[2](checks)
    check_polar_recovery[3](checks)
    check_polar_recovery[8](checks)
    check_polar_recovery[32](checks)

    check_negative_control[2](checks)
    check_negative_control[3](checks)
    check_negative_control[8](checks)
    check_negative_control[32](checks)

    print("dims exercised     : 2, 3, 8, 32")
    print("assertions compared:", checks)
    print(
        "tolerances         : ortho <=",
        ORTHO_TOL,
        " det <=",
        DET_TOL,
        " recover <=",
        RECOVER_TOL,
    )
    print("PASS: G1 O(D) algebra + polar Procrustes factor")
