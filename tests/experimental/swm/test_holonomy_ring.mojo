"""G2 + G3 — SWM Phase 1 gates: ring holonomy and the sheaf-Laplacian spectrum.

The claim these two gates pin is the one that decides which observable the
runtime reads (docs/SHEAF_WORLD_MODELS_V2.md §1.2, last paragraph):

  - the holonomy of a Mobius ring is `||H - I||_F = 2` and `det H = -1`
    **at every ring length** — an O(1) signal;
  - the spectral signature of the same obstruction is the gap
    `lambda_2 - lambda_1 = 2(1 - cos(pi/N))`, which decays as 1/N^2.

So on a long corridor the spectrum is under the noise and the holonomy is not.
`2(1 - cos(pi/N))` is a closed form, which makes G3 the cheapest gate in the
plan that can still catch a wrong Laplacian assembly.

Validates:
  G2  Mobius ring, N in {12, 24, 48, 96}: ||H-I||_F = 2 and det H = -1,
      independent of N; the orientable control gives det H = +1 and H ~ I.
  G2b `dim ker L` = 1 for Mobius (only the reflection's axis admits a global
      frame) vs D = 2 for the orientable control — the cohomological statement
      of the same fact.
  G3  lambda_1 = 0 and lambda_2 = 2(1 - cos(pi/N)) to 1e-9, at every N; and the
      measured gap shrinks by ~4x per doubling of N while ||H - I|| does not.
      The spectral leg stops at N=96 (Jacobi on 384x384 is too slow for a fast
      manifest); N=192 is exercised on the holonomy alone, which is exactly the
      cost asymmetry being claimed.
  NEGATIVE CONTROL: an orientable ring must NOT be reported as obstructed, and
      the Laplacian must be symmetric. Without the orientable leg, code that
      always answered "det = -1" would pass every assertion.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_holonomy_ring.mojo
"""

from std.collections import InlineArray
from std.math import abs, cos, sin, pi
from std.random import seed, random_float64
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat, householder
from mojo_rl.experimental.swm.place_graph import PlaceGraph, Edge
from mojo_rl.experimental.swm.sheaf_laplacian import (
    build_sheaf_laplacian,
    eigenvalues_ascending,
    kernel_dimension,
)

comptime DT = DType.float64
comptime D = 2


def rot2(t: Float64) -> SqMat[D, DT]:
    var m = SqMat[D, DT]()
    m[0, 0] = Scalar[DT](cos(t))
    m[0, 1] = Scalar[DT](-sin(t))
    m[1, 0] = Scalar[DT](sin(t))
    m[1, 1] = Scalar[DT](cos(t))
    return m^


def build_ring(n: Int, mobius: Bool) raises -> PlaceGraph[D, DT]:
    """Ring of `n` places, flat (angles sum to zero), optionally with a seam.

    Angles summing to zero means an ORIENTABLE ring has trivial holonomy
    exactly; any non-triviality then comes from the seam and nothing else.
    """
    var angles = List[Float64]()
    var total = Float64(0)
    for _ in range(n - 1):
        var t = (random_float64() * 2.0 - 1.0) * 0.6
        angles.append(t)
        total += t
    angles.append(-total)

    var refl = SqMat[D, DT].identity()
    refl[1, 1] = Scalar[DT](-1.0)  # diag(1, -1): the Mobius seam

    var g = PlaceGraph[D, DT]()
    for _ in range(n):
        _ = g.add_place()
    for i in range(n):
        var r = rot2(angles[i])
        if mobius and i == n - 1:
            r = refl * r
        _ = g.add_edge(Edge.action_edge(i, (i + 1) % n, 0), r)
    g.rebuild_gauge(0)
    return g^


def main() raises:
    seed(20260904)
    comptime FRO_TOL = 1e-10
    comptime DET_TOL = 1e-10
    comptime LAMBDA_TOL = 1e-9

    var sizes: List[Int] = [12, 24, 48, 96]
    var checks = 0
    var prev_gap = Float64(0)

    print("N | ||H-I||_F | det H | lambda_1 | lambda_2 | 2(1-cos(pi/N)) | dim ker L")
    for si in range(len(sizes)):
        var n = sizes[si]

        # ---- Mobius --------------------------------------------------------
        var g = build_ring(n, True)
        var cycles = g.fundamental_cycle_edges()
        checks += 1
        assert_true(
            len(cycles) == 1,
            "a ring must have exactly one fundamental cycle, got "
            + String(len(cycles)),
        )
        var e = cycles[0]
        var fro = g.holonomy_dist_to_identity(e)
        var det = g.holonomy_det(e)

        var lap = build_sheaf_laplacian[D, DT](g)
        checks += 1
        assert_true(
            lap.symmetry_error() <= 1e-12,
            "sheaf Laplacian is not symmetric at N=" + String(n),
        )
        var eigs = eigenvalues_ascending[DT](lap)
        var kdim = kernel_dimension(eigs)
        var closed = 2.0 * (1.0 - cos(pi / Float64(n)))

        print(
            n, " |H-I|=", fro, " det=", det,
            " l1=", eigs[0], " l2=", eigs[1],
            " closed=", closed, " dimker=", kdim,
        )

        # G2: the holonomy is O(1) in N.
        checks += 1
        assert_true(
            abs(fro - 2.0) <= FRO_TOL,
            "Mobius ||H-I||_F != 2 at N=" + String(n) + " (got " + String(fro) + ")",
        )
        checks += 1
        assert_true(
            abs(det + 1.0) <= DET_TOL,
            "Mobius det H != -1 at N=" + String(n) + " (got " + String(det) + ")",
        )
        # G2b: only the reflection's axis admits a global section.
        checks += 1
        assert_true(
            kdim == 1,
            "Mobius dim ker L must be 1, got " + String(kdim) + " at N=" + String(n),
        )
        # G3: closed-form spectrum.
        checks += 1
        assert_true(
            abs(eigs[0]) <= LAMBDA_TOL,
            "lambda_1 != 0 at N=" + String(n) + " (got " + String(eigs[0]) + ")",
        )
        checks += 1
        assert_true(
            abs(eigs[1] - closed) <= LAMBDA_TOL,
            "lambda_2 != 2(1-cos(pi/N)) at N="
            + String(n)
            + ": got "
            + String(eigs[1])
            + " want "
            + String(closed),
        )
        # The decay itself: ~4x per doubling. This is the claim that makes the
        # spectrum unusable on long loops, so assert it rather than assume it.
        if si > 0:
            var ratio = prev_gap / eigs[1]
            checks += 1
            assert_true(
                ratio > 3.5 and ratio < 4.5,
                "lambda_2 must fall ~4x per doubling of N, got " + String(ratio),
            )
        prev_gap = eigs[1]

        # ---- NEGATIVE CONTROL: orientable ring ------------------------------
        var go = build_ring(n, False)
        var ce = go.fundamental_cycle_edges()[0]
        var fro_o = go.holonomy_dist_to_identity(ce)
        var det_o = go.holonomy_det(ce)
        checks += 1
        assert_true(
            abs(det_o - 1.0) <= DET_TOL,
            "NEGATIVE CONTROL FAILED: orientable ring reported det H = "
            + String(det_o)
            + " at N="
            + String(n),
        )
        checks += 1
        assert_true(
            fro_o <= FRO_TOL,
            "NEGATIVE CONTROL FAILED: orientable ring has non-trivial holonomy "
            + String(fro_o)
            + " at N="
            + String(n),
        )
        var eigs_o = eigenvalues_ascending[DT](build_sheaf_laplacian[D, DT](go))
        checks += 1
        assert_true(
            kernel_dimension(eigs_o) == D,
            "NEGATIVE CONTROL FAILED: orientable ring must have dim ker L = D",
        )

    # N = 192, holonomy only. The Laplacian leg stops at 96 because Jacobi on
    # the 384x384 matrix costs ~8x the N=96 sweep and this gate belongs in a
    # fast manifest; the holonomy is O(D^3) regardless of N, which is the very
    # asymmetry under test, so exercising it alone here is the point and not a
    # shortcut.
    var g192 = build_ring(192, True)
    var e192 = g192.fundamental_cycle_edges()[0]
    var fro192 = g192.holonomy_dist_to_identity(e192)
    var det192 = g192.holonomy_det(e192)
    print("192  |H-I|=", fro192, " det=", det192, " (holonomy only)")
    checks += 2
    assert_true(
        abs(fro192 - 2.0) <= FRO_TOL,
        "Mobius ||H-I||_F != 2 at N=192 (got " + String(fro192) + ")",
    )
    assert_true(
        abs(det192 + 1.0) <= DET_TOL,
        "Mobius det H != -1 at N=192 (got " + String(det192) + ")",
    )
    var go192 = build_ring(192, False)
    checks += 1
    assert_true(
        abs(go192.holonomy_det(go192.fundamental_cycle_edges()[0]) - 1.0) <= DET_TOL,
        "NEGATIVE CONTROL FAILED: orientable ring obstructed at N=192",
    )

    print()
    print("ring sizes compared:", len(sizes), "+ N=192 holonomy-only",
          "(Mobius + orientable control each)")
    print("assertions compared:", checks)
    print("PASS: G2 ring holonomy + G3 sheaf-Laplacian spectrum")
