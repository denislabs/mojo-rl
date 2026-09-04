"""G30 — SWM Phase 11: gauge coincidence against the frame's dimension.

Phase 7 left a residue that no consistency test can remove. A false
identification's holonomy is the transport along the walk it spans; when that
walk's holonomy happens to land within tolerance of `I` or of the world's
monodromy `M`, the closure is indistinguishable from a true one — PCM accepts
it, the clique keeps it, and `det H` reads it as a fact. Measured: 10 of 95
false closures survive on the ring (G18) and 41 % on the flat 2D grid (G19).
G19 also refuted the obvious hypothesis: the second DIMENSION OF THE BASE does
not help, because on a flat bundle the transport between two places is the
same along every homotopic path.

The remaining lead is the dimension of the FIBRE. In `O(2)` a holonomy is one
angle, so a walk's holonomy is a random walk on a circle and returns near the
identity often. In `O(D)` it wanders a `D(D-1)/2`-dimensional group, and the
tolerance ball shrinks against the group's volume. This gate measures that
directly and needs no learning: plant a ring in `O(D)`, enumerate every
non-returning walk (each is a candidate FALSE closure), and count how many
land within tolerance of `{I, M}`.

Two tolerance conventions, both reported, because the choice is not neutral:
`‖H − I‖_F` of a random element grows like `sqrt(D)`, and the residual noise
on a learned transport is per-coordinate, so **`tol ∝ sqrt(D)` is the fair
convention** and a fixed `tol` flatters higher `D`. The claim is gated under
the fair one.

Controls: the TRUE closures (walks that do return) must be accepted at every
`D`, otherwise a falling false rate would just mean the test rejects
everything; and the per-edge step size is held constant in Frobenius norm
across `D`, so the walk is not simply slower in higher dimensions.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_gauge_coincidence_dim.mojo
"""

from std.math import sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import (
    SqMat,
    skew_from_vector,
    expm_skew,
    householder,
)
from mojo_rl.experimental.swm.rng import Rng

comptime DT = DType.float64
comptime N = 12
comptime STEP = 0.6
"""Frobenius norm of every edge's generator — held constant across D."""
comptime TOL = 0.3


@fieldwise_init
struct DimResult(Copyable, ImplicitlyCopyable, Movable):
    var d: Int
    var true_ok: Int
    var true_n: Int
    var false_hit_fixed: Int
    var false_hit_fair: Int
    var false_n: Int
    var mean_dist: Float64


def measure[D: Int](seed: UInt64) raises -> DimResult:
    """Plant a Mobius ring in `O(D)` and count coincidences among the walks
    that do NOT return to their start — the candidate false closures."""
    comptime NGEN = D * (D - 1) // 2
    var rng = Rng(seed)

    # A FRAME PER CELL, with the edge transports as their differences, so the
    # product around the ring telescopes to the identity EXACTLY in any
    # dimension. The ring's "angles sum to zero" trick is abelian and does NOT
    # generalise: in O(D>2), exp(A)exp(B) != exp(A+B), and a first version of
    # this gate built that way had a Baker-Campbell-Hausdorff residue large
    # enough that NO true closure was accepted at D >= 3 (measured: 0/36) —
    # which the true-closure control caught. Same lesson as the flat vs curved
    # Klein bundle in G19.
    var frames = List[SqMat[D, DT]]()
    for _ in range(N):
        var sp = List[Scalar[DT]](length=NGEN, fill=0)
        var nrm = Float64(0)
        var v = List[Float64](length=NGEN, fill=0)
        for k in range(NGEN):
            v[k] = rng.normal()
            nrm += v[k] * v[k]
        nrm = sqrt(nrm)
        for k in range(NGEN):
            sp[k] = Scalar[DT](v[k] * STEP / nrm)
        frames.append(expm_skew[D, DT](skew_from_vector[D, DT](Span(sp))))

    var refl_v = List[Float64](length=D, fill=0)
    refl_v[0] = 1.0
    var refl = householder[D, DT](Span(refl_v))

    var edge = List[SqMat[D, DT]]()
    for i in range(N):
        var nxt = (i + 1) % N
        var inv = frames[i].transpose()
        if i == N - 1:
            # the seam as a DECK TRANSFORMATION: flat everywhere, the whole
            # obstruction in the identification
            edge.append((frames[nxt] * refl) * inv)
        else:
            edge.append(frames[nxt] * inv)

    # Transport from step 0 to step t along a forward walk of 3 laps.
    var tr = List[SqMat[D, DT]]()
    tr.append(SqMat[D, DT].identity())
    for t in range(3 * N):
        tr.append(edge[t % N] * tr[t])

    var m = tr[N].copy()  # the monodromy, based at the root
    var ident = SqMat[D, DT].identity()
    var fair = TOL * sqrt(Float64(D) / 2.0)

    var true_ok = 0
    var true_n = 0
    var f_fixed = 0
    var f_fair = 0
    var false_n = 0
    var dist_sum = Float64(0)
    for s in range(2 * N):
        for t in range(s + 1, 3 * N):
            var h = tr[t].transpose() * tr[s]
            var di = Float64((h - ident).frobenius_norm())
            var dm = Float64((h - m).frobenius_norm())
            var best = di if di < dm else dm
            if (t - s) % N == 0:
                true_n += 1
                if best <= fair:
                    true_ok += 1
            else:
                false_n += 1
                dist_sum += best
                if best <= TOL:
                    f_fixed += 1
                if best <= fair:
                    f_fair += 1
    return DimResult(
        D, true_ok, true_n, f_fixed, f_fair, false_n,
        dist_sum / Float64(false_n),
    )


def main() raises:
    var checks = 0
    var r2 = measure[2](11)
    var r3 = measure[3](11)
    var r4 = measure[4](11)
    var r6 = measure[6](11)
    var rs = [r2^, r3^, r4^, r6^]

    print("D | true closures accepted | false closures within tol of {I, M}")
    print("  |                        |  fixed tol 0.3   fair tol 0.3*sqrt(D/2)   mean dist")
    for i in range(4):
        var r = rs[i]
        var fx = Float64(r.false_hit_fixed) / Float64(r.false_n)
        var fr = Float64(r.false_hit_fair) / Float64(r.false_n)
        print(r.d, "|", r.true_ok, "/", r.true_n, "            |",
              r.false_hit_fixed, "/", r.false_n, "=", fx, "  ",
              r.false_hit_fair, "/", r.false_n, "=", fr, "  ", r.mean_dist)

    var f2 = Float64(rs[0].false_hit_fair) / Float64(rs[0].false_n)
    var f6 = Float64(rs[3].false_hit_fair) / Float64(rs[3].false_n)
    checks += 3
    for i in range(4):
        checks += 1
        assert_true(
            rs[i].true_ok == rs[i].true_n and rs[i].true_n > 0,
            "CONTROL: every TRUE closure must be accepted at D = "
            + String(rs[i].d) + ", else a falling false rate would only mean "
            + "the test rejects everything. got " + String(rs[i].true_ok)
            + "/" + String(rs[i].true_n),
        )
    assert_true(
        f2 > 0.05,
        "the premise: gauge coincidence must be COMMON in O(2), else there is "
        + "nothing to shrink. got " + String(f2),
    )
    assert_true(
        f6 < 0.5 * f2,
        "THE ANSWER: a higher-dimensional FIBRE must shrink gauge coincidence "
        + "under the fair tolerance, where a higher-dimensional BASE did not "
        + "(G19). D=2 " + String(f2) + " -> D=6 " + String(f6),
    )
    assert_true(
        rs[3].mean_dist > rs[0].mean_dist,
        "and the mechanism: a walk's holonomy sits FURTHER from {I, M} on "
        + "average as the group grows. " + String(rs[0].mean_dist) + " -> "
        + String(rs[3].mean_dist),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G30 gauge coincidence shrinks with the FIBRE's dimension")
