"""G9 — SWM Phase 4 gate (P4 / E3): the fault classification, with its counts.

The asymmetry this gate defends is the whole reason the method is worth
building. A `det H = -1` cycle with nominal residuals is a FACT ABOUT THE WORLD,
to be recorded and handed to the planner, never corrected away. A broken sensor
is an edge to be down-weighted. A constant sensor BIAS is neither: it produces a
perfectly coherent continuous holonomy that a single cycle cannot distinguish
from real curvature, so it must come out UNDECIDED and must NEVER be filed as a
topological obstruction. That last one is a zero-tolerance claim, not a rate.

Four worlds, `TRIALS` independent draws each:
  mobius        flat ring + a reflection on the seam      -> OBSTRUCTION
  noisy edge    orientable, one edge with large state noise -> ABERRANT
  biased edge   orientable, one edge with a constant extra rotation -> UNDECIDED
  clean         orientable, uniform noise                 -> NOMINAL

Scope: this exercises the CLASSIFIER on frame-level observations, which is the
setting the numpy prototype's D / D' controls use and the input the classifier
actually consumes. Fault injection routed through a learned encoder is E3, and
is not claimed here.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_faults.mojo
"""

from std.math import abs, sqrt, cos, sin
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.rng import Rng
from mojo_rl.experimental.swm.procrustes import (
    PairBatch,
    procrustes_o_d,
    mean_squared_residual,
)
from mojo_rl.experimental.swm.ablations import holonomy_product
from mojo_rl.experimental.swm.observables import (
    gnc_weights,
    estimate_c_bar,
    classify,
    class_name,
    confirm_by_independent_cycles,
    CLASS_NOMINAL,
    CLASS_ABERRANT,
    CLASS_OBSTRUCTION,
    CLASS_UNDECIDED,
    CLASS_CURVATURE_CONFIRMED,
)

comptime DT = DType.float64
comptime N = 12
comptime TRIALS = 40
comptime NOISE = 0.02
comptime BAD_EDGE = 5
comptime ANGLE_TOL = 0.2

comptime WORLD_MOBIUS = 0
comptime WORLD_NOISY = 1
comptime WORLD_BIASED = 2
comptime WORLD_CLEAN = 3


def rot2(t: Float64) -> SqMat[2, DT]:
    var m = SqMat[2, DT]()
    m[0, 0] = Scalar[DT](cos(t))
    m[0, 1] = Scalar[DT](-sin(t))
    m[1, 0] = Scalar[DT](sin(t))
    m[1, 1] = Scalar[DT](cos(t))
    return m^


@fieldwise_init
struct TrialResult(Copyable, ImplicitlyCopyable, Movable):
    var verdict: UInt8
    var det_h: Float64
    var fro_h: Float64
    var worst_w: Float64
    var bad_edge_w: Float64


def run_trial(world: Int, mut rng: Rng) raises -> TrialResult:
    # ---- plant the world -------------------------------------------------
    var truth = List[SqMat[2, DT]]()
    var total = Float64(0)
    for i in range(N):
        var t = rng.uniform_range(-0.6, 0.6)
        if i == N - 1:
            t = -total
        else:
            total += t
        truth.append(rot2(t))
    if world == WORLD_MOBIUS:
        var refl = SqMat[2, DT].identity()
        refl[1, 1] = Scalar[DT](-1)
        truth[N - 1] = refl * truth[N - 1]
    elif world == WORLD_BIASED:
        # A constant extra rotation on one edge: the holonomy becomes a
        # non-trivial ROTATION, learned as a perfectly coherent transport.
        truth[BAD_EDGE] = rot2(0.3) * truth[BAD_EDGE]

    # ---- roll out --------------------------------------------------------
    var batches = List[PairBatch[2, DT]]()
    for _ in range(N):
        batches.append(PairBatch[2, DT]())
    for _ in range(40):
        var a = rng.uniform_range(0.0, 6.283185307179586)
        var x = InlineArray[Scalar[DT], 2](fill=0)
        x[0] = Scalar[DT](cos(a))
        x[1] = Scalar[DT](sin(a))
        for step in range(4 * N):
            var e = step % N
            var s = NOISE
            if world == WORLD_NOISY and e == BAD_EDGE:
                s = 0.6
            var y = InlineArray[Scalar[DT], 2](fill=0)
            for i in range(2):
                var v = Scalar[DT](0)
                for j in range(2):
                    v += truth[e][i, j] * x[j]
                y[i] = v + Scalar[DT](rng.normal() * s)
            batches[e].push(x, y)
            for i in range(2):
                x[i] = y[i]

    # ---- fit, then READ --------------------------------------------------
    var fits = List[SqMat[2, DT]]()
    var res_norm = List[Float64]()
    for e in range(N):
        var r = procrustes_o_d[2, DT](batches[e])
        res_norm.append(
            sqrt(Float64(mean_squared_residual[2, DT](batches[e], r)))
        )
        fits.append(r^)
    var h = holonomy_product[2, DT](fits)
    var det_h = Float64(h.det())
    var fro_h = Float64(h.dist_to_identity())

    var w = gnc_weights(res_norm)
    var worst_w = 1.0
    for e in range(N):
        if w[e] < worst_w:
            worst_w = w[e]

    var r_nominal = estimate_c_bar(res_norm, 1.0)  # the median itself
    var worst_r = Float64(0)
    for e in range(N):
        if res_norm[e] > worst_r:
            worst_r = res_norm[e]
    # A single cycle can never CONFIRM a continuous holonomy (v2 §4.4).
    var verdict = classify(worst_r, r_nominal, det_h, fro_h, ANGLE_TOL, False)
    return TrialResult(verdict, det_h, fro_h, worst_w, w[BAD_EDGE])


def main() raises:
    var checks = 0
    var names: List[String] = ["mobius", "noisy edge", "biased edge", "clean"]
    var expect: List[UInt8] = [
        CLASS_OBSTRUCTION, CLASS_ABERRANT, CLASS_UNDECIDED, CLASS_NOMINAL
    ]

    print("world       | trials | NOMINAL ABERRANT OBSTR UNDEC | false OBSTR")
    var total_trials = 0
    for world in range(4):
        var counts = List[Int](length=5, fill=0)
        var fro_sum = Float64(0)
        var badw_sum = Float64(0)
        var rng = Rng(UInt64(31337 + world * 977))
        for _ in range(TRIALS):
            var r = run_trial(world, rng)
            counts[Int(r.verdict)] += 1
            fro_sum += r.fro_h
            badw_sum += r.bad_edge_w
            total_trials += 1
        var hit = counts[Int(expect[world])]
        var false_obstr = counts[Int(CLASS_OBSTRUCTION)]
        if world == WORLD_MOBIUS:
            false_obstr = 0
        print(
            names[world], "|", TRIALS, "|",
            counts[0], counts[1], counts[2], counts[3], "|", false_obstr,
            " mean |H-I| =", fro_sum / Float64(TRIALS),
            " w[bad] =", badw_sum / Float64(TRIALS),
        )

        checks += 1
        assert_true(
            Float64(hit) / Float64(TRIALS) >= 0.95,
            names[world] + ": expected " + class_name(expect[world]) + " in "
            + String(hit) + "/" + String(TRIALS) + " trials",
        )
        # ZERO TOLERANCE: a sensor fault must never be filed as a world fact.
        if world != WORLD_MOBIUS:
            checks += 1
            assert_true(
                false_obstr == 0,
                "ZERO-TOLERANCE LEG FAILED: " + names[world] + " was filed as "
                + "a TOPOLOGICAL OBSTRUCTION " + String(false_obstr)
                + " times. A sensor fault must never be recorded as a fact "
                + "about the world",
            )
        # ...and the converse: an obstruction must never be corrected away.
        if world == WORLD_MOBIUS:
            checks += 1
            assert_true(
                counts[Int(CLASS_ABERRANT)] == 0,
                "the Mobius obstruction was explained away as an ABERRANT edge "
                + String(counts[Int(CLASS_ABERRANT)]) + " times — it must be "
                + "recorded, not corrected",
            )

    # ---- vacuity guards ---------------------------------------------------
    # The biased world must actually have a non-trivial holonomy, and the clean
    # world must not: without this, "UNDECIDED vs NOMINAL" could be decided by
    # a threshold that nothing ever crosses.
    var rng_b = Rng(555)
    var rng_c = Rng(556)
    var fro_biased = Float64(0)
    var fro_clean = Float64(0)
    for _ in range(TRIALS):
        fro_biased += run_trial(WORLD_BIASED, rng_b).fro_h
        fro_clean += run_trial(WORLD_CLEAN, rng_c).fro_h
    fro_biased /= Float64(TRIALS)
    fro_clean /= Float64(TRIALS)
    checks += 2
    assert_true(
        fro_biased > ANGLE_TOL,
        "the biased world must exceed the angle tolerance, got "
        + String(fro_biased),
    )
    assert_true(
        fro_clean < 0.5 * ANGLE_TOL,
        "the clean world must sit well below it, got " + String(fro_clean),
    )

    # ---- cross-confirmation: one cycle can never confirm a curvature -------
    var edges_a: List[Int] = [0, 1, 2, 3]
    var edges_b: List[Int] = [4, 5, 6, 7]
    var edges_overlap: List[Int] = [3, 4, 5, 6]
    var one_cycle = List[List[Int]]()
    one_cycle.append(edges_a.copy())
    var fro_one: List[Float64] = [0.43]
    checks += 3
    assert_true(
        not confirm_by_independent_cycles(one_cycle, fro_one, ANGLE_TOL),
        "a single cycle must NEVER confirm a continuous holonomy",
    )
    var overlapping = List[List[Int]]()
    overlapping.append(edges_a.copy())
    overlapping.append(edges_overlap.copy())
    var fro_two: List[Float64] = [0.43, 0.41]
    assert_true(
        not confirm_by_independent_cycles(overlapping, fro_two, ANGLE_TOL),
        "two cycles SHARING an edge must not confirm — one biased edge "
        + "explains both",
    )
    var disjoint = List[List[Int]]()
    disjoint.append(edges_a.copy())
    disjoint.append(edges_b.copy())
    assert_true(
        confirm_by_independent_cycles(disjoint, fro_two, ANGLE_TOL),
        "two EDGE-DISJOINT cycles both reporting a holonomy must confirm — "
        + "no single biased edge can explain both",
    )

    print()
    print("trials compared     :", total_trials, "(", TRIALS, "per world )")
    print("mean |H-I|: biased", fro_biased, " clean", fro_clean,
          " (tol", ANGLE_TOL, ")")
    print("assertions compared :", checks)
    print("PASS: G9 fault classification, zero false obstructions")
