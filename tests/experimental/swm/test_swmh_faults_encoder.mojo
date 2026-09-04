"""G20 — SWM Phase 7: E3 as the design doc claims it — faults through the
learned encoder.

G9 exercised the classifier on FRAME-LEVEL observations: the fault was injected
on the transition pairs the Procrustes fit consumed, which is the numpy
prototype's setting and not the method's. Here the fault lives in the WORLD and
reaches the classifier only through `mlp.mojo` and the learned transport
table: a broken sensor is a cell whose observations carry 30x the noise; a
biased sensor is an edge whose crossing applies an extra rotation that nothing
compensates. Four worlds, several training seeds each, and the same asymmetry
gated as counts:

  mobius, clean            -> OBSTRUCTION (and never ABERRANT)
  orientable, noisy cell   -> ABERRANT   (and never OBSTRUCTION)
  orientable, biased edge  -> UNDECIDED  (and NEVER OBSTRUCTION — zero tolerance)
  orientable, clean        -> NOMINAL

The per-edge residual is the transport table's own pre-consensus EWMA on the
selected branch, the nominal scale is the median across entries, and the GNC
weight is Geman-McClure at convergence — i.e. the readings the runtime would
have. The frame channel's validity (landmark R^2) is printed beside every
verdict so a verdict from a collapsed channel cannot pass unnoticed.

**The encoder compresses the fault.** At frame level (G9) a sensor with 30x
the noise gave a residual 920x the median. Through the learned encoder the
same sensor gives ~4.7x: an MLP over 16 mixed coordinates averages the noise
down, and the transport's EWMA smooths what is left. The clean worlds sit at
1.5-1.7x. So the ABERRANT band that separates a fault from a nominal edge is
one order of magnitude wide here rather than three, and the 10x outlier factor
G9 used (a frame-level calibration) misses the fault outright (measured: 0/3).
The factor is therefore set to 3 and GATED from both sides — every clean
reading below it, every faulty reading above it — so the calibration is a
measurement rather than a knob turned until the gate passed.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_faults_encoder.mojo
"""

from std.math import sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import (
    MobiusRing,
    MobiusConfig,
    ACTION_FORWARD,
)
from mojo_rl.experimental.swm.observables import (
    gnc_weight,
    estimate_c_bar,
    classify,
    class_name,
    CLASS_NOMINAL,
    CLASS_ABERRANT,
    CLASS_OBSTRUCTION,
    CLASS_UNDECIDED,
)

comptime DT = DType.float64
comptime N = 12
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]
comptime EnvT = MobiusRing[12, 6, 16, DT]
comptime SEEDS = 3
comptime ANGLE_TOL = 0.2
comptime BAD_EDGE = 5
comptime BAD_CELL = 6
comptime OUTLIER = 3.0
"""Outlier factor AND GNC inlier multiple, for encoder-level residuals."""


@fieldwise_init
struct Reading(Copyable, ImplicitlyCopyable, Movable):
    var verdict: UInt8
    var det_h: Float64
    var fro_h: Float64
    var worst_ratio: Float64
    var w_bad: Float64
    var landmark_r2: Float64


def read(ecfg: MobiusConfig, cfg: Phase3Config, bad_entry: Int) raises -> Reading:
    var m = TrainerT.train(ecfg, cfg)
    var res = List[Float64]()
    var h = SqMat[2, DT].identity()
    for i in range(N):
        var e = m.table.index(ACTION_FORWARD, i)
        var sq = m.table.res_plus[e] if m.table.sigma[e] > 0 else m.table.res_minus[e]
        res.append(sqrt(sq))
        h = m.table.transport_for(ACTION_FORWARD, i) * h
    var r_nominal = estimate_c_bar(res, 1.0)
    var worst = Float64(0)
    for i in range(N):
        if res[i] > worst:
            worst = res[i]
    var c_bar = estimate_c_bar(res, OUTLIER)
    var w = List[Float64](length=N, fill=1.0)
    for i in range(N):
        w[i] = gnc_weight(res[i], 1.0, c_bar)
    var det_h = Float64(h.det())
    var fro_h = Float64(h.dist_to_identity())
    var v = classify(worst, r_nominal, det_h, fro_h, ANGLE_TOL, False, OUTLIER)
    var env = EnvT(ecfg)
    var steps = cfg.laps * N
    var ev = TrainerT._evaluate(env, m.enc, m.table, cfg, steps, steps + 1)
    return Reading(v, det_h, fro_h, worst / r_nominal, w[bad_entry], ev.landmark_r2)


def main() raises:
    var checks = 0
    var names: List[String] = [
        "mobius clean     ", "noisy cell       ", "biased edge      ",
        "orientable clean ",
    ]
    var expect: List[UInt8] = [
        CLASS_OBSTRUCTION, CLASS_ABERRANT, CLASS_UNDECIDED, CLASS_NOMINAL
    ]
    var worlds = List[MobiusConfig]()
    worlds.append(MobiusConfig.default_mobius())
    worlds.append(MobiusConfig.default_orientable().with_noisy_cell(BAD_CELL, 0.6))
    worlds.append(MobiusConfig.default_orientable().with_biased_edge(BAD_EDGE, 0.3))
    worlds.append(MobiusConfig.default_orientable())

    print("world             | seed | verdict     | det H  |H-I|   worst/median  w[bad]  landmark R^2")
    var total_false_obstr = 0
    var clean_max_ratio = Float64(0)
    var noisy_min_ratio = 1e300
    for wi in range(4):
        var counts = List[Int](length=5, fill=0)
        for s in range(SEEDS):
            var cfg = Phase3Config.default()
            cfg.seed = UInt64(4242 + s * 7717)
            var r = read(worlds[wi], cfg, BAD_EDGE if wi != 1 else BAD_CELL)
            counts[Int(r.verdict)] += 1
            if wi == 1:
                if r.worst_ratio < noisy_min_ratio:
                    noisy_min_ratio = r.worst_ratio
            elif r.worst_ratio > clean_max_ratio:
                clean_max_ratio = r.worst_ratio
            print(names[wi], "|", s, "|", class_name(r.verdict), "|", r.det_h,
                  r.fro_h, r.worst_ratio, r.w_bad, r.landmark_r2)
            checks += 1
            assert_true(
                r.landmark_r2 > 0.8,
                names[wi] + " seed " + String(s) + ": frame channel invalid "
                + "(landmark R^2 " + String(r.landmark_r2) + "), verdict "
                + "meaningless",
            )
        var hit = counts[Int(expect[wi])]
        print(names[wi], "| counts NOMINAL", counts[0], "ABERRANT", counts[1],
              "OBSTRUCTION", counts[2], "UNDECIDED", counts[3], " expected",
              class_name(expect[wi]), hit, "/", SEEDS)
        checks += 1
        assert_true(
            hit == SEEDS,
            names[wi] + ": expected " + class_name(expect[wi]) + " in "
            + String(hit) + "/" + String(SEEDS),
        )
        if wi > 0:
            total_false_obstr += counts[Int(CLASS_OBSTRUCTION)]
    print("residual ratio worst/median: clean worlds max", clean_max_ratio,
          "  noisy cell min", noisy_min_ratio, "  outlier factor", OUTLIER)
    checks += 2
    assert_true(
        clean_max_ratio < OUTLIER and noisy_min_ratio > OUTLIER,
        "the outlier factor must separate EVERY clean reading from EVERY faulty "
        + "one, from both sides: clean max " + String(clean_max_ratio)
        + ", noisy min " + String(noisy_min_ratio) + ", factor " + String(OUTLIER),
    )
    assert_true(
        noisy_min_ratio < 20.0,
        "RECORDED: the encoder COMPRESSES the fault (920x at frame level in G9). "
        + "If this ever reads like G9, the docstring is stale: "
        + String(noisy_min_ratio),
    )
    assert_true(
        total_false_obstr == 0,
        "ZERO TOLERANCE: a sensor fault reached the classifier through the "
        + "encoder and was filed as a world fact " + String(total_false_obstr)
        + " times",
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G20 faults through the learned encoder keep the asymmetry")
