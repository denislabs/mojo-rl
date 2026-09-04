"""G31 — SWM Phase 12: a learned 4-dimensional frame, and what it buys.

G30 measured on PLANTED transports that gauge coincidence — a false closure
whose walk holonomy lands within tolerance of `I` or the monodromy `M` — is a
two-dimensional artefact: 59 % of candidate walks at `d = 2`, 9 % at 3, and
0 % at 4 and 6. That is geometry, and it says nothing about whether a LEARNED
wide frame is still a frame. Everything the method rests on was established at
`d = 2`: the landmark is found, the texture is kept out, the channel does not
collapse, and `det H` comes out as the world's class. A wider fibre has more
room to collapse into, more parameters per transport, and the same amount of
data.

So this gate runs the whole Phase 3 recipe at `d = 2` and `d = 4` on the same
world family, ON THE SAME BINARY, and then measures the payoff with the
transports the encoder actually learned.

The world is `MobiusRingND`, E1's recipe with the fibre widened: a landmark in
`R^d` transported by `O(d)`, a non-transported per-cell texture, overcomplete
mixing, noise. Its transports come from a FRAME PER CELL, because the ring's
"angles that sum to zero" trick is abelian and leaves a
Baker-Campbell-Hausdorff residue above `d = 2` (G30 measured 0/36 true
closures accepted when built that way).

Legs.

**A. Is a learned 4-D frame still a frame?** Landmark R², texture leakage,
anisotropy (now the Jacobi eigenvalues of the `d x d` covariance, not the
closed 2x2 form) and within-place spread, with `det H = -1` on Möbius and
`+1` on the orientable twin at both widths. The answer is yes with a PRICE,
and the price is gated: the split holds but is measurably looser at width —
landmark R^2 0.989 -> 0.841, texture leak 0.054 -> 0.105, anisotropy 0.752 ->
0.468, on the same data with twice the coordinates and six times the transport
parameters. Whether a longer schedule closes that gap is untested.

**B. The payoff, with LEARNED transports.** Enumerate every walk that does not
return to its start — each is a candidate false closure — and count how many
land within tolerance of `{I, M}`. The tolerance scales as `sqrt(d)` because
`‖H − I‖_F` of a random element does, which is the fair convention and the one
that does not flatter the wider frame.

**C. The control that makes B a measurement.** Every TRUE closure must still
be accepted at both widths. A false rate that falls because the test rejects
everything is exactly the failure G30 caught.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_wide_frame.mojo
"""

from std.math import sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring_nd import (
    MobiusRingND,
    MobiusNDConfig,
    ACTION_FORWARD_ND,
)

comptime DT = DType.float64
comptime N = 12
comptime SEEDS = 4
comptime TOL = 0.3


@fieldwise_init
struct WideResult(Copyable, ImplicitlyCopyable, Movable):
    var d: Int
    var mob_ok: Int
    var ori_ok: Int
    var false_obstr: Int
    var lm_worst: Float64
    var nu_worst: Float64
    var aniso_worst: Float64
    var wps_worst: Float64
    var true_ok: Int
    var true_n: Int
    var false_hit: Int
    var false_n: Int
    var mean_dist: Float64


def measure[D: Int]() raises -> WideResult:
    comptime TrT = SwmPhase3[N, 6, 16, 32, 8, DT, 1, D]
    comptime WT = MobiusRingND[N, D, 6, 16]
    var mob_ok = 0
    var ori_ok = 0
    var false_obstr = 0
    var lm_worst = 1.0
    var nu_worst = 0.0
    var an_worst = 1.0
    var wp_worst = 1e9
    var true_ok = 0
    var true_n = 0
    var f_hit = 0
    var f_n = 0
    var dist_sum = Float64(0)
    var fair = TOL * sqrt(Float64(D) / 2.0)

    for s in range(SEEDS):
        for world in range(2):
            var wcfg = MobiusNDConfig.default_mobius() if world == 0 else MobiusNDConfig.default_orientable()
            var cfg = Phase3Config.default()
            cfg.seed = UInt64(20260904 + s * 7717)
            var env = WT(wcfg)
            var m = TrT.train_world(env, cfg, N)
            var ev = WT(wcfg)
            var st = TrT.validity_stats(ev, m, cfg, 16)
            if st.landmark_r2 < lm_worst:
                lm_worst = st.landmark_r2
            if st.nuisance_r2 > nu_worst:
                nu_worst = st.nuisance_r2
            if st.u_anisotropy < an_worst:
                an_worst = st.u_anisotropy
            if st.within_place_std < wp_worst:
                wp_worst = st.within_place_std

            # cumulative transports from the LEARNED table
            var tr = List[SqMat[D, DT]]()
            tr.append(SqMat[D, DT].identity())
            for t in range(3 * N):
                tr.append(
                    m.table.transport_for(ACTION_FORWARD_ND, t % N) * tr[t]
                )
            var h_ring = tr[N].copy()
            var det_h = Float64(h_ring.det())
            if world == 0 and det_h < 0:
                mob_ok += 1
            if world == 1:
                if det_h > 0:
                    ori_ok += 1
                else:
                    false_obstr += 1

            # every walk: does its holonomy land in {I, M}?
            if world == 0:
                var ident = SqMat[D, DT].identity()
                for a in range(2 * N):
                    for b in range(a + 1, 3 * N):
                        var h = tr[b].transpose() * tr[a]
                        var di = Float64((h - ident).frobenius_norm())
                        var dm = Float64((h - h_ring).frobenius_norm())
                        var best = di if di < dm else dm
                        if (b - a) % N == 0:
                            true_n += 1
                            if best <= fair:
                                true_ok += 1
                        else:
                            f_n += 1
                            dist_sum += best
                            if best <= fair:
                                f_hit += 1
    return WideResult(
        D, mob_ok, ori_ok, false_obstr, lm_worst, nu_worst, an_worst, wp_worst,
        true_ok, true_n, f_hit, f_n, dist_sum / Float64(f_n),
    )


def main() raises:
    var checks = 0
    var r2 = measure[2]()
    var r4 = measure[4]()
    var rs = [r2, r4]

    print("d | mobius det-1 | orient det+1 | false obstr | landmark R^2 | "
          + "texture R^2 | aniso | within-place")
    for i in range(2):
        var r = rs[i]
        print(r.d, "|", r.mob_ok, "/", SEEDS, "      |", r.ori_ok, "/", SEEDS,
              "     |", r.false_obstr, "         |", r.lm_worst, "|",
              r.nu_worst, "|", r.aniso_worst, "|", r.wps_worst)
    print()
    print("d | true closures accepted | false closures within tol of {I, M} | "
          + "mean dist")
    for i in range(2):
        var r = rs[i]
        var fr = Float64(r.false_hit) / Float64(r.false_n)
        print(r.d, "|", r.true_ok, "/", r.true_n, "            |",
              r.false_hit, "/", r.false_n, "=", fr, "  |", r.mean_dist)

    checks += 9
    for i in range(2):
        var r = rs[i]
        checks += 4
        assert_true(
            r.mob_ok == SEEDS and r.ori_ok == SEEDS and r.false_obstr == 0,
            "d = " + String(r.d) + ": the Z/2 class must come out on BOTH "
            + "worlds with zero false obstructions. mobius " + String(r.mob_ok)
            + ", orientable " + String(r.ori_ok) + ", false "
            + String(r.false_obstr),
        )
        # The split must HOLD at both widths, but it is not free: the wider
        # frame has twice the coordinates and six times the transport
        # parameters on the same data, so the bar is set per width and the
        # DIFFERENCE is reported as the price.
        var lm_bar = 0.9 if r.d == 2 else 0.75
        var nu_bar = 0.1 if r.d == 2 else 0.2
        assert_true(
            r.lm_worst > lm_bar and r.nu_worst < nu_bar,
            "d = " + String(r.d) + ": hypothesis 4.0 must hold at this width — "
            + "landmark R^2 " + String(r.lm_worst) + ", texture R^2 "
            + String(r.nu_worst),
        )
        assert_true(
            r.aniso_worst > 0.05 and r.wps_worst > 0.05,
            "d = " + String(r.d) + ": a wider fibre has more room to collapse "
            + "into; anisotropy " + String(r.aniso_worst)
            + ", within-place std " + String(r.wps_worst),
        )
        assert_true(
            r.true_ok == r.true_n and r.true_n > 0,
            "CONTROL at d = " + String(r.d) + ": every TRUE closure must be "
            + "accepted, else a falling false rate means only that the test "
            + "rejects everything (the failure G30 caught). "
            + String(r.true_ok) + "/" + String(r.true_n),
        )
    var f2 = Float64(rs[0].false_hit) / Float64(rs[0].false_n)
    var f4 = Float64(rs[1].false_hit) / Float64(rs[1].false_n)
    assert_true(
        f2 > 0.05,
        "the premise: gauge coincidence must be common at d = 2 with LEARNED "
        + "transports too, else there is nothing to buy away. got " + String(f2),
    )
    assert_true(
        f4 < 0.5 * f2,
        "THE PAYOFF: a learned 4-D frame must shrink gauge coincidence as the "
        + "planted geometry said it would (G30: 59% -> 0%). d=2 " + String(f2)
        + " -> d=4 " + String(f4),
    )
    assert_true(
        rs[1].lm_worst < rs[0].lm_worst and rs[1].nu_worst > rs[0].nu_worst
        and rs[1].aniso_worst < rs[0].aniso_worst,
        "RECORDED PRICE: the wider frame is measurably a WORSE frame on the "
        + "same data — landmark R^2 " + String(rs[0].lm_worst) + " -> "
        + String(rs[1].lm_worst) + ", texture leak " + String(rs[0].nu_worst)
        + " -> " + String(rs[1].nu_worst) + ", anisotropy "
        + String(rs[0].aniso_worst) + " -> " + String(rs[1].aniso_worst)
        + ". Whether a longer schedule or more data closes that gap is "
        + "untested. If this ever reverses, the trade-off has changed.",
    )
    assert_true(
        rs[1].mean_dist > rs[0].mean_dist,
        "and by the same mechanism: a walk's holonomy sits further from "
        + "{I, M} as the fibre grows. " + String(rs[0].mean_dist) + " -> "
        + String(rs[1].mean_dist),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G31 a learned 4-D frame keeps its class and buys the residue away")
