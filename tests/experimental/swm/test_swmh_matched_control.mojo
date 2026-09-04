"""G28 — SWM Phase 10: the matched control on E1, and the reconciliation.

Phase 9 (G27) found on Pendulum that lifting the orthogonality constraint —
same channel, same slots, same data — cost nothing over 24 rolled steps, and
concluded that 6a's reading was unsupported. 6a had compared an orthogonal
frame roll against a free NONLINEAR content roll: different quantities in
different spaces, so the difference was never attributable to the isometry.
This gate runs the missing control on the world where the claim was made, and
the answer is not the one Pendulum suggested.

**A. The constraint DOES buy rollout accuracy on E1 — at the loop closure,
and only there.** Per-edge Procrustes `O(2)` against per-edge unconstrained
least squares (`fit_free_lsq`: no cocycle term, no descent — ablation C's
collapse is what a cocycle loss does, not what the constraint buys), fitted
and rolled on DISJOINT episodes, over 4 seeds:

    horizon        1      6     12     24
    orthogonal   0.105  0.151  0.171  0.040
    free LSQ     0.092  0.147  0.182  0.119

At ONE step the free fit is BETTER (0.87x) — it has more degrees of freedom
and the constraint costs something locally. At 6 and 12 they are level. At 24,
which on a 12-cell ring is TWO LAPS, the orthogonal arm is 3.0x better.

**The mechanism is exact algebra, and it is measured.** Two laps of the
constrained transports compose to the identity EXACTLY — a reflection squared
— `|P - I|_F = 6.1e-16`; the free product does not, 0.095. So the rolled frame
returns when the walk returns, and the free one does not. That is why the
advantage appears at 24 and nowhere else.

**And it reconciles G27.** Pendulum closes no loop in 24 steps, so there was
nothing there for exact closure to buy, and the matched control correctly
found nothing. The two results agree once the mechanism is named: the isometry
pays at loop closures, not at arbitrary horizons. 6a's numbers stand and its
reading needed narrowing, not withdrawing.

**B. Does the constraint buy the OBSERVABLE?** The sharper question, never
asked before: if a free 2x2 can represent a reflection, can the sign of `det`
of the product around the ring carry the `Z/2` class on its own? Measured: yes
on this world — 4/4 negative on Mobius, 4/4 positive on the orientable twin.
But the margin is eroded: `|det H|` falls to 0.85 and the minimum singular
value to 0.85, against exactly 1 by construction for the constrained fits. A
sign whose magnitude is drifting toward zero is a sign that will eventually
mean nothing, which is precisely what G8 measured under a cocycle loss
(det -1.00 -> -0.02, min singular value 1.0 -> 0.64). The constraint is what
makes the class exact rather than merely usually right.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_matched_control.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.procrustes import procrustes_o_d
from mojo_rl.experimental.swm.ablations import (
    fit_free_lsq,
    orthogonality_defect,
    holonomy_product,
    min_singular_value,
)
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import MobiusConfig

comptime DT = DType.float64
comptime N = 12
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]
comptime SEEDS = 4


def roll_error(
    ms: List[SqMat[2, DT]],
    seq: List[Scalar[DT]],
    n_episodes: Int,
    n_frames: Int,
    horizons: List[Int],
) -> List[Float64]:
    """Roll `u` from frame 0 of each episode and score against the encoder's
    own trajectory. Edge `t % N` is the one crossed at step `t`, matching how
    the trainer indexes transports along a forward walk."""
    var acc = List[Float64](length=len(horizons), fill=0)
    var cnt = List[Float64](length=len(horizons), fill=0)
    for ep in range(n_episodes):
        var u = List[Float64](length=2, fill=0)
        for i in range(2):
            u[i] = Float64(seq[(ep * n_frames + 0) * 2 + i])
        for t in range(n_frames - 1):
            var r = ms[t % N]
            var un = List[Float64](length=2, fill=0)
            for i in range(2):
                var s = Float64(0)
                for j in range(2):
                    s += Float64(r[i, j]) * u[j]
                un[i] = s
            u = un^
            var d = Float64(0)
            for i in range(2):
                var e = u[i] - Float64(seq[(ep * n_frames + t + 1) * 2 + i])
                d += e * e
            d = sqrt(d)
            for hi in range(len(horizons)):
                if t + 1 == horizons[hi]:
                    acc[hi] += d
                    cnt[hi] += 1.0
    for hi in range(len(horizons)):
        if cnt[hi] > 0:
            acc[hi] /= cnt[hi]
    return acc^


def main() raises:
    var checks = 0
    var horizons: List[Int] = [1, 6, 12, 24]
    var e_orth = List[Float64](length=4, fill=0)
    var e_free = List[Float64](length=4, fill=0)
    var defect_sum = Float64(0)
    var det_free_mob = List[Float64]()
    var det_free_ori = List[Float64]()
    var det_orth_mob = List[Float64]()
    var sv_free_mob = List[Float64]()
    var mob_sign_ok = 0
    var ori_sign_ok = 0
    var close_o = Float64(0)
    var close_f = Float64(0)

    print("seed | world      | det H orth | det H free | min sv free | "
          + "|M^T M - I|")
    for s in range(SEEDS):
        for world in range(2):
            var ecfg = MobiusConfig.default_mobius() if world == 0 else MobiusConfig.default_orientable()
            var cfg = Phase3Config.default()
            cfg.seed = UInt64(20260904 + s * 7717)
            var m = TrainerT.train(ecfg, cfg)
            # fit and roll on DISJOINT episodes
            var fit = TrainerT.encode_rollouts(m, ecfg, cfg, 24, 0xF17_0000)
            var ev = TrainerT.encode_rollouts(m, ecfg, cfg, 24, 0xE7A1_0000)

            var orth = List[SqMat[2, DT]]()
            for e in range(N):
                orth.append(procrustes_o_d[2, DT](fit.batches[e]))
            var free = fit_free_lsq[DT](fit.batches)
            var defect = orthogonality_defect[DT](free)

            var h_o = holonomy_product[2, DT](orth)
            var h_f = holonomy_product[2, DT](free)
            var d_o = Float64(h_o.det())
            var d_f = Float64(h_f.det())
            var sv = min_singular_value[DT](h_f)
            print(s, "|", "mobius    " if world == 0 else "orientable",
                  "|", d_o, "|", d_f, "|", sv, "|", defect)

            if world == 0:
                # The mechanism: what does the 24-step (TWO LAP) product do?
                # A reflection squared is the identity EXACTLY; a free product
                # whose |det| is 0.85-0.98 neither returns nor preserves area.
                var p_o = SqMat[2, DT].identity()
                var p_f = SqMat[2, DT].identity()
                for t in range(24):
                    p_o = orth[t % N] * p_o
                    p_f = free[t % N] * p_f
                close_o += Float64(p_o.dist_to_identity()) / Float64(SEEDS)
                close_f += Float64(p_f.dist_to_identity()) / Float64(SEEDS)
                det_free_mob.append(d_f)
                det_orth_mob.append(d_o)
                sv_free_mob.append(sv)
                defect_sum += defect
                if d_f < 0:
                    mob_sign_ok += 1
                var eo = roll_error(orth, ev.seq_u, ev.n_episodes, ev.n_frames, horizons)
                var ef = roll_error(free, ev.seq_u, ev.n_episodes, ev.n_frames, horizons)
                for hi in range(4):
                    e_orth[hi] += eo[hi] / Float64(SEEDS)
                    e_free[hi] += ef[hi] / Float64(SEEDS)
            else:
                det_free_ori.append(d_f)
                if d_f > 0:
                    ori_sign_ok += 1

    print()
    print("A | rollout error on E1 (Mobius), mean over", SEEDS, "seeds")
    var lo = String("A |   orthogonal (Procrustes) |")
    var lf = String("A |   free least squares      |")
    for hi in range(4):
        lo += " " + String(Int(e_orth[hi] * 1000)) + "e-3"
        lf += " " + String(Int(e_free[hi] * 1000)) + "e-3"
    print("A |   horizon                 |     1      6     12     24")
    print(lo)
    print(lf)
    var ratio24 = e_free[3] / e_orth[3]
    var ratio1 = e_free[0] / e_orth[0]
    print("A |   free/orthogonal:", ratio1, "at 1 step,", ratio24, "at 24")
    print("A | MECHANISM — the 24-step product is TWO LAPS: |P - I|_F "
          + "orthogonal", close_o, " free", close_f,
          "   (a reflection squared is the identity EXACTLY)")
    print("B | det H sign from FREE fits: mobius", mob_sign_ok, "/", SEEDS,
          " negative;  orientable", ori_sign_ok, "/", SEEDS, " positive")
    print("B | mean |M^T M - I| of the free fits:", defect_sum / Float64(SEEDS))

    checks += 6
    assert_true(
        ratio24 > 2.0,
        "THE MATCHED CONTROL, on the world where 6a's claim was made: over 24 "
        + "steps — TWO LAPS, i.e. a loop closure — the constraint DOES buy "
        + "accuracy. free/orthogonal = " + String(ratio24),
    )
    assert_true(
        ratio1 < 1.05,
        "...and it is not simply that the constrained fit is better: at ONE "
        + "step the free fit, having more degrees of freedom, is at least as "
        + "good. free/orthogonal at 1 step = " + String(ratio1),
    )
    assert_true(
        close_o < 0.01 and close_f > 5.0 * close_o + 0.05,
        "MECHANISM: the win is the LOOP CLOSURE. Two laps of the constrained "
        + "transports compose to the identity exactly (a reflection squared), "
        + "|P - I| = " + String(close_o) + "; the free product does not, "
        + String(close_f) + ". That is why the advantage appears at 24 steps "
        + "and not at 1, 6 or 12 — and why Pendulum, which closes no loop in "
        + "24 steps, showed none (G27).",
    )
    assert_true(
        defect_sum / Float64(SEEDS) < 0.35,
        "ATTRIBUTION, as on Pendulum: the unconstrained fit is ALREADY nearly "
        + "an isometry, so the constraint has nothing to buy on this axis. "
        + "|M^T M - I| = " + String(defect_sum / Float64(SEEDS)),
    )
    assert_true(
        mob_sign_ok == SEEDS and ori_sign_ok == SEEDS,
        "and the OBSERVABLE survives without the constraint too: the sign of "
        + "det of the free fits' product must be negative on Mobius and "
        + "positive on the twin in every seed. got " + String(mob_sign_ok)
        + ", " + String(ori_sign_ok),
    )
    var worst_sv = 1e300
    var worst_mag = 1e300
    for i in range(len(det_free_mob)):
        if sv_free_mob[i] < worst_sv:
            worst_sv = sv_free_mob[i]
        var mag = abs(det_free_mob[i])
        if mag < worst_mag:
            worst_mag = mag
    print("B | worst |det H| and min singular value over Mobius seeds:",
          worst_mag, worst_sv)
    assert_true(
        worst_mag > 0.7 and worst_sv > 0.7,
        "...and the free determinant must not be DRIFTING TOWARD ZERO, which "
        + "is what makes a sign meaningless (G8 measured exactly that under a "
        + "cocycle loss: det -1.00 -> -0.02, min sv 1.0 -> 0.64). got |det| "
        + String(worst_mag) + " min sv " + String(worst_sv),
    )
    var worst_orth = 1e300
    for i in range(len(det_orth_mob)):
        if abs(det_orth_mob[i]) < worst_orth:
            worst_orth = abs(det_orth_mob[i])
    assert_true(
        worst_orth > 0.99,
        "CONTROL: the constrained fits' determinant is exactly +-1 by "
        + "construction, which is the difference the free arm has to earn: "
        + String(worst_orth),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G28 matched control on E1")
