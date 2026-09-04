"""G27 — SWM Phase 9: the transfer test. What survives a world that was not
built for the method, and what a matched control does to our own claim.

Every world through Phase 8 was constructed to carry the structure. Pendulum
is not: it is picked because it contains an exact `SO(2)` action —
`(cos theta, sin theta)` is carried by `R(theta_dot' dt)`, gated to 4e-16
below — beside a nonlinearly evolving velocity, so the split hypothesis 4.0
asks for exists analytically in a physical system.

The transport is selected by the VELOCITY, which is content, so the design's
rule (v3 §4.2: never condition on the latent) is bent through its own escape
hatch: the place index becomes a velocity BIN, `log2(12)` bits. The trainer's
per-place anti-collapse hinge then reads "per velocity bin", which is the
right analogue — at a fixed velocity what varies is the angle.

**What transfers.** The split is found through an overcomplete mixing:
landmark R^2 **0.990**, speed leakage R^2 **0.013**, frame not collapsed, and
`det H = +1` on every cycle of the bin graph — no obstruction is manufactured
on a physical world, which is the only thing the `Z/2` machinery may be asked
here (every transport is a rotation, so there is nothing to detect).

**What does not.** The frame rollout is NOT flat. E1 measured 0.092 -> 0.107
over 12 steps; here it is 0.083 -> 0.485 -> 0.520 over 24. Two candidate
causes, both measured:

  - *Quantization of the bottleneck* — REFUTED. Widening it 4x (12 -> 48 bins)
    cuts the per-step binning angle error exactly 4x (0.0166 -> 0.0042 rad)
    and leaves the 24-step frame error where it was (0.520 -> 0.604).
  - *The orthogonality constraint* — a contributor, not the driver. On the
    same held-out pairs a general linear fit beats the orthogonal one by
    1.47x at 12 bins and 1.14x at 48.

**The matched control, which is the finding.** A difference between an
orthogonal frame roll and a free NONLINEAR content roll is not attributable to
the isometry: the two predict different quantities in different spaces. The
matched arm is the same channel, the same slots, the same data, with the
orthogonality LIFTED — a free 2x2 per (action, bin), rolled forward:

    24-step error   orthogonal   linear      ratio
      12 bins          0.520      0.442       0.85
      48 bins          0.604      0.641       1.06

So in matched units **the isometry buys nothing here**, in either direction.
And the reason is measured rather than argued: the unconstrained
least-squares fit is ALREADY nearly an isometry, `|M^T M - I|_F` = 0.126 at
12 bins and 0.067 at 48. The data is a rotation; an honest fit finds one, and
a constraint that the solution already satisfies cannot buy anything. On E1
the free arm was driven OFF the manifold by a cocycle loss (G8: minimum
singular value 1.0 -> 0.64), which is a force no least-squares fit applies.

**What that says about 6a.** Phase 6a's numbers stand as measured — the frame
rollout is flat and the content rollout drifts 11x — but its INTERPRETATION,
that the orthogonal constraint buys the only channel still trustworthy after
a long rollout, has never had this matched control run against it. On the one
world where it has, the constraint is not what does the work. The matched
control belongs on E1 too, and is not built.

Legs: A the world's exact rotation (with a 5%-wrong-angle control); B the
split and `det H`; C the rollout table incl. the matched linear arm; D the
raw-observation easy mode, so a failure in B cannot be blamed on the mixing;
E the bin sweep and the fit comparison; F growth and attribution.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_pendulum_transfer.mojo
"""

from std.math import abs, cos, sin, sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.place_graph import PlaceGraph, Edge
from mojo_rl.experimental.swm.procrustes import PairBatch, procrustes_o_d
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.pendulum_swm import (
    PendulumSwm,
    PendulumSwmConfig,
    MAX_SPEED,
    DT,
    TORQUE_NONE,
)

comptime DT_T = DType.float64
comptime BINS = 12
comptime OBS = 12
comptime CDIM = 8
comptime TrainerT = SwmPhase3[BINS, 1, OBS, 32, CDIM, DT_T, 3]
comptime WorldT = PendulumSwm[BINS, OBS]
comptime SEEDS = 2
comptime EVAL_EPISODES = 24


def enc_full[
    NB: Int
](mut env: PendulumSwm[NB, OBS], m: SwmPhase3[NB, 1, OBS, 32, CDIM, DT_T, 3].ModelT) raises -> List[Float64]:
    var o = env.observation()
    var hid = List[Scalar[DT_T]](length=32, fill=0)
    var lat = List[Scalar[DT_T]](length=2 + CDIM, fill=0)
    m.enc.forward(o, hid, lat)
    var out = List[Float64](length=2 + CDIM, fill=0)
    for i in range(2 + CDIM):
        out[i] = Float64(lat[i])
    return out^


def nearest_bin[NB: Int](h: List[Float64], cent: List[Float64]) -> Int:
    var best = 1e300
    var arg = 0
    for b in range(NB):
        var d = Float64(0)
        for i in range(CDIM):
            var e = cent[b * CDIM + i] - h[i]
            d += e * e
        if d < best:
            best = d
            arg = b
    return arg


@fieldwise_init
struct ArmResult(Copyable, Movable):
    """Rollout error at horizons 1/6/12/24 for the three arms, plus the
    validity numbers and the measured per-step binning angle error."""

    var e: List[Float64]
    """`arm * 4 + horizon_index`."""
    var landmark_r2: Float64
    var leak_r2: Float64
    var aniso: Float64
    var within_std: Float64
    var det_h: Float64
    var bin_angle_err: Float64
    """Mean |true angle - bin-centre angle| per step, radians."""
    var res_orth: Float64
    """Held-out per-slot residual of the best ORTHOGONAL fit."""
    var res_lin: Float64
    """...and of the best GENERAL LINEAR fit, on the same pairs. If the linear
    fit is much better the ORTHOGONALITY CONSTRAINT is what binds — i.e. the
    encoder's gauge is not conformal, so the world's rotation does not conjugate
    to a rotation."""
    var pairs_per_slot: Float64
    var lin_nonorth: Float64
    """Mean `|M^T M - I|_F` of the UNCONSTRAINED per-slot fits. Small means the
    least-squares solution is already an isometry, i.e. the orthogonality
    constraint is not binding and can buy nothing."""


def measure[
    NB: Int
](cfgw: PendulumSwmConfig, seed: UInt64, laps: Int) raises -> ArmResult:
    comptime TrT = SwmPhase3[NB, 1, OBS, 32, CDIM, DT_T, 3]
    comptime WT = PendulumSwm[NB, OBS]
    var tcfg = Phase3Config.with_content()
    tcfg.seed = seed
    tcfg.laps = laps
    var w = WT(cfgw)
    var m = TrT.train_world(w, tcfg, NB)

    # ---- validity ------------------------------------------------------
    var ev = WT(cfgw)
    var st = TrT.validity_stats(ev, m, tcfg, 16)
    var us = List[Float64]()
    var sp = List[Float64]()
    var bin_err = Float64(0)
    var bin_n = Float64(0)
    var e2 = WT(cfgw)
    for ep in range(16):
        e2.reset(UInt64(61000 + ep))
        for _ in range(60):
            var l = enc_full[NB](e2, m)
            us.append(l[0])
            us.append(l[1])
            sp.append(e2.speed() / MAX_SPEED)
            e2.step(e2.explore_action())
            bin_err += abs(e2.speed() - e2.bin_speed(e2.place_id())) * DT
            bin_n += 1.0
    var leak = TrT._explained_variance(us, sp, len(sp), 1)

    # every transport is a rotation: no obstruction may appear
    var g = PlaceGraph[2, DT_T]()
    for _ in range(NB):
        _ = g.add_place()
    for b in range(NB):
        _ = g.add_edge(
            Edge.action_edge(b, (b + 1) % NB, TORQUE_NONE),
            m.table.transport_for(TORQUE_NONE, b),
        )
    g.rebuild_gauge(0)
    var cyc = g.fundamental_cycle_edges()
    var det_h = g.holonomy_det(cyc[0])

    # ---- per-bin content centroids: the rolled h -> transport index -----
    var acc = List[Float64](length=NB * CDIM, fill=0)
    var cnt = List[Float64](length=NB, fill=0)
    var e3 = WT(cfgw)
    for ep in range(EVAL_EPISODES):
        e3.reset(UInt64(77000 + ep))
        for _ in range(60):
            var l = enc_full[NB](e3, m)
            var b = e3.place_id()
            cnt[b] += 1
            for i in range(CDIM):
                acc[b * CDIM + i] += l[2 + i]
            e3.step(e3.explore_action())
    for b in range(NB):
        if cnt[b] > 0:
            for i in range(CDIM):
                acc[b * CDIM + i] /= cnt[b]

    # ---- is the orthogonality constraint binding in the encoder's gauge?
    #      Fit each (action, bin) slot on half the pairs and score the other
    #      half, orthogonal vs general linear. Same pairs, same slots.
    var xs = List[Float64]()
    var ys = List[Float64]()
    var slot = List[Int]()
    var e4 = WT(cfgw)
    for ep in range(EVAL_EPISODES):
        e4.reset(UInt64(91000 + ep))
        var prev = enc_full[NB](e4, m)
        for _ in range(60):
            var b = e4.place_id()
            var a = e4.explore_action()
            e4.step(a)
            var cur = enc_full[NB](e4, m)
            xs.append(prev[0])
            xs.append(prev[1])
            ys.append(cur[0])
            ys.append(cur[1])
            slot.append(a * NB + b)
            prev = cur^
    var n_slots = 3 * NB
    var lin_maps = List[SqMat[2, DT_T]]()
    for _ in range(n_slots):
        lin_maps.append(SqMat[2, DT_T].identity())
    var lin_ok = List[Bool](length=n_slots, fill=False)
    var res_o = Float64(0)
    var res_l = Float64(0)
    var nonorth = Float64(0)
    var res_n = Float64(0)
    var used_slots = Float64(0)
    var pairs_tot = Float64(0)
    for sl in range(n_slots):
        var fit = PairBatch[2, DT_T]()
        var n_here = 0
        for t in range(len(slot)):
            if slot[t] != sl:
                continue
            n_here += 1
            if n_here % 2 == 1:  # odd -> fit, even -> score
                var xa = InlineArray[Scalar[DT_T], 2](fill=0)
                var ya = InlineArray[Scalar[DT_T], 2](fill=0)
                for i in range(2):
                    xa[i] = Scalar[DT_T](xs[t * 2 + i])
                    ya[i] = Scalar[DT_T](ys[t * 2 + i])
                fit.push(xa, ya)
        if fit.count() < 8:
            continue
        used_slots += 1.0
        pairs_tot += Float64(fit.count())
        var r_o = procrustes_o_d[2, DT_T](fit)
        # general linear: M = (Y X^T)(X X^T)^-1 on the same fit half
        var yxt = SqMat[2, DT_T]()
        var xxt = SqMat[2, DT_T]()
        for k in range(fit.count()):
            var xk = fit.x_at(k)
            var yk = fit.y_at(k)
            for i in range(2):
                for j in range(2):
                    yxt[i, j] = yxt[i, j] + yk[i] * xk[j]
                    xxt[i, j] = xxt[i, j] + xk[i] * xk[j]
        for i in range(2):
            xxt[i, i] = xxt[i, i] + Scalar[DT_T](1e-9)
        var m_lin = yxt * xxt.inverse()
        lin_maps[sl] = m_lin.copy()
        lin_ok[sl] = True
        nonorth += Float64(
            (m_lin.transpose() * m_lin - SqMat[2, DT_T].identity()).frobenius_norm()
        )
        var n2 = 0
        for t in range(len(slot)):
            if slot[t] != sl:
                continue
            n2 += 1
            if n2 % 2 == 1:
                continue
            var d_o = Float64(0)
            var d_l = Float64(0)
            for i in range(2):
                var po = Float64(0)
                var pl = Float64(0)
                for j in range(2):
                    po += Float64(r_o[i, j]) * xs[t * 2 + j]
                    pl += Float64(m_lin[i, j]) * xs[t * 2 + j]
                d_o += (po - ys[t * 2 + i]) * (po - ys[t * 2 + i])
                d_l += (pl - ys[t * 2 + i]) * (pl - ys[t * 2 + i])
            res_o += sqrt(d_o)
            res_l += sqrt(d_l)
            res_n += 1.0
    if res_n > 0:
        res_o /= res_n
        res_l /= res_n

    # ---- the rollout table ---------------------------------------------
    var horizons: List[Int] = [1, 6, 12, 24]
    var err = List[Float64](length=4 * 4, fill=0)
    var errn = List[Float64](length=4 * 4, fill=0)
    var roll = WT(cfgw)
    for ep in range(EVAL_EPISODES):
        for arm in range(4):
            roll.reset(UInt64(83000 + ep))
            var l0 = enc_full[NB](roll, m)
            var u = List[Float64](length=2, fill=0)
            for i in range(2):
                u[i] = l0[i]
            var h = List[Scalar[DT_T]](length=CDIM, fill=0)
            for i in range(CDIM):
                h[i] = Scalar[DT_T](l0[2 + i])
            for k in range(24):
                var a = roll.explore_action()
                var b_oracle = roll.place_id()
                var hf = List[Float64](length=CDIM, fill=0)
                for i in range(CDIM):
                    hf[i] = Float64(h[i])
                var b_pred = nearest_bin[NB](hf, acc)
                var hn = m.content.predict_next(h, a)
                if arm != 2:
                    # arm 3 is the MATCHED control: the same channel, the same
                    # slots, the same data, with the orthogonality constraint
                    # LIFTED — the only way to attribute a difference to the
                    # isometry rather than to what is being predicted.
                    var b = b_oracle if arm == 1 else b_oracle
                    if arm == 1:
                        b = b_pred
                    var r = m.table.transport_for(a, b)
                    if arm == 3 and lin_ok[a * NB + b]:
                        r = lin_maps[a * NB + b].copy()
                    var un = List[Float64](length=2, fill=0)
                    for i in range(2):
                        var s2 = Float64(0)
                        for j in range(2):
                            s2 += Float64(r[i, j]) * u[j]
                        un[i] = s2
                    u = un^
                for i in range(CDIM):
                    h[i] = hn[i]
                roll.step(a)
                var lt = enc_full[NB](roll, m)
                var d = Float64(0)
                if arm == 2:
                    for i in range(CDIM):
                        var e = Float64(h[i]) - lt[2 + i]
                        d += e * e
                else:
                    for i in range(2):
                        var e = u[i] - lt[i]
                        d += e * e
                d = sqrt(d)
                for hi in range(4):
                    if k + 1 == horizons[hi]:
                        err[arm * 4 + hi] += d
                        errn[arm * 4 + hi] += 1.0
    for i in range(16):
        if errn[i] > 0:
            err[i] /= errn[i]
    return ArmResult(
        err^, st.landmark_r2, leak, st.u_anisotropy, st.within_place_std,
        det_h, bin_err / bin_n, res_o, res_l,
        pairs_tot / used_slots if used_slots > 0 else 0.0,
        nonorth / used_slots if used_slots > 0 else 0.0,
    )


def row(name: String, r: ArmResult, arm: Int) -> String:
    var line = name + " |"
    for hi in range(4):
        line += " " + String(Int(r.e[arm * 4 + hi] * 1000)) + "e-3"
    return line


def main() raises:
    var checks = 0

    # =====================================================================
    # A. The world's claim: the frame rotation is EXACT.
    # =====================================================================
    var wcfg = PendulumSwmConfig.default()
    var env = WorldT(wcfg)
    env.reset(4242)
    var worst = Float64(0)
    var worst_wrong = Float64(0)
    for _ in range(400):
        var f0 = env.true_landmark()
        env.step(env.explore_action())
        var r = env.true_transport()
        var f1 = env.true_landmark()
        var aw = env.speed() * DT * 1.05
        var rw = SqMat[2, DT_T]()
        rw[0, 0] = Scalar[DT_T](cos(aw))
        rw[0, 1] = Scalar[DT_T](-sin(aw))
        rw[1, 0] = Scalar[DT_T](sin(aw))
        rw[1, 1] = Scalar[DT_T](cos(aw))
        var d = Float64(0)
        var dw = Float64(0)
        for i in range(2):
            var pred = Float64(0)
            var pw = Float64(0)
            for j in range(2):
                pred += Float64(r[i, j] * f0[j])
                pw += Float64(rw[i, j] * f0[j])
            d += abs(pred - Float64(f1[i]))
            dw += abs(pw - Float64(f1[i]))
        if d > worst:
            worst = d
        if dw > worst_wrong:
            worst_wrong = dw
    print("A | frame rotation exact over 400 steps: worst |F' - R F| =", worst,
          "  (5% wrong angle:", worst_wrong, ")")
    checks += 2
    assert_true(
        worst < 1e-12,
        "the pendulum's frame must rotate EXACTLY by R(theta_dot dt), got "
        + String(worst),
    )
    assert_true(
        worst_wrong > 1e-4,
        "CONTROL: a 5% wrong angle must be REJECTED by the same test, else it "
        + "measures nothing. got " + String(worst_wrong),
    )

    # =====================================================================
    # B/C/D. The split, the rollout table, and the easy-mode control.
    # =====================================================================
    var mixed = measure[BINS](PendulumSwmConfig.default(), 52000, 5)
    var raw = measure[BINS](PendulumSwmConfig.raw_obs(), 52000, 5)
    print()
    print("B | mixed obs | landmark R^2", mixed.landmark_r2, " speed leak R^2",
          mixed.leak_r2, " aniso", mixed.aniso, " within-bin std",
          mixed.within_std, " det H", mixed.det_h)
    print("B | raw obs   | landmark R^2", raw.landmark_r2, " speed leak R^2",
          raw.leak_r2, " aniso", raw.aniso, " within-bin std",
          raw.within_std, " det H", raw.det_h)
    print()
    print("C | rollout error vs horizon (mean |predicted - encoded|)")
    print("C | arm                          |    1      6     12     24")
    print("C |", row("mixed frame, ORACLE bins   ", mixed, 0))
    print("C |", row("mixed frame, PREDICTED bins", mixed, 1))
    print("C |", row("mixed free content roll    ", mixed, 2))
    print("C |", row("mixed frame, LINEAR maps   ", mixed, 3),
          "   <- matched control: same channel, same slots, orthogonality LIFTED")
    print("C |", row("raw   frame, ORACLE bins   ", raw, 0))
    print("C |", row("raw   frame, PREDICTED bins", raw, 1))
    print("C |", row("raw   free content roll    ", raw, 2))

    # =====================================================================
    # E. Attribution: is the degradation the BOTTLENECK's quantization?
    #    Widen the bottleneck 4x and the per-step angle error must fall with
    #    it, and the frame rollout with it. That is the mechanism, measured
    #    rather than argued.
    # =====================================================================
    var wide = measure[4 * BINS](PendulumSwmConfig.default(), 52000, 2)
    var pred_12 = mixed.bin_angle_err * 24.0
    var pred_48 = wide.bin_angle_err * 24.0
    print()
    print("E | bins", BINS, ": per-step binning angle err", mixed.bin_angle_err,
          " rad; coherent over 24 steps", pred_12,
          " ; frame(oracle bins) at 24 =", mixed.e[3])
    print("E | bins", 4 * BINS, ": per-step binning angle err",
          wide.bin_angle_err, " rad; coherent over 24 steps", pred_48,
          " ; frame(oracle bins) at 24 =", wide.e[3])
    print("E |", row("wide frame, ORACLE bins    ", wide, 0))
    print("E |", row("wide frame, PREDICTED bins ", wide, 1))
    print("E |", row("wide frame, LINEAR maps    ", wide, 3))
    print("E | landmark R^2", wide.landmark_r2, " leak", wide.leak_r2,
          " det H", wide.det_h)

    var sqrt24 = sqrt(24.0)
    var frame_ratio = mixed.e[3] / mixed.e[0]
    var free_ratio = mixed.e[2 * 4 + 3] / mixed.e[2 * 4 + 0]
    var wide_ratio = wide.e[3] / wide.e[0]
    print()
    print("F | error GROWTH over 24 steps (error(24)/error(1); a random walk "
          + "gives sqrt(24) = 4.90, an amplifying map more)")
    print("F |   frame, oracle bins, 12:", frame_ratio, "  48:", wide_ratio)
    print("F |   free content roll     :", free_ratio)
    print("F | MATCHED (same channel, orthogonality lifted): orthogonal",
          mixed.e[3], " linear", mixed.e[3 * 4 + 3], " at 24 steps;  1 step:",
          mixed.e[0], "vs", mixed.e[3 * 4 + 0])
    print("F |   wide (48 bins): orthogonal", wide.e[3], " linear",
          wide.e[3 * 4 + 3])
    print("F | mean |M^T M - I|_F of the UNCONSTRAINED fits:", mixed.lin_nonorth,
          " (12 bins) ", wide.lin_nonorth, " (48 bins)")
    print()
    print("E | held-out per-slot residual, ORTHOGONAL vs GENERAL LINEAR fit "
          + "(same pairs, same slots)")
    print("E |  bins", BINS, ": orthogonal", mixed.res_orth, " linear",
          mixed.res_lin, "  ratio", mixed.res_orth / mixed.res_lin,
          "  pairs/slot", mixed.pairs_per_slot)
    print("E |  bins", 4 * BINS, ": orthogonal", wide.res_orth, " linear",
          wide.res_lin, "  ratio", wide.res_orth / wide.res_lin,
          "  pairs/slot", wide.pairs_per_slot)

    checks += 8
    assert_true(
        mixed.det_h > 0 and raw.det_h > 0 and wide.det_h > 0,
        "NO OBSTRUCTION may be manufactured on a physical world: det H = "
        + String(mixed.det_h) + ", " + String(raw.det_h) + ", "
        + String(wide.det_h),
    )
    assert_true(
        mixed.landmark_r2 > 0.9 and mixed.leak_r2 < 0.2,
        "hypothesis 4.0 on a PHYSICAL world: the frame channel must find "
        + "(cos, sin) through the mixing and keep the speed out. R^2 "
        + String(mixed.landmark_r2) + " leak " + String(mixed.leak_r2),
    )
    assert_true(
        mixed.aniso > 0.05 and mixed.within_std > 0.05,
        "frame channel must not be collapsed or bin-indexed-constant",
    )
    assert_true(
        raw.landmark_r2 > 0.9,
        "EASY MODE control (raw observation) must also find the split, else "
        + "a failure above could be blamed on the mixing: "
        + String(raw.landmark_r2),
    )
    assert_true(
        mixed.e[3] > 3.0 * mixed.e[0],
        "RECORDED NEGATIVE: unlike E1 (flat 0.092 -> 0.107 over 12 steps), the "
        + "frame rollout on a physical world DEGRADES, because the transport "
        + "angle is itself uncertain. If this ever goes flat the docstring is "
        + "stale: " + String(mixed.e[0]) + " -> " + String(mixed.e[3]),
    )
    assert_true(
        wide.bin_angle_err < 0.5 * mixed.bin_angle_err
        and wide.e[3] > 0.75 * mixed.e[3],
        "REFUTED, and recorded: quantization of the velocity bottleneck is NOT "
        + "the cause. Widening it 4x cuts the per-step binning angle error 4x "
        + "(" + String(mixed.bin_angle_err) + " -> " + String(wide.bin_angle_err)
        + ") and leaves the 24-step frame error where it was ("
        + String(mixed.e[3]) + " -> " + String(wide.e[3])
        + "). If this ever improves with bin count the docstring is stale.",
    )
    assert_true(
        mixed.res_orth > 1.2 * mixed.res_lin and mixed.res_orth < 2.0 * mixed.res_lin,
        "the orthogonality constraint is a CONTRIBUTOR, not the driver: a "
        + "general linear fit on the same held-out pairs does better by a "
        + "modest factor (measured 1.47x at 12 bins, 1.14x at 48). got "
        + String(mixed.res_orth / mixed.res_lin),
    )
    var m_ratio = mixed.e[3 * 4 + 3] / mixed.e[3]
    var w_ratio = wide.e[3 * 4 + 3] / wide.e[3]
    assert_true(
        m_ratio > 0.7 and m_ratio < 1.4 and w_ratio > 0.7 and w_ratio < 1.4,
        "RECORDED NEGATIVE, and the point of the whole gate: in MATCHED units "
        + "the isometry buys NOTHING here. Same channel, same slots, same "
        + "data, orthogonality lifted: 24-step error ratio linear/orthogonal "
        + String(m_ratio) + " at 12 bins and " + String(w_ratio) + " at 48. "
        + "E1's frame-vs-content comparison was never matched like this.",
    )
    assert_true(
        mixed.lin_nonorth < 0.35 and wide.lin_nonorth < 0.35,
        "ATTRIBUTION: the constraint cannot buy anything because the "
        + "UNCONSTRAINED least-squares fit is ALREADY nearly an isometry — the "
        + "data is a rotation and nothing pushes the fit off the manifold. "
        + "|M^T M - I|_F = " + String(mixed.lin_nonorth) + ", "
        + String(wide.lin_nonorth) + ". On E1 the free arm was driven off the "
        + "manifold by a COCYCLE LOSS (G8: min singular value 1.0 -> 0.64); "
        + "an honest least-squares fit has no such force.",
    )
    assert_true(
        mixed.e[(1) * 4 + 3] > mixed.e[3],
        "predicting the bin must cost something on top of quantization: "
        + String(mixed.e[4 + 3]) + " vs " + String(mixed.e[3]),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G27 pendulum transfer measured")
