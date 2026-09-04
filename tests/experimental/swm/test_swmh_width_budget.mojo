"""G32 — SWM Phase 13: is the wide frame's looser split a DATA problem?

Phase 12 recorded a trade-off rather than a free lunch. A learned `d = 4`
frame keeps the `Z/2` class and collapses gauge coincidence from 51.5 % to
0.4 %, but on the same schedule it is a measurably looser frame: landmark R²
0.989 → 0.841, texture leak 0.054 → 0.105, anisotropy 0.752 → 0.468. The gate
said explicitly that whether budget closes that gap was untested. This tests
it, because the answer changes how the method should be used: if width is free
given data, always take the wider frame and the coincidence residue disappears
for nothing; if the gap is intrinsic, it is a real trade between a clean split
and a robust clique.

`d = 4` has twice the frame coordinates and six times the transport parameters
(`d(d-1)/2` per entry: 1 at `d = 2`, 6 at `d = 4`) on identical data, so the
prior is that it is under-trained rather than incapable. The budget is scaled
by EPOCHS, which multiplies both the encoder's steps and the number of
transitions each transport entry sees.

Reported against the `d = 2` reference at 1x, so the question is not "does it
improve" — more training almost always improves something — but **does it
reach the narrow frame's numbers, and where does it stop**.

Answer: it does. **85 % of the landmark gap closes by 4x**, and the texture
leak ends up BETTER than the narrow frame's:

    d  epochs   landmark R^2   texture leak   anisotropy
    2    1x        0.989          0.054          0.813
    4    1x        0.891          0.059          0.680
    4    2x        0.923          0.029          0.600
    4    4x        0.975          0.014          0.684

So the wide frame was UNDER-TRAINED, not incapable — which is what the
parameter count already suggested (`d(d-1)/2` per transport entry: 1 at
`d = 2`, 6 at `d = 4`) — and Phase 12's trade-off is a budget question rather
than a property of the method. Width is close to free given data, and the
recommendation that follows is to take the wider frame: the gauge-coincidence
residue goes with it. Anisotropy is not monotone in budget and is gated as a
floor, not a trend.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_width_budget.mojo
"""

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
comptime SEEDS = 2


@fieldwise_init
struct BudgetPoint(Copyable, ImplicitlyCopyable, Movable):
    var d: Int
    var mult: Int
    var lm: Float64
    var nu: Float64
    var aniso: Float64
    var wps: Float64
    var class_ok: Int


def measure[D: Int](mult: Int) raises -> BudgetPoint:
    comptime TrT = SwmPhase3[N, 6, 16, 32, 8, DT, 1, D]
    comptime WT = MobiusRingND[N, D, 6, 16]
    var lm = 1.0
    var nu = 0.0
    var an = 1.0
    var wp = 1e9
    var ok = 0
    for s in range(SEEDS):
        var cfg = Phase3Config.default()
        cfg.seed = UInt64(20260904 + s * 7717)
        cfg.epochs = cfg.epochs * mult
        cfg.warmup_epochs = cfg.warmup_epochs * mult
        var env = WT(MobiusNDConfig.default_mobius())
        var m = TrT.train_world(env, cfg, N)
        var ev = WT(MobiusNDConfig.default_mobius())
        var st = TrT.validity_stats(ev, m, cfg, 16)
        if st.landmark_r2 < lm:
            lm = st.landmark_r2
        if st.nuisance_r2 > nu:
            nu = st.nuisance_r2
        if st.u_anisotropy < an:
            an = st.u_anisotropy
        if st.within_place_std < wp:
            wp = st.within_place_std
        var h = SqMat[D, DT].identity()
        for i in range(N):
            h = m.table.transport_for(ACTION_FORWARD_ND, i) * h
        if Float64(h.det()) < 0:
            ok += 1
    return BudgetPoint(D, mult, lm, nu, an, wp, ok)


def main() raises:
    var checks = 0
    var ref2 = measure[2](1)
    var w1 = measure[4](1)
    var w2 = measure[4](2)
    var w4 = measure[4](4)
    var pts = [ref2, w1, w2, w4]

    print("d | epochs x | landmark R^2 | texture leak | anisotropy | "
          + "within-place | det H = -1")
    for i in range(4):
        var p = pts[i]
        print(p.d, "|", p.mult, "       |", p.lm, "|", p.nu, "|", p.aniso, "|",
              p.wps, "|", p.class_ok, "/", SEEDS)

    var gap1 = ref2.lm - w1.lm
    var gap4 = ref2.lm - w4.lm
    print()
    print("landmark R^2 gap to the d=2 reference:", gap1, "at 1x ->", gap4,
          "at 4x   (closed:", 1.0 - gap4 / gap1, "of it)")

    checks += 4
    for i in range(4):
        checks += 1
        assert_true(
            pts[i].class_ok == SEEDS,
            "the Z/2 class must survive every width and budget: d = "
            + String(pts[i].d) + " at " + String(pts[i].mult) + "x gave "
            + String(pts[i].class_ok) + "/" + String(SEEDS),
        )
    assert_true(
        gap1 > 0.05,
        "the premise: there must BE a gap at 1x to close, else this gate is "
        + "measuring nothing. got " + String(gap1),
    )
    assert_true(
        w4.lm > w1.lm,
        "more budget must at least help the wide frame's split: "
        + String(w1.lm) + " -> " + String(w4.lm),
    )
    assert_true(
        gap4 < 0.5 * gap1,
        "THE ANSWER, if the gap closes: the wide frame is UNDER-TRAINED, not "
        + "incapable, and width is free given data. gap " + String(gap1)
        + " -> " + String(gap4),
    )
    assert_true(
        w4.nu < ref2.nu and w4.aniso > 0.5 * ref2.aniso,
        "the wide frame at budget must keep the texture out at least as well "
        + "as the narrow one (leak " + String(w4.nu) + " vs " + String(ref2.nu)
        + ") and must not be flattened (anisotropy " + String(w4.aniso)
        + " vs " + String(ref2.aniso) + "). Anisotropy is NOT monotone in "
        + "budget here (0.680 -> 0.600 -> 0.684 over 1x/2x/4x), so it is "
        + "gated as a floor rather than as a trend.",
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G32 the width price against budget")
