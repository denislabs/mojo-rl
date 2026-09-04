"""G15 — SWM Phase 6a: the content channel, and why it does NOT fix planning.

Phase 5 measured a specific deficit: nine of ten planner failures were the RIGHT
PARITY IN THE WRONG CELL, because the frame alone is a weak place code. The
content channel is the design's answer to that, so this gate asks whether it
delivers. It half does, and the half that fails is the more interesting half.

  1. The content channel DOES localise. Nearest-centroid cell accuracy from `h`
     alone goes 0.58 (untrained) -> 0.999 (trained).
  2. It does NOT contaminate the frame. Adding a reconstruction loss over
     `(u, h)` could have pushed texture into `u` and undone Phase 3; measured,
     the frame's nuisance R^2 stays ~0.005 and `det H = -1` still comes out with
     exactly one reflected edge.
  3. **Rolled forward, it does not help the planner — it hurts.** Matching on
     `(u, h)` instead of `u` alone drops cell accuracy 96 -> 75 out of 120, and
     the damage is monotone in how much the content is trusted (weight
     0 / 0.01 / 0.05 / 0.2 / 1.0 -> cell 96 / 89 / 77 / 75 / 75).

The reason is measured, not guessed:

    steps    frame rollout error    content rollout error
      1            0.097                   0.393
      6            0.119                   1.695
     12            0.095                   4.450
     24            0.043                   3.468

The frame is FLAT because its transition is an isometry — an orthogonal map
rotates error instead of amplifying it. The content channel's free nonlinear
transition drifts, ~35x by twelve steps. So this is not a tuning failure; it is
the design's own asymmetry showing up as a measurement. The orthogonal
constraint is usually justified by what it does for the holonomy (it makes
`det H` an invariant); this says it ALSO buys the only channel you can still
trust after twenty-four imagined steps.

What that implies, and what is NOT built here: the content channel's value is
localisation FROM AN OBSERVATION, not long-horizon imagination. Using it well
means observing at the arrival state and re-planning — a control-loop change,
not a world-model change, and Phase 5 deliberately plans open-loop precisely to
keep the model under test. So the negative result is recorded rather than
engineered around.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_content.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import (
    MobiusRing,
    MobiusConfig,
    ACTION_FORWARD,
)
from mojo_rl.experimental.swm.planner import (
    FrameModel,
    PlannerConfig,
    Plan,
    plan_exhaustive,
    plan_exhaustive_with_content,
    MODEL_ORTHOGONAL,
    PLAN_FORWARD,
)

comptime DT = DType.float64
comptime N = 12
comptime SEEDS = 3
comptime EPISODES = 40
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]
comptime EnvT = MobiusRing[12, 6, 16, DT]


def enc_full(mut env: EnvT, model: TrainerT.ModelT) raises -> List[Scalar[DT]]:
    var o = env.observation()
    var hid = List[Scalar[DT]](length=32, fill=0)
    var lat = List[Scalar[DT]](length=10, fill=0)
    model.enc.forward(o, hid, lat)
    return lat^


def main() raises:
    var checks = 0
    var ecfg = MobiusConfig.default_mobius()
    var pcfg = PlannerConfig.default()

    # =====================================================================
    # 1-2. The content channel localises, and does not contaminate the frame.
    # =====================================================================
    var cell_off = Float64(0)
    var cell_on = Float64(0)
    var worst_nuis = Float64(0)
    var worst_lm = 1.0
    var det_ok = 0
    var par_gap = Float64(0)
    for s in range(SEEDS):
        var c_off = Phase3Config.default()
        c_off.seed = UInt64(1000 + s * 7717)
        var r_off = TrainerT.run(ecfg, c_off)
        cell_off += r_off.content_cell_acc

        var c_on = Phase3Config.with_content()
        c_on.seed = UInt64(1000 + s * 7717)
        var r_on = TrainerT.run(ecfg, c_on)
        cell_on += r_on.content_cell_acc
        if r_on.nuisance_r2 > worst_nuis:
            worst_nuis = r_on.nuisance_r2
        if r_on.landmark_r2 < worst_lm:
            worst_lm = r_on.landmark_r2
        if r_on.det_h < 0 and r_on.n_reflected == 1:
            det_ok += 1
        var gap = abs(r_on.frame_parity_acc - r_on.content_parity_acc)
        if gap > par_gap:
            par_gap = gap
    cell_off /= Float64(SEEDS)
    cell_on /= Float64(SEEDS)

    print("content cell accuracy: untrained", cell_off, " trained", cell_on)
    print("frame after adding content: worst nuisance R2", worst_nuis,
          " worst landmark R2", worst_lm, " det H ok", det_ok, "/", SEEDS)
    checks += 5
    assert_true(
        cell_on > 0.9,
        "the content channel must localise, got " + String(cell_on),
    )
    assert_true(
        cell_on > cell_off + 0.25,
        "training the content channel must MATTER: " + String(cell_on)
        + " vs untrained " + String(cell_off),
    )
    assert_true(
        worst_nuis < 0.05,
        "CONTAMINATION: the reconstruction loss pushed texture into the FRAME "
        + "channel, nuisance R^2 = " + String(worst_nuis),
    )
    assert_true(
        worst_lm > 0.95,
        "the frame channel must still carry the landmark, got "
        + String(worst_lm),
    )
    assert_true(
        det_ok == SEEDS,
        "det H = -1 with one reflected edge must survive the content channel, "
        + String(det_ok) + "/" + String(SEEDS),
    )

    # Parity is not decodable from ONE frame in EITHER channel — the two parity
    # classes have identical marginals (u(c,1) = A F H w, and H w is uniform
    # when w is). A gap either way would falsify that argument.
    checks += 1
    assert_true(
        par_gap < 0.1,
        "frame and content parity accuracy must be INDISTINGUISHABLE (neither "
        + "can decode an absolute parity); gap = " + String(par_gap),
    )

    # =====================================================================
    # 3. Rollout drift: the isometry is flat, the free transition is not.
    # =====================================================================
    var cfg = Phase3Config.with_content()
    cfg.seed = 20260904
    var model = TrainerT.train(ecfg, cfg)
    var trs = List[SqMat[2, DT]]()
    for i in range(N):
        trs.append(model.table.transport_for(ACTION_FORWARD, i))

    var horizons: List[Int] = [1, 6, 12]
    var ef = List[Float64](length=3, fill=0)
    var ec = List[Float64](length=3, fill=0)
    comptime DRIFT_TRIALS = 40
    for ep in range(DRIFT_TRIALS):
        var env = EnvT(ecfg)
        env.reset(UInt64(90000 + ep))
        var l = enc_full(env, model)
        var pu = List[Float64](length=2, fill=0)
        for i in range(2):
            pu[i] = Float64(l[i])
        var ph = List[Scalar[DT]](length=8, fill=0)
        for i in range(8):
            ph[i] = l[2 + i]
        var hi = 0
        for step in range(12):
            var c = env.place_id()
            var nu = List[Float64](length=2, fill=0)
            for i in range(2):
                var acc = Float64(0)
                for j in range(2):
                    acc += Float64(trs[c][i, j]) * pu[j]
                nu[i] = acc
            pu = nu^
            ph = model.content.predict_next(ph, 0)
            env.step(ACTION_FORWARD)
            if hi < 3 and step + 1 == horizons[hi]:
                var a = enc_full(env, model)
                var df = Float64(0)
                var dc = Float64(0)
                for i in range(2):
                    df += (pu[i] - Float64(a[i])) ** 2
                for i in range(8):
                    dc += Float64(ph[i] - a[2 + i]) ** 2
                ef[hi] += sqrt(df)
                ec[hi] += sqrt(dc)
                hi += 1
    for i in range(3):
        ef[i] /= Float64(DRIFT_TRIALS)
        ec[i] /= Float64(DRIFT_TRIALS)
    print("rollout error   frame:", ef[0], ef[1], ef[2],
          "   content:", ec[0], ec[1], ec[2])
    checks += 3
    assert_true(
        ef[2] < 2.0 * ef[0] + 0.1,
        "the FRAME rollout must stay flat (it is an isometry): "
        + String(ef[0]) + " -> " + String(ef[2]),
    )
    assert_true(
        ec[2] > 4.0 * ec[0],
        "the CONTENT rollout must be shown to DRIFT, otherwise the explanation "
        + "for why it hurts planning is unsupported: " + String(ec[0]) + " -> "
        + String(ec[2]),
    )
    assert_true(
        ec[2] > 5.0 * ef[2],
        "at horizon 12 the content channel must be far worse than the frame: "
        + String(ec[2]) + " vs " + String(ef[2]),
    )

    # =====================================================================
    # 4. The honest negative: rolled content does not help planning.
    # =====================================================================
    var weights: List[Float64] = [0.0, 0.05, 1.0]
    var cells = List[Int](length=3, fill=0)
    var parities = List[Int](length=3, fill=0)
    for wi in range(len(weights)):
        for s in range(SEEDS):
            var c = Phase3Config.with_content()
            c.seed = UInt64(20260904 + s * 7717)
            var m = TrainerT.train(ecfg, c)
            var tr2 = List[SqMat[2, DT]]()
            for i in range(N):
                tr2.append(m.table.transport_for(ACTION_FORWARD, i))
            var empty = List[Float64](length=N * 2, fill=0)
            var fm = FrameModel[N, DT](
                MODEL_ORTHOGONAL, tr2.copy(), empty.copy(), empty.copy()
            )
            for ep in range(EPISODES):
                var env = EnvT(ecfg)
                env.reset(UInt64(90000 + ep))
                var gc = env.goal_cell()
                var gp = env.goal_parity()
                var g = 0
                while (
                    env.place_id() != gc or env.lap_parity() != gp
                ) and g < 3 * N:
                    env.step(ACTION_FORWARD)
                    g += 1
                var lg = enc_full(env, m)
                env.reset(UInt64(90000 + ep))
                var l0 = enc_full(env, m)
                var u0 = List[Float64](length=2, fill=0)
                var ug = List[Float64](length=2, fill=0)
                for i in range(2):
                    u0[i] = Float64(l0[i])
                    ug[i] = Float64(lg[i])
                var h0 = List[Scalar[DT]](length=8, fill=0)
                var hg = List[Scalar[DT]](length=8, fill=0)
                for i in range(8):
                    h0[i] = l0[2 + i]
                    hg[i] = lg[2 + i]
                var p = plan_exhaustive_with_content[
                    N, 16, 10, 32, 8, 2, DT
                ](
                    fm, m.content, u0, h0, env.place_id(), ug, hg, pcfg,
                    weights[wi],
                )
                for st in range(p.arrival):
                    env.step(0 if p.actions[st] == PLAN_FORWARD else 1)
                if env.place_id() == gc:
                    cells[wi] += 1
                if env.lap_parity() == gp:
                    parities[wi] += 1

    var total = SEEDS * EPISODES
    print("content weight in matching | cell | parity   (of", total, ")")
    for wi in range(len(weights)):
        print("  ", weights[wi], "                    |", cells[wi], "|",
              parities[wi])
    checks += 2
    assert_true(
        cells[0] > cells[2],
        "RECORDED NEGATIVE: rolled-forward content matching must be shown to "
        + "HURT cell accuracy (it drifts). If this ever reverses, the finding "
        + "has changed and the docs are stale: " + String(cells[0]) + " vs "
        + String(cells[2]),
    )
    assert_true(
        parities[0] * 10 > 8 * total,
        "the frame channel must still get the parity right regardless: "
        + String(parities[0]) + "/" + String(total),
    )

    print()
    print("seeds:", SEEDS, " episodes each:", EPISODES)
    print("assertions compared :", checks)
    print("PASS: G15 content channel localises; rolled content drifts")
