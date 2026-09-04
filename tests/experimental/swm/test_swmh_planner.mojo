"""G13 + G14 — SWM Phase 5 gates: planning through the seam.

The task is goal-conditioned, and that is what makes it GAUGE-FREE. The encoder
learns some basis, so "the landmark appears on the left" is a direction the
planner cannot name. Handing it the goal's frame encoding instead — "reach a
state that looks like this" — is well posed in any gauge: matching `u` means
matching `F_{cell,parity}`, i.e. reaching the right cell on the right LAP.

**PARITY accuracy is the headline metric**, not raw goal success, because parity
is exactly what the double cover buys and a parity-blind model can only guess
it. Cell accuracy is reported beside it and is deliberately NOT the claim: the
frame alone is a weak place code (adjacent cells differ by ~0.3 rad, and the
frame channel identifies the goal in ~93-95% of episodes against ~98% for frame
+ content). Measured: of SWM's failures, 9 out of 10 are the RIGHT parity in the
wrong cell. That residue is the concrete argument for the content channel the
design carries beside the frame — it is not a failure of the frame channel at
the job the frame channel has.

**Planning is exhaustive over monotone walks, not CEM.** On a ring, walking
forward `k` steps for `k` in `[0, 2N)` reaches every state of the double cover
exactly once, so the optimum is a scan of `4N` rollouts. This separates MODEL
error from SEARCH error, and that separation turned out to matter: the CEM
planner's step penalty is a path-length prior that trades near goals against far
ones (penalty 0 gives 14/14 on far goals and 16/26 on near ones; penalty 0.01
gives 11/14 and 22/26). Those numbers are properties of the search, and a gate
on the world model should not be reading them.

Validates:
  G13  SWM gets the parity right; both parity-blind baselines sit near chance —
       the translation model (the constant sheaf: can locate the seam, cannot
       represent it) and a place-lookup model (frame as a function of the cell
       alone, i.e. what a model without the double cover is limited to).
  G14  Applying the monodromy a SECOND time, on top of edge transports that
       already carry the reflection, computes `H^2 = I` and destroys the parity
       while leaving cell tracking largely intact. That asymmetric damage is the
       signature; if it broke everything the ablation would just be "a broken
       model" and would say nothing about the monodromy.
  Confidence reaches the planner (§4.7): a distrusted edge is crossed less.

NOT claimed: a recurrent baseline. An RSSM or GRU can learn the parity bit, so
that comparison is about sample efficiency at a matched budget — a study, not a
gate, and the plan defers it with P5/CSCG. The claim made here is structural:
models that cannot REPRESENT the seam cannot get the parity, whatever their
capacity.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_planner.mojo
"""

from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.ablations import fit_translations
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import (
    MobiusRing,
    MobiusConfig,
    ACTION_FORWARD,
)
from mojo_rl.experimental.swm.planner import (
    FrameModel,
    PlannerConfig,
    plan_exhaustive,
    MODEL_ORTHOGONAL,
    MODEL_TRANSLATION,
    MODEL_PLACE_LOOKUP,
    PLAN_FORWARD,
)

comptime DT = DType.float64
comptime N = 12
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]
comptime EnvT = MobiusRing[12, 6, 16, DT]
comptime TRAIN_SEEDS = 3
comptime EPISODES = 40


def encode_frame(mut env: EnvT, model: TrainerT.ModelT) raises -> List[Float64]:
    var o = env.observation()
    var hid = List[Scalar[DT]](length=32, fill=0)
    var lat = List[Scalar[DT]](length=10, fill=0)
    model.enc.forward(o, hid, lat)
    var u = List[Float64](length=2, fill=0)
    for i in range(2):
        u[i] = Float64(lat[i])
    return u^


def evaluate(
    mut fm: FrameModel[N, DT],
    model: TrainerT.ModelT,
    ecfg: MobiusConfig,
    pcfg: PlannerConfig,
    mut counts: List[Int],
    mut seam_crossings: Int,
) raises:
    """`counts = [parity_ok, n, cell_ok, goal_ok]`, judged at the arrival step."""
    for ep in range(EPISODES):
        var env = EnvT(ecfg)
        env.reset(UInt64(90000 + ep))
        var gc = env.goal_cell()
        var gp = env.goal_parity()

        # Show the agent what the goal looks like (one noisy observation of it).
        var guard = 0
        while (
            env.place_id() != gc or env.lap_parity() != gp
        ) and guard < 3 * N:
            env.step(ACTION_FORWARD)
            guard += 1
        var u_goal = encode_frame(env, model)

        env.reset(UInt64(90000 + ep))
        var u0 = encode_frame(env, model)
        var p = plan_exhaustive[N, DT](fm, u0, env.place_id(), u_goal, pcfg)
        for s in range(p.arrival):
            var a = p.actions[s]
            if fm.edge_of(env.place_id(), a) == N - 1:
                seam_crossings += 1
            env.step(0 if a == PLAN_FORWARD else 1)

        counts[1] += 1
        if env.lap_parity() == gp:
            counts[0] += 1
        if env.place_id() == gc:
            counts[2] += 1
        if env.place_id() == gc and env.lap_parity() == gp:
            counts[3] += 1


def main() raises:
    var checks = 0
    var ecfg = MobiusConfig.default_mobius()
    var pcfg = PlannerConfig.default()

    var labels: List[String] = [
        "SWM (orthogonal)   ",
        "translation (B)    ",
        "place lookup       ",
        "SWM + monodromy x2 ",
    ]
    var tot = List[Int](length=4 * 4, fill=0)

    for seed_i in range(TRAIN_SEEDS):
        var cfg = Phase3Config.default()
        cfg.seed = UInt64(20260904 + seed_i * 7717)
        var model = TrainerT.train(ecfg, cfg)
        var roll = TrainerT.encode_rollouts(model, ecfg, cfg, 24)

        var trs = List[SqMat[2, DT]]()
        for i in range(N):
            trs.append(model.table.transport_for(ACTION_FORWARD, i))
        var ts = fit_translations[DT](roll.batches)
        var tsf = List[Float64](length=N * 2, fill=0)
        for i in range(N * 2):
            tsf[i] = Float64(ts[i])
        var lk = List[Float64](length=N * 2, fill=0)
        var cnt = List[Float64](length=N, fill=0)
        for ep in range(roll.n_episodes):
            for t in range(roll.n_frames):
                var c = t % N
                cnt[c] += 1.0
                for i in range(2):
                    lk[c * 2 + i] += Float64(
                        roll.seq_u[(ep * roll.n_frames + t) * 2 + i]
                    )
        for c in range(N):
            for i in range(2):
                lk[c * 2 + i] /= cnt[c]

        var empty = List[Float64](length=N * 2, fill=0)
        var eye = List[SqMat[2, DT]]()
        for _ in range(N):
            eye.append(SqMat[2, DT].identity())

        for k in range(4):
            var fm: FrameModel[N, DT]
            if k == 0 or k == 3:
                fm = FrameModel[N, DT](
                    MODEL_ORTHOGONAL, trs.copy(), empty.copy(), empty.copy()
                )
                if k == 3:
                    fm.apply_monodromy_twice = True
            elif k == 1:
                fm = FrameModel[N, DT](
                    MODEL_TRANSLATION, eye.copy(), tsf.copy(), empty.copy()
                )
            else:
                fm = FrameModel[N, DT](
                    MODEL_PLACE_LOOKUP, eye.copy(), empty.copy(), lk.copy()
                )
            var c4 = List[Int](length=4, fill=0)
            var seam = 0
            evaluate(fm, model, ecfg, pcfg, c4, seam)
            for j in range(4):
                tot[k * 4 + j] += c4[j]

    print("model               | PARITY correct      | cell | goal")
    var parity_rate = List[Float64](length=4, fill=0)
    for k in range(4):
        parity_rate[k] = Float64(tot[k * 4 + 0]) / Float64(tot[k * 4 + 1])
        print(
            labels[k], "|", tot[k * 4 + 0], "/", tot[k * 4 + 1],
            "=", parity_rate[k], "|", tot[k * 4 + 2], "|", tot[k * 4 + 3],
        )

    checks += 1
    assert_true(
        parity_rate[0] > 0.9,
        "SWM must get the PARITY right — that is the whole claim. got "
        + String(tot[0]) + "/" + String(tot[1]),
    )
    for k in range(1, 3):
        checks += 2
        assert_true(
            parity_rate[k] < 0.7,
            labels[k] + "is parity-blind and must be near chance, got "
            + String(parity_rate[k]),
        )
        assert_true(
            parity_rate[0] > parity_rate[k] + 0.25,
            "SWM must beat " + labels[k] + "decisively on parity: "
            + String(parity_rate[0]) + " vs " + String(parity_rate[k]),
        )
    checks += 1
    assert_true(
        tot[2] > tot[1 * 4 + 2] and tot[2] > tot[2 * 4 + 2],
        "SWM must also localise better than the baselines — otherwise the "
        + "parity result could be coming from a model that has stopped "
        + "tracking where it is at all",
    )

    # G14 -------------------------------------------------------------------
    checks += 2
    assert_true(
        parity_rate[3] < 0.7,
        "applying the monodromy twice computes H^2 = I and must destroy the "
        + "PARITY prediction; got " + String(parity_rate[3]),
    )
    assert_true(
        tot[3 * 4 + 2] > tot[3 * 4 + 3],
        "the double-application must damage the PARITY specifically while "
        + "leaving cell tracking largely intact — otherwise the ablation is "
        + "just 'a broken model' and says nothing about the monodromy",
    )

    # ---- confidence reaches the planner (§4.7) -----------------------------
    var cfg = Phase3Config.default()
    cfg.seed = 20260904
    var model = TrainerT.train(ecfg, cfg)
    var trs2 = List[SqMat[2, DT]]()
    for i in range(N):
        trs2.append(model.table.transport_for(ACTION_FORWARD, i))
    var empty2 = List[Float64](length=N * 2, fill=0)
    var seam_trusted = 0
    var seam_distrusted = 0
    for arm in range(2):
        var fm = FrameModel[N, DT](
            MODEL_ORTHOGONAL, trs2.copy(), empty2.copy(), empty2.copy()
        )
        var pc = pcfg
        if arm == 1:
            fm.edge_w[N - 1] = 0.0
            pc.trust_lambda = 40.0
        var c4 = List[Int](length=4, fill=0)
        var seam = 0
        evaluate(fm, model, ecfg, pc, c4, seam)
        if arm == 0:
            seam_trusted = seam
        else:
            seam_distrusted = seam
    print("seam crossings in executed plans: trusted", seam_trusted,
          " distrusted", seam_distrusted)
    checks += 2
    assert_true(
        seam_trusted > 0,
        "the trusted arm must cross the seam at all, else this is vacuous",
    )
    assert_true(
        seam_distrusted < seam_trusted,
        "a distrusted edge must be crossed LESS often: "
        + String(seam_distrusted) + " vs " + String(seam_trusted),
    )

    print()
    print("training seeds:", TRAIN_SEEDS, " episodes each:", EPISODES)
    print("assertions compared :", checks)
    print("PASS: G13 parity through the seam, G14 no double monodromy")
