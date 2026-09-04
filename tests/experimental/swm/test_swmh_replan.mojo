"""G21 — SWM Phase 7: the content channel as a place code, and one
observation at arrival.

6a measured that the content channel localises from an observation (cell
accuracy 0.998) and drifts when rolled forward (0.405 -> 4.398 over 12 steps),
so matching on rolled content HURT the planner. Phase 7 first tried the obvious
control-loop answer — re-plan every step — and it does not work; the numbers
are kept here as recorded negatives because they say what the channel is for:

  naive replan, frame + content (rolled, weight 1)   cell 103, parity DOWN to 106
  naive replan, frame only                           cell 28, budget exhausted

The frame-only loop OSCILLATES: "stay when the planner's best is to stay" is
unstable on a noisy frame match, so the agent wanders for the whole budget.
Rolled content in the loop is no better than open-loop. And a +-1 probe at
arrival helps nothing (97 vs 96), because the open-loop failures are 2-3 cells
away, never adjacent — the diagnostic that led here.

What works is to ask the content channel only what 6a showed it can do. The
rollout's content term is a LOOKUP of the stored centroid of the cell the
rollout stands in, matched against the goal's observed `h`; the frame still
decides the parity, the content decides the cell, and nothing that drifts is
rolled. Then ONE observation at arrival, and a single re-plan if the observed
content disagrees with the planned cell.

  open-loop, frame only              -- Phase 5's number
  naive replan, frame + content      -- recorded negative
  naive replan, frame only           -- recorded negative
  place-code content + verify        -- GATED: cell >= 118/120
  place-code, u-only (weight 0)      -- CONTROL: must equal the open-loop arm

**What is left is structural, and it is G22's reading.** With the cell pinned
(120/120) the frame has to choose between exactly two states one lap apart,
which differ by the reflection `M`. That choice is undecidable when the goal's
landmark lies along `M`'s axis — the holonomy's FIXED SUBSPACE, the one
direction that admits a global frame and therefore carries no parity. Measured:
every parity failure (10/120) falls in the lowest third of the frame's own
margin `|u_k - u_{k+N}|^2`, none in the upper two thirds. Gated as such: the
failures must be confined to the low-margin band, so the residue is the fixed
subspace and not the planner.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_replan.mojo
"""

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
    plan_exhaustive,
    plan_exhaustive_with_content,
    plan_exhaustive_with_place_code,
    MODEL_ORTHOGONAL,
    PLAN_FORWARD,
)

comptime DT = DType.float64
comptime N = 12
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]
comptime EnvT = MobiusRing[12, 6, 16, DT]
comptime SEEDS = 3
comptime EPISODES = 40
comptime BUDGET = 4 * N


def enc_full(mut env: EnvT, m: TrainerT.ModelT) raises -> List[Scalar[DT]]:
    var o = env.observation()
    var hid = List[Scalar[DT]](length=32, fill=0)
    var lat = List[Scalar[DT]](length=10, fill=0)
    m.enc.forward(o, hid, lat)
    return lat^


def split_u(l: List[Scalar[DT]]) -> List[Float64]:
    var u = List[Float64](length=2, fill=0)
    for i in range(2):
        u[i] = Float64(l[i])
    return u^


def split_h(l: List[Scalar[DT]]) -> List[Scalar[DT]]:
    var h = List[Scalar[DT]](length=8, fill=0)
    for i in range(8):
        h[i] = l[2 + i]
    return h^


def nearest_cell(
    h: List[Scalar[DT]], centroids: List[Scalar[DT]]
) -> Int:
    var best = 1e300
    var arg = 0
    for c in range(N):
        var d = Float64(0)
        for i in range(8):
            var x = Float64(centroids[c * 8 + i] - h[i])
            d += x * x
        if d < best:
            best = d
            arg = c
    return arg


def learn_centroids(
    m: TrainerT.ModelT, ecfg: MobiusConfig
) raises -> List[Scalar[DT]]:
    """Per-cell mean of `h` over the agent's own visits (oracle cell, v2 §4.1)."""
    var acc = List[Scalar[DT]](length=N * 8, fill=0)
    var cnt = List[Float64](length=N, fill=0)
    for ep in range(24):
        var env = EnvT(ecfg)
        env.reset(UInt64(70000 + ep))
        for _ in range(3 * N + 1):
            var l = enc_full(env, m)
            var c = env.place_id()
            for i in range(8):
                acc[c * 8 + i] += l[2 + i]
            cnt[c] += 1
            env.step(ACTION_FORWARD)
    for c in range(N):
        for i in range(8):
            acc[c * 8 + i] = acc[c * 8 + i] / Scalar[DT](cnt[c])
    return acc^


def run_arm(
    arm: Int,
    m: TrainerT.ModelT,
    mut fm: FrameModel[N, DT],
    ecfg: MobiusConfig,
    pcfg: PlannerConfig,
    centroids: List[Scalar[DT]],
    mut counts: List[Int],
    mut seps: List[Float64],
    mut par_ok: List[Bool],
) raises:
    """`counts = [parity_ok, cell_ok, goal_ok, n, steps_used, replans]`. For
    arm 3, `seps`/`par_ok` record the parity margin and outcome per episode."""
    for ep in range(EPISODES):
        var env = EnvT(ecfg)
        env.reset(UInt64(90000 + ep))
        var gc = env.goal_cell()
        var gp = env.goal_parity()
        var g = 0
        while (env.place_id() != gc or env.lap_parity() != gp) and g < 3 * N:
            env.step(ACTION_FORWARD)
            g += 1
        var lg = enc_full(env, m)
        var ug = split_u(lg)
        var hg = split_h(lg)
        env.reset(UInt64(90000 + ep))

        if arm == 0:
            var l0 = enc_full(env, m)
            var p = plan_exhaustive[N, DT](fm, split_u(l0), env.place_id(), ug, pcfg)
            for st in range(p.arrival):
                env.step(0 if p.actions[st] == PLAN_FORWARD else 1)
            counts[4] += p.arrival
        elif arm >= 3:
            var w = 1.0 if arm == 3 else 0.0
            var goal_cell_est = nearest_cell(hg, centroids)
            var first = True
            for _ in range(2):
                var l0 = enc_full(env, m)
                var u0 = split_u(l0)
                var c0 = env.place_id()
                var p = plan_exhaustive_with_place_code[N, DT](
                    fm, u0, c0, ug, hg, centroids, 8, pcfg, w
                )
                if first and arm == 3:
                    # The frame's parity margin at the goal cell: distance
                    # between the two candidates one lap apart. Small when the
                    # goal's landmark lies along the reflection axis — the
                    # holonomy's FIXED SUBSPACE, where parity is invisible.
                    var dir = p.actions[0] if p.arrival > 0 else PLAN_FORWARD
                    var ua = u0.copy()
                    var cell = c0
                    for _ in range(p.arrival):
                        ua = fm.step(ua, cell, dir)
                        cell = fm.next_cell(cell, dir)
                    var ub = ua.copy()
                    for _ in range(N):
                        ub = fm.step(ub, cell, dir)
                        cell = fm.next_cell(cell, dir)
                    var sep = Float64(0)
                    for i in range(2):
                        sep += (ua[i] - ub[i]) * (ua[i] - ub[i])
                    seps.append(sep)
                    first = False
                for st in range(p.arrival):
                    env.step(0 if p.actions[st] == PLAN_FORWARD else 1)
                counts[4] += p.arrival
                # One observation at arrival: does the content agree?
                var la = enc_full(env, m)
                if arm != 3 or nearest_cell(split_h(la), centroids) == goal_cell_est:
                    break
                counts[5] += 1
            if arm == 3:
                par_ok.append(env.lap_parity() == gp)
        else:
            var used = 0
            while used < BUDGET:
                var l0 = enc_full(env, m)
                var p = plan_exhaustive_with_content[N, 16, 10, 32, 8, 2, DT](
                    fm, m.content, split_u(l0), split_h(l0), env.place_id(),
                    ug, hg, pcfg, 1.0 if arm == 1 else 0.0,
                )
                if p.arrival == 0:
                    break
                env.step(0 if p.actions[0] == PLAN_FORWARD else 1)
                used += 1
            counts[4] += used

        counts[3] += 1
        if env.lap_parity() == gp:
            counts[0] += 1
        if env.place_id() == gc:
            counts[1] += 1
        if env.place_id() == gc and env.lap_parity() == gp:
            counts[2] += 1


def main() raises:
    var checks = 0
    var ecfg = MobiusConfig.default_mobius()
    var pcfg = PlannerConfig.default()
    var names: List[String] = [
        "open-loop, frame only        ",
        "naive replan, frame + content",
        "naive replan, frame only     ",
        "place-code content + verify  ",
        "place-code, u-only (control) ",
    ]
    var tot = List[Int](length=5 * 6, fill=0)
    var seps = List[Float64]()
    var par_ok = List[Bool]()
    for s in range(SEEDS):
        var cfg = Phase3Config.with_content()
        cfg.seed = UInt64(20260904 + s * 7717)
        var m = TrainerT.train(ecfg, cfg)
        var trs = List[SqMat[2, DT]]()
        for i in range(N):
            trs.append(m.table.transport_for(ACTION_FORWARD, i))
        var empty = List[Float64](length=N * 2, fill=0)
        var centroids = learn_centroids(m, ecfg)
        for arm in range(5):
            var fm = FrameModel[N, DT](
                MODEL_ORTHOGONAL, trs.copy(), empty.copy(), empty.copy()
            )
            var c = List[Int](length=6, fill=0)
            run_arm(arm, m, fm, ecfg, pcfg, centroids, c, seps, par_ok)
            for j in range(6):
                tot[arm * 6 + j] += c[j]

    var total = SEEDS * EPISODES
    print("arm                           | parity | cell | goal | mean steps | re-plans  (of", total, ")")
    for arm in range(5):
        print(names[arm], "|", tot[arm * 6 + 0], "|", tot[arm * 6 + 1], "|",
              tot[arm * 6 + 2], "|", Float64(tot[arm * 6 + 4]) / Float64(total),
              "|", tot[arm * 6 + 5])
    # Parity failures against the frame's own margin, in thirds.
    var sorted = seps.copy()
    for i in range(1, len(sorted)):
        var x = sorted[i]
        var j = i - 1
        while j >= 0 and sorted[j] > x:
            sorted[j + 1] = sorted[j]
            j -= 1
        sorted[j + 1] = x
    var t1 = sorted[len(sorted) // 3]
    var t2 = sorted[(2 * len(sorted)) // 3]
    var band_n = List[Int](length=3, fill=0)
    var band_fail = List[Int](length=3, fill=0)
    for i in range(len(seps)):
        var b = 0 if seps[i] < t1 else (1 if seps[i] < t2 else 2)
        band_n[b] += 1
        if not par_ok[i]:
            band_fail[b] += 1
    print("parity margin |u_k - u_{k+N}|^2 in thirds: low <", t1, " mid <", t2)
    print("  parity failures by band: low", band_fail[0], "/", band_n[0],
          " mid", band_fail[1], "/", band_n[1], " high", band_fail[2], "/",
          band_n[2])

    var pc_cell = tot[3 * 6 + 1]
    var pc_par = tot[3 * 6 + 0]
    var ctrl_cell = tot[4 * 6 + 1]
    var open_cell = tot[0 * 6 + 1]
    checks += 6
    assert_true(
        pc_cell >= 118,
        "place-code content + verify must pin the cell (>= 118/120): "
        + String(pc_cell),
    )
    assert_true(
        pc_cell > open_cell + 8 and pc_cell > tot[1 * 6 + 1] + 8,
        "...and must beat both the open-loop baseline and ROLLED content by a "
        + "visible margin: " + String(pc_cell) + " vs " + String(open_cell)
        + " and " + String(tot[1 * 6 + 1]),
    )
    assert_true(
        band_fail[1] == 0 and band_fail[2] == 0 and band_fail[0] > 0
        and pc_par * 10 >= 9 * total,
        "the parity residue must be the FIXED SUBSPACE: every failure in the "
        + "lowest margin third, none above, and >= 90% overall. got low "
        + String(band_fail[0]) + " mid " + String(band_fail[1]) + " high "
        + String(band_fail[2]) + ", parity " + String(pc_par),
    )
    assert_true(
        ctrl_cell == open_cell,
        "CONTROL: the place-code planner at content weight 0 must be exactly "
        + "the open-loop frame planner: " + String(ctrl_cell) + " vs "
        + String(open_cell),
    )
    assert_true(
        tot[0 * 6 + 0] * 10 >= 9 * total,
        "the open-loop baseline must still get the parity (Phase 5's claim): "
        + String(tot[0 * 6 + 0]) + "/" + String(total),
    )
    assert_true(
        tot[2 * 6 + 1] < open_cell and tot[1 * 6 + 1] < 115,
        "RECORDED NEGATIVE: the naive loops must be shown to be the wrong "
        + "design (frame-only oscillates below the open-loop baseline; rolled "
        + "content does not reach the gate). If this reverses, the docstring is "
        + "stale: " + String(tot[2 * 6 + 1]) + ", " + String(tot[1 * 6 + 1]),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G21 the content channel is a place code, not a dynamics")
