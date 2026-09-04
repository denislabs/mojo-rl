"""G26 — SWM Phase 8d: the end-to-end task, with no oracle in the loop.

Everything before this gate was a diagnostic. This is the capability: on the
aliased 2D world the agent explores, labels its visits with its own content
channel, clones the labels the graph contradicts, fits transports per clone,
is shown ONE observation of a goal, and plans on its own map through the
double cover to reach that goal — the right cell AND the right parity, where
parity is the homotopy class of the path through the reversing seam and the
goal is the state showing the landmark furthest left (an argmax over all
2 x 30 states, as on the ring). The oracle appears only in the judge.

Under aliasing the goal's content matches TWO clones; the frame channel picks
between them exactly as it picks the parity, so the goal is reached in either
case. Episodes whose goal sits on the aliased pair are counted separately.

Arms, on the same trained models and the same learned map:

  SWM on the learned map           -- GATED: goal reached (cell AND parity)
                                      in >= 85 % of episodes
  constant sheaf on the learned map -- frames never transported: parity-blind,
                                      must sit near chance on parity
  SWM on the ORACLE map            -- true cells for the vertices, transports
                                      fitted on the same encoded pairs (the
                                      planted ones live in the world's gauge,
                                      not the encoder's): the upper bound;
                                      the learned map must be within a few
                                      episodes of it

The parity misses are attributed the way G21 attributed the ring's: by the
frame's own margin, the gap between the chosen goal state and the runner-up
(the same cell at the other parity). Every miss must fall in the lowest
third of that margin — the goal's landmark along the reflection axis, the
holonomy's fixed subspace, where the two parities look alike — and none
above.

Then a second leg with the goal SET on the aliased pair — both cells, both
parities, ten landmark draws each — because the argmax goal happened to land
there in 0 of 80 episodes and the claim would otherwise go untested.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_klein_task.mojo
"""

from std.testing import assert_true

from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.klein_grid import (
    KleinWorld,
    KleinWorldConfig,
    ALIAS_LOCAL,
    ACT_X,
    ACT_Y,
)
from mojo_rl.experimental.swm.place_graph import PlaceGraph
from mojo_rl.experimental.swm.map_builder import (
    label_walk,
    count_labels,
    split_until_stable,
    score_map,
    clone_graph,
)
from mojo_rl.experimental.swm.graph_planner import (
    plan_double_cover,
    nearest_centroid,
    clone_centroids,
)

comptime DT = DType.float64
comptime W = 6
comptime H = 5
comptime NP = W * H
comptime TrainerT = SwmPhase3[NP, 6, 16, 32, 8, DT]
comptime WorldT = KleinWorld[W, H, 6, 16]
comptime SEEDS = 2
comptime EPISODES = 40
comptime LABEL_THRESHOLD = 2.0
comptime FRAME_TOL = 0.5
comptime ALIASED_A = 1 * W + 1
comptime ALIASED_B = 3 * W + 4


def encode(mut env: WorldT, m: TrainerT.ModelT) raises -> List[Float64]:
    var o = env.observation()
    var hid = List[Scalar[DT]](length=32, fill=0)
    var lat = List[Scalar[DT]](length=10, fill=0)
    m.enc.forward(o, hid, lat)
    var out = List[Float64](length=10, fill=0)
    for i in range(10):
        out[i] = Float64(lat[i])
    return out^


def split(l: List[Float64], mut u: List[Float64], mut h: List[Float64]):
    u = List[Float64](length=2, fill=0)
    h = List[Float64](length=8, fill=0)
    for i in range(2):
        u[i] = l[i]
    for i in range(8):
        h[i] = l[2 + i]


def main() raises:
    var checks = 0
    var wcfg = KleinWorldConfig.flat_klein().with_alias(ALIAS_LOCAL)
    var names: List[String] = [
        "SWM, learned map     ",
        "constant sheaf       ",
        "SWM, oracle map      ",
    ]
    # counts per arm: [goal, cell, parity, n, goal_on_aliased_n, goal_on_aliased_ok, not_found]
    var tot = List[Int](length=3 * 7, fill=0)
    var map_clones = 0
    var map_purity = Float64(0)
    var alias_n = 0
    var alias_ok = 0
    var alias_cell_only = 0
    var alias_wrong_twin = 0
    var alias_goal_clones = 0
    var margins = List[Float64]()
    var margin_ok = List[Bool]()
    var oracle_sep = List[Float64]()

    for s in range(SEEDS):
        var cfg = Phase3Config.with_content()
        cfg.seed = UInt64(31000 + s * 7717)
        var env = WorldT(wcfg)
        var m = TrainerT.train_world(env, cfg, NP)

        # ---- the agent's own map --------------------------------------
        var rec = TrainerT.record_walks(env, m, 16, 4 * NP, 41000)
        var labels = label_walk(rec, LABEL_THRESHOLD)
        var clones = split_until_stable(labels, rec.action, 2)
        var sc = score_map(clones, rec.true_place, NP)
        map_clones += sc.n_labels
        map_purity += sc.purity
        var g = clone_graph(rec, clones, 2)
        var cent = clone_centroids(rec.h, clones, 8, sc.n_labels)

        # ---- the oracle map: true cells, transports fitted on encoded pairs
        var g_true = clone_graph(rec, rec.true_place, 2)
        var cent_true = List[Float64](length=NP * 8, fill=0)
        var cnt_true = List[Float64](length=NP, fill=0)
        for t in range(rec.size()):
            var p = rec.true_place[t]
            cnt_true[p] += 1
            for i in range(8):
                cent_true[p * 8 + i] += rec.h[t * 8 + i]
        for p in range(NP):
            for i in range(8):
                cent_true[p * 8 + i] /= cnt_true[p]

        for arm in range(3):
            for ep in range(EPISODES):
                env.reset(UInt64(90000 + ep))
                var gc = env.goal_cell()
                var gp = env.goal_parity()
                # show the agent the goal: walk there along the reference path
                var guard = 0
                # reference path: x-loops for parity, then +x to the column,
                # then +y up it
                while (env.place_id() != gc or env.lap_parity() != gp) and guard < 8 * NP:
                    var a = ACT_X
                    if env.lap_parity() == gp and env.x == gc % W:
                        a = ACT_Y
                    env.step(a)
                    guard += 1
                var lg = encode(env, m)
                var ug = List[Float64]()
                var hg = List[Float64]()
                split(lg, ug, hg)
                env.reset(UInt64(90000 + ep))
                var l0 = encode(env, m)
                var u0 = List[Float64]()
                var h0 = List[Float64]()
                split(l0, u0, h0)

                var use_true = arm == 2
                var start: Int
                if use_true:
                    start = env.place_id()
                else:
                    start = nearest_centroid(h0, cent, 8, sc.n_labels)
                var p = plan_double_cover(
                    g_true if use_true else g,
                    cent_true if use_true else cent,
                    8, start, u0, ug, hg, LABEL_THRESHOLD, FRAME_TOL,
                    arm != 1,
                )
                if not p.found:
                    tot[arm * 7 + 6] += 1
                for k in range(len(p.actions)):
                    env.step(p.actions[k])
                var on_alias = gc == ALIASED_A or gc == ALIASED_B
                var ok_cell = env.place_id() == gc
                var ok_par = env.lap_parity() == gp
                if arm == 0:
                    # No runner-up = the two parities' frames fell within
                    # FRAME_TOL of each other and were merged into one search
                    # state: the frame had no choice to make. Margin zero.
                    var mg = p.u_err_runner_up - p.u_err
                    if p.u_err_runner_up > 1e100:
                        mg = 0.0
                    margins.append(mg)
                    margin_ok.append(ok_par)
                    # The oracle's own separation of the two parity frames at
                    # the goal cell: 2 |w_perp| to the reflection axis, the
                    # distance to the holonomy's fixed subspace.
                    var f0 = env.frame_at(gc, 0)
                    var f1 = env.frame_at(gc, 1)
                    var sep = Float64(0)
                    for i in range(2):
                        var a0 = Float64(f0[i, 0] * env.w[0] + f0[i, 1] * env.w[1])
                        var a1 = Float64(f1[i, 0] * env.w[0] + f1[i, 1] * env.w[1])
                        sep += (a0 - a1) * (a0 - a1)
                    oracle_sep.append(sep)
                tot[arm * 7 + 3] += 1
                if ok_cell:
                    tot[arm * 7 + 1] += 1
                if ok_par:
                    tot[arm * 7 + 2] += 1
                if ok_cell and ok_par:
                    tot[arm * 7 + 0] += 1
                if on_alias:
                    tot[arm * 7 + 4] += 1
                    if ok_cell and ok_par:
                        tot[arm * 7 + 5] += 1

        # ---- leg B: goals SET on the aliased pair, both cells, both parities
        for gi in range(4):
            var gc = ALIASED_A if gi < 2 else ALIASED_B
            var gp = gi % 2
            for ep in range(10):
                env.reset(UInt64(95000 + ep))
                var guard = 0
                while (env.place_id() != gc or env.lap_parity() != gp) and guard < 8 * NP:
                    var a = ACT_X
                    if env.lap_parity() == gp and env.x == gc % W:
                        a = ACT_Y
                    env.step(a)
                    guard += 1
                var lg = encode(env, m)
                var ug = List[Float64]()
                var hg = List[Float64]()
                split(lg, ug, hg)
                env.reset(UInt64(95000 + ep))
                var l0 = encode(env, m)
                var u0 = List[Float64]()
                var h0 = List[Float64]()
                split(l0, u0, h0)
                var start = nearest_centroid(h0, cent, 8, sc.n_labels)
                var p = plan_double_cover(
                    g, cent, 8, start, u0, ug, hg, LABEL_THRESHOLD, FRAME_TOL, True
                )
                alias_goal_clones += p.n_goal_clones
                for k in range(len(p.actions)):
                    env.step(p.actions[k])
                alias_n += 1
                if env.place_id() == gc and env.lap_parity() == gp:
                    alias_ok += 1
                elif env.place_id() == gc:
                    alias_cell_only += 1
                elif env.place_id() == ALIASED_A or env.place_id() == ALIASED_B:
                    alias_wrong_twin += 1

    var total = SEEDS * EPISODES
    print("learned map: clones", map_clones, "/", SEEDS * NP, " purity",
          map_purity / Float64(SEEDS))
    print("arm                   | goal | cell | parity | goal on aliased pair | no plan   (of", total, ")")
    for arm in range(3):
        print(names[arm], "|", tot[arm * 7 + 0], "|", tot[arm * 7 + 1], "|",
              tot[arm * 7 + 2], "|", tot[arm * 7 + 5], "/", tot[arm * 7 + 4],
              "|", tot[arm * 7 + 6])
    # ---- attribute the parity misses: the frame's own margin, in thirds ----
    var sorted = margins.copy()
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
    var fail_margins = String("")
    for i in range(len(margins)):
        var b = 0 if margins[i] < t1 else (1 if margins[i] < t2 else 2)
        band_n[b] += 1
        if not margin_ok[i]:
            band_fail[b] += 1
            fail_margins += " " + String(Int(margins[i] * 1000))
    print("parity margin (runner-up - best |u - u_goal|, 0 = parities merged) in thirds: low <", t1,
          " mid <", t2, "  max", sorted[len(sorted) - 1])
    print("  parity misses by band: low", band_fail[0], "/", band_n[0], " mid",
          band_fail[1], "/", band_n[1], " high", band_fail[2], "/", band_n[2],
          "  miss margins x1000:" + fail_margins)
    # independent check: the ORACLE separation of the two parity frames
    var osorted = oracle_sep.copy()
    for i in range(1, len(osorted)):
        var x = osorted[i]
        var j = i - 1
        while j >= 0 and osorted[j] > x:
            osorted[j + 1] = osorted[j]
            j -= 1
        osorted[j + 1] = x
    var o1 = osorted[len(osorted) // 3]
    var o2 = osorted[(2 * len(osorted)) // 3]
    var oband_fail = List[Int](length=3, fill=0)
    var miss_seps = String("")
    for i in range(len(oracle_sep)):
        var b = 0 if oracle_sep[i] < o1 else (1 if oracle_sep[i] < o2 else 2)
        if not margin_ok[i]:
            oband_fail[b] += 1
            miss_seps += " " + String(Int(oracle_sep[i] * 1000))
    print("  oracle |F0 w - F1 w|^2 at the goal, thirds: low <", o1, " mid <", o2,
          "  misses by band: low", oband_fail[0], " mid", oband_fail[1], " high",
          oband_fail[2], "  miss separations x1000:" + miss_seps)
    print("goal SET on the aliased pair:", alias_ok, "/", alias_n,
          " reached;", alias_cell_only, "right cell wrong parity;",
          alias_wrong_twin, "went to the aliased TWIN;  goal clones per plan",
          Float64(alias_goal_clones) / Float64(alias_n))
    checks += 7
    assert_true(
        band_fail[1] == 0 and band_fail[2] == 0 and band_fail[0] > 0,
        "the parity misses must be the FIXED SUBSPACE: every miss in the "
        + "lowest third of the frame's margin, none above. got low "
        + String(band_fail[0]) + " mid " + String(band_fail[1]) + " high "
        + String(band_fail[2]),
    )
    assert_true(
        oband_fail[1] == 0 and oband_fail[2] == 0,
        "...and independently, every miss must be a goal whose two parity "
        + "frames the ORACLE places closest together: mid " + String(oband_fail[1])
        + " high " + String(oband_fail[2]),
    )
    assert_true(
        map_clones == SEEDS * NP and map_purity / Float64(SEEDS) > 0.97,
        "the learned map must be exact before the task means anything",
    )
    assert_true(
        tot[0] * 100 >= 85 * total,
        "SWM on its own map must reach the goal (cell AND parity) in >= 85%: "
        + String(tot[0]) + "/" + String(total),
    )
    assert_true(
        tot[0] >= tot[2 * 7 + 0] - 6,
        "the learned map must cost at most a few episodes against the oracle "
        + "map: " + String(tot[0]) + " vs " + String(tot[2 * 7 + 0]),
    )
    assert_true(
        tot[1 * 7 + 2] * 100 <= 65 * total and tot[1 * 7 + 0] * 100 <= 65 * total,
        "CONTROL: the constant sheaf is parity-blind and must sit near chance: "
        + "parity " + String(tot[1 * 7 + 2]) + " goal " + String(tot[1 * 7 + 0]),
    )
    assert_true(
        alias_goal_clones * 10 >= 19 * alias_n and alias_ok * 100 >= 80 * alias_n,
        "with the goal on the aliased pair the content offers ~2 goal clones "
        + "and the FRAME must pick the right one and the right parity in >= 80%: "
        + String(alias_ok) + "/" + String(alias_n) + ", goal clones per plan "
        + String(Float64(alias_goal_clones) / Float64(alias_n)),
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G26 the agent maps, plans and reaches a parity-dependent goal with no oracle")
