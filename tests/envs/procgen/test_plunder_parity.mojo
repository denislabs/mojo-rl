"""Plunder reset+step parity vs the Qt-free C++ probe.

Combined. Validates the reset (image_permutation, lane dirs/vels, panels with
collision-retry, reposition_agent with retries) and the step (lane spawning,
firing UP, bullet↔ship/panel collision, juice decay, target quota) over a
fire-heavy 400-step tape. Ground truth = `scratchpad/plunder_probe.cpp`.
Asset-free/fast. See `docs/PROCGEN_PLUNDER_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import PlunderGame, PANEL
from mojo_rl.envs.procgen.games.plunder import DIST_EASY, DIST_HARD

comptime STEPS = 400


def _tape(step: Int) -> Int:
    var t: List[Int] = [9, 4, 7, 9, 1, 9, 4, 7, 9, 4, 1, 9, 7, 9]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var perm_sig: Int
    var lane_sig: Int
    var n_panel: Int
    var rax: Float32
    var ray: Float32
    var final_x: Float32
    var reward_x10: Int
    var hits: Int
    var juice_x1000: Int
    var done_count: Int
    var complete: Int
    var pos_sig: Int


def test_plunder_parity() raises:
    var c = List[Expect]()
    # From plunder_probe (400-step fire-heavy tape).
    c.append(Expect(DIST_EASY, 0, 46, 27140, 0, 15.3069, 1.8761, 18.21556, 80, 8, -2060, 274, 0, 1461917291193465))
    c.append(Expect(DIST_HARD, 0, 46, 27140, 2, 17.6607, 1.5841, 18.71556, 80, 8, -1860, 254, 0, 1503777438355200))
    c.append(Expect(DIST_EASY, 1, 46, 33677, 0, 6.8256, 1.8761, 18.21556, 100, 10, -1760, 302, 0, 1361433688743562))
    c.append(Expect(DIST_HARD, 1, 46, 33677, 2, 16.8061, 1.5841, 18.71556, 140, 14, -1360, 253, 0, 1503316127971273))
    c.append(Expect(DIST_EASY, 7, 50, 26923, 0, 16.2696, 1.8761, 18.21556, 100, 10, -1760, 262, 0, 1463192874020202))
    c.append(Expect(DIST_HARD, 7, 50, 26923, 0, 15.4673, 1.5841, 18.71556, 110, 11, -1060, 218, 0, 1501325587999671))
    c.append(Expect(DIST_EASY, 42, 67, 28343, 0, 5.5000, 1.8761, 18.21556, 100, 10, -1560, 245, 0, 1321643255372620))
    c.append(Expect(DIST_HARD, 42, 67, 28343, 1, 15.9840, 1.5841, 18.71556, 90, 9, -1860, 251, 0, 1502295630909791))
    c.append(Expect(DIST_EASY, 123, 42, 28428, 0, 12.1644, 1.8761, 18.21556, 90, 9, -1560, 197, 0, 1448077020672778))
    c.append(Expect(DIST_HARD, 123, 42, 28428, 2, 5.0000, 1.5841, 18.71556, 60, 6, -1960, 260, 0, 1325674413047727))

    var easy = PlunderGame(DIST_EASY)
    var hard = PlunderGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: PlunderGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    var perm_sig = 0
    for i in range(len(g.image_perm)):
        perm_sig += g.image_perm[i] * (i + 1)
    assert_equal(perm_sig, e.perm_sig, "perm_sig" + tag)

    var lane_sig = 0
    for i in range(g.num_lanes):
        lane_sig += (
            (1 if g.lane_dirs[i] else 0) * 3 + Int(floor(g.lane_vels[i] * 10000))
        ) * (i + 1)
    assert_equal(lane_sig, e.lane_sig, "lane_sig" + tag)

    var n_panel = 0
    for i in range(len(g.entities)):
        if g.entities[i].type == PANEL:
            n_panel += 1
    assert_equal(n_panel, e.n_panel, "n_panel" + tag)
    assert_true(abs(g.agent.x - e.rax) < 1e-3, "rax" + tag)
    assert_true(abs(g.agent.y - e.ray) < 1e-3, "ray" + tag)

    var done_count = 0
    var pos_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        var cx = Int(floor(g.agent.x * 1000))
        var cy = Int(floor(g.agent.y * 1000))
        pos_sig += (cx * 1000003 + cy) * (s + 1)

    assert_true(abs(g.agent.x - e.final_x) < 1e-3, "final_x" + tag)
    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(g.targets_hit, e.hits, "hits" + tag)
    assert_equal(Int(round(g.juice * 1000.0)), e.juice_x1000, "juice" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(1 if g.level_complete else 0, e.complete, "complete" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
