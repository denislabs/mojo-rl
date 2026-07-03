"""Jumper reset+step parity vs the Qt-free C++ probe.

Combined P0+P1. Validates the MazeGen-seeded roomgen carving (maze fill → 2x CA →
border → find_best_room → goal/agent choose_one → find_path → prune → spike
placement → wall thinning → spike entities → top walls) and the double-jump step
(jump_count/cooldown, gravity, wall collision, spike/goal collision, trails) over a
jump tape. Ground truth = `scratchpad/jumper_probe.cpp`. Asset-free/fast.
See `docs/PROCGEN_JUMPER_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import JumperGame
from mojo_rl.envs.procgen.games.jumper import DIST_EASY, DIST_HARD, CAVEWALL, CAVEWALL_TOP


comptime STEPS = 200


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 5, 8, 7, 4, 8, 1, 7, 5, 8, 7]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax: Float32
    var ray: Float32
    var goal_x: Int
    var goal_y: Int
    var n_cavewall: Int
    var n_space: Int
    var free_after: Int
    var path_len: Int
    var n_spike: Int
    var gsig: Int
    var final_x: Float32
    var final_y: Float32
    var reward_x10: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var pos_sig: Int


def test_jumper_parity() raises:
    var c = List[Expect]()
    # From jumper_probe (200-step jump tape).
    c.append(Expect(DIST_EASY, 0, 3.5, 11.4, 3, 3, 296, 104, 144, 17, 1, 2047068, 9.4960, 8.4, 0, 2, 0, 10, 194647417961504))
    c.append(Expect(DIST_HARD, 0, 9.5, 29.4, 15, 36, 1467, 133, 994, 14, 0, 25289081, 16.4960, 30.4, 0, 0, 0, 8, 334856233243495))
    c.append(Expect(DIST_EASY, 1, 7.5, 1.4, 9, 7, 304, 96, 171, 9, 4, 1691872, 11.4960, 1.4, 0, 3, 0, 13, 235185050372856))
    c.append(Expect(DIST_HARD, 1, 36.5, 23.4, 1, 25, 1195, 405, 933, 70, 3, 32455384, 39.4960, 25.45, 0, 0, 0, 7, 797980773865363))
    c.append(Expect(DIST_EASY, 7, 5.5, 6.4, 13, 12, 285, 115, 153, 15, 1, 2779525, 9.4960, 8.45, 0, 1, 0, 5, 194917626985580))
    c.append(Expect(DIST_HARD, 7, 17.5, 1.4, 7, 17, 1430, 170, 963, 33, 1, 14568916, 20.4960, 1.4, 0, 1, 0, 10, 416175762343363))
    c.append(Expect(DIST_EASY, 42, 9.5, 12.4, 14, 14, 343, 57, 126, 8, 0, 1887134, 15.4960, 12.45, 500, 5, 0, 4, 314921768644537))
    c.append(Expect(DIST_HARD, 42, 16.5, 4.4, 26, 5, 1460, 140, 939, 14, 1, 11883928, 31.4960, 5.4, 200, 2, 0, 9, 626927103973514))
    c.append(Expect(DIST_EASY, 123, 15.5, 2.4, 6, 7, 301, 99, 165, 15, 1, 1792303, 16.4960, 2.45, 0, 0, 0, 5, 335850957052900))
    c.append(Expect(DIST_HARD, 123, 16.5, 12.4, 2, 23, 1343, 257, 996, 44, 1, 28602509, 27.4960, 16.45, 0, 1, 0, 5, 552010560154640))

    var easy = JumperGame(DIST_EASY)
    var hard = JumperGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: JumperGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax) < 1e-3, "rax" + tag)
    assert_true(abs(g.agent.y - e.ray) < 1e-3, "ray" + tag)
    assert_equal(g.goal_cell % g.w, e.goal_x, "goal_x" + tag)
    assert_equal(g.goal_cell // g.w, e.goal_y, "goal_y" + tag)

    var ncw = 0
    var nspace = 0
    var gsig = 0
    for i in range(g.w * g.h):
        var v = g.grid.data[i]
        if v == CAVEWALL or v == CAVEWALL_TOP:
            ncw += 1
        elif v == 100:  # SPACE
            nspace += 1
        gsig += (v + 1) * (i + 1)
    assert_equal(ncw, e.n_cavewall, "n_cavewall" + tag)
    assert_equal(nspace, e.n_space, "n_space" + tag)
    assert_equal(gsig, e.gsig, "gsig" + tag)
    assert_equal(len(g._free_cells), e.free_after, "free_after" + tag)
    assert_equal(len(g._goal_path), e.path_len, "path_len" + tag)

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
    assert_true(abs(g.agent.y - e.final_y) < 1e-3, "final_y" + tag)
    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(1 if g.level_complete else 0, e.complete, "complete" + tag)
    assert_equal(len(g.entities), e.n_ent, "n_ent" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
