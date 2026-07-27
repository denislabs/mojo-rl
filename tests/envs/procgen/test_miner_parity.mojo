"""Miner reset+step parity vs the Qt-free C++ probe.

Combined. Validates the reset (agent/boulder/diamond placement + exit) and the
Boulder-Dash grid cellular automaton over a 150-step tape: grid-step agent
movement, digging/collecting, boulder pushing, and the per-step falling/rolling
scan (which mutates the grid). Ground truth = `scratchpad/miner_probe.cpp`.
Asset-free/fast. See `docs/PROCGEN_MINER_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import MinerGame, BOULDER, DIAMOND
from mojo_rl.envs.procgen.games.miner import DIST_EASY, DIST_HARD

comptime STEPS = 150


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 3, 3, 1, 1, 5, 7, 3, 4, 1, 3, 7, 5]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var agx: Float32
    var agy: Float32
    var exit_x: Int
    var exit_y: Int
    var nb: Int
    var nd: Int
    var grid0: Int
    var final_x: Float32
    var final_y: Float32
    var reward_x10: Int
    var drem: Int
    var done_count: Int
    var complete: Int
    var pos_sig: Int
    var grid_sig: Int


def test_miner_parity() raises:
    var c = List[Expect]()
    # From miner_probe: Easy (wd=10) + Hard (wd=20) × 5 seeds.
    c.append(Expect(DIST_EASY, 0, 3.5, 6.5, 2, 0, 19, 3, 41411, 4.5, 5.5, 0, 3, 0, 0, 48668708431000, 972068089))
    c.append(Expect(DIST_HARD, 0, 3.5, 8.5, 1, 9, 80, 12, 600451, 4.5, 6.5, 10, 11, 0, 0, 50978726752000, 8197159049))
    c.append(Expect(DIST_EASY, 1, 3.5, 6.5, 5, 2, 17, 3, 47084, 8.5, 1.5, 0, 3, 1, 0, 93669798031000, 1227323092))
    c.append(Expect(DIST_HARD, 1, 3.5, 3.5, 13, 11, 78, 12, 618693, 13.5, 4.5, 20, 10, 6, 0, 141307970980000, 8771540861))
    c.append(Expect(DIST_EASY, 7, 3.5, 8.5, 3, 9, 19, 3, 52360, 9.5, 0.5, 10, 2, 1, 0, 102596822791000, 1973645808))
    c.append(Expect(DIST_HARD, 7, 3.5, 4.5, 6, 1, 79, 12, 599823, 5.5, 3.5, 0, 12, 0, 0, 62183726419000, 7700773687))
    c.append(Expect(DIST_EASY, 42, 6.5, 2.5, 8, 5, 19, 3, 44590, 8.5, 0.5, 0, 3, 0, 0, 96263798507000, 649923547))
    c.append(Expect(DIST_HARD, 42, 6.5, 1.5, 19, 7, 79, 12, 595557, 10.5, 1.5, 10, 11, 1, 0, 115369862724000, 7093811937))
    c.append(Expect(DIST_EASY, 123, 0.5, 8.5, 4, 3, 20, 3, 47282, 8.5, 0.5, 10, 2, 0, 0, 86099772928000, 1902716802))
    c.append(Expect(DIST_HARD, 123, 0.5, 19.5, 9, 13, 79, 12, 638085, 2.5, 17.5, 10, 11, 0, 0, 22853766952000, 10194637334))

    var easy = MinerGame(DIST_EASY)
    var hard = MinerGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: MinerGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.agx) < 1e-4, "agx" + tag)
    assert_true(abs(g.agent.y - e.agy) < 1e-4, "agy" + tag)
    assert_equal(g.exit_x, e.exit_x, "exit_x" + tag)
    assert_equal(g.exit_y, e.exit_y, "exit_y" + tag)

    var nb = 0
    var nd = 0
    var grid0 = 0
    for k in range(g.w * g.h):
        grid0 += (k + 1) * g.grid[k]
        if g.grid[k] == BOULDER:
            nb += 1
        if g.grid[k] == DIAMOND:
            nd += 1
    assert_equal(nb, e.nb, "nb" + tag)
    assert_equal(nd, e.nd, "nd" + tag)
    assert_equal(grid0, e.grid0, "grid0" + tag)

    var done_count = 0
    var complete = 0
    var pos_sig = 0
    var grid_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        if g.level_complete:
            complete += 1
        var cx = Int(floor(g.agent.x * 1000))
        var cy = Int(floor(g.agent.y * 1000))
        pos_sig += (cx * 1000003 + cy) * (s + 1)
        var gc = 0
        for k in range(g.w * g.h):
            gc += (k + 1) * g.grid[k]
        grid_sig += gc * (s + 1)

    assert_true(abs(g.agent.x - e.final_x) < 1e-4, "final_x" + tag)
    assert_true(abs(g.agent.y - e.final_y) < 1e-4, "final_y" + tag)
    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(g.diamonds_remaining, e.drem, "drem" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(complete, e.complete, "complete" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)
    assert_equal(grid_sig, e.grid_sig, "grid_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
