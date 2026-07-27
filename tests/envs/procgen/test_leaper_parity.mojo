"""Leaper reset+step parity vs the Qt-free C++ probe.

Combined. Validates the heavy reset (lane layout RNG + steady-state spawn loop) via
lane counts / row positions / goal / post-reset entity+grid+speed signatures, then
replays a fixed 200-step tape and compares the frog trajectory + done/finish counts
+ surviving entity count. Exercises the frog-hop movement, car/log drift + auto-erase,
collision/drown/log-ride/finish. Ground truth = `scratchpad/leaper_probe.cpp`.
Asset-free/fast. See `docs/PROCGEN_LEAPER_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import LeaperGame
from mojo_rl.envs.procgen.games.leaper import DIST_EASY, DIST_HARD

comptime STEPS = 200


def _tape(step: Int) -> Int:
    var t: List[Int] = [5, 5, 7, 5, 1, 5, 3, 4, 5, 7, 5, 5, 4, 1]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var road: Int
    var water: Int
    var broad: Int
    var bwater: Int
    var goal: Int
    var n_reset: Int
    var sp_sig: Int
    var grid_sig: Int
    var ent_sig: Int
    var final_x: Float32
    var final_y: Float32
    var n_end: Int
    var done_count: Int
    var complete: Int
    var pos_sig: Int


def test_leaper_parity() raises:
    var c = List[Expect]()
    # From leaper_probe: Easy (wd=9) + Hard (wd=15) × 5 seeds.
    c.append(Expect(DIST_EASY, 0, 3, 3, 1, 5, 9, 35, 15555, 116721, 108814428, 1.1, 8.6, 14, 38, 24, 71562639275681))
    c.append(Expect(DIST_HARD, 0, 4, 5, 2, 7, 13, 70, 1656, 1146435, 1044264248, 9.152689, 14.6, 39, 23, 3, 217164659484698))
    c.append(Expect(DIST_EASY, 1, 3, 3, 1, 5, 9, 24, 35111, 116721, 72794048, 0.4, 8.599993, 17, 51, 21, 58169452476864))
    c.append(Expect(DIST_HARD, 1, 3, 4, 2, 7, 12, 47, 41654, 1520160, 442286311, 7.887013, 14.6, 32, 28, 3, 201084992245841))
    c.append(Expect(DIST_EASY, 7, 3, 3, 1, 5, 9, 25, 74931, 116721, 86432163, 2.214411, 8.6, 17, 36, 23, 71779242354905))
    c.append(Expect(DIST_HARD, 7, 2, 3, 2, 6, 10, 28, -13208, 1915485, 185152101, 8.474847, 14.6, 17, 24, 6, 214544767625046))
    c.append(Expect(DIST_EASY, 42, 2, 2, 1, 4, 7, 13, -18245, 220023, 9726009, 0.4, 8.6, 13, 8, 2, 93055516754120))
    c.append(Expect(DIST_HARD, 42, 0, 0, 1, 2, 3, 1, 0, 2542500, 275035, 10.900143, 14.6, 1, 3, 3, 263307513912846))
    c.append(Expect(DIST_EASY, 123, 0, 0, 1, 2, 3, 1, 0, 332100, 182035, 7.599998, 8.6, 1, 3, 3, 75475881912227))
    c.append(Expect(DIST_HARD, 123, 3, 2, 1, 6, 9, 37, -48298, 2067915, 127530195, 1.275583, 14.6, 25, 15, 4, 70013600032844))

    var easy = LeaperGame(DIST_EASY)
    var hard = LeaperGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: LeaperGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_equal(len(g.road_speeds), e.road, "road" + tag)
    assert_equal(len(g.water_speeds), e.water, "water" + tag)
    assert_equal(g.bottom_road_y, e.broad, "broad" + tag)
    assert_equal(g.bottom_water_y, e.bwater, "bwater" + tag)
    assert_equal(g.goal_y, e.goal, "goal" + tag)
    assert_equal(len(g.entities), e.n_reset, "n_reset" + tag)

    var sp_sig = 0
    for i in range(len(g.road_speeds)):
        sp_sig += Int(floor(g.road_speeds[i] * 100000)) * (i + 1)
    for i in range(len(g.water_speeds)):
        sp_sig += Int(floor(g.water_speeds[i] * 100000)) * (i + 7)
    assert_equal(sp_sig, e.sp_sig, "sp_sig" + tag)

    var grid_sig = 0
    for k in range(g.w * g.h):
        grid_sig += (k + 1) * g.grid[k]
    assert_equal(grid_sig, e.grid_sig, "grid_sig" + tag)

    var ent_sig = 0
    for i in range(len(g.entities)):
        ref en = g.entities[i]
        ent_sig += (
            Int(floor(en.x * 1000)) * 31
            + Int(floor(en.y * 1000)) * 17
            + en.type * 7
            + en.image_theme
        ) * (i + 1)
    assert_equal(ent_sig, e.ent_sig, "ent_sig" + tag)

    var done_count = 0
    var complete = 0
    var pos_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        if g.level_complete:
            complete += 1
        var cx = Int(floor(g.agent.x * 1000))
        var cy = Int(floor(g.agent.y * 1000))
        pos_sig += (cx * 1000003 + cy) * (s + 1)

    assert_true(abs(g.agent.x - e.final_x) < 1e-3, "final_x" + tag)
    assert_true(abs(g.agent.y - e.final_y) < 1e-3, "final_y" + tag)
    assert_equal(len(g.entities), e.n_end, "n_end" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(complete, e.complete, "complete" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
