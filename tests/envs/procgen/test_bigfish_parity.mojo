"""Bigfish reset+step parity vs the Qt-free C++ probe.

Combined (reset is trivial for bigfish). Checks the reset RNG (bg_pct_x,
background_index, agent start), then replays a fixed 300-step tape (no reset) and
compares float-robust signatures against `scratchpad/bigfish_probe.cpp`: final
agent position + radius, fish eaten, reward, surviving fish count, done-fire count,
and a per-step agent position signature. Exercises the base momentum physics,
edge-blocking, fish spawn RNG (randn(10) gate → rand01×4 → theme randn(3)), fish
drift + auto-erase, match_aspect_ratio (fish ry), and eat/grow/die. Asset-free/fast.
See `docs/PROCGEN_BIGFISH_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import BigfishGame
from mojo_rl.envs.procgen.games.bigfish import DIST_EASY, DIST_HARD

comptime STEPS = 300


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 5, 5, 1, 3, 8, 6, 2, 0, 4, 7, 1, 5, 3]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var bg_pct_x: Float32
    var bg_idx: Int
    var agent_x: Float32
    var agent_y: Float32
    var final_x: Float32
    var final_y: Float32
    var agent_r: Float32
    var fish_eaten: Int
    var reward_x100: Int
    var n_fish: Int
    var done_count: Int
    var pos_sig: Int


def test_bigfish_parity() raises:
    # From bigfish_probe. probe easy=1 → DIST_EASY (start_r 1); easy=0 → DIST_HARD.
    var cases = List[Expect]()
    cases.append(Expect(DIST_EASY, 0, 0.548813522, 3, 14.131636, 2.0, 14.390791, 11.284462, 1.066667, 2, 200, 10, 35, 648510765954554))
    cases.append(Expect(DIST_HARD, 0, 0.548813522, 3, 14.131636, 1.5, 14.390791, 10.784451, 0.550000, 1, 100, 10, 14, 648510743379031))
    cases.append(Expect(DIST_EASY, 1, 0.417021990, 3, 14.230231, 2.0, 14.489387, 11.284462, 1.033333, 1, 100, 5, 42, 652967737325428))
    cases.append(Expect(DIST_HARD, 1, 0.417021990, 3, 14.230231, 1.5, 14.489387, 10.784451, 0.550000, 1, 100, 5, 23, 652967714749905))
    cases.append(Expect(DIST_EASY, 7, 0.076308288, 6, 15.374441, 2.0, 15.633596, 11.284462, 1.100000, 3, 300, 6, 115, 704624138294166))
    cases.append(Expect(DIST_HARD, 7, 0.076308288, 6, 15.374441, 1.5, 15.633596, 10.784451, 0.500000, 0, 0, 6, 59, 704624115718643))
    cases.append(Expect(DIST_EASY, 42, 0.374540120, 2, 18.653715, 2.0, 18.503891, 11.284462, 1.133333, 4, 400, 8, 30, 834226227099266))
    cases.append(Expect(DIST_HARD, 42, 0.374540120, 2, 18.653715, 1.5, 18.912947, 10.784451, 0.550000, 1, 100, 8, 47, 852685256900734))
    cases.append(Expect(DIST_EASY, 123, 0.696469188, 1, 5.893876, 2.0, 6.152850, 11.284462, 1.133333, 4, 400, 6, 38, 276568837132115))
    cases.append(Expect(DIST_HARD, 123, 0.696469188, 1, 5.893876, 1.5, 6.152850, 10.784451, 0.500000, 0, 0, 6, 32, 276568814556592))

    var easy = BigfishGame(DIST_EASY)
    var hard = BigfishGame(DIST_HARD)
    for ci in range(len(cases)):
        var e = cases[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: BigfishGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    var diff = g.bg_pct_x - e.bg_pct_x
    if diff < 0:
        diff = -diff
    assert_true(diff < 1e-6, "bg_pct_x" + tag)
    assert_equal(g.background_index, e.bg_idx, "bg_idx" + tag)
    assert_true(abs(g.agent.x - e.agent_x) < 1e-4, "agent_x" + tag)
    assert_true(abs(g.agent.y - e.agent_y) < 1e-4, "agent_y" + tag)

    var done_count = 0
    var pos_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        var cx = Int(floor(g.agent.x * 1000.0))
        var cy = Int(floor(g.agent.y * 1000.0))
        pos_sig += (cx * 1000003 + cy) * (s + 1)

    var reward_x100 = Int(round(g.episode_reward * 100.0))
    assert_true(abs(g.agent.x - e.final_x) < 1e-4, "final_x" + tag)
    assert_true(abs(g.agent.y - e.final_y) < 1e-4, "final_y" + tag)
    assert_true(abs(g.agent.rx - e.agent_r) < 1e-4, "agent_r" + tag)
    assert_equal(g.fish_eaten, e.fish_eaten, "fish_eaten" + tag)
    assert_equal(reward_x100, e.reward_x100, "reward" + tag)
    assert_equal(len(g.entities), e.n_fish, "n_fish" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
