"""Ninja reset+step parity vs the Qt-free C++ probe.

Combined P0+P1. Validates the platformer level gen (generate_coin_to_the_right RNG:
num_sections/edges/dy/dx/gap/bomb draws → wall/fire/bomb tile counts + grid checksum
+ goal pos) and the step (charged jump, gravity, wall collision, throwing stars that
stick to walls / detonate bombs, fire/bomb/explosion death) over a jump+fire tape.
Ground truth = `scratchpad/ninja_probe.cpp`. Asset-free/fast. See `docs/PROCGEN_NINJA_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import NinjaGame
from mojo_rl.envs.procgen.games.ninja import DIST_EASY, DIST_HARD, WALL_MID, FIRE, BOMB


comptime STEPS = 200


def _tape(step: Int) -> Int:
    var t: List[Int] = [8, 8, 4, 7, 5, 4, 9, 7, 8, 4, 7, 11]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax: Float32
    var ray: Float32
    var difficulty: Int
    var goal_x: Int
    var goal_y: Int
    var n_wall: Int
    var n_fire: Int
    var n_bomb: Int
    var gsig: Int
    var final_x: Float32
    var final_y: Float32
    var reward_x10: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var pos_sig: Int


def test_ninja_parity() raises:
    var c = List[Expect]()
    # From ninja_probe (200-step jump+fire tape).
    c.append(Expect(DIST_EASY, 0, 1.5, 32.5, 3, 29, 36, 3650, 25, 2, 250563520, 29.5, 28.25, 0, 37, 0, 5, 519333065583152))
    c.append(Expect(DIST_HARD, 0, 1.5, 32.5, 3, 25, 35, 3706, 21, 2, 241172440, 25.5, 27.0, 0, 83, 0, 5, 467895591614415))
    c.append(Expect(DIST_EASY, 1, 1.5, 32.5, 1, 11, 31, 3976, 7, 0, 197275526, 11.5, 36.0, 3900, 39, 0, 4, 226795162743737))
    c.append(Expect(DIST_HARD, 1, 1.5, 32.5, 1, 9, 31, 3994, 5, 0, 194485776, 9.5, 33.0, 8400, 84, 0, 4, 189332866856544))
    c.append(Expect(DIST_EASY, 7, 1.5, 32.5, 2, 20, 32, 3855, 16, 1, 217300894, 20.5, 37.0, 2900, 31, 0, 5, 385506933813613))
    c.append(Expect(DIST_HARD, 7, 1.5, 32.5, 2, 17, 28, 3855, 13, 1, 215362684, 17.5, 26.0, 0, 92, 0, 5, 331528402786535))
    c.append(Expect(DIST_EASY, 42, 1.5, 32.5, 2, 15, 29, 3925, 11, 1, 204510324, 15.5, 34.0, 3600, 39, 0, 4, 301392089612211))
    c.append(Expect(DIST_HARD, 42, 1.5, 32.5, 2, 13, 25, 3927, 9, 1, 202796122, 13.5, 23.5, 0, 143, 0, 4, 261975467697594))
    c.append(Expect(DIST_EASY, 123, 1.5, 32.5, 1, 10, 33, 3996, 6, 0, 193643870, 10.5, 38.0, 4100, 41, 0, 4, 208211210907086))
    c.append(Expect(DIST_HARD, 123, 1.5, 32.5, 1, 8, 35, 4012, 4, 0, 191196396, 8.5, 32.0, 0, 91, 0, 4, 169782806134099))

    var easy = NinjaGame(DIST_EASY)
    var hard = NinjaGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: NinjaGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax) < 1e-3, "rax" + tag)
    assert_true(abs(g.agent.y - e.ray) < 1e-3, "ray" + tag)
    assert_equal(g.difficulty, e.difficulty, "difficulty" + tag)
    assert_equal(g.goal_x, e.goal_x, "goal_x" + tag)
    assert_equal(g.goal_y, e.goal_y, "goal_y" + tag)

    var nwall = 0
    var nfire = 0
    var nbomb = 0
    var gsig = 0
    for i in range(g.w * g.h):
        var v = g.grid[i]
        if v == WALL_MID:
            nwall += 1
        elif v == FIRE:
            nfire += 1
        elif v == BOMB:
            nbomb += 1
        gsig += (v + 1) * (i + 1)
    assert_equal(nwall, e.n_wall, "n_wall" + tag)
    assert_equal(nfire, e.n_fire, "n_fire" + tag)
    assert_equal(nbomb, e.n_bomb, "n_bomb" + tag)
    assert_equal(gsig, e.gsig, "gsig" + tag)

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
