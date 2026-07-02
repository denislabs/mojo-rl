"""Climber reset+step parity vs the Qt-free C++ probe.

Combined P0+P1. Validates the platformer-reuse level gen (generate_platforms RNG:
difficulty/num_platforms + per-platform delta_y/enemy/plat_len/direction/coin/next_x
draws → WALL_TOP count + grid checksum + coin/enemy counts) and the climbing step
(gravity/jump/has_support, wall collision, patrolling enemies, coin collection +
coin_quota completion) over a jump-heavy tape. Ground truth =
`scratchpad/climber_probe.cpp`. Asset-free/fast. See `docs/PROCGEN_CLIMBER_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import ClimberGame
from mojo_rl.envs.procgen.games.climber import DIST_EASY, DIST_HARD, WALL_TOP, COIN, ENEMY


comptime STEPS = 200


def _tape(step: Int) -> Int:
    var t: List[Int] = [8, 7, 5, 8, 7, 4, 8, 1, 8, 5, 7, 8]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax: Float32
    var ray: Float32
    var difficulty: Int
    var num_platforms: Int
    var coin_quota: Int
    var n_wall_top: Int
    var gsig: Int
    var n_coin: Int
    var n_enemy: Int
    var final_x: Float32
    var final_y: Float32
    var collected: Int
    var reward_x10: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var pos_sig: Int


def test_climber_parity() raises:
    var c = List[Expect]()
    # From climber_probe (200-step jump-heavy tape).
    c.append(Expect(DIST_EASY, 0, 1.5, 1.5, 1, 5, 4, 42, 45733393, 4, 0, 14.4625, 7.5, 1, 10, 0, 0, 3, 286685298228191))
    c.append(Expect(DIST_HARD, 0, 1.5, 1.5, 1, 5, 4, 51, 73215891, 4, 3, 18.4625, 7.9, 0, 0, 2, 0, 7, 362265765908824))
    c.append(Expect(DIST_EASY, 1, 1.5, 1.5, 1, 3, 3, 31, 46050661, 3, 1, 14.4625, 7.0, 0, 0, 11, 0, 4, 287243918131785))
    c.append(Expect(DIST_HARD, 1, 1.5, 1.5, 1, 3, 3, 37, 73637571, 3, 2, 18.4625, 7.0, 0, 0, 5, 0, 5, 362438003713365))
    c.append(Expect(DIST_EASY, 7, 1.5, 1.5, 1, 5, 5, 40, 45696517, 5, 0, 14.4798, 7.8, 1, 10, 0, 0, 4, 286925655527929))
    c.append(Expect(DIST_HARD, 7, 1.5, 1.5, 1, 5, 4, 43, 73213623, 4, 4, 18.4798, 4.3, 0, 0, 0, 0, 8, 361792676794005))
    c.append(Expect(DIST_EASY, 42, 1.5, 1.5, 0, 2, 2, 21, 46146589, 2, 2, 14.4798, 4.3, 0, 0, 0, 0, 4, 287586935850864))
    c.append(Expect(DIST_HARD, 42, 1.5, 1.5, 0, 2, 2, 32, 73681167, 2, 2, 18.2594, 1.5, 0, 0, 2, 0, 4, 360973853587156))
    c.append(Expect(DIST_EASY, 123, 1.5, 1.5, 1, 4, 3, 42, 45773209, 3, 0, 14.4625, 12.0, 0, 0, 0, 0, 3, 287053456890897))
    c.append(Expect(DIST_HARD, 123, 1.5, 1.5, 1, 4, 3, 45, 73363983, 3, 1, 18.2594, 1.5, 0, 0, 0, 0, 4, 362181627922455))

    var easy = ClimberGame(DIST_EASY)
    var hard = ClimberGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: ClimberGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax) < 1e-3, "rax" + tag)
    assert_true(abs(g.agent.y - e.ray) < 1e-3, "ray" + tag)
    assert_equal(g.difficulty, e.difficulty, "difficulty" + tag)
    assert_equal(g.num_platforms, e.num_platforms, "num_platforms" + tag)
    assert_equal(g.coin_quota, e.coin_quota, "coin_quota" + tag)

    var nwt = 0
    var gsig = 0
    for i in range(g.w * g.h):
        var v = g.grid[i]
        if v == WALL_TOP:
            nwt += 1
        gsig += (v + 1) * (i + 1)
    assert_equal(nwt, e.n_wall_top, "n_wall_top" + tag)
    assert_equal(gsig, e.gsig, "gsig" + tag)

    var ncoin = 0
    var nen = 0
    for i in range(len(g.entities)):
        var t = g.entities[i].type
        if t == COIN:
            ncoin += 1
        elif t == ENEMY:
            nen += 1
    assert_equal(ncoin, e.n_coin, "n_coin" + tag)
    assert_equal(nen, e.n_enemy, "n_enemy" + tag)

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
    assert_equal(g.coins_collected, e.collected, "collected" + tag)
    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(1 if g.level_complete else 0, e.complete, "complete" + tag)
    assert_equal(len(g.entities), e.n_ent, "n_ent" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
