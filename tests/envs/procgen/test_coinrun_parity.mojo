"""Coinrun reset+step parity vs the Qt-free C++ probe.

Combined P0+P1 — the PLATFORMER SUBSTRATE gate. Validates level gen
(generate_coin_to_the_right RNG order: dif/num_sections/danger_type + per-section
dy/dx/pit/saw/enemy/crate draws → tile counts + grid checksum + goal pos + entity
counts) and the full platformer step (has_support + jump gating, gravity via
update_agent_velocity, basic_step_object over walls + one-way crates, enemy walk +
reflect, saw/lava/goal grid collision, agent death) over a jump-heavy 200-step
tape. Ground truth = `scratchpad/coinrun_probe.cpp`. Asset-free/fast.
See `docs/PROCGEN_COINRUN_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import CoinrunGame
from mojo_rl.envs.procgen.games.coinrun import (
    DIST_EASY, DIST_HARD, WALL_MID, WALL_TOP, LAVA_MID, LAVA_TOP,
    ENEMY_BARRIER, GOAL, CRATE, ENEMY, SAW, W, H,
)


comptime STEPS = 200


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 7, 8, 7, 7, 5, 7, 7, 8, 7, 7, 4, 8, 7]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax: Float32
    var dif: Int
    var num_sections: Int
    var danger_type: Int
    var goal_x: Int
    var goal_y: Int
    var n_wall_mid: Int
    var n_wall_top: Int
    var n_lava: Int
    var n_barrier: Int
    var gsig: Int
    var n_crate: Int
    var n_enemy: Int
    var n_saw: Int
    var reward_x10: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var final_x: Float32
    var final_y: Float32
    var pos_sig: Int


def test_coinrun_parity() raises:
    var c = List[Expect]()
    # From coinrun_probe (200-step jump-heavy tape).
    c.append(Expect(DIST_EASY, 0, 1.5, 3, 3, 1, 21, 6, 2872, 21, 0, 3, 356538879, 6, 0, 2, 3200, 33, 0, 8, 21.5073, 12.0787, 421558618521238))
    c.append(Expect(DIST_HARD, 0, 1.5, 2, 3, 1, 21, 9, 2881, 21, 0, 3, 356113195, 9, 0, 3, 2600, 32, 0, 12, 21.5073, 15.0787, 415418307617194))
    c.append(Expect(DIST_EASY, 1, 1.5, 1, 1, 1, 9, 3, 3539, 9, 0, 1, 230030295, 2, 0, 1, 3500, 35, 0, 3, 9.5073, 9.0787, 190084149651596))
    c.append(Expect(DIST_HARD, 1, 1.5, 2, 3, 2, 20, 8, 2910, 20, 0, 3, 346374651, 1, 1, 2, 2700, 33, 0, 13, 20.5073, 11.3787, 400999395693650))
    c.append(Expect(DIST_EASY, 7, 1.5, 2, 2, 1, 16, 7, 3148, 16, 0, 2, 303861559, 0, 0, 0, 3000, 30, 0, 0, 16.5073, 10.3787, 326740240786905))
    c.append(Expect(DIST_HARD, 7, 1.5, 2, 3, 2, 16, 12, 3187, 16, 0, 3, 301957150, 2, 0, 1, 2900, 29, 1, 3, 16.5000, 12.5787, 320877577954685))
    c.append(Expect(DIST_EASY, 42, 1.5, 2, 2, 0, 11, 2, 3419, 11, 0, 2, 251262403, 0, 0, 0, 3400, 34, 0, 0, 11.5073, 8.0787, 229522313495791))
    c.append(Expect(DIST_HARD, 42, 1.5, 1, 1, 1, 8, 2, 3596, 8, 0, 1, 219428590, 2, 0, 0, 3500, 35, 0, 2, 8.5073, 8.0787, 170244817168237))
    c.append(Expect(DIST_EASY, 123, 1.5, 1, 1, 1, 8, 4, 3604, 8, 0, 1, 219292162, 3, 0, 0, 3500, 35, 0, 3, 8.5073, 10.0787, 170262641170189))
    c.append(Expect(DIST_HARD, 123, 1.5, 2, 2, 0, 14, 6, 3246, 14, 0, 2, 283076880, 1, 1, 0, 3200, 32, 0, 11, 14.5073, 9.3787, 288136940011810))

    var easy = CoinrunGame(DIST_EASY)
    var hard = CoinrunGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: CoinrunGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax) < 1e-3, "rax" + tag)
    assert_equal(g.goal_x, e.goal_x, "goal_x" + tag)
    assert_equal(g.goal_y, e.goal_y, "goal_y" + tag)

    var nwm = 0
    var nwt = 0
    var nlava = 0
    var nbar = 0
    var ngoal = 0
    var gsig = 0
    for idx in range(W * H):
        var v = g.grid[idx]
        if v == WALL_MID:
            nwm += 1
        elif v == WALL_TOP:
            nwt += 1
        elif v == LAVA_MID or v == LAVA_TOP:
            nlava += 1
        elif v == ENEMY_BARRIER:
            nbar += 1
        elif v == GOAL:
            ngoal += 1
        gsig += (v + 1) * (idx + 1)
    assert_equal(nwm, e.n_wall_mid, "n_wall_mid" + tag)
    assert_equal(nwt, e.n_wall_top, "n_wall_top" + tag)
    assert_equal(nlava, e.n_lava, "n_lava" + tag)
    assert_equal(nbar, e.n_barrier, "n_barrier" + tag)
    assert_equal(ngoal, 1, "n_goal" + tag)
    assert_equal(gsig, e.gsig, "gsig" + tag)

    var ncrate = 0
    var nen = 0
    var nsaw = 0
    for i in range(len(g.entities)):
        var t = g.entities[i].type
        if t == CRATE:
            ncrate += 1
        elif t == ENEMY:
            nen += 1
        elif t == SAW:
            nsaw += 1
    assert_equal(ncrate, e.n_crate, "n_crate" + tag)
    assert_equal(nen, e.n_enemy, "n_enemy" + tag)
    assert_equal(nsaw, e.n_saw, "n_saw" + tag)

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
