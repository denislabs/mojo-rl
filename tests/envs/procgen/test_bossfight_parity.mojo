"""Bossfight reset+step parity vs the Qt-free C++ probe.

Combined P0+P1. Validates reset (boss theme/aspect, round_health, num_rounds,
invuln, boss health, player/boss laser themes, attack_modes, reposition_agent
against boss+shields, meteor barriers) and the full game_step (boss drift/dest +
shields swap, 4 attack modes, firing, reflection, meteor↔bullet collisions, agent
death, laser trails) over a fire-heavy 300-step tape. boss/shields/agent are
separate fields; entities hold bullets/trails/barriers/explosions. 4 rand01/step.
Ground truth = `scratchpad/bossfight_probe.cpp`. Asset-free/fast.
See `docs/PROCGEN_BOSSFIGHT_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import BossfightGame
from mojo_rl.envs.procgen.games.bossfight import DIST_EASY, DIST_HARD, BARRIER


comptime STEPS = 300


def _tape(step: Int) -> Int:
    var t: List[Int] = [9, 4, 7, 9, 1, 9, 5, 9, 3, 9, 7, 4]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax0: Float32
    var round_health: Int
    var num_rounds: Int
    var invuln: Int
    var boss_hp0: Int
    var player_laser: Int
    var boss_laser: Int
    var am_sig: Int
    var n_barrier: Int
    var reward_x10: Int
    var boss_hp: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var pos_sig: Int


def test_bossfight_parity() raises:
    var c = List[Expect]()
    # From bossfight_probe (300-step fire-heavy tape).
    c.append(Expect(DIST_EASY, 0, 17.2478, 7, 3, 3, 21, 0, 2, 11, 1, 0, 16, 11, 0, 30, 853066202108333))
    c.append(Expect(DIST_HARD, 0, 17.2478, 7, 3, 5, 21, 0, 2, 11, 1, 0, 16, 18, 0, 23, 853066202108333))
    c.append(Expect(DIST_EASY, 1, 18.0575, 2, 2, 3, 4, 0, 2, 1, 1, 10, 2, 11, 0, 24, 853314186119465))
    c.append(Expect(DIST_HARD, 1, 18.0575, 2, 2, 3, 4, 0, 2, 1, 1, 10, 2, 6, 0, 24, 853314186119465))
    c.append(Expect(DIST_EASY, 7, 2.0829, 7, 3, 3, 21, 2, 1, 6, 2, 10, 14, 8, 0, 62, 609046308050847))
    c.append(Expect(DIST_HARD, 7, 2.0829, 7, 3, 5, 21, 2, 1, 6, 2, 10, 14, 1, 0, 156, 609046308050847))
    c.append(Expect(DIST_EASY, 42, 1.8245, 3, 1, 2, 3, 2, 1, 2, 2, 130, -6, 7, 0, 64, 597729914121876))
    c.append(Expect(DIST_HARD, 42, 1.8245, 3, 1, 2, 3, 2, 1, 2, 2, 130, -6, 4, 0, 64, 597729914121876))
    c.append(Expect(DIST_EASY, 123, 3.3391, 6, 2, 3, 12, 1, 2, 5, 2, 10, 5, 3, 0, 46, 659408188403217))
    c.append(Expect(DIST_HARD, 123, 3.3391, 6, 2, 3, 12, 1, 2, 5, 2, 10, 5, 4, 0, 47, 659408188403217))

    var easy = BossfightGame(DIST_EASY)
    var hard = BossfightGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: BossfightGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax0) < 1e-3, "rax0" + tag)
    assert_equal(g.round_health, e.round_health, "round_health" + tag)
    assert_equal(g.num_rounds, e.num_rounds, "num_rounds" + tag)
    assert_equal(g.invuln, e.invuln, "invuln" + tag)
    assert_equal(Int(g.boss.health), e.boss_hp0, "boss_hp0" + tag)
    assert_equal(g.player_laser, e.player_laser, "player_laser" + tag)
    assert_equal(g.boss_laser, e.boss_laser, "boss_laser" + tag)

    var am_sig = 0
    for i in range(len(g.attack_modes)):
        am_sig += g.attack_modes[i] * (i + 1)
    assert_equal(am_sig, e.am_sig, "am_sig" + tag)

    var n_barrier = 0
    for i in range(len(g.entities)):
        if g.entities[i].type == BARRIER:
            n_barrier += 1
    assert_equal(n_barrier, e.n_barrier, "n_barrier" + tag)

    var done_count = 0
    var pos_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        var cx = Int(floor(g.agent.x * 1000))
        var cy = Int(floor(g.agent.y * 1000))
        pos_sig += (cx * 1000003 + cy) * (s + 1)

    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(Int(g.boss.health), e.boss_hp, "boss_hp" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(1 if g.level_complete else 0, e.complete, "complete" + tag)
    assert_equal(len(g.entities), e.n_ent, "n_ent" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
