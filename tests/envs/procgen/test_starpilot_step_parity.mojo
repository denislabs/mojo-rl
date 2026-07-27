"""Starpilot step-physics parity vs the Qt-free C++ probe (the projectile substrate).

Replays a fire-heavy 250-step tape (no reset, < SHOOTER_WIN_TIME so no finish line)
through `StarpilotGame.step` and compares float-robust signatures against
`scratchpad/starpilot_step_probe.cpp`: agent position, reward (enemies killed),
kill count, done-fire count, surviving entity/spawner counts, and per-step agent
pos + entity-count signatures. Exercises momentum movement, spawner activation,
player + enemy firing, the collides_with_entities bullet→enemy collision, enemy
destruction, and explosions. See `docs/PROCGEN_STARPILOT_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import StarpilotGame
from mojo_rl.envs.procgen.games.starpilot import DIST_EASY, DIST_HARD

comptime STEPS = 250


def _tape(step: Int) -> Int:
    var t: List[Int] = [9, 4, 7, 9, 5, 9, 3, 9, 10, 4, 9, 7, 9, 5]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var final_x: Float32
    var final_y: Float32
    var reward_x10: Int
    var kills: Int
    var done_count: Int
    var n_ent: Int
    var n_spawn_left: Int
    var pos_sig: Int
    var ent_sig: Int


def test_starpilot_step_parity() raises:
    var c = List[Expect]()
    # From starpilot_step_probe (250-step fire-heavy tape).
    c.append(Expect(DIST_EASY, 0, 15.6, 14.99146, 10, 1, 44, 12, 35, 487616204537810, 302231))
    c.append(Expect(DIST_HARD, 0, 15.6, 14.99146, 10, 1, 8, 6, 28, 487616204537810, 185623))
    c.append(Expect(DIST_EASY, 1, 15.6, 14.99146, 20, 2, 56, 14, 40, 487718151782411, 402986))
    c.append(Expect(DIST_HARD, 1, 15.6, 14.99146, 0, 0, 16, 10, 31, 487718151782411, 242159))
    c.append(Expect(DIST_EASY, 7, 15.6, 14.99146, 30, 3, 64, 9, 36, 488578070398436, 422840))
    c.append(Expect(DIST_HARD, 7, 15.6, 14.99146, 20, 2, 30, 8, 44, 488578070398436, 170867))
    c.append(Expect(DIST_EASY, 42, 15.6, 13.62403, 10, 1, 15, 13, 42, 489439073623222, 292678))
    c.append(Expect(DIST_HARD, 42, 15.6, 13.62403, 10, 1, 26, 6, 26, 489439073623222, 223294))
    c.append(Expect(DIST_EASY, 123, 15.6, 14.99146, 10, 1, 33, 8, 49, 461158127681491, 380140))
    c.append(Expect(DIST_HARD, 123, 15.6, 14.99146, 30, 3, 24, 12, 29, 461158127681491, 217097))

    var easy = StarpilotGame(DIST_EASY)
    var hard = StarpilotGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: StarpilotGame, e: Expect) raises:
    g.reset(e.seed)
    var done_count = 0
    var kills = 0
    var pos_sig = 0
    var ent_sig = 0
    for s in range(STEPS):
        var r = g.step(_tape(s))
        if g.done:
            done_count += 1
        # kills = number of +1 rewards (excluding the +10 completion, absent here).
        if r >= 1.0:
            kills += Int(r + 0.5)
        var cx = Int(floor(g.agent.x * 1000))
        var cy = Int(floor(g.agent.y * 1000))
        pos_sig += (cx * 1000003 + cy) * (s + 1)
        ent_sig += len(g.entities) * (s + 1)

    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)
    assert_true(abs(g.agent.x - e.final_x) < 1e-4, "final_x" + tag)
    assert_true(abs(g.agent.y - e.final_y) < 1e-4, "final_y" + tag)
    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(kills, e.kills, "kills" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(len(g.entities), e.n_ent, "n_ent" + tag)
    assert_equal(len(g.spawners), e.n_spawn_left, "n_spawn_left" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)
    assert_equal(ent_sig, e.ent_sig, "ent_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
