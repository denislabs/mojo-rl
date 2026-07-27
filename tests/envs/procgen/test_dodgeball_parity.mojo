"""Dodgeball reset+step parity vs the Qt-free C++ probe.

Combined P0+P1 — the last procgen game. Validates the recursive room-split gen
(rand01/randn per split placing LAVA_WALL entities → wall count + entity signature),
collision-checked door/agent/enemy spawns, enemy setup (theme + choose_vel), and
the step (agent movement, ball firing, enemy AI + reflection off lava walls, ball
collisions/kills, ball edge-erase) over a fire tape. Ground truth =
`scratchpad/dodgeball_probe.cpp`. Asset-free/fast. See `docs/PROCGEN_DODGEBALL_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import DodgeballGame
from mojo_rl.envs.procgen.games.dodgeball import DIST_EASY, DIST_HARD, LAVA_WALL, ENEMY


comptime STEPS = 200


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 9, 5, 1, 9, 7, 3, 9, 7, 4, 9, 7]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax: Float32
    var ray: Float32
    var exit_wall_choice: Int
    var n_lava: Int
    var n_enemy: Int
    var esig: Int
    var final_x: Float32
    var final_y: Float32
    var reward_x10: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var numen_end: Int
    var pos_sig: Int


def test_dodgeball_parity() raises:
    var c = List[Expect]()
    # From dodgeball_probe (200-step fire tape).
    c.append(Expect(DIST_EASY, 0, 17.0519, 2.0208, 1, 2, 3, 1951673274, 18.9108, 17.2383, 20, 0, 0, 7, 2, 377960885704593))
    c.append(Expect(DIST_HARD, 0, 11.2588, 8.0165, 2, 4, 4, 5035194615, 19.1905, 8.0907, 40, 21, 0, 11, 2, 382138339498870))
    c.append(Expect(DIST_EASY, 1, 7.9824, 7.2201, 0, 2, 3, 1712773367, 18.0000, 18.0212, 20, 11, 0, 7, 2, 374321133227163))
    c.append(Expect(DIST_HARD, 1, 8.5051, 6.5456, 2, 4, 5, 5941726564, 19.1905, 6.7045, 40, 46, 0, 10, 3, 378918026919262))
    c.append(Expect(DIST_EASY, 7, 10.0202, 2.5614, 2, 2, 5, 3783348973, 18.9108, 15.2383, 60, 23, 0, 7, 2, 376485480530262))
    c.append(Expect(DIST_HARD, 7, 13.3158, 16.8228, 3, 4, 6, 6416954414, 19.1905, 16.9817, 20, 25, 0, 11, 5, 383348334780490))
    c.append(Expect(DIST_EASY, 42, 2.0455, 9.2665, 2, 2, 6, 4461279965, 18.0000, 18.0212, 60, 61, 0, 8, 3, 367593284263936))
    c.append(Expect(DIST_HARD, 42, 1.1308, 1.7936, 3, 4, 4, 5263299769, 19.1905, 1.9525, 20, 12, 0, 10, 3, 359286164508938))
    c.append(Expect(DIST_EASY, 123, 18.6538, 8.3966, 3, 2, 3, 1704968168, 18.0000, 18.0212, 0, 0, 0, 8, 3, 376325619477430))
    c.append(Expect(DIST_HARD, 123, 7.0988, 12.3554, 1, 4, 6, 6358447023, 19.1905, 12.5142, 40, 13, 0, 11, 4, 376502979447130))

    var easy = DodgeballGame(DIST_EASY)
    var hard = DodgeballGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: DodgeballGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax) < 1e-3, "rax" + tag)
    assert_true(abs(g.agent.y - e.ray) < 1e-3, "ray" + tag)
    assert_equal(g.exit_wall_choice, e.exit_wall_choice, "exit_wall_choice" + tag)

    var nlava = 0
    var nen = 0
    var esig = 0
    for i in range(len(g.entities)):
        var t = g.entities[i].type
        if t == LAVA_WALL:
            nlava += 1
        elif t == ENEMY:
            nen += 1
        esig += (
            t * 7 + Int(floor(g.entities[i].x * 100.0)) * 100003
            + Int(floor(g.entities[i].y * 100.0))
        ) * (i + 1)
    assert_equal(nlava, e.n_lava, "n_lava" + tag)
    assert_equal(nen, e.n_enemy, "n_enemy" + tag)
    assert_equal(esig, e.esig, "esig" + tag)

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
    assert_equal(g.num_enemies, e.numen_end, "numen_end" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
