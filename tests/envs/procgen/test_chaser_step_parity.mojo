"""Chaser step-physics parity vs the Qt-free C++ probe.

Replays a fixed action tape (120 steps, no reset) through `ChaserGame.step` and
compares float-robust signatures against `scratchpad/chaser_step_probe.cpp`:
final agent position, accumulated reward, orbs collected, entity count, done-fire
count, and a per-step cell + sub-cell position signature. This exercises the whole
continuous substrate (basic_step_object sub-stepping, grid collision, Pac-Man
agent velocity) + chaser's enemy AI / egg hatch / respawn / pickup / completion,
with RNG (one step_rand_int/step) consumed in the reference order.

See `docs/PROCGEN_CHASER_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import ChaserGame, DIST_EASY, DIST_HARD, DIST_EXTREME

comptime STEPS = 120


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 5, 5, 1, 1, 3, 3, 8, 6, 2, 0, 4, 7, 5, 3, 1]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var mode: Int
    var seed: Int
    var final_x: Float32
    var final_y: Float32
    var reward_x100: Int
    var orbs: Int
    var n_ents: Int
    var done_count: Int
    var cell_sig: Int
    var pos_sig: Int


def test_chaser_step_parity() raises:
    var cases = List[Expect]()
    # From chaser_step_probe (120 steps, tape as above).
    cases.append(Expect(0, 0, 2.875, 0.5, 32, 8, 7, 17, 39241, 334001135875))
    cases.append(Expect(1, 0, 3.0, 10.5, 36, 8, 5, 0, 981297, 311002251250))
    cases.append(Expect(2, 0, 14.5, 12.5, 12, 3, 10, 5, 1898244, 1736506804000))
    cases.append(Expect(0, 7, 3.0, 0.5, 52, 13, 7, 18, 79410, 319501158750))
    cases.append(Expect(1, 7, 11.0, 2.5, 40, 10, 6, 1, 288259, 1297504271250))
    cases.append(Expect(2, 7, 14.0, 8.5, 60, 15, 10, 20, 1485108, 1524755931375))
    cases.append(Expect(0, 42, 6.5, 8.5, 12, 3, 7, 17, 767316, 776503444000))
    cases.append(Expect(1, 42, 12.5, 4.5, 48, 11, 5, 2, 520069, 1378754715375))
    cases.append(Expect(2, 42, 18.5, 16.5, 48, 12, 10, 17, 2417089, 2101758321875))

    var easy = ChaserGame(DIST_EASY)
    var hard = ChaserGame(DIST_HARD)
    var extreme = ChaserGame(DIST_EXTREME)

    for ci in range(len(cases)):
        var e = cases[ci]
        if e.mode == 0:
            _run_and_check(easy, e)
        elif e.mode == 1:
            _run_and_check(hard, e)
        else:
            _run_and_check(extreme, e)


def _run_and_check(mut g: ChaserGame, e: Expect) raises:
    g.reset(e.seed)
    var total_reward: Float32 = 0.0
    var done_count = 0
    var cell_sig = 0
    var pos_sig = 0
    for s in range(STEPS):
        var r = g.step(_tape(s))
        total_reward += r
        if g.done:
            done_count += 1
        var cx = Int(floor(g.agent.x * 1000.0))
        var cy = Int(floor(g.agent.y * 1000.0))
        pos_sig += cx * 1000003 + cy
        var fcx = Int(floor(g.agent.x))
        var fcy = Int(floor(g.agent.y))
        cell_sig += (fcx + g.w * fcy + 1) * (s + 1)

    var tag = " mode " + String(e.mode) + " seed " + String(e.seed)
    assert_true(abs(g.agent.x - e.final_x) < 1e-4, "final_x" + tag)
    assert_true(abs(g.agent.y - e.final_y) < 1e-4, "final_y" + tag)
    assert_equal(Int(round(total_reward * 100.0)), e.reward_x100, "reward" + tag)
    assert_equal(g.orbs_collected, e.orbs, "orbs" + tag)
    assert_equal(len(g.entities), e.n_ents, "n_ents" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(cell_sig, e.cell_sig, "cell_sig" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
