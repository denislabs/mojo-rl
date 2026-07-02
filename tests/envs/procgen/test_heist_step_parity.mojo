"""Heist step-physics parity vs the Qt-free C++ probe.

Replays a fixed 150-step action tape (no reset) through `HeistGame.step` and
compares float-robust signatures against `scratchpad/heist_step_probe.cpp`: final
agent position, keys collected, doors opened, entity count, done-fire count, and a
per-step sub-cell position signature. Exercises the base momentum movement
(update_agent_velocity + basic_step_object sub-stepping), grid collision, the
locked-door entity-blocking path (sub_step entity loop + push_obj), and key/door/
exit handling — with RNG consumed in the reference order. See
`docs/PROCGEN_HEIST_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import HeistGame
from mojo_rl.envs.procgen.games.heist import DIST_EASY, DIST_HARD

comptime STEPS = 150


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 5, 5, 1, 1, 3, 3, 8, 6, 2, 0, 4, 7, 5, 3, 1]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var final_x: Float32
    var final_y: Float32
    var keys: Int
    var doors: Int
    var n_ents: Int
    var done_count: Int
    var pos_sig: Int


def test_heist_step_parity() raises:
    # From heist_step_probe (150 steps, tape as above). keys = KEY entities that
    # were collected (initial keys − remaining); doors = doors opened.
    var cases = List[Expect]()
    cases.append(Expect(DIST_EASY, 0, 2.902803, 8.375, 0, 0, 5, 0, 34372349143900))
    cases.append(Expect(DIST_HARD, 0, 6.625, 11.437847, 0, 0, 7, 0, 73684034790508))
    cases.append(Expect(DIST_EASY, 1, 5.800550, 3.375, 0, 0, 3, 0, 67191833970279))
    cases.append(Expect(DIST_HARD, 1, 4.800550, 2.375, 0, 0, 7, 0, 55866788670279))
    cases.append(Expect(DIST_EASY, 7, 7.800550, 4.375, 1, 0, 4, 0, 89841913245279))
    cases.append(Expect(DIST_HARD, 7, 8.625, 8.540115, 0, 0, 7, 0, 96298274647159))
    cases.append(Expect(DIST_EASY, 42, 4.800550, 1.375, 0, 0, 3, 0, 55866777352229))
    cases.append(Expect(DIST_HARD, 42, 8.902843, 9.375, 1, 0, 4, 0, 102322564318900))
    cases.append(Expect(DIST_EASY, 123, 1.800541, 7.375, 0, 0, 3, 0, 22048289833823))
    cases.append(Expect(DIST_HARD, 123, 9.902843, 1.375, 0, 0, 1, 0, 113647507693900))

    var easy = HeistGame(DIST_EASY)
    var hard = HeistGame(DIST_HARD)
    for ci in range(len(cases)):
        var e = cases[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: HeistGame, e: Expect) raises:
    g.reset(e.seed)
    # keys/doors are inferred from entity-count deltas by type.
    var init_keys = 0
    var init_doors = 0
    for i in range(len(g.entities)):
        if g.entities[i].type == 2:  # KEY
            init_keys += 1
        elif g.entities[i].type == 1:  # LOCKED_DOOR
            init_doors += 1

    var done_count = 0
    var pos_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        var cx = Int(floor(g.agent.x * 1000.0))
        var cy = Int(floor(g.agent.y * 1000.0))
        pos_sig += (cx * 1000003 + cy) * (s + 1)

    var rem_keys = 0
    var rem_doors = 0
    for i in range(len(g.entities)):
        if g.entities[i].type == 2:
            rem_keys += 1
        elif g.entities[i].type == 1:
            rem_doors += 1
    var keys = init_keys - rem_keys
    var doors = init_doors - rem_doors

    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)
    assert_true(abs(g.agent.x - e.final_x) < 1e-4, "final_x" + tag)
    assert_true(abs(g.agent.y - e.final_y) < 1e-4, "final_y" + tag)
    assert_equal(keys, e.keys, "keys" + tag)
    assert_equal(doors, e.doors, "doors" + tag)
    assert_equal(len(g.entities), e.n_ents, "n_ents" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
