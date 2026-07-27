"""Fruitbot reset+step parity vs the Qt-free C++ probe.

Combined. Validates the heaviest reset (height partition, walls with doors/locks,
presents, ~20-40 collision-retry object spawns, theme+fit_aspect) and the step
(auto-scroll, collect/crash, fire, bullet↔lock/barrier) over a 300-step tape.
Ground truth = `scratchpad/fruitbot_probe.cpp`. Asset-free/fast.
See `docs/PROCGEN_FRUITBOT_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import FruitbotGame
from mojo_rl.envs.procgen.games.fruitbot import (
    DIST_EASY, DIST_HARD, GOOD_OBJ, BAD_OBJ, BARRIER, LOCKED_DOOR, LOCK,
)

comptime STEPS = 300


def _tape(step: Int) -> Int:
    var t: List[Int] = [4, 1, 4, 7, 9, 4, 1, 7, 4, 9, 7, 1]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax0: Float32
    var good: Int
    var bad: Int
    var bar: Int
    var door: Int
    var lock: Int
    var ent_sig: Int
    var final_y: Float32
    var reward_x10: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var pos_sig: Int


def test_fruitbot_parity() raises:
    var c = List[Expect]()
    # From fruitbot_probe (300-step tape).
    c.append(Expect(DIST_EASY, 0, 6.9797, 10, 13, 10, 0, 0, 33543544, 42.0131, -130, 25, 0, 36, 310017464027894))
    c.append(Expect(DIST_HARD, 0, 14.1316, 17, 13, 20, 0, 0, 113139507, 42.0131, -20, 59, 0, 68, 632931232766294))
    c.append(Expect(DIST_EASY, 1, 7.0270, 13, 11, 10, 0, 0, 38524039, 42.0131, -80, 13, 0, 43, 312154506439002))
    c.append(Expect(DIST_HARD, 1, 14.2302, 18, 19, 20, 1, 1, 148003333, 42.0131, -130, 70, 0, 73, 637391618147410))
    c.append(Expect(DIST_EASY, 7, 7.5753, 10, 17, 10, 0, 0, 46825872, 42.0131, -60, 16, 0, 44, 336911856710830))
    c.append(Expect(DIST_HARD, 7, 15.3744, 11, 14, 20, 0, 0, 110318361, 42.0131, 0, 41, 0, 65, 689045396108279))
    c.append(Expect(DIST_EASY, 42, 9.1466, 19, 15, 10, 0, 0, 57272693, 42.0131, -150, 27, 0, 49, 407856856545191))
    c.append(Expect(DIST_HARD, 42, 18.6537, 13, 13, 20, 1, 1, 116753303, 42.0131, -40, 49, 0, 67, 837094705254874))
    c.append(Expect(DIST_EASY, 123, 3.0325, 12, 13, 10, 0, 0, 44436568, 42.0131, -170, 30, 0, 38, 131802424384379))
    c.append(Expect(DIST_HARD, 123, 5.8939, 12, 18, 20, 1, 1, 132446950, 42.0131, -110, 42, 0, 67, 260988272940762))

    var easy = FruitbotGame(DIST_EASY)
    var hard = FruitbotGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: FruitbotGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax0) < 1e-3, "rax0" + tag)

    var good = 0
    var bad = 0
    var bar = 0
    var door = 0
    var lock = 0
    var ent_sig = 0
    for i in range(len(g.entities)):
        ref en = g.entities[i]
        ent_sig += (
            en.type * 3
            + Int(floor(en.x * 100)) * 7
            + Int(floor(en.y * 100)) * 11
            + en.image_theme
        ) * (i + 1)
        if en.type == GOOD_OBJ:
            good += 1
        elif en.type == BAD_OBJ:
            bad += 1
        elif en.type == BARRIER:
            bar += 1
        elif en.type == LOCKED_DOOR:
            door += 1
        elif en.type == LOCK:
            lock += 1
    assert_equal(good, e.good, "good" + tag)
    assert_equal(bad, e.bad, "bad" + tag)
    assert_equal(bar, e.bar, "bar" + tag)
    assert_equal(door, e.door, "door" + tag)
    assert_equal(lock, e.lock, "lock" + tag)
    assert_equal(ent_sig, e.ent_sig, "ent_sig" + tag)

    var done_count = 0
    var pos_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        var cx = Int(floor(g.agent.x * 1000))
        var cy = Int(floor(g.agent.y * 1000))
        pos_sig += (cx * 1000003 + cy) * (s + 1)

    assert_true(abs(g.agent.y - e.final_y) < 1e-3, "final_y" + tag)
    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(1 if g.level_complete else 0, e.complete, "complete" + tag)
    assert_equal(len(g.entities), e.n_ent, "n_ent" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
