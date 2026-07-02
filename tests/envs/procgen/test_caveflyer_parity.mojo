"""Caveflyer reset+step parity vs the Qt-free C++ probe.

Combined P0+P1 — the ROOMGEN SUBSTRATE gate. Validates roomgen (random fill →
4× CA update → find_best_room flood fill → simple_choose agent/goal → find_path
BFS → expand_room dilation → 4× update → object placement) and the rotate-and-
thrust flyer step (rotation dynamics, thrust, projectile firing, entity/grid
collisions, enemy reflect off cave walls) over a thrust+fire tape. Ground truth =
`scratchpad/caveflyer_probe.cpp`. Asset-free/fast. See `docs/PROCGEN_CAVEFLYER_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import CaveflyerGame
from mojo_rl.envs.procgen.games.caveflyer import (
    DIST_EASY, DIST_HARD, CAVEWALL, OBSTACLE, TARGET, ENEMY,
)


comptime STEPS = 200


def _tape(step: Int) -> Int:
    var t: List[Int] = [5, 5, 8, 5, 9, 5, 2, 5, 9, 8, 5, 7]
    return t[step % len(t)]


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var rax: Float32
    var ray: Float32
    var goal_x: Int
    var goal_y: Int
    var n_cavewall: Int
    var n_space: Int
    var gsig: Int
    var free_after: Int
    var chunk: Int
    var num_objs: Int
    var path_len: Int
    var n_obstacle: Int
    var n_target: Int
    var n_enemy: Int
    var final_x: Float32
    var final_y: Float32
    var rot: Float32
    var reward_x10: Int
    var done_count: Int
    var complete: Int
    var n_ent: Int
    var pos_sig: Int
    var rot_sig: Int


def test_caveflyer_parity() raises:
    var c = List[Expect]()
    # From caveflyer_probe (200-step thrust+fire tape). rot_sig identical (tape-driven).
    c.append(Expect(DIST_EASY, 0, 25.5, 20.5, 16, 7, 748, 152, 8135706, 127, 1, 3, 25, 1, 1, 1, 28.5, 13.5, 8.4383, 0, 33, 0, 38, 513037074580430, 114190747))
    c.append(Expect(DIST_HARD, 0, 8.5, 38.5, 26, 19, 1427, 173, 30800924, 135, 1, 3, 38, 1, 1, 1, 10.7561, 36.9588, 8.4383, 0, 4, 0, 40, 215182279205904, 114190747))
    c.append(Expect(DIST_EASY, 1, 7.5, 17.5, 21, 27, 738, 162, 14010274, 135, 1, 3, 27, 1, 1, 1, 13.5, 24.5, 8.4383, 0, 6, 0, 39, 154726643137802, 114190747))
    c.append(Expect(DIST_HARD, 1, 20.5, 26.5, 27, 20, 1504, 96, 19865804, 82, 1, 3, 14, 1, 1, 1, 22.5, 28.5, 8.4383, 0, 0, 0, 40, 403442277697847, 114190747))
    c.append(Expect(DIST_EASY, 7, 4.5, 12.5, 16, 7, 804, 96, 6755062, 76, 0, 0, 20, 0, 0, 0, 14.5, 12.6423, 8.4383, 0, 0, 0, 37, 250920931783900, 114190747))
    c.append(Expect(DIST_HARD, 7, 12.5, 14.5, 24, 20, 1462, 138, 20552952, 119, 1, 3, 19, 1, 1, 1, 24.5, 17.9693, 8.4383, 0, 33, 0, 40, 297006086920664, 114190747))
    c.append(Expect(DIST_EASY, 42, 3.5, 11.5, 6, 23, 755, 145, 11077774, 125, 1, 3, 20, 1, 1, 1, 11.5, 20.2180, 8.4383, 0, 72, 0, 37, 134809089631016, 114190747))
    c.append(Expect(DIST_HARD, 42, 19.5, 8.5, 27, 2, 1530, 70, 12999292, 55, 0, 0, 15, 0, 0, 0, 22.7561, 9.9588, 8.4383, 0, 0, 0, 37, 449293191089029, 114190747))
    c.append(Expect(DIST_EASY, 123, 23.5, 13.5, 25, 10, 845, 55, 5621346, 49, 0, 0, 6, 0, 0, 0, 28.5, 14.7884, 8.4383, 900, 9, 0, 37, 523285263744612, 114190747))
    c.append(Expect(DIST_HARD, 123, 35.5, 31.5, 26, 35, 1462, 138, 28591268, 124, 1, 3, 14, 1, 1, 1, 35.7158, 36.0243, 8.4383, 0, 14, 0, 40, 613876720855095, 114190747))

    var easy = CaveflyerGame(DIST_EASY)
    var hard = CaveflyerGame(DIST_HARD)
    for ci in range(len(c)):
        var e = c[ci]
        if e.dist_mode == DIST_EASY:
            _run_and_check(easy, e)
        else:
            _run_and_check(hard, e)


def _run_and_check(mut g: CaveflyerGame, e: Expect) raises:
    g.reset(e.seed)
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    assert_true(abs(g.agent.x - e.rax) < 1e-3, "rax" + tag)
    assert_true(abs(g.agent.y - e.ray) < 1e-3, "ray" + tag)
    assert_equal(g.goal_cell % g.w, e.goal_x, "goal_x" + tag)
    assert_equal(g.goal_cell // g.w, e.goal_y, "goal_y" + tag)

    var ncw = 0
    var nsp = 0
    var gsig = 0
    for i in range(g.w * g.h):
        var v = g.grid.data[i]
        if v == CAVEWALL:
            ncw += 1
        elif v == 100:  # SPACE
            nsp += 1
        gsig += (v + 1) * (i + 1)
    assert_equal(ncw, e.n_cavewall, "n_cavewall" + tag)
    assert_equal(nsp, e.n_space, "n_space" + tag)
    assert_equal(gsig, e.gsig, "gsig" + tag)

    var nobs = 0
    var ntar = 0
    var nen = 0
    for i in range(len(g.entities)):
        var t = g.entities[i].type
        if t == OBSTACLE:
            nobs += 1
        elif t == TARGET:
            ntar += 1
        elif t == ENEMY:
            nen += 1
    assert_equal(nobs, e.n_obstacle, "n_obstacle" + tag)
    assert_equal(ntar, e.n_target, "n_target" + tag)
    assert_equal(nen, e.n_enemy, "n_enemy" + tag)

    var done_count = 0
    var pos_sig = 0
    var rot_sig = 0
    for s in range(STEPS):
        _ = g.step(_tape(s))
        if g.done:
            done_count += 1
        var cx = Int(floor(g.agent.x * 1000))
        var cy = Int(floor(g.agent.y * 1000))
        pos_sig += (cx * 1000003 + cy) * (s + 1)
        rot_sig += Int(floor(g.agent.rotation * 1000)) * (s + 1)

    assert_true(abs(g.agent.x - e.final_x) < 1e-3, "final_x" + tag)
    assert_true(abs(g.agent.y - e.final_y) < 1e-3, "final_y" + tag)
    assert_true(abs(g.agent.rotation - e.rot) < 1e-3, "rot" + tag)
    assert_equal(Int(round(g.episode_reward * 10.0)), e.reward_x10, "reward" + tag)
    assert_equal(done_count, e.done_count, "done_count" + tag)
    assert_equal(1 if g.level_complete else 0, e.complete, "complete" + tag)
    assert_equal(len(g.entities), e.n_ent, "n_ent" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)
    assert_equal(rot_sig, e.rot_sig, "rot_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
