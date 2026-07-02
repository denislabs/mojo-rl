"""Chaser reset level-exactness vs the Qt-free C++ probe.

Exercises the `BasicAbstractGame` base-reset + `ChaserGame::game_reset` RNG order
(bg draws → agent-spawn draws → maze gen → extra_quad → per-quadrant orb
`simple_choose` → enemy `simple_choose`) so a level seed reproduces reference
Procgen's chaser layout exactly. Ground truth = `scratchpad/chaser_reset_probe.cpp`
(Qt-free: only the background count, 1, is needed). See `docs/PROCGEN_CHASER_SCOPE.md`.

Asset-free (reset builds no sprites) so this runs in milliseconds.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import (
    ChaserGame,
    DIST_EASY,
    DIST_HARD,
    DIST_EXTREME,
    LARGE_ORB,
    ENEMY_EGG,
    MAZE_WALL,
    ORB,
)
from mojo_rl.envs.procgen.core.object_ids import SPACE


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var bg_pct_x: Float32
    var maze_dim: Int
    var agent_x: Float32
    var agent_y: Float32
    var total_orbs: Int
    var orb_cells: Int
    var wall: Int
    var space: Int
    var grid_cksum: Int
    var n_large: Int
    var n_egg: Int
    var ent_cksum: Int


def test_chaser_reset_parity() raises:
    # From chaser_reset_probe: 3 modes × 5 seeds.
    var cases = List[Expect]()
    # Easy (maze_dim=11, enemies=3, extra_orb_sign=0)
    cases.append(Expect(DIST_EASY, 0, 0.548813522, 11, 2.5, 2.5, 75, 75, 42, 4, 4607877, 4, 3, 1326))
    cases.append(Expect(DIST_EASY, 1, 0.417021990, 11, 10.5, 0.5, 75, 75, 42, 4, 4618273, 4, 3, 1036))
    cases.append(Expect(DIST_EASY, 7, 0.076308288, 11, 4.5, 1.5, 77, 77, 40, 4, 4847819, 4, 3, 870))
    cases.append(Expect(DIST_EASY, 42, 0.374540120, 11, 6.5, 8.5, 77, 77, 40, 4, 4744799, 4, 3, 1326))
    cases.append(Expect(DIST_EASY, 123, 0.696469188, 11, 2.5, 6.5, 75, 75, 42, 4, 4693145, 4, 3, 1430))
    # Hard (maze_dim=13, enemies=3, extra_orb_sign=-1)
    cases.append(Expect(DIST_HARD, 0, 0.548813522, 13, 0.5, 12.5, 108, 108, 58, 3, 9174216, 3, 3, 1848))
    cases.append(Expect(DIST_HARD, 1, 0.417021990, 13, 8.5, 11.5, 107, 107, 59, 3, 9260566, 3, 3, 910))
    cases.append(Expect(DIST_HARD, 7, 0.076308288, 13, 9.5, 4.5, 106, 106, 60, 3, 9048258, 3, 3, 1886))
    cases.append(Expect(DIST_HARD, 42, 0.374540120, 13, 10.5, 4.5, 107, 107, 59, 3, 9223916, 3, 3, 1398))
    cases.append(Expect(DIST_HARD, 123, 0.696469188, 13, 8.5, 2.5, 107, 107, 59, 3, 9128538, 3, 3, 1548))
    # Extreme (maze_dim=19, enemies=5, extra_orb_sign=+1)
    cases.append(Expect(DIST_EXTREME, 0, 0.548813522, 19, 14.5, 12.5, 220, 220, 136, 5, 40272175, 5, 5, 4412))
    cases.append(Expect(DIST_EXTREME, 1, 0.417021990, 19, 3.5, 16.5, 217, 217, 139, 5, 39256181, 5, 5, 5868))
    cases.append(Expect(DIST_EXTREME, 7, 0.076308288, 19, 12.5, 12.5, 222, 222, 134, 5, 40669697, 5, 5, 5478))
    cases.append(Expect(DIST_EXTREME, 42, 0.374540120, 19, 18.5, 15.5, 217, 217, 139, 5, 39507327, 5, 5, 4252))
    cases.append(Expect(DIST_EXTREME, 123, 0.696469188, 19, 0.5, 11.5, 219, 219, 137, 5, 40007833, 5, 5, 5868))

    var easy = ChaserGame(DIST_EASY)
    var hard = ChaserGame(DIST_HARD)
    var extreme = ChaserGame(DIST_EXTREME)

    for ci in range(len(cases)):
        var e = cases[ci]
        if e.dist_mode == DIST_EASY:
            easy.reset(e.seed)
            _check(easy, e)
        elif e.dist_mode == DIST_HARD:
            hard.reset(e.seed)
            _check(hard, e)
        else:
            extreme.reset(e.seed)
            _check(extreme, e)


def _check(g: ChaserGame, e: Expect) raises:
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    var diff = g.bg_pct_x - e.bg_pct_x
    if diff < 0:
        diff = -diff
    assert_true(diff < 1e-6, "bg_pct_x mismatch" + tag)
    assert_equal(g.background_index, 0, "bg_idx" + tag)
    assert_equal(g.maze_dim, e.maze_dim, "maze_dim" + tag)

    assert_true(abs(g.agent.x - e.agent_x) < 1e-4, "agent_x" + tag)
    assert_true(abs(g.agent.y - e.agent_y) < 1e-4, "agent_y" + tag)
    assert_equal(g.total_orbs, e.total_orbs, "total_orbs" + tag)

    var grid_cksum = 0
    var orb_cells = 0
    var wall = 0
    var space = 0
    for k in range(g.w * g.h):
        var v = g.grid[k]
        grid_cksum += (k + 1) * v
        if v == ORB:
            orb_cells += 1
        elif v == MAZE_WALL:
            wall += 1
        elif v == SPACE:
            space += 1
    assert_equal(orb_cells, e.orb_cells, "orb_cells" + tag)
    assert_equal(wall, e.wall, "wall" + tag)
    assert_equal(space, e.space, "space" + tag)
    assert_equal(grid_cksum, e.grid_cksum, "grid_cksum" + tag)

    var ent_cksum = 0
    var n_large = 0
    var n_egg = 0
    for ei in range(len(g.entities)):
        ref ent = g.entities[ei]
        var cell = Int(ent.y) * g.maze_dim + Int(ent.x)
        ent_cksum += ent.type * (cell + 1)
        if ent.type == LARGE_ORB:
            n_large += 1
        elif ent.type == ENEMY_EGG:
            n_egg += 1
    assert_equal(n_large, e.n_large, "n_large" + tag)
    assert_equal(n_egg, e.n_egg, "n_egg" + tag)
    assert_equal(ent_cksum, e.ent_cksum, "ent_cksum" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
