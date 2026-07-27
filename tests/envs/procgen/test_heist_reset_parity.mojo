"""Heist reset level-exactness vs the Qt-free C++ probe.

Exercises the BasicAbstractGame base-reset + HeistGame::game_reset RNG order (bg
draws → agent-spawn draws → difficulty → num_keys → generate_maze_with_doors →
off_x/off_y → collision-checked KEY/EXIT spawns) so a level seed reproduces
reference Procgen's heist layout. Ground truth = `scratchpad/heist_reset_probe.cpp`
(uses the real mazegen.cpp; reroll==0 confirmed → spawn RNG is asset-independent).
Asset-free/fast. See `docs/PROCGEN_HEIST_SCOPE.md`.
"""

from std.math import floor
from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import HeistGame, KEY, LOCKED_DOOR, EXIT
from mojo_rl.envs.procgen.games.heist import DIST_EASY, DIST_HARD
from mojo_rl.envs.procgen.core.object_ids import SPACE


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var bg_pct_x: Float32
    var bg_idx: Int
    var num_keys: Int
    var maze_dim: Int
    var off_x: Int
    var off_y: Int
    var agent_x: Float32
    var agent_y: Float32
    var space: Int
    var grid_cksum: Int
    var n_key: Int
    var n_door: Int
    var pos_sig: Int
    var theme_sig: Int


def test_heist_reset_parity() raises:
    # From heist_reset_probe: Easy (wd=9) + Hard (wd=13) × 5 seeds.
    var cases = List[Expect]()
    cases.append(Expect(DIST_EASY, 0, 0.548813522, 0, 2, 7, 1, 2, 2.5, 8.5, 31, 245664, 2, 2, 75411290271, 802))
    cases.append(Expect(DIST_HARD, 0, 0.548813522, 0, 3, 11, 0, 2, 6.5, 12.5, 71, 1068951, 3, 3, 114996522746, 751))
    cases.append(Expect(DIST_EASY, 1, 0.417021990, 5, 2, 7, 0, 1, 6.5, 2.5, 31, 228710, 1, 1, 25309114609, 409))
    cases.append(Expect(DIST_HARD, 1, 0.417021990, 5, 3, 11, 1, 0, 5.5, 1.5, 71, 991874, 3, 3, 129047618929, 1062))
    cases.append(Expect(DIST_EASY, 7, 0.076308288, 4, 2, 7, 2, 2, 8.5, 3.5, 31, 244145, 2, 2, 94749354880, 437))
    cases.append(Expect(DIST_HARD, 7, 0.076308288, 4, 3, 11, 2, 1, 8.5, 8.5, 71, 1038130, 3, 3, 297637114364, 1135))
    cases.append(Expect(DIST_EASY, 42, 0.374540120, 5, 1, 5, 3, 1, 5.5, 1.5, 17, 196860, 1, 1, 41867149335, 409))
    cases.append(Expect(DIST_HARD, 42, 0.374540120, 5, 2, 7, 4, 3, 8.5, 9.5, 31, 862122, 2, 2, 111020440924, 526))
    cases.append(Expect(DIST_EASY, 123, 0.696469188, 1, 1, 7, 0, 1, 2.5, 7.5, 31, 230964, 1, 1, 29028110080, 409))
    cases.append(Expect(DIST_HARD, 123, 0.696469188, 1, 0, 5, 7, 1, 9.5, 1.5, 17, 772256, 0, 0, 7557024127, 118))

    var easy = HeistGame(DIST_EASY)
    var hard = HeistGame(DIST_HARD)
    for ci in range(len(cases)):
        var e = cases[ci]
        if e.dist_mode == DIST_EASY:
            easy.reset(e.seed)
            _check(easy, e)
        else:
            hard.reset(e.seed)
            _check(hard, e)


def _check(g: HeistGame, e: Expect) raises:
    var tag = " mode " + String(e.dist_mode) + " seed " + String(e.seed)

    var diff = g.bg_pct_x - e.bg_pct_x
    if diff < 0:
        diff = -diff
    assert_true(diff < 1e-6, "bg_pct_x" + tag)
    assert_equal(g.background_index, e.bg_idx, "bg_idx" + tag)
    assert_equal(g.num_keys, e.num_keys, "num_keys" + tag)
    assert_equal(g.maze_dim, e.maze_dim, "maze_dim" + tag)
    assert_equal(g.off_x, e.off_x, "off_x" + tag)
    assert_equal(g.off_y, e.off_y, "off_y" + tag)
    assert_true(abs(g.agent.x - e.agent_x) < 1e-4, "agent_x" + tag)
    assert_true(abs(g.agent.y - e.agent_y) < 1e-4, "agent_y" + tag)

    var grid_cksum = 0
    var space = 0
    for k in range(g.w * g.h):
        grid_cksum += (k + 1) * g.grid[k]
        if g.grid[k] == SPACE:
            space += 1
    assert_equal(space, e.space, "space" + tag)
    assert_equal(grid_cksum, e.grid_cksum, "grid_cksum" + tag)

    var pos_sig = 0
    var theme_sig = 0
    var n_key = 0
    var n_door = 0
    var n_exit = 0
    for i in range(len(g.entities)):
        ref en = g.entities[i]
        var fx = Int(floor(en.x * 1000.0))
        var fy = Int(floor(en.y * 1000.0))
        pos_sig += (fx * 1000003 + fy) * (i + 1)
        theme_sig += (en.type * 13 + en.image_theme + 1) * (i + 1)
        if en.type == KEY:
            n_key += 1
        elif en.type == LOCKED_DOOR:
            n_door += 1
        elif en.type == EXIT:
            n_exit += 1
    assert_equal(n_key, e.n_key, "n_key" + tag)
    assert_equal(n_door, e.n_door, "n_door" + tag)
    assert_equal(n_exit, 1, "n_exit" + tag)
    assert_equal(pos_sig, e.pos_sig, "pos_sig" + tag)
    assert_equal(theme_sig, e.theme_sig, "theme_sig" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
