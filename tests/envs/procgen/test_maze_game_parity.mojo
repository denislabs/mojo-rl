"""Game-level level-exactness: full maze reset vs the C++ game probe.

Unlike `test_mt19937_parity` (which validates the generator in isolation), this
exercises the *game* reset RNG order — the base-class draws (`bg_pct_x`,
`background_index`) that precede `maze_dim` — so a given level seed reproduces
reference Procgen's maze exactly. Ground truth from `scratchpad/maze_game_probe.cpp`
(Qt-free: only the topdown-background count, 9, is needed). See `docs/PROCGEN_PORT.md`.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import (
    MazeGame,
    DIST_EASY,
    DIST_HARD,
    DIST_MEMORY,
    world_dim_for,
)
from mojo_rl.envs.procgen.core.object_ids import SPACE

comptime ASSET_ROOT = String("references/procgen-master/procgen/data/assets/")


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var seed: Int
    var bg_pct_x: Float32
    var bg_idx: Int
    var maze_dim: Int
    var spaces: Int
    var checksum: Int


def test_maze_game_reset_parity() raises:
    # From maze_game_probe: seed, bg_pct_x, bg_idx, maze_dim, spaces, checksum.
    var cases = List[Expect]()
    cases.append(Expect(0, 0.548813522, 0, 13, 96, 11432616))
    cases.append(Expect(1, 0.417021990, 5, 3, 6, 10052237))
    cases.append(Expect(7, 0.076308288, 4, 5, 16, 10213300))
    cases.append(Expect(42, 0.374540120, 5, 11, 70, 11044095))
    cases.append(Expect(123, 0.696469188, 1, 15, 126, 11887973))

    var game = MazeGame(ASSET_ROOT)
    for ci in range(len(cases)):
        var e = cases[ci]
        game.reset(e.seed)
        var diff = game.bg_pct_x - e.bg_pct_x
        if diff < 0:
            diff = -diff
        assert_true(diff < 1e-6, "bg_pct_x mismatch seed " + String(e.seed))
        assert_equal(game.background_index, e.bg_idx)
        assert_equal(game.maze_dim, e.maze_dim)
        var checksum = 0
        var spaces = 0
        for k in range(game.w * game.h):
            checksum += (k + 1) * game.grid[k]
            if game.grid[k] == SPACE:
                spaces += 1
        assert_equal(spaces, e.spaces)
        assert_equal(checksum, e.checksum)


@fieldwise_init
struct ModeExpect(ImplicitlyCopyable, Movable):
    var dist_mode: Int
    var seed: Int
    var maze_dim: Int
    var spaces: Int
    var checksum: Int


def test_maze_distribution_modes() raises:
    # From maze_game_probe (Easy=15 / Hard=25 / Memory=31). Base draws
    # (bg_pct_x, bg_idx) are mode-independent and covered above; here we confirm
    # maze_dim + full grid (spaces + weighted checksum) per mode.
    var cases = List[ModeExpect]()
    cases.append(ModeExpect(DIST_EASY, 0, 3, 6, 1323723))
    cases.append(ModeExpect(DIST_EASY, 7, 15, 126, 1977971))
    cases.append(ModeExpect(DIST_EASY, 42, 11, 70, 1684265))
    cases.append(ModeExpect(DIST_HARD, 0, 13, 96, 11432616))
    cases.append(ModeExpect(DIST_HARD, 7, 5, 16, 10213300))
    cases.append(ModeExpect(DIST_HARD, 42, 11, 70, 11044095))
    cases.append(ModeExpect(DIST_MEMORY, 0, 19, 198, 28208711))
    cases.append(ModeExpect(DIST_MEMORY, 7, 5, 16, 23935666))
    cases.append(ModeExpect(DIST_MEMORY, 42, 5, 16, 23926454))

    # One game per mode (reused across its seeds) to limit asset reloads.
    var easy = MazeGame(ASSET_ROOT, DIST_EASY)
    var hard = MazeGame(ASSET_ROOT, DIST_HARD)
    var mem = MazeGame(ASSET_ROOT, DIST_MEMORY)

    for ci in range(len(cases)):
        var e = cases[ci]
        var wd = world_dim_for(e.dist_mode)
        if e.dist_mode == DIST_EASY:
            easy.reset(e.seed)
        elif e.dist_mode == DIST_HARD:
            hard.reset(e.seed)
        else:
            mem.reset(e.seed)

        var checksum = 0
        var spaces = 0
        for k in range(wd * wd):
            var v: Int
            if e.dist_mode == DIST_EASY:
                v = easy.grid[k]
            elif e.dist_mode == DIST_HARD:
                v = hard.grid[k]
            else:
                v = mem.grid[k]
            checksum += (k + 1) * v
            if v == SPACE:
                spaces += 1

        var md: Int
        if e.dist_mode == DIST_EASY:
            md = easy.maze_dim
        elif e.dist_mode == DIST_HARD:
            md = hard.maze_dim
        else:
            md = mem.maze_dim
        assert_equal(md, e.maze_dim)
        assert_equal(spaces, e.spaces)
        assert_equal(checksum, e.checksum)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
