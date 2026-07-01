"""Game-level level-exactness: full maze reset vs the C++ game probe.

Unlike `test_mt19937_parity` (which validates the generator in isolation), this
exercises the *game* reset RNG order — the base-class draws (`bg_pct_x`,
`background_index`) that precede `maze_dim` — so a given level seed reproduces
reference Procgen's maze exactly. Ground truth from `scratchpad/maze_game_probe.cpp`
(Qt-free: only the topdown-background count, 9, is needed). See `docs/PROCGEN_PORT.md`.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import MazeSpikeGame
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

    var game = MazeSpikeGame(ASSET_ROOT)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
