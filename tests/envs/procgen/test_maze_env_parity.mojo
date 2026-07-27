"""Level-selection parity: LevelScheduler vs the C++ env probe.

Validates the train/test level-scheduling layer (vecgame.cpp per-env seeding +
Game::reset's `current_level_seed` draw) independently of asset loading — so this
gate is fast. Ground truth from `scratchpad/maze_env_probe.cpp` (Qt-free).
See `docs/PROCGEN_PORT.md`.
"""

from std.testing import assert_equal, TestSuite

from mojo_rl.envs.procgen.core import LevelScheduler


def _seq(rand_seed: Int, num_levels: Int, start_level: Int, k: Int) -> List[Int]:
    var sch = LevelScheduler(rand_seed, num_levels, start_level)
    var out = List[Int]()
    for _ in range(k):
        out.append(sch.next_level_seed())
    return out^


def _check(
    rand_seed: Int, num_levels: Int, start_level: Int, expected: List[Int]
) raises:
    var got = _seq(rand_seed, num_levels, start_level, len(expected))
    for i in range(len(expected)):
        assert_equal(got[i], expected[i])


def test_finite_levels() raises:
    # rand_seed=0, num_levels=200, start_level=0 → levels in [0,200).
    _check(0, 200, 0, [71, 165, 125, 72, 126, 58, 117, 12])


def test_offset_levels() raises:
    # rand_seed=42, num_levels=500, start_level=100 → levels in [100,600).
    _check(42, 500, 100, [271, 241, 454, 438, 360, 574, 597, 239])


def test_unbounded_levels() raises:
    # num_levels=0 → levels in [0, INT32_MAX).
    _check(
        0, 0, 0,
        [1263407424, 1641956518, 686441525, 1851516025,
         1678418526, 1517557411, 1874689470, 2122579412],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
