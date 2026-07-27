"""Phase-0 parity: MT19937 + RandGen + MazeGen vs the C++ ground-truth probe.

Reference vectors were produced by compiling the real Procgen sources
(`randgen.cpp`, `mazegen.cpp`, `cpp-utils.cpp` — none pull in Qt) into a probe
and dumping draw sequences + maze grids. See `docs/PROCGEN_PORT.md` §6.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.core import MT19937, RandGen, MazeGen


def test_mt19937_standard_checkpoint() raises:
    # The canonical std::mt19937 conformance value: seed 5489, 10000th draw.
    var g = MT19937(5489)
    var v: UInt32 = 0
    for _ in range(10000):
        v = g.next_u32()
    assert_equal(Int(v), 4123659995)


def test_randgen_raw_seed0() raises:
    var expected: List[UInt32] = [
        2357136044, 2546248239, 3071714933, 3626093760,
        2588848963, 3684848379, 2340255427, 3638918503,
    ]
    var rg = RandGen()
    rg.seed(0)
    for i in range(len(expected)):
        assert_equal(Int(rg.randint()), Int(expected[i]))


def test_randgen_raw_seed42() raises:
    var expected: List[UInt32] = [
        1608637542, 3421126067, 4083286876, 787846414,
        3143890026, 3348747335, 2571218620, 2563451924,
    ]
    var rg = RandGen()
    rg.seed(42)
    for i in range(len(expected)):
        assert_equal(Int(rg.randint()), Int(expected[i]))


def test_randgen_randn() raises:
    var expected: List[Int] = [2, 9, 2, 0, 0, 2, 6, 7]
    var rg = RandGen()
    rg.seed(123)
    for i in range(len(expected)):
        assert_equal(rg.randn(10), expected[i])


def test_randgen_randint_range() raises:
    var expected: List[Int] = [7, 14, 7, 5, 5, 7, 11, 12]
    var rg = RandGen()
    rg.seed(123)
    for i in range(len(expected)):
        assert_equal(rg.randint(5, 15), expected[i])


def test_randgen_rand01() raises:
    var expected: List[Float32] = [
        0.696469188, 0.712955296, 0.286139339,
        0.428470939, 0.226851448, 0.690884829,
    ]
    var rg = RandGen()
    rg.seed(123)
    for i in range(len(expected)):
        var got = rg.rand01()
        var diff = got - expected[i]
        if diff < 0:
            diff = -diff
        assert_true(diff < 1e-6, "rand01 mismatch at " + String(i))


def test_randgen_choose_n() raises:
    var elems: List[Int] = [10, 11, 12, 13, 14, 15, 16, 17]
    var rg = RandGen()
    rg.seed(123)
    var chosen = rg.choose_n(elems, 4)
    var expected: List[Int] = [16, 11, 10, 12]
    assert_equal(len(chosen), len(expected))
    for i in range(len(expected)):
        assert_equal(chosen[i], expected[i])


def _maze_grid_string(seed: Int) -> String:
    # Replicates MazeGame::game_reset's maze construction (HardMode world_dim=25).
    var rg = RandGen()
    rg.seed(seed)
    var world_dim = 25
    var maze_dim = rg.randn((world_dim - 1) // 2) * 2 + 3
    var mg = MazeGen(maze_dim)
    mg.generate_maze(rg)
    mg.place_objects(rg, 2, 1)  # GOAL = 2
    var s = String("")
    for y in range(mg.grid.h):
        for x in range(mg.grid.w):
            s += String(mg.grid.get(x, y))
            s += " "
        s += "\n"
    return s


def test_maze_seed1_full_grid() raises:
    comptime EXPECTED = String(
        "51 51 51 51 51 51 51 \n"
        "51 100 100 100 51 100 51 \n"
        "51 51 51 100 51 100 51 \n"
        "51 100 100 100 51 100 51 \n"
        "51 100 51 2 51 100 51 \n"
        "51 100 51 100 100 100 51 \n"
        "51 51 51 51 51 51 51 \n"
    )
    assert_equal(_maze_grid_string(1), EXPECTED)


def test_maze_seed7_full_grid() raises:
    comptime EXPECTED = String(
        "51 51 51 51 51 51 51 51 51 51 51 \n"
        "51 100 100 100 100 100 51 100 100 100 51 \n"
        "51 51 51 51 51 2 51 100 51 51 51 \n"
        "51 100 100 100 100 100 100 100 100 100 51 \n"
        "51 51 51 51 51 51 51 100 51 100 51 \n"
        "51 100 100 100 100 100 100 100 51 100 51 \n"
        "51 100 51 51 51 100 51 51 51 51 51 \n"
        "51 100 100 100 51 100 100 100 100 100 51 \n"
        "51 51 51 100 51 51 51 51 51 51 51 \n"
        "51 100 100 100 100 100 100 100 100 100 51 \n"
        "51 51 51 51 51 51 51 51 51 51 51 \n"
    )
    assert_equal(_maze_grid_string(7), EXPECTED)


def test_maze_seed0_dim_and_goal() raises:
    # world_dim 25 → maze_dim 19 → array_dim 21. Exactly one GOAL(2) placed.
    var rg = RandGen()
    rg.seed(0)
    var maze_dim = rg.randn(12) * 2 + 3
    assert_equal(maze_dim, 19)
    var mg = MazeGen(maze_dim)
    mg.generate_maze(rg)
    mg.place_objects(rg, 2, 1)
    var goals = 0
    for y in range(mg.grid.h):
        for x in range(mg.grid.w):
            if mg.grid.get(x, y) == 2:
                goals += 1
    assert_equal(goals, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
