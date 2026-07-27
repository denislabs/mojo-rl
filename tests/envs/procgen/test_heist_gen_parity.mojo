"""`MazeGen.generate_maze_with_doors` parity vs the Qt-free C++ probe.

Validates the lock-and-key maze generator (heist) in isolation — the array_dim²
grid checksum + object counts (doors/keys/exit/agent/space/wall) must match
reference Procgen bit-for-bit, which pins down the `expand_to_type` frontier
ordering and the `choose_n`/`choose_one` draw sequence. Ground truth =
`scratchpad/heist_gen_probe.cpp` (uses the real `mazegen.cpp`). Asset-free/fast.

See `docs/PROCGEN_HEIST_SCOPE.md`.
"""

from std.testing import assert_equal, TestSuite

from mojo_rl.envs.procgen.core.randgen import RandGen
from mojo_rl.envs.procgen.core.mazegen import MazeGen
from mojo_rl.envs.procgen.core.object_ids import (
    SPACE,
    WALL_OBJ,
    DOOR_OBJ,
    KEY_OBJ,
    EXIT_OBJ,
    AGENT_OBJ,
)


@fieldwise_init
struct Expect(ImplicitlyCopyable, Movable):
    var maze_dim: Int
    var num_doors: Int
    var seed: Int
    var cksum: Int
    var door: Int
    var key: Int


def test_heist_gen_parity() raises:
    # From heist_gen_probe: (maze_dim, num_doors) ∈ {(5,1),(7,2),(9,3),(13,3)}.
    var cases = List[Expect]()
    cases.append(Expect(5, 1, 0, 90890, 1, 1))
    cases.append(Expect(7, 2, 0, 253783, 2, 2))
    cases.append(Expect(9, 3, 0, 569062, 3, 3))
    cases.append(Expect(13, 3, 0, 1882236, 3, 3))
    cases.append(Expect(5, 1, 1, 89759, 1, 1))
    cases.append(Expect(7, 2, 1, 248879, 2, 2))
    cases.append(Expect(9, 3, 1, 575593, 3, 3))
    cases.append(Expect(13, 3, 1, 1951970, 3, 3))
    cases.append(Expect(5, 1, 7, 85033, 1, 1))
    cases.append(Expect(7, 2, 7, 248927, 2, 2))
    cases.append(Expect(9, 3, 7, 573314, 3, 3))
    cases.append(Expect(13, 3, 7, 1923820, 3, 3))
    cases.append(Expect(5, 1, 42, 87580, 1, 1))
    cases.append(Expect(7, 2, 42, 251622, 2, 2))
    cases.append(Expect(9, 3, 42, 566467, 3, 3))
    cases.append(Expect(13, 3, 42, 1920703, 3, 3))
    cases.append(Expect(5, 1, 123, 84804, 1, 1))
    cases.append(Expect(7, 2, 123, 253043, 2, 2))
    cases.append(Expect(9, 3, 123, 560909, 3, 3))
    cases.append(Expect(13, 3, 123, 1957831, 3, 3))

    for ci in range(len(cases)):
        var e = cases[ci]
        var tag = (
            " md " + String(e.maze_dim) + " nd " + String(e.num_doors)
            + " seed " + String(e.seed)
        )
        var rg = RandGen()
        rg.seed(e.seed)
        var mg = MazeGen(e.maze_dim)
        mg.generate_maze_with_doors(rg, e.num_doors)

        var ad = e.maze_dim + 2
        var cksum = 0
        var door = 0
        var key = 0
        var exit = 0
        var agent = 0
        for i in range(ad * ad):
            var v = mg.grid.get_index(i)
            cksum += (i + 1) * v
            if v >= DOOR_OBJ and v < KEY_OBJ:
                door += 1
            elif v >= KEY_OBJ:
                key += 1
            elif v == EXIT_OBJ:
                exit += 1
            elif v == AGENT_OBJ:
                agent += 1
        assert_equal(cksum, e.cksum, "cksum" + tag)
        assert_equal(door, e.door, "door" + tag)
        assert_equal(key, e.key, "key" + tag)
        assert_equal(exit, 1, "exit" + tag)
        assert_equal(agent, 1, "agent" + tag)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
