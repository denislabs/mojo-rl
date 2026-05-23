"""Test: ComputeGraph.set_node_attr[NAME, ATTR](value).

Phase 1.5 verification. Builds a tiny graph with a Scale node, runs
forward at the default multiplier (1.0), then mutates the multiplier
via `set_node_attr["alpha_scale", "multiplier"](2.5)` and re-runs forward.

Asserts: output[i] = input[i] × 2.5 after the mutation.

This is the API SACActorLossCG will use post-Phase-3 instead of the
hard-coded `self._post_graph.nodes[5].op.multiplier = alpha`.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.initializer import Zero


def test_set_node_attr_multiplier() raises:
    print("test_set_node_attr_multiplier ...")
    comptime DIM = 4
    comptime BATCH = 2
    comptime ScaleGraph = ComputeGraph[
        DIM,
        InputSlot["in", DIM],
        Node["alpha_scale", Scale[DIM], "in"],
    ]
    var g = ScaleGraph.make[target="cpu", INIT=Zero]()

    # Buffers.
    var x = alloc[Scalar[DT]](BATCH * DIM)
    var y = alloc[Scalar[DT]](BATCH * DIM)
    for i in range(BATCH * DIM):
        x[i] = Scalar[DT](Float64(i + 1))  # x = [1, 2, 3, ...]
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())

    # Forward at default multiplier (1.0). Should pass input through.
    g.set_input["in", BATCH](x_t)
    g.forward["cpu", BATCH](y_t)
    print("  default-multiplier forward: y[0..3] =", y[0], y[1], y[2], y[3])
    for i in range(BATCH * DIM):
        assert_true(
            y[i] == x[i],
            "default multiplier (1.0): output should match input",
        )

    # Mutate multiplier via set_node_attr.
    g.set_node_attr["alpha_scale", "multiplier"](Scalar[DT](2.5))

    # Forward again. Output should be input × 2.5.
    g.forward["cpu", BATCH](y_t)
    print("  after-set forward:           y[0..3] =", y[0], y[1], y[2], y[3])
    for i in range(BATCH * DIM):
        var expected = x[i] * Scalar[DT](2.5)
        assert_true(
            y[i] == expected,
            "after set_node_attr(2.5): output must be input * 2.5"
        )

    # Mutate again — confirm the field is genuinely live, not just
    # initialised once.
    g.set_node_attr["alpha_scale", "multiplier"](Scalar[DT](-0.5))
    g.forward["cpu", BATCH](y_t)
    print("  after second set:            y[0..3] =", y[0], y[1], y[2], y[3])
    for i in range(BATCH * DIM):
        var expected = x[i] * Scalar[DT](-0.5)
        assert_true(
            y[i] == expected,
            "after set_node_attr(-0.5): output must be input * -0.5"
        )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ComputeGraph.set_node_attr (Phase 1.5)")
    print("=" * 70)
    test_set_node_attr_multiplier()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
