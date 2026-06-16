"""ComputeGraph v2 smoke test.

Builds a small graph: input → Scale[1.0] = a → Scale[2.0] = b → Sub(b, a)
                          → output (scalar).

input ∈ R, a = 1·input, b = 2·input, output = b - a = input.

So the graph computes identity; forward(x) should equal x, and the
chain rule says backward(grad_output) should equal grad_output, with
fan-out gradient accumulation through `a` (consumed once by `b` and once
by `sub`).

Exit criteria:
  * forward output equals input ± 1e-5 across the batch.
  * backward grad_input equals grad_output ± 1e-5 (identity gradient).
  * The graph compiles, runs without error, and the `_grad_input_buf`
    scatter-add works through the two paths feeding `"input"`.

This is the post-Phase-F2 sanity check: confirms the retrofit graph
stack (compute_graph + graph_nodes + binary trait default for_each_param)
links and runs end-to-end on a non-trivial DAG.
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators import (
    ComputeGraph, InputSlot, Node, Node,
)
from mojo_rl.nn.primitives.scale import Scale
from mojo_rl.nn.primitives.binary_sub import BinarySub
from mojo_rl.nn.initializer import Kaiming

from layout import TileTensor, row_major


def test_compute_graph_identity() raises:
    comptime BATCH = 4

    comptime IdentityGraph = ComputeGraph[
        1,
        InputSlot["input", 1],
        Node["a",   Scale[1], "input"],
        Node["b",   Scale[1], "input"],
        Node["sub", BinarySub[1], "b", "a"],
    ]

    var g = IdentityGraph.make[target="cpu", INIT=Kaiming]()
    # Override the Scale multipliers manually.
    # nodes[0] is the InputSlot; nodes[1] = "a", nodes[2] = "b".
    g.nodes[1].op.multiplier = Scalar[DT](1.0)
    g.nodes[2].op.multiplier = Scalar[DT](2.0)

    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    for b in range(BATCH):
        in_buf[b] = Scalar[DT](b + 1)
        go_buf[b] = Scalar[DT](0.3 + 0.1 * Float64(b))

    var in_t = TileTensor(in_buf, row_major[BATCH, 1]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())
    g.set_input["input", BATCH](in_t)
    g.forward["cpu", BATCH](out_t)

    print("forward outputs:")
    for b in range(BATCH):
        print(
            "  b=", b, " in=", Float64(in_buf[b]),
            " out=", Float64(out_buf[b]),
        )
        assert_true(
            (out_buf[b] - in_buf[b]).__abs__() < Scalar[DT](1e-5),
            "forward output must equal input (b - a = 2·in - 1·in = in)",
        )

    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    g.vjp["cpu", BATCH](go_t)
    var gi_p = g.grad_input_ptr["input"]()

    print("backward grad_inputs:")
    for b in range(BATCH):
        print(
            "  b=", b, " go=", Float64(go_buf[b]),
            " gi=", Float64(gi_p[b]),
        )
        # d(out)/d(in) = d(2·in - 1·in)/d(in) = 1; grad_input = grad_output.
        assert_true(
            (gi_p[b] - go_buf[b]).__abs__() < Scalar[DT](1e-5),
            "grad_input must equal grad_output (identity)",
        )

    in_buf.free()
    out_buf.free()
    go_buf.free()
    print("  test_compute_graph_identity PASSED")


def main() raises:
    print("=" * 60)
    print("ComputeGraph v2 smoke (Phase F2 — post-retrofit)")
    print("=" * 60)
    test_compute_graph_identity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
