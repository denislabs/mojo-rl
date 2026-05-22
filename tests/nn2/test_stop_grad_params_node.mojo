"""Test: StopGradParams as a ComputeGraph node — frozen inner params.

Phase 1.2 verification. Confirms that `UnaryNode["...", StopGradParams[Linear], "..."]`
in a ComputeGraph (a) forwards correctly through Linear and (b) leaves
Linear's `weight.grad` / `bias.grad` untouched after a full forward+backward
through the graph.

Two sub-tests:

  1. **control**: graph with `UnaryNode["lin", Linear[3, 1], "x"]` — after
     forward+backward, Linear's `weight.grad` must be non-zero
     (sanity-check that the test's gradient seed is actually flowing).

  2. **stop-grad-params**: graph with `UnaryNode["sgp",
     StopGradParams[Linear[3, 1]], "x"]` — after the SAME forward+backward
     shape, Linear's `weight.grad` must be ALL-ZERO (StopGradParams
     forced `mode="input_only"` on inner.backward).

If both sub-tests pass, Phase 3 (FullGraph SAC actor loss) can use
`StopGradParams[CRITIC]` as a graph node directly — no external mode
coordination needed.
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, UnaryNode
from mojo_rl.nn2.combinators.stop_grad_params import StopGradParams
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Xavier


def test_control_linear_grads_flow() raises:
    """Control: a graph with raw Linear MUST accumulate weight.grad ≠ 0."""
    print("test_control_linear_grads_flow ...")
    comptime BATCH = 2
    comptime ControlGraph = ComputeGraph[
        1,
        InputSlot["x", 3],
        UnaryNode["lin", Linear[3, 1], "x"],
    ]
    var g = ControlGraph.make[target="cpu", INIT=Xavier]()

    # Forward + backward with a non-zero input + non-zero grad seed.
    var x_buf = alloc[Scalar[DT]](BATCH * 3)
    var y_buf = alloc[Scalar[DT]](BATCH * 1)
    var go_buf = alloc[Scalar[DT]](BATCH * 1)
    for i in range(BATCH * 3):
        x_buf[i] = Scalar[DT](0.5 + 0.1 * Float64(i))
    for i in range(BATCH):
        go_buf[i] = Scalar[DT](1.0)  # non-zero grad seed

    var x_t = TileTensor(x_buf, row_major[BATCH, 3]())
    var y_t = TileTensor(y_buf, row_major[BATCH, 1]())
    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    g.set_input["x", BATCH](x_t)
    g.forward["cpu", BATCH](y_t)
    g.backward["cpu", BATCH](go_t)

    # Inspect the Linear leaf at node index 1 (InputSlot is 0).
    var max_w_grad: Scalar[DT] = 0.0
    for i in range(3):  # IN_DIM * OUT_DIM = 3 * 1 = 3
        var v = g.nodes[1].op.weight.grad[i]
        var av = v if v >= Scalar[DT](0) else -v
        if av > max_w_grad:
            max_w_grad = av
    print("  control max |weight.grad| =", max_w_grad)
    assert_true(
        max_w_grad > Scalar[DT](1e-6),
        "control: weight.grad must be non-zero (gradient seed isn't flowing?)"
    )
    print("  ok")


def test_stop_grad_params_freezes_inner() raises:
    """StopGradParams[Linear] in a graph — Linear's grads must stay ZERO."""
    print("test_stop_grad_params_freezes_inner ...")
    comptime BATCH = 2
    comptime FrozenGraph = ComputeGraph[
        1,
        InputSlot["x", 3],
        UnaryNode["sgp", StopGradParams[Linear[3, 1]], "x"],
    ]
    var g = FrozenGraph.make[target="cpu", INIT=Xavier]()

    var x_buf = alloc[Scalar[DT]](BATCH * 3)
    var y_buf = alloc[Scalar[DT]](BATCH * 1)
    var go_buf = alloc[Scalar[DT]](BATCH * 1)
    for i in range(BATCH * 3):
        x_buf[i] = Scalar[DT](0.5 + 0.1 * Float64(i))
    for i in range(BATCH):
        go_buf[i] = Scalar[DT](1.0)

    var x_t = TileTensor(x_buf, row_major[BATCH, 3]())
    var y_t = TileTensor(y_buf, row_major[BATCH, 1]())
    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    g.set_input["x", BATCH](x_t)
    g.forward["cpu", BATCH](y_t)
    g.backward["cpu", BATCH](go_t)

    # Path: graph.nodes[1].op = StopGradParams; .op.inner = Linear.
    var max_w_grad: Scalar[DT] = 0.0
    for i in range(3):
        var v = g.nodes[1].op.inner.weight.grad[i]
        var av = v if v >= Scalar[DT](0) else -v
        if av > max_w_grad:
            max_w_grad = av
    var max_b_grad: Scalar[DT] = 0.0
    for i in range(1):
        var v = g.nodes[1].op.inner.bias.grad[i]
        var av = v if v >= Scalar[DT](0) else -v
        if av > max_b_grad:
            max_b_grad = av
    print(
        "  frozen max |weight.grad| =", max_w_grad,
        " max |bias.grad| =", max_b_grad,
    )
    assert_true(
        max_w_grad == Scalar[DT](0),
        "StopGradParams: weight.grad must be zero after backward",
    )
    assert_true(
        max_b_grad == Scalar[DT](0),
        "StopGradParams: bias.grad must be zero after backward",
    )

    # Sanity: grad_input still flowed (StopGradParams routes grad_in via
    # inner.backward[mode="input_only"], so the slot's grad accumulator
    # must be non-zero).
    var gx_p = g.grad_input_ptr["x"]()
    var max_gx: Scalar[DT] = 0.0
    for i in range(BATCH * 3):
        var v = gx_p[i]
        var av = v if v >= Scalar[DT](0) else -v
        if av > max_gx:
            max_gx = av
    print("  frozen max |grad_input(x)| =", max_gx)
    assert_true(
        max_gx > Scalar[DT](1e-6),
        "StopGradParams: grad_input must still flow (input-only mode)",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("StopGradParams-as-GraphNode verification (Phase 1.2)")
    print("=" * 70)
    test_control_linear_grads_flow()
    test_stop_grad_params_freezes_inner()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
