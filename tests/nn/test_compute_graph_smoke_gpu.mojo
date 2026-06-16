"""ComputeGraph v2 GPU smoke test (Block A).

Same identity DAG as `test_compute_graph_smoke.mojo` but executed on
GPU. Verifies:
  * Node / Node allocate DeviceBuffer storage when made via
    target="gpu".
  * `_forward_gpu` / `_backward_gpu` walk the topo + reverse-topo
    correctly, with kernels for the inter-node wiring (zero/add/copy).
  * Fan-out (`a` consumed by both `b`'s sibling and `sub`) scatters into
    the external grad_input_buf correctly.

Graph:
  input → a = 1·input
        → b = 2·input
        → sub = b - a  →  output

Identity: forward(x) ≈ x, backward(go) ≈ go.

Tolerance is 1e-5 in fp32 (matches the CPU smoke test).
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators import (
    ComputeGraph, InputSlot, Node, Node,
)
from mojo_rl.nn.primitives.scale import Scale
from mojo_rl.nn.primitives.binary_sub import BinarySub
from mojo_rl.nn.initializer import Kaiming

from layout import TileTensor, row_major


def test_compute_graph_identity_gpu() raises:
    comptime BATCH = 4
    var ctx = DeviceContext()

    comptime IdentityGraph = ComputeGraph[
        1,
        InputSlot["input", 1],
        Node["a",   Scale[1], "input"],
        Node["b",   Scale[1], "input"],
        Node["sub", BinarySub[1], "b", "a"],
    ]

    var g = IdentityGraph.make[target="gpu", INIT=Kaiming](ctx)
    # Override the Scale multipliers manually.
    # nodes[0] is the InputSlot; nodes[1] = "a", nodes[2] = "b".
    g.nodes[1].op.multiplier = Scalar[DT](1.0)
    g.nodes[2].op.multiplier = Scalar[DT](2.0)

    # Host scratch.
    var in_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    var out_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    var go_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    var gi_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()

    for b in range(BATCH):
        in_h.unsafe_ptr()[b] = Scalar[DT](b + 1)
        go_h.unsafe_ptr()[b] = Scalar[DT](0.3 + 0.1 * Float64(b))

    # Device buffers.
    var in_d = ctx.enqueue_create_buffer[DT](BATCH)
    var out_d = ctx.enqueue_create_buffer[DT](BATCH)
    var go_d = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(in_d, in_h)
    ctx.enqueue_copy(go_d, go_h)
    ctx.synchronize()

    var in_t = TileTensor(in_d, row_major[BATCH, 1]())
    var out_t = TileTensor(out_d, row_major[BATCH, 1]())
    g.set_input["input", BATCH](in_t)
    g.forward["gpu", BATCH](out_t)
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    print("forward outputs (GPU):")
    for b in range(BATCH):
        var ip = in_h.unsafe_ptr()[b]
        var op_ = out_h.unsafe_ptr()[b]
        print("  b=", b, " in=", Float64(ip), " out=", Float64(op_))
        assert_true(
            (op_ - ip).__abs__() < Scalar[DT](1e-5),
            "forward output must equal input (b - a = 2·in - 1·in = in)",
        )

    var go_t = TileTensor(go_d, row_major[BATCH, 1]())
    g.vjp["gpu", BATCH](go_t)
    # The slot's grad_out_buf is the input-gradient accumulator. We need
    # the underlying DeviceBuffer to copy back to the host — the slot
    # owns it as `nodes[0]._grad_out_buf_dev`.
    ctx.enqueue_copy(
        gi_h, g.nodes[0]._grad_out_buf_dev.value(),
    )
    ctx.synchronize()

    print("backward grad_inputs (GPU):")
    for b in range(BATCH):
        var go = go_h.unsafe_ptr()[b]
        var gi = gi_h.unsafe_ptr()[b]
        print("  b=", b, " go=", Float64(go), " gi=", Float64(gi))
        assert_true(
            (gi - go).__abs__() < Scalar[DT](1e-5),
            "grad_input must equal grad_output (identity)",
        )

    print("  test_compute_graph_identity_gpu PASSED")


def main() raises:
    print("=" * 60)
    print("ComputeGraph v2 GPU smoke (Block A — Phase A4)")
    print("=" * 60)
    test_compute_graph_identity_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
