"""ComputeGraph v2 multi-input smoke (Block B).

Exercises the multi-external-input feature shipped by Block B
(2026-05-22). Builds a graph with two named InputSlots and verifies
that:

  * forward consumes both slots independently
  * backward writes per-slot gradient accumulators that the caller
    can read via `grad_input_ptr[NAME]()`
  * scatter-add through fan-out across multiple slots is correct

Graph (one slot fans out, one is single-use):
  a:        InputSlot["a", 1]
  b:        InputSlot["b", 1]
  scaled_a: a * 3.0
  scaled_b: b * 5.0
  out:      scaled_a - scaled_b    (BinarySub)

Math:
  forward(a, b) = 3a - 5b
  d/da = +3 · grad_out
  d/db = -5 · grad_out

Pick grad_out = 1.0 across the batch — then grad_a should be +3 per
sample and grad_b should be -5. (The minus sign on grad_b is what
makes the test non-trivial: it exercises the BinarySub backward in
combination with the per-slot scatter accumulator.)
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators import (
    ComputeGraph, InputSlot, UnaryNode, BinaryNode,
)
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.nn2.initializer import Kaiming

from layout import TileTensor, row_major


def test_multi_input_independence() raises:
    comptime BATCH = 4

    comptime AffineGraph = ComputeGraph[
        1,
        InputSlot["a", 1],
        InputSlot["b", 1],
        UnaryNode["scaled_a", Scale[1], "a"],
        UnaryNode["scaled_b", Scale[1], "b"],
        BinaryNode["out",     BinarySub[1], "scaled_a", "scaled_b"],
    ]

    var g = AffineGraph.make[target="cpu", INIT=Kaiming]()
    # Indices: 0=slot a, 1=slot b, 2=scaled_a, 3=scaled_b, 4=out.
    g.nodes[2].op.multiplier = Scalar[DT](3.0)
    g.nodes[3].op.multiplier = Scalar[DT](5.0)

    var a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    for k in range(BATCH):
        a_buf[k] = Scalar[DT](Float64(k) + 1.0)        # a = 1, 2, 3, 4
        b_buf[k] = Scalar[DT](Float64(k) * 0.5 + 0.1)  # b = 0.1, 0.6, 1.1, 1.6
        go_buf[k] = Scalar[DT](1.0)                    # uniform unit grad-out

    var a_t = TileTensor(a_buf, row_major[BATCH, 1]())
    var b_t = TileTensor(b_buf, row_major[BATCH, 1]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())

    g.set_input["a", BATCH](a_t)
    g.set_input["b", BATCH](b_t)
    g.forward["cpu", BATCH](out_t)

    print("forward (CPU): out[i] = 3·a[i] - 5·b[i]")
    for k in range(BATCH):
        var expected = Scalar[DT](3.0) * a_buf[k] - Scalar[DT](5.0) * b_buf[k]
        print(
            "  k=", k,
            " a=", Float64(a_buf[k]),
            " b=", Float64(b_buf[k]),
            " out=", Float64(out_buf[k]),
            " expected=", Float64(expected),
        )
        assert_true(
            (out_buf[k] - expected).__abs__() < Scalar[DT](1e-5),
            "forward output mismatch",
        )

    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    g.backward["cpu", BATCH](go_t)
    var ga_p = g.grad_input_ptr["a"]()
    var gb_p = g.grad_input_ptr["b"]()

    print("backward (CPU): grad_a = +3, grad_b = -5  (per sample)")
    for k in range(BATCH):
        print(
            "  k=", k,
            " grad_a=", Float64(ga_p[k]),
            " grad_b=", Float64(gb_p[k]),
        )
        assert_true(
            (ga_p[k] - Scalar[DT](3.0)).__abs__() < Scalar[DT](1e-5),
            "grad_a must equal +3",
        )
        assert_true(
            (gb_p[k] - Scalar[DT](-5.0)).__abs__() < Scalar[DT](1e-5),
            "grad_b must equal -5",
        )

    a_buf.free()
    b_buf.free()
    out_buf.free()
    go_buf.free()
    print("  test_multi_input_independence PASSED")


def test_multi_input_fanout() raises:
    """A consumed by two compute nodes — exercises per-slot scatter-add.

    Graph: out = (3·a) - (2·a) - b   →   out = a - b
      grad_a should be (+3) + (-2) = +1
      grad_b should be -1
    """
    comptime BATCH = 4

    comptime FanoutGraph = ComputeGraph[
        1,
        InputSlot["a", 1],
        InputSlot["b", 1],
        UnaryNode["a3", Scale[1], "a"],
        UnaryNode["a2", Scale[1], "a"],
        BinaryNode["ab", BinarySub[1], "a3", "a2"],   # 3·a - 2·a = a
        BinaryNode["out", BinarySub[1], "ab", "b"],   # a - b
    ]

    var g = FanoutGraph.make[target="cpu", INIT=Kaiming]()
    g.nodes[2].op.multiplier = Scalar[DT](3.0)
    g.nodes[3].op.multiplier = Scalar[DT](2.0)

    var a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    for k in range(BATCH):
        a_buf[k] = Scalar[DT](2.0 + Float64(k))
        b_buf[k] = Scalar[DT](Float64(k) - 0.5)
        go_buf[k] = Scalar[DT](1.0)

    var a_t = TileTensor(a_buf, row_major[BATCH, 1]())
    var b_t = TileTensor(b_buf, row_major[BATCH, 1]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())

    g.set_input["a", BATCH](a_t)
    g.set_input["b", BATCH](b_t)
    g.forward["cpu", BATCH](out_t)

    for k in range(BATCH):
        var expected = a_buf[k] - b_buf[k]
        assert_true(
            (out_buf[k] - expected).__abs__() < Scalar[DT](1e-5),
            "fanout forward mismatch",
        )

    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    g.backward["cpu", BATCH](go_t)
    var ga_p = g.grad_input_ptr["a"]()
    var gb_p = g.grad_input_ptr["b"]()

    print("fanout backward: grad_a (=+3-2=+1), grad_b (=-1)")
    for k in range(BATCH):
        print(
            "  k=", k,
            " grad_a=", Float64(ga_p[k]),
            " grad_b=", Float64(gb_p[k]),
        )
        assert_true(
            (ga_p[k] - Scalar[DT](1.0)).__abs__() < Scalar[DT](1e-5),
            "fanout grad_a must equal +1 (scatter-add of +3 and -2)",
        )
        assert_true(
            (gb_p[k] - Scalar[DT](-1.0)).__abs__() < Scalar[DT](1e-5),
            "fanout grad_b must equal -1",
        )

    a_buf.free()
    b_buf.free()
    out_buf.free()
    go_buf.free()
    print("  test_multi_input_fanout PASSED")


def main() raises:
    print("=" * 60)
    print("ComputeGraph v2 multi-input smoke (Block B — 2026-05-22)")
    print("=" * 60)
    test_multi_input_independence()
    test_multi_input_fanout()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
