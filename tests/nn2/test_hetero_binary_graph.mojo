"""Quick smoke for the hetero-binary variadic workaround.

Builds a graph with BinaryConcat (IN0_DIM=3, IN1_DIM=5, OUT_DIM=8) and
verifies forward + backward both compile and produce correct results.
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators import (
    ComputeGraph, InputSlot, BinaryNode,
)
from mojo_rl.nn2.primitives.binary_concat import BinaryConcat
from mojo_rl.nn2.initializer import Kaiming

from layout import TileTensor, row_major


def test_hetero_binary_concat() raises:
    """Concat two inputs of different dim sizes."""
    comptime BATCH = 2
    comptime IN0_DIM = 3
    comptime IN1_DIM = 5
    comptime OUT_DIM = IN0_DIM + IN1_DIM

    comptime ConcatGraph = ComputeGraph[
        OUT_DIM,
        InputSlot["a", IN0_DIM],
        InputSlot["b", IN1_DIM],
        BinaryNode["out", BinaryConcat[IN0_DIM, IN1_DIM], "a", "b"],
    ]

    var g = ConcatGraph.make[target="cpu", INIT=Kaiming]()

    var a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN0_DIM
    )
    var b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN1_DIM
    )
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT_DIM
    )
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT_DIM
    )

    # Fill a = [100*b + d for d in IN0_DIM], b = [1000*b + d for d in IN1_DIM]
    for b in range(BATCH):
        for d in range(IN0_DIM):
            a_buf[b * IN0_DIM + d] = Scalar[DT](Float64(b) * 100.0 + Float64(d))
        for d in range(IN1_DIM):
            b_buf[b * IN1_DIM + d] = Scalar[DT](Float64(b) * 1000.0 + Float64(d))
    # Unique sentinel grad_out so we can verify split.
    for k in range(BATCH * OUT_DIM):
        go_buf[k] = Scalar[DT](Float64(k) + 1.0)

    var a_t = TileTensor(a_buf, row_major[BATCH, IN0_DIM]())
    var b_t = TileTensor(b_buf, row_major[BATCH, IN1_DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT_DIM]())

    g.set_input["a", BATCH](a_t)
    g.set_input["b", BATCH](b_t)
    g.forward["cpu", BATCH](out_t)

    print("forward: out[b, :IN0_DIM] = a[b, :]; out[b, IN0_DIM:] = b[b, :]")
    for b in range(BATCH):
        for d in range(IN0_DIM):
            var got = out_buf[b * OUT_DIM + d]
            var want = a_buf[b * IN0_DIM + d]
            assert_true(
                (got - want).__abs__() < Scalar[DT](1e-6),
                "out[b, d<IN0_DIM] must equal a[b, d]",
            )
        for d in range(IN1_DIM):
            var got = out_buf[b * OUT_DIM + IN0_DIM + d]
            var want = b_buf[b * IN1_DIM + d]
            assert_true(
                (got - want).__abs__() < Scalar[DT](1e-6),
                "out[b, IN0_DIM+d] must equal b[b, d]",
            )

    var go_t = TileTensor(go_buf, row_major[BATCH, OUT_DIM]())
    g.vjp["cpu", BATCH](go_t)

    var ga_p = g.grad_input_ptr["a"]()
    var gb_p = g.grad_input_ptr["b"]()

    print("backward: grad_a = grad_out[b, :IN0_DIM]; grad_b = grad_out[b, IN0_DIM:]")
    for b in range(BATCH):
        for d in range(IN0_DIM):
            var got = ga_p[b * IN0_DIM + d]
            var want = go_buf[b * OUT_DIM + d]
            assert_true(
                (got - want).__abs__() < Scalar[DT](1e-6),
                "grad_a[b, d] must equal grad_out[b, d]",
            )
        for d in range(IN1_DIM):
            var got = gb_p[b * IN1_DIM + d]
            var want = go_buf[b * OUT_DIM + IN0_DIM + d]
            assert_true(
                (got - want).__abs__() < Scalar[DT](1e-6),
                "grad_b[b, d] must equal grad_out[b, IN0_DIM+d]",
            )

    a_buf.free()
    b_buf.free()
    out_buf.free()
    go_buf.free()
    print("  test_hetero_binary_concat PASSED")


def main() raises:
    print("=" * 60)
    print("Hetero-binary variadic workaround smoke")
    print("=" * 60)
    test_hetero_binary_concat()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
