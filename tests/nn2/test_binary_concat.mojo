"""CPU tests for the Cleanup-13 BinaryConcat[IN0_DIM, IN1_DIM] primitive.

Mirrors the structure of tests/nn2/test_binary_ops.mojo:
  - forward correctness against hand-computed values
  - backward correctness against analytic formulas
  - FD gradcheck on each input (virtual loss = Σ go·output)
  - Phase 10A buffer surface round-trip
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import BinaryModule
from mojo_rl.nn2.primitives.binary_concat import BinaryConcat
from mojo_rl.nn2.initializer import Zero


def test_binary_concat_forward_backward() raises:
    comptime IN0 = 3
    comptime IN1 = 2
    comptime OUT = IN0 + IN1
    comptime BATCH = 4

    var in0_buf = alloc[Scalar[DT]](BATCH * IN0)
    var in1_buf = alloc[Scalar[DT]](BATCH * IN1)
    var out_buf = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf = alloc[Scalar[DT]](BATCH * OUT)
    var gi0_buf = alloc[Scalar[DT]](BATCH * IN0)
    var gi1_buf = alloc[Scalar[DT]](BATCH * IN1)

    for i in range(BATCH * IN0):
        in0_buf[i] = Scalar[DT](Float32(i) * 0.21 - 0.6)
    for i in range(BATCH * IN1):
        in1_buf[i] = Scalar[DT](Float32(i) * 0.07 + 0.3)
    for i in range(BATCH * OUT):
        go_buf[i] = Scalar[DT](Float32(i) * 0.13 + 0.1)

    var op = BinaryConcat[IN0, IN1].make[target="cpu", INIT=Zero]()
    var in0_t = TileTensor(in0_buf, row_major[BATCH, IN0]())
    var in1_t = TileTensor(in1_buf, row_major[BATCH, IN1]())
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT]())
    op.forward["cpu", BATCH](in0_t, in1_t, out_t)

    # Forward: output[b, d] is in0 for d<IN0, in1 shifted otherwise.
    for b in range(BATCH):
        for d in range(IN0):
            assert_almost_equal(
                out_buf[b * OUT + d], in0_buf[b * IN0 + d], atol=1e-7
            )
        for d in range(IN1):
            assert_almost_equal(
                out_buf[b * OUT + IN0 + d], in1_buf[b * IN1 + d], atol=1e-7
            )

    var go_t = TileTensor(go_buf, row_major[BATCH, OUT]())
    var gi0_t = TileTensor(gi0_buf, row_major[BATCH, IN0]())
    var gi1_t = TileTensor(gi1_buf, row_major[BATCH, IN1]())
    op.backward["cpu", BATCH](go_t, gi0_t, gi1_t)

    # Backward: grad_in0 reads the IN0 prefix; grad_in1 reads the IN1 suffix.
    for b in range(BATCH):
        for d in range(IN0):
            assert_almost_equal(
                gi0_buf[b * IN0 + d], go_buf[b * OUT + d], atol=1e-7
            )
        for d in range(IN1):
            assert_almost_equal(
                gi1_buf[b * IN1 + d], go_buf[b * OUT + IN0 + d], atol=1e-7
            )

    # FD gradcheck on in0
    var eps: Scalar[DT] = 1e-3
    var max_rel_0: Scalar[DT] = 0.0
    for idx in range(BATCH * IN0):
        var orig = in0_buf[idx]
        in0_buf[idx] = orig + eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_plus: Scalar[DT] = 0.0
        for k in range(BATCH * OUT):
            L_plus += go_buf[k] * out_buf[k]
        in0_buf[idx] = orig - eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_minus: Scalar[DT] = 0.0
        for k in range(BATCH * OUT):
            L_minus += go_buf[k] * out_buf[k]
        in0_buf[idx] = orig
        var num = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
        var ana = gi0_buf[idx]
        var ae = fabs(num - ana)
        var denom = fabs(num) + fabs(ana) + Scalar[DT](1e-6)
        var rel = ae / denom
        if rel > max_rel_0:
            max_rel_0 = rel
    print("  BinaryConcat FD max_rel(in0)=", max_rel_0)
    assert_true(max_rel_0 < Scalar[DT](1e-3), "BinaryConcat FD in0 too loose")

    # FD gradcheck on in1
    var max_rel_1: Scalar[DT] = 0.0
    for idx in range(BATCH * IN1):
        var orig = in1_buf[idx]
        in1_buf[idx] = orig + eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_plus: Scalar[DT] = 0.0
        for k in range(BATCH * OUT):
            L_plus += go_buf[k] * out_buf[k]
        in1_buf[idx] = orig - eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_minus: Scalar[DT] = 0.0
        for k in range(BATCH * OUT):
            L_minus += go_buf[k] * out_buf[k]
        in1_buf[idx] = orig
        var num = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
        var ana = gi1_buf[idx]
        var ae = fabs(num - ana)
        var denom = fabs(num) + fabs(ana) + Scalar[DT](1e-6)
        var rel = ae / denom
        if rel > max_rel_1:
            max_rel_1 = rel
    print("  BinaryConcat FD max_rel(in1)=", max_rel_1)
    assert_true(max_rel_1 < Scalar[DT](1e-3), "BinaryConcat FD in1 too loose")

    in0_buf.free(); in1_buf.free(); out_buf.free()
    go_buf.free(); gi0_buf.free(); gi1_buf.free()
    print("  test_binary_concat_forward_backward PASSED")


def test_binary_concat_buffer_surface() raises:
    """`ensure_buffers[BATCH]` allocates non-null buffers of the right size,
    and `out_ptr` / `grad_in0_ptr` / `grad_in1_ptr` / `grad_out_ptr` are
    usable as `TileTensor` data pointers for a round-trip forward+backward."""
    comptime IN0 = 5
    comptime IN1 = 3
    comptime OUT = IN0 + IN1
    comptime BATCH = 4
    var op = BinaryConcat[IN0, IN1].make[target="cpu", INIT=Zero]()
    op.ensure_buffers[BATCH]()

    var out_p = op.out_ptr()
    var gi0_p = op.grad_in0_ptr()
    var gi1_p = op.grad_in1_ptr()
    var go_p = op.grad_out_ptr()

    var in0_buf = alloc[Scalar[DT]](BATCH * IN0)
    var in1_buf = alloc[Scalar[DT]](BATCH * IN1)
    for i in range(BATCH * IN0):
        in0_buf[i] = Scalar[DT](Float32(i) + 1.0)
    for i in range(BATCH * IN1):
        in1_buf[i] = Scalar[DT](Float32(i) * 0.5 - 1.0)
    var in0_t = TileTensor(in0_buf, row_major[BATCH, IN0]())
    var in1_t = TileTensor(in1_buf, row_major[BATCH, IN1]())
    var out_t = TileTensor(out_p, row_major[BATCH, OUT]())
    op.forward["cpu", BATCH](in0_t, in1_t, out_t)
    for b in range(BATCH):
        for d in range(IN0):
            assert_almost_equal(
                out_p[b * OUT + d], in0_buf[b * IN0 + d], atol=1e-7
            )
        for d in range(IN1):
            assert_almost_equal(
                out_p[b * OUT + IN0 + d], in1_buf[b * IN1 + d], atol=1e-7
            )

    for i in range(BATCH * OUT):
        go_p[i] = Scalar[DT](Float32(i) * 0.11 + 0.3)
    var go_t = TileTensor(go_p, row_major[BATCH, OUT]())
    var gi0_t = TileTensor(gi0_p, row_major[BATCH, IN0]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, IN1]())
    op.backward["cpu", BATCH](go_t, gi0_t, gi1_t)
    for b in range(BATCH):
        for d in range(IN0):
            assert_almost_equal(gi0_p[b * IN0 + d], go_p[b * OUT + d], atol=1e-7)
        for d in range(IN1):
            assert_almost_equal(
                gi1_p[b * IN1 + d], go_p[b * OUT + IN0 + d], atol=1e-7
            )

    in0_buf.free(); in1_buf.free()
    print("  test_binary_concat_buffer_surface PASSED")


def main() raises:
    print("=" * 70)
    print("nn2 Cleanup 13 — BinaryConcat[IN0, IN1] CPU tests")
    print("=" * 70)
    test_binary_concat_forward_backward()
    test_binary_concat_buffer_surface()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
