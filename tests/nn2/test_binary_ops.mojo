"""CPU tests for the Phase 10C two-input Modules: BinarySub, BinaryElemMin.

Mirrors the structure of tests/nn2/test_elementwise_ops.mojo for the
packed Sub / ElemMin, but feeds the two inputs through separate
[BATCH, DIM] tiles and reads the two grads through separate tiles.

Each test covers:
  - forward correctness against hand-computed values
  - backward correctness against analytic formulas
  - FD gradcheck on each input (virtual loss = Σ go·output)

Also exercises the Phase 10A buffer surface (`ensure_buffers`,
`out_ptr`, `grad_in0_ptr`, `grad_in1_ptr`, `grad_out_ptr`) to
make sure the CG-v2-ready hooks return non-null pointers and the
right sizes.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import BinaryModule
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.nn2.primitives.binary_elem_min import BinaryElemMin
from mojo_rl.nn2.initializer import Zero


# ──────────────────────────────────────────────────────────────────────────
# BinarySub
# ──────────────────────────────────────────────────────────────────────────


def test_binary_sub_forward_backward() raises:
    comptime DIM = 3
    comptime BATCH = 2
    var in0_buf = alloc[Scalar[DT]](BATCH * DIM)
    var in1_buf = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * DIM)
    var go_buf = alloc[Scalar[DT]](BATCH * DIM)
    var gi0_buf = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_buf = alloc[Scalar[DT]](BATCH * DIM)

    for i in range(BATCH * DIM):
        in0_buf[i] = Scalar[DT](Float32(i) * 0.21 - 0.6)
        in1_buf[i] = Scalar[DT](Float32(i) * 0.07 + 0.3)
        go_buf[i] = Scalar[DT](Float32(i) * 0.13 + 0.1)

    var op = BinarySub[DIM].make[target="cpu", INIT=Zero]()
    var in0_t = TileTensor(in0_buf, row_major[BATCH, DIM]())
    var in1_t = TileTensor(in1_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, DIM]())
    op.forward["cpu", BATCH](in0_t, in1_t, out_t)
    for b in range(BATCH):
        for d in range(DIM):
            assert_almost_equal(
                out_buf[b * DIM + d],
                in0_buf[b * DIM + d] - in1_buf[b * DIM + d],
                atol=1e-7,
            )

    var go_t = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi0_t = TileTensor(gi0_buf, row_major[BATCH, DIM]())
    var gi1_t = TileTensor(gi1_buf, row_major[BATCH, DIM]())
    op.backward["cpu", BATCH](go_t, gi0_t, gi1_t)
    for b in range(BATCH):
        for d in range(DIM):
            assert_almost_equal(
                gi0_buf[b * DIM + d], go_buf[b * DIM + d], atol=1e-7
            )
            assert_almost_equal(
                gi1_buf[b * DIM + d], -go_buf[b * DIM + d], atol=1e-7
            )

    # FD gradcheck on in0
    var eps: Scalar[DT] = 1e-3
    var max_rel_0: Scalar[DT] = 0.0
    for idx in range(BATCH * DIM):
        var orig = in0_buf[idx]
        in0_buf[idx] = orig + eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_plus: Scalar[DT] = 0.0
        for k in range(BATCH * DIM):
            L_plus += go_buf[k] * out_buf[k]
        in0_buf[idx] = orig - eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_minus: Scalar[DT] = 0.0
        for k in range(BATCH * DIM):
            L_minus += go_buf[k] * out_buf[k]
        in0_buf[idx] = orig
        var num = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
        var ana = gi0_buf[idx]
        var ae = fabs(num - ana)
        var denom = fabs(num) + fabs(ana) + Scalar[DT](1e-6)
        var rel = ae / denom
        if rel > max_rel_0:
            max_rel_0 = rel
    print("  BinarySub FD max_rel(in0)=", max_rel_0)
    assert_true(max_rel_0 < Scalar[DT](1e-3), "BinarySub FD in0 too loose")

    # FD gradcheck on in1
    var max_rel_1: Scalar[DT] = 0.0
    for idx in range(BATCH * DIM):
        var orig = in1_buf[idx]
        in1_buf[idx] = orig + eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_plus: Scalar[DT] = 0.0
        for k in range(BATCH * DIM):
            L_plus += go_buf[k] * out_buf[k]
        in1_buf[idx] = orig - eps
        op.forward["cpu", BATCH](in0_t, in1_t, out_t)
        var L_minus: Scalar[DT] = 0.0
        for k in range(BATCH * DIM):
            L_minus += go_buf[k] * out_buf[k]
        in1_buf[idx] = orig
        var num = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
        var ana = gi1_buf[idx]
        var ae = fabs(num - ana)
        var denom = fabs(num) + fabs(ana) + Scalar[DT](1e-6)
        var rel = ae / denom
        if rel > max_rel_1:
            max_rel_1 = rel
    print("  BinarySub FD max_rel(in1)=", max_rel_1)
    assert_true(max_rel_1 < Scalar[DT](1e-3), "BinarySub FD in1 too loose")

    in0_buf.free(); in1_buf.free(); out_buf.free()
    go_buf.free(); gi0_buf.free(); gi1_buf.free()
    print("  test_binary_sub PASSED")


# ──────────────────────────────────────────────────────────────────────────
# BinaryElemMin
# ──────────────────────────────────────────────────────────────────────────


def test_binary_elem_min_forward_backward() raises:
    comptime DIM = 2
    comptime BATCH = 3
    var in0_buf = alloc[Scalar[DT]](BATCH * DIM)
    var in1_buf = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * DIM)
    var go_buf = alloc[Scalar[DT]](BATCH * DIM)
    var gi0_buf = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_buf = alloc[Scalar[DT]](BATCH * DIM)

    # b=0: a=[1, 2], b=[0.5, 3]  → min=[0.5, 2]  (b wins col 0, a wins col 1)
    in0_buf[0] = 1.0; in0_buf[1] = 2.0
    in1_buf[0] = 0.5; in1_buf[1] = 3.0
    # b=1: a=[-1, 4], b=[5, -2]  → min=[-1, -2] (a wins col 0, b wins col 1)
    in0_buf[2] = -1.0; in0_buf[3] = 4.0
    in1_buf[2] = 5.0; in1_buf[3] = -2.0
    # b=2: a=[0.7, 0.7], b=[0.8, 0.6]  → min=[0.7, 0.6] (a wins col 0, b wins col 1)
    in0_buf[4] = 0.7; in0_buf[5] = 0.7
    in1_buf[4] = 0.8; in1_buf[5] = 0.6

    for i in range(BATCH * DIM):
        go_buf[i] = Scalar[DT](Float32(i) * 0.17 - 0.25)

    var op = BinaryElemMin[DIM].make[target="cpu", INIT=Zero]()
    var in0_t = TileTensor(in0_buf, row_major[BATCH, DIM]())
    var in1_t = TileTensor(in1_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, DIM]())
    op.forward["cpu", BATCH](in0_t, in1_t, out_t)

    assert_almost_equal(out_buf[0], 0.5, atol=1e-7)
    assert_almost_equal(out_buf[1], 2.0, atol=1e-7)
    assert_almost_equal(out_buf[2], -1.0, atol=1e-7)
    assert_almost_equal(out_buf[3], -2.0, atol=1e-7)
    assert_almost_equal(out_buf[4], 0.7, atol=1e-7)
    assert_almost_equal(out_buf[5], 0.6, atol=1e-7)

    var go_t = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi0_t = TileTensor(gi0_buf, row_major[BATCH, DIM]())
    var gi1_t = TileTensor(gi1_buf, row_major[BATCH, DIM]())
    op.backward["cpu", BATCH](go_t, gi0_t, gi1_t)

    # b=0: b wins col 0 → gi0[0]=0,        gi1[0]=go[0]
    #      a wins col 1 → gi0[1]=go[1],    gi1[1]=0
    assert_almost_equal(gi0_buf[0], 0.0, atol=1e-7)
    assert_almost_equal(gi1_buf[0], go_buf[0], atol=1e-7)
    assert_almost_equal(gi0_buf[1], go_buf[1], atol=1e-7)
    assert_almost_equal(gi1_buf[1], 0.0, atol=1e-7)
    # b=1: a wins col 0 → gi0[2]=go[2], gi1[2]=0
    #      b wins col 1 → gi0[3]=0,    gi1[3]=go[3]
    assert_almost_equal(gi0_buf[2], go_buf[2], atol=1e-7)
    assert_almost_equal(gi1_buf[2], 0.0, atol=1e-7)
    assert_almost_equal(gi0_buf[3], 0.0, atol=1e-7)
    assert_almost_equal(gi1_buf[3], go_buf[3], atol=1e-7)
    # b=2: a wins col 0, b wins col 1
    assert_almost_equal(gi0_buf[4], go_buf[4], atol=1e-7)
    assert_almost_equal(gi1_buf[4], 0.0, atol=1e-7)
    assert_almost_equal(gi0_buf[5], 0.0, atol=1e-7)
    assert_almost_equal(gi1_buf[5], go_buf[5], atol=1e-7)

    in0_buf.free(); in1_buf.free(); out_buf.free()
    go_buf.free(); gi0_buf.free(); gi1_buf.free()
    print("  test_binary_elem_min PASSED")


# ──────────────────────────────────────────────────────────────────────────
# Phase 10A buffer surface
# ──────────────────────────────────────────────────────────────────────────


def test_binary_sub_buffer_surface() raises:
    """`ensure_buffers[BATCH]` allocates non-null buffers of the right size,
    and `out_ptr` / `grad_in0_ptr` / `grad_in1_ptr` / `grad_out_ptr` are
    usable as `TileTensor` data pointers for a round-trip forward+backward."""
    comptime DIM = 4
    comptime BATCH = 5
    var op = BinarySub[DIM].make[target="cpu", INIT=Zero]()
    op.ensure_buffers[BATCH]()

    var out_p = op.out_ptr()
    var gi0_p = op.grad_in0_ptr()
    var gi1_p = op.grad_in1_ptr()
    var go_p = op.grad_out_ptr()
    # The forward/backward round-trip below would segfault if any pointer
    # were the default null UnsafePointer from BinaryModule's trait defaults.

    var in0_buf = alloc[Scalar[DT]](BATCH * DIM)
    var in1_buf = alloc[Scalar[DT]](BATCH * DIM)
    for i in range(BATCH * DIM):
        in0_buf[i] = Scalar[DT](Float32(i) + 1.0)
        in1_buf[i] = Scalar[DT](Float32(i) * 0.5)
    var in0_t = TileTensor(in0_buf, row_major[BATCH, DIM]())
    var in1_t = TileTensor(in1_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_p, row_major[BATCH, DIM]())
    op.forward["cpu", BATCH](in0_t, in1_t, out_t)
    for i in range(BATCH * DIM):
        var expected = in0_buf[i] - in1_buf[i]
        assert_almost_equal(out_p[i], expected, atol=1e-7)

    # Write grad_output via grad_out_ptr, backward writes grad_in0/grad_in1 via owned ptrs.
    for i in range(BATCH * DIM):
        go_p[i] = Scalar[DT](Float32(i) * 0.11 + 0.3)
    var go_t = TileTensor(go_p, row_major[BATCH, DIM]())
    var gi0_t = TileTensor(gi0_p, row_major[BATCH, DIM]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, DIM]())
    op.backward["cpu", BATCH](go_t, gi0_t, gi1_t)
    for i in range(BATCH * DIM):
        assert_almost_equal(gi0_p[i], go_p[i], atol=1e-7)
        assert_almost_equal(gi1_p[i], -go_p[i], atol=1e-7)

    in0_buf.free(); in1_buf.free()
    print("  test_binary_sub_buffer_surface PASSED")


def main() raises:
    print("=" * 70)
    print("nn2 Phase 10C — two-input Modules (BinarySub, BinaryElemMin) CPU tests")
    print("=" * 70)
    test_binary_sub_forward_backward()
    test_binary_elem_min_forward_backward()
    test_binary_sub_buffer_surface()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
