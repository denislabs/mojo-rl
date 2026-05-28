"""ARITY=3 instantiations of the merged Concat + Add primitives (Block D-7)."""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.concat import Concat
from mojo_rl.nn2.primitives.add import Add
from mojo_rl.nn2.initializer import Kaiming


def test_concat_forward_backward() raises:
    comptime BATCH = 2
    comptime D0 = 2
    comptime D1 = 3
    comptime D2 = 1
    comptime OUT = D0 + D1 + D2
    var c = Concat[D0, D1, D2].make[target="cpu", INIT=Kaiming]()

    var i0_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D0)
    var i1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D1)
    var i2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D2)
    var o_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * D0):
        i0_p[k] = Scalar[DT](100.0 + k)
    for k in range(BATCH * D1):
        i1_p[k] = Scalar[DT](200.0 + k)
    for k in range(BATCH * D2):
        i2_p[k] = Scalar[DT](300.0 + k)

    # Hetero-ternary variadic workaround: every variadic element shares
    # the same comptime Layout (row_major[BATCH, D0]). The leaf body
    # recovers per-input shape via typed_view[BATCH, IN<i>_DIM]; the
    # Layout carried by a variadic TileTensor is dead metadata after
    # leaf unpack. See feedback_mojo_variadic_hetero_shape_workaround.
    var i0_t = TileTensor(i0_p, row_major[BATCH, D0]())
    var i1_t = TileTensor(i1_p, row_major[BATCH, D0]())
    var i2_t = TileTensor(i2_p, row_major[BATCH, D0]())
    var o_t  = TileTensor(o_p,  row_major[BATCH, OUT]())
    c.forward["cpu", BATCH](i0_t, i1_t, i2_t, output=o_t)

    for b in range(BATCH):
        for d in range(D0):
            assert_true(o_p[b * OUT + d] == i0_p[b * D0 + d])
        for d in range(D1):
            assert_true(o_p[b * OUT + D0 + d] == i1_p[b * D1 + d])
        for d in range(D2):
            assert_true(o_p[b * OUT + D0 + D1 + d] == i2_p[b * D2 + d])

    # Backward: distinct values per output slot; verify split.
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi0_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D0)
    var gi1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D1)
    var gi2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D2)
    for k in range(BATCH * OUT):
        go_p[k] = Scalar[DT](k + 1)
    # Hetero-ternary variadic workaround: see forward block above.
    var go_t = TileTensor(go_p, row_major[BATCH, OUT]())
    var gi0_t = TileTensor(gi0_p, row_major[BATCH, D0]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, D0]())
    var gi2_t = TileTensor(gi2_p, row_major[BATCH, D0]())
    c.vjp["cpu", BATCH](go_t, gi0_t, gi1_t, gi2_t)

    for b in range(BATCH):
        for d in range(D0):
            assert_true(gi0_p[b * D0 + d] == go_p[b * OUT + d])
        for d in range(D1):
            assert_true(gi1_p[b * D1 + d] == go_p[b * OUT + D0 + d])
        for d in range(D2):
            assert_true(gi2_p[b * D2 + d] == go_p[b * OUT + D0 + D1 + d])

    i0_p.free()
    i1_p.free()
    i2_p.free()
    o_p.free()
    go_p.free()
    gi0_p.free()
    gi1_p.free()
    gi2_p.free()
    print("  test_concat_forward_backward PASSED")


def test_fused_add_forward_backward() raises:
    comptime BATCH = 3
    comptime DIM = 5
    var a = Add[DIM, 3].make[target="cpu", INIT=Kaiming]()
    var i0_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var i1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var i2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var o_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        i0_p[k] = Scalar[DT](0.1 * Float64(k))
        i1_p[k] = Scalar[DT](-0.05 * Float64(k))
        i2_p[k] = Scalar[DT](0.2)
    var i0_t = TileTensor(i0_p, row_major[BATCH, DIM]())
    var i1_t = TileTensor(i1_p, row_major[BATCH, DIM]())
    var i2_t = TileTensor(i2_p, row_major[BATCH, DIM]())
    var o_t  = TileTensor(o_p,  row_major[BATCH, DIM]())
    a.forward["cpu", BATCH](i0_t, i1_t, i2_t, output=o_t)
    for k in range(BATCH * DIM):
        var expected = i0_p[k] + i1_p[k] + i2_p[k]
        assert_true(fabs(o_p[k] - expected) < 1e-6, "fused add mismatch")

    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi0_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_p[k] = Scalar[DT](0.7 - 0.03 * Float64(k))
    var go_t  = TileTensor(go_p,  row_major[BATCH, DIM]())
    var gi0_t = TileTensor(gi0_p, row_major[BATCH, DIM]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, DIM]())
    var gi2_t = TileTensor(gi2_p, row_major[BATCH, DIM]())
    a.vjp["cpu", BATCH](go_t, gi0_t, gi1_t, gi2_t)

    for k in range(BATCH * DIM):
        assert_true(fabs(gi0_p[k] - go_p[k]) < 1e-6)
        assert_true(fabs(gi1_p[k] - go_p[k]) < 1e-6)
        assert_true(fabs(gi2_p[k] - go_p[k]) < 1e-6)

    i0_p.free()
    i1_p.free()
    i2_p.free()
    o_p.free()
    go_p.free()
    gi0_p.free()
    gi1_p.free()
    gi2_p.free()
    print("  test_fused_add_forward_backward PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 ARITY=3 Concat + Add tests (Block D-7)")
    print("=" * 60)
    test_concat_forward_backward()
    test_fused_add_forward_backward()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
