"""Parity test: BinaryElementwise[DIM, OP] vs standalone Binary{Add,Sub,ElemMin}.

Phase 4.5 migration gate. The hand-written binary leaves are the
regression oracle until the alias-swap completes; `BinaryElementwise[DIM,
OP]` should produce bit-identical output + bit-identical (gi0, gi1) for
identical inputs (both paths run the same scalar/SIMD arithmetic).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.nn2.primitives.binary_elem_min import BinaryElemMin
from mojo_rl.nn2.primitives.binary_elementwise import BinaryElementwise
from mojo_rl.nn2.primitives.ops.binary_sub_op import BinarySubOp
from mojo_rl.nn2.primitives.ops.binary_elem_min_op import BinaryElemMinOp
from mojo_rl.nn2.initializer import Zero


def _max_diff(a: UnsafePointer[Scalar[DT], MutAnyOrigin],
              b: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Scalar[DT]:
    var max_d: Scalar[DT] = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_d:
            max_d = ad
    return max_d


def test_binary_sub_parity() raises:
    print("test_binary_sub_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_op = BinarySub[DIM].make[target="cpu", INIT=Zero]()
    var new_op = BinaryElementwise[DIM, BinarySubOp].make[
        target="cpu", INIT=Zero
    ]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var z_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var z_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gx_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gx_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gy_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gy_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-2.0 + 0.13 * Float64(i))
        y[i] = Scalar[DT](0.5 + 0.07 * Float64(i))
        go[i] = Scalar[DT](0.1 + 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var z_old_t = TileTensor(z_old, row_major[BATCH, DIM]())
    var z_new_t = TileTensor(z_new, row_major[BATCH, DIM]())
    var gx_old_t = TileTensor(gx_old, row_major[BATCH, DIM]())
    var gx_new_t = TileTensor(gx_new, row_major[BATCH, DIM]())
    var gy_old_t = TileTensor(gy_old, row_major[BATCH, DIM]())
    var gy_new_t = TileTensor(gy_new, row_major[BATCH, DIM]())

    old_op.forward["cpu", BATCH](x_t, y_t, output=z_old_t)
    new_op.forward["cpu", BATCH](x_t, y_t, output=z_new_t)
    old_op.vjp["cpu", BATCH](go_t, gx_old_t, gy_old_t)
    new_op.vjp["cpu", BATCH](go_t, gx_new_t, gy_new_t)

    var df = _max_diff(z_old, z_new, N)
    var dgx = _max_diff(gx_old, gx_new, N)
    var dgy = _max_diff(gy_old, gy_new, N)
    print("  forward max diff =", df,
          " gx max diff =", dgx, " gy max diff =", dgy)
    assert_true(df == Scalar[DT](0), "BinarySub forward mismatch")
    assert_true(dgx == Scalar[DT](0), "BinarySub gx mismatch")
    assert_true(dgy == Scalar[DT](0), "BinarySub gy mismatch")
    print("  ok")


def test_binary_elem_min_parity() raises:
    print("test_binary_elem_min_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_op = BinaryElemMin[DIM].make[target="cpu", INIT=Zero]()
    var new_op = BinaryElementwise[DIM, BinaryElemMinOp].make[
        target="cpu", INIT=Zero
    ]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var z_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var z_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gx_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gx_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gy_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gy_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Interleave so the per-lane choice alternates between branches.
    for i in range(N):
        x[i] = Scalar[DT](-2.0 + 0.13 * Float64(i))
        y[i] = Scalar[DT](-1.5 + 0.09 * Float64(i))   # crosses x near i ~ 12
        go[i] = Scalar[DT](0.1 + 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var z_old_t = TileTensor(z_old, row_major[BATCH, DIM]())
    var z_new_t = TileTensor(z_new, row_major[BATCH, DIM]())
    var gx_old_t = TileTensor(gx_old, row_major[BATCH, DIM]())
    var gx_new_t = TileTensor(gx_new, row_major[BATCH, DIM]())
    var gy_old_t = TileTensor(gy_old, row_major[BATCH, DIM]())
    var gy_new_t = TileTensor(gy_new, row_major[BATCH, DIM]())

    old_op.forward["cpu", BATCH](x_t, y_t, output=z_old_t)
    new_op.forward["cpu", BATCH](x_t, y_t, output=z_new_t)
    old_op.vjp["cpu", BATCH](go_t, gx_old_t, gy_old_t)
    new_op.vjp["cpu", BATCH](go_t, gx_new_t, gy_new_t)

    var df = _max_diff(z_old, z_new, N)
    var dgx = _max_diff(gx_old, gx_new, N)
    var dgy = _max_diff(gy_old, gy_new, N)
    print("  forward max diff =", df,
          " gx max diff =", dgx, " gy max diff =", dgy)
    assert_true(df == Scalar[DT](0), "BinaryElemMin forward mismatch")
    assert_true(dgx == Scalar[DT](0), "BinaryElemMin gx mismatch")
    assert_true(dgy == Scalar[DT](0), "BinaryElemMin gy mismatch")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BinaryElementwise[OP] vs standalone parity (Phase 4.5)")
    print("=" * 70)
    test_binary_sub_parity()
    test_binary_elem_min_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
