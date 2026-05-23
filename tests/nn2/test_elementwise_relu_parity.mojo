"""Parity test: Elementwise[DIM, ReLUOp] vs ReLU[DIM].

Phase 2 Track A migration gate. The hand-written `ReLU[DIM]` is the
regression oracle until the migration completes; `Elementwise[DIM,
ReLUOp]` should produce bit-identical forward output and bit-identical
grad_input for the same inputs (both paths use the same scalar / SIMD
arithmetic).

Two sub-tests:
  1. Forward parity — same input ⇒ same output buffer.
  2. Backward parity — same forward input + same grad_output ⇒ same
     grad_input buffer (input-alias backward).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn2.initializer import Zero


def test_forward_parity() raises:
    print("test_forward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_relu = ReLU[DIM].make[target="cpu", INIT=Zero]()
    var new_relu = Elementwise[DIM, ReLUOp].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Span negatives and positives to hit both branches in every lane.
    for i in range(N):
        x[i] = Scalar[DT](-2.0 + 0.13 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_old_t = TileTensor(y_old, row_major[BATCH, DIM]())
    var y_new_t = TileTensor(y_new, row_major[BATCH, DIM]())
    old_relu.forward["cpu", BATCH](x_t, output=y_old_t)
    new_relu.forward["cpu", BATCH](x_t, output=y_new_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = y_old[i] - y_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y_old - y_new| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[ReLUOp] forward should be bit-identical to ReLU"
    )
    print("  ok")


def test_backward_parity() raises:
    print("test_backward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_relu = ReLU[DIM].make[target="cpu", INIT=Zero]()
    var new_relu = Elementwise[DIM, ReLUOp].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-2.0 + 0.13 * Float64(i))
        go[i] = Scalar[DT](0.5 + 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_old_t = TileTensor(y_old, row_major[BATCH, DIM]())
    var y_new_t = TileTensor(y_new, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_old_t = TileTensor(gi_old, row_major[BATCH, DIM]())
    var gi_new_t = TileTensor(gi_new, row_major[BATCH, DIM]())

    # Forward both (the input-alias path caches `x.ptr` for backward).
    old_relu.forward["cpu", BATCH](x_t, output=y_old_t)
    new_relu.forward["cpu", BATCH](x_t, output=y_new_t)

    old_relu.vjp["cpu", BATCH](go_t, gi_old_t)
    new_relu.vjp["cpu", BATCH](go_t, gi_new_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = gi_old[i] - gi_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |gi_old - gi_new| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[ReLUOp] backward should be bit-identical to ReLU"
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Elementwise[ReLUOp] vs ReLU parity (Phase 2 Track A #1)")
    print("=" * 70)
    test_forward_parity()
    test_backward_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
