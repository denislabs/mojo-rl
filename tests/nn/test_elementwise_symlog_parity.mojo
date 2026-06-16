"""Parity test: Elementwise[DIM, SymlogOp] vs Symlog[DIM].

Phase 2 Track A migration gate. The hand-written `Symlog[DIM]` is the
regression oracle until the migration completes; `Elementwise[DIM,
SymlogOp]` should produce bit-identical forward output and bit-identical
grad_input for the same inputs (both paths share `std.math.log` /
`std.math.abs` SIMD specialisations).

Two sub-tests:
  1. Forward parity — same input ⇒ same output buffer.
  2. Backward parity — same forward input + same grad_output ⇒ same
     grad_input buffer (input-alias backward).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.symlog import Symlog
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.symlog_op import SymlogOp
from mojo_rl.nn.initializer import Zero


def test_forward_parity() raises:
    print("test_forward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_sym = Symlog[DIM].make[target="cpu", INIT=Zero]()
    var new_sym = Elementwise[DIM, SymlogOp].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    # Span large negatives → large positives to exercise both branches
    # plus the magnitude-compression path.
    for i in range(N):
        x[i] = Scalar[DT](-50.0 + 3.3 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_old_t = TileTensor(y_old, row_major[BATCH, DIM]())
    var y_new_t = TileTensor(y_new, row_major[BATCH, DIM]())
    old_sym.forward["cpu", BATCH](x_t, output=y_old_t)
    new_sym.forward["cpu", BATCH](x_t, output=y_new_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = y_old[i] - y_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y_old - y_new| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[SymlogOp] forward should be bit-identical to Symlog"
    )
    print("  ok")


def test_backward_parity() raises:
    print("test_backward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_sym = Symlog[DIM].make[target="cpu", INIT=Zero]()
    var new_sym = Elementwise[DIM, SymlogOp].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_old: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi_new: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-50.0 + 3.3 * Float64(i))
        go[i] = Scalar[DT](0.5 + 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_old_t = TileTensor(y_old, row_major[BATCH, DIM]())
    var y_new_t = TileTensor(y_new, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_old_t = TileTensor(gi_old, row_major[BATCH, DIM]())
    var gi_new_t = TileTensor(gi_new, row_major[BATCH, DIM]())

    old_sym.forward["cpu", BATCH](x_t, output=y_old_t)
    new_sym.forward["cpu", BATCH](x_t, output=y_new_t)

    old_sym.vjp["cpu", BATCH](go_t, gi_old_t)
    new_sym.vjp["cpu", BATCH](go_t, gi_new_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = gi_old[i] - gi_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |gi_old - gi_new| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[SymlogOp] backward should be bit-identical to Symlog"
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Elementwise[SymlogOp] vs Symlog parity (Phase 2 Track A #3)")
    print("=" * 70)
    test_forward_parity()
    test_backward_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
