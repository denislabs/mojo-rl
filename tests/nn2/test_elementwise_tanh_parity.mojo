"""Parity test: Elementwise[DIM, TanhOp] vs Tanh[DIM].

Phase 1.3 validation. The hand-written `Tanh[DIM]` is the regression
oracle. `Elementwise[DIM, TanhOp]` should produce bit-identical forward
output and bit-identical grad_input for the same inputs.

Two sub-tests:

  1. **Forward parity**: Same input ⇒ same output buffer (every element
     bit-identical).
  2. **Backward parity**: Same forward input + same grad_output ⇒ same
     grad_input buffer (every element bit-identical).

Mojo nightly's `tanh` is shared between both paths so bit-identity is
the right bar — anything else means the orchestration changed the math.
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.tanh_op import TanhOp
from mojo_rl.nn2.initializer import Zero


def test_forward_parity() raises:
    print("test_forward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_tanh = Tanh[DIM].make[target="cpu", INIT=Zero]()
    var new_tanh = Elementwise[DIM, TanhOp].make[target="cpu", INIT=Zero]()

    var x = alloc[Scalar[DT]](N)
    var y_old = alloc[Scalar[DT]](N)
    var y_new = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-2.0 + 0.1 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_old_t = TileTensor(y_old, row_major[BATCH, DIM]())
    var y_new_t = TileTensor(y_new, row_major[BATCH, DIM]())
    old_tanh.forward["cpu", BATCH](x_t, y_old_t)
    new_tanh.forward["cpu", BATCH](x_t, y_new_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = y_old[i] - y_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y_old - y_new| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[TanhOp] forward should be bit-identical to Tanh"
    )
    print("  ok")


def test_backward_parity() raises:
    print("test_backward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_tanh = Tanh[DIM].make[target="cpu", INIT=Zero]()
    var new_tanh = Elementwise[DIM, TanhOp].make[target="cpu", INIT=Zero]()

    var x = alloc[Scalar[DT]](N)
    var y_old = alloc[Scalar[DT]](N)
    var y_new = alloc[Scalar[DT]](N)
    var go = alloc[Scalar[DT]](N)
    var gi_old = alloc[Scalar[DT]](N)
    var gi_new = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-2.0 + 0.1 * Float64(i))
        go[i] = Scalar[DT](0.5 + 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_old_t = TileTensor(y_old, row_major[BATCH, DIM]())
    var y_new_t = TileTensor(y_new, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_old_t = TileTensor(gi_old, row_major[BATCH, DIM]())
    var gi_new_t = TileTensor(gi_new, row_major[BATCH, DIM]())

    # Forward both (cache y).
    old_tanh.forward["cpu", BATCH](x_t, y_old_t)
    new_tanh.forward["cpu", BATCH](x_t, y_new_t)

    # Backward both with same grad_output.
    old_tanh.backward["cpu", BATCH](go_t, gi_old_t)
    new_tanh.backward["cpu", BATCH](go_t, gi_new_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = gi_old[i] - gi_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |gi_old - gi_new| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[TanhOp] backward should be bit-identical to Tanh"
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Elementwise[TanhOp] vs Tanh parity (Phase 1.3)")
    print("=" * 70)
    test_forward_parity()
    test_backward_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
