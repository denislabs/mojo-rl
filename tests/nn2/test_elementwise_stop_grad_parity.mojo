"""Parity test: Elementwise[DIM, StopGradOp] vs StopGrad[DIM].

Phase 2 Track A migration gate. Forward = identity, backward = zero.
Both paths must agree bit-identically.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.stop_grad import StopGrad
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.stop_grad_op import StopGradOp
from mojo_rl.nn2.initializer import Zero


def test_forward_parity() raises:
    print("test_forward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_sg = StopGrad[DIM].make[target="cpu", INIT=Zero]()
    var new_sg = Elementwise[DIM, StopGradOp].make[target="cpu", INIT=Zero]()

    var x = alloc[Scalar[DT]](N)
    var y_old = alloc[Scalar[DT]](N)
    var y_new = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-1.5 + 0.17 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_old_t = TileTensor(y_old, row_major[BATCH, DIM]())
    var y_new_t = TileTensor(y_new, row_major[BATCH, DIM]())
    old_sg.forward["cpu", BATCH](x_t, y_old_t)
    new_sg.forward["cpu", BATCH](x_t, y_new_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = y_old[i] - y_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |y_old - y_new| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[StopGradOp] forward should be bit-identical to StopGrad"
    )
    print("  ok")


def test_backward_parity() raises:
    print("test_backward_parity ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM
    var old_sg = StopGrad[DIM].make[target="cpu", INIT=Zero]()
    var new_sg = Elementwise[DIM, StopGradOp].make[target="cpu", INIT=Zero]()

    var x = alloc[Scalar[DT]](N)
    var y_buf = alloc[Scalar[DT]](N)
    var go = alloc[Scalar[DT]](N)
    var gi_old = alloc[Scalar[DT]](N)
    var gi_new = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT](-1.5 + 0.17 * Float64(i))
        go[i] = Scalar[DT](0.5 + 0.05 * Float64(i))
        # Pre-fill grad_input with garbage to confirm it gets overwritten with 0.
        gi_old[i] = Scalar[DT](42.0)
        gi_new[i] = Scalar[DT](-7.0)

    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y_buf, row_major[BATCH, DIM]())
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi_old_t = TileTensor(gi_old, row_major[BATCH, DIM]())
    var gi_new_t = TileTensor(gi_new, row_major[BATCH, DIM]())

    old_sg.forward["cpu", BATCH](x_t, y_t)
    new_sg.forward["cpu", BATCH](x_t, y_t)

    old_sg.vjp["cpu", BATCH](go_t, gi_old_t)
    new_sg.vjp["cpu", BATCH](go_t, gi_new_t)

    var max_diff: Scalar[DT] = 0.0
    var max_abs_old: Scalar[DT] = 0.0
    var max_abs_new: Scalar[DT] = 0.0
    for i in range(N):
        var d = gi_old[i] - gi_new[i]
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
        var a_old = gi_old[i] if gi_old[i] >= Scalar[DT](0) else -gi_old[i]
        var a_new = gi_new[i] if gi_new[i] >= Scalar[DT](0) else -gi_new[i]
        if a_old > max_abs_old:
            max_abs_old = a_old
        if a_new > max_abs_new:
            max_abs_new = a_new
    print("  max |gi_old - gi_new| =", max_diff)
    print("  max |gi_old| =", max_abs_old, "  max |gi_new| =", max_abs_new)
    assert_true(
        max_diff == Scalar[DT](0),
        "Elementwise[StopGradOp] backward should be bit-identical to StopGrad"
    )
    assert_true(
        max_abs_new == Scalar[DT](0),
        "Elementwise[StopGradOp] backward should zero grad_input"
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Elementwise[StopGradOp] vs StopGrad parity (Phase 2 Track A #4)")
    print("=" * 70)
    test_forward_parity()
    test_backward_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
