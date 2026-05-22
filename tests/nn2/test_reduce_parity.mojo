"""Parity + correctness test: `Reduce[DIM, OP]` Sum/Mean leaves.

Phase 2.5.C validation. The pre-Phase-2.5 `Sum` / `Mean` structs are
gone — both are now one-line aliases for `Reduce[DIM, SumOp]` /
`Reduce[DIM, MeanOp]`. This test pins down the math directly:

  1. Sum forward — sums input row, no scaling.
  2. Sum backward — broadcasts grad_out unchanged.
  3. Mean forward — `(1/DIM)·Σ x`.
  4. Mean backward — broadcasts `grad_out / DIM`.
  5. Sequential composition still picks `Sum` up as a Module.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.reduce import Sum, Mean, Reduce
from mojo_rl.nn2.primitives.ops.sum_op import SumOp
from mojo_rl.nn2.primitives.ops.mean_op import MeanOp
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Zero


def test_sum_forward() raises:
    print("test_sum_forward ...")
    comptime BATCH = 2
    comptime DIM = 4
    var op = Sum[DIM].make[target="cpu", INIT=Zero]()
    var x = alloc[Scalar[DT]](BATCH * DIM)
    var y = alloc[Scalar[DT]](BATCH * 1)
    # Row 0: 1+2+3+4 = 10; Row 1: 5+6+7+8 = 26.
    for i in range(BATCH * DIM):
        x[i] = Scalar[DT](Float64(i + 1))
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, 1]())
    op.forward["cpu", BATCH](x_t, y_t)
    assert_true(y[0] == Scalar[DT](10.0), "Sum row 0")
    assert_true(y[1] == Scalar[DT](26.0), "Sum row 1")
    print("  ok")


def test_sum_backward() raises:
    print("test_sum_backward ...")
    comptime BATCH = 2
    comptime DIM = 4
    var op = Sum[DIM].make[target="cpu", INIT=Zero]()
    var go = alloc[Scalar[DT]](BATCH * 1)
    var gi = alloc[Scalar[DT]](BATCH * DIM)
    go[0] = Scalar[DT](3.0)
    go[1] = Scalar[DT](-2.0)
    var go_t = TileTensor(go, row_major[BATCH, 1]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())
    op.backward["cpu", BATCH](go_t, gi_t)
    for d in range(DIM):
        assert_true(gi[0 * DIM + d] == Scalar[DT](3.0), "Sum.bwd row 0")
        assert_true(gi[1 * DIM + d] == Scalar[DT](-2.0), "Sum.bwd row 1")
    print("  ok")


def test_mean_forward() raises:
    print("test_mean_forward ...")
    comptime BATCH = 2
    comptime DIM = 4
    var op = Mean[DIM].make[target="cpu", INIT=Zero]()
    var x = alloc[Scalar[DT]](BATCH * DIM)
    var y = alloc[Scalar[DT]](BATCH * 1)
    for i in range(BATCH * DIM):
        x[i] = Scalar[DT](Float64(i + 1))
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, 1]())
    op.forward["cpu", BATCH](x_t, y_t)
    # Row 0: 10/4 = 2.5; Row 1: 26/4 = 6.5.
    assert_true(y[0] == Scalar[DT](2.5), "Mean row 0")
    assert_true(y[1] == Scalar[DT](6.5), "Mean row 1")
    print("  ok")


def test_mean_backward() raises:
    print("test_mean_backward ...")
    comptime BATCH = 2
    comptime DIM = 4
    var op = Mean[DIM].make[target="cpu", INIT=Zero]()
    var go = alloc[Scalar[DT]](BATCH * 1)
    var gi = alloc[Scalar[DT]](BATCH * DIM)
    go[0] = Scalar[DT](4.0)
    go[1] = Scalar[DT](-8.0)
    var go_t = TileTensor(go, row_major[BATCH, 1]())
    var gi_t = TileTensor(gi, row_major[BATCH, DIM]())
    op.backward["cpu", BATCH](go_t, gi_t)
    # 4 / 4 = 1, -8 / 4 = -2.
    for d in range(DIM):
        assert_true(gi[0 * DIM + d] == Scalar[DT](1.0), "Mean.bwd row 0")
        assert_true(gi[1 * DIM + d] == Scalar[DT](-2.0), "Mean.bwd row 1")
    print("  ok")


def test_sequential_with_sum() raises:
    """Confirm Sum still composes as a Module inside Sequential."""
    print("test_sequential_with_sum ...")
    comptime BATCH = 2
    comptime IN = 3
    comptime HID = 4
    comptime MLP = Sequential[Linear[IN, HID], Sum[HID]]
    var net = MLP.make[target="cpu", INIT=Zero]()
    var x = alloc[Scalar[DT]](BATCH * IN)
    var y = alloc[Scalar[DT]](BATCH * 1)
    for i in range(BATCH * IN):
        x[i] = Scalar[DT](0.1 * Float64(i + 1))
    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var y_t = TileTensor(y, row_major[BATCH, 1]())
    net.forward["cpu", BATCH](x_t, y_t)
    # Linear[IN, HID] with Zero init outputs 0 (bias=0, weight=0), so
    # Sum produces 0. We're checking the call path, not the math.
    assert_true(y[0] == Scalar[DT](0.0), "Sequential Linear->Sum yields 0 with Zero init")
    assert_true(y[1] == Scalar[DT](0.0), "Sequential Linear->Sum yields 0 with Zero init")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Reduce[DIM, OP] Sum/Mean (Phase 2.5.C)")
    print("=" * 70)
    test_sum_forward()
    test_sum_backward()
    test_mean_forward()
    test_mean_backward()
    test_sequential_with_sum()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
