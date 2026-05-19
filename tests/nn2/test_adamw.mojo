"""AdamW optimizer CPU tests — Phase 4.

Covers:
  - One-step update with hand-computed expected weight + bias.
    Critically: weight gets λ-decay applied; bias does NOT.
  - Init walks param tree, populates offsets + apply_decay table per
    PyTorch convention (weight=True, bias=False — Linear.for_each_param).
  - Convergence on a trivial overfit task.
"""

from std.math import abs as fabs, sqrt as fsqrt
from std.memory import alloc
from std.testing import assert_equal, assert_almost_equal, assert_true
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.optimizer import AdamW


def test_one_step_smoke() raises:
    """One AdamW step on Linear[1, 1] with known grads. Hand-compute the
    expected weight + bias after one step.

    Setup:
      weight = 1.0, bias = 2.0
      grad_w = 0.5, grad_b = 0.1
      lr=0.1, β₁=0.9, β₂=0.999, eps=1e-8, weight_decay=0.5

    After step 1 (weight, apply_decay=True):
      m = 0.1 * 0.5 = 0.05
      v = 0.001 * 0.25 = 0.00025
      bc1 = 0.1, bc2 = 0.001
      m_hat = 0.5, v_hat = 0.25, sqrt(v_hat) = 0.5
      update_no_decay = lr * m_hat / (sqrt(v_hat) + eps) = 0.1 * 0.5/0.5 ≈ 0.1
      update_with_decay = 0.1 + lr * λ * w = 0.1 + 0.1 * 0.5 * 1.0 = 0.15
      new_weight = 1.0 - 0.15 ≈ 0.85

    After step 1 (bias, apply_decay=False):
      m = 0.1 * 0.1 = 0.01
      v = 0.001 * 0.01 = 1e-5
      m_hat = 0.1, v_hat = 0.01, sqrt(v_hat) = 0.1
      update = lr * 0.1 / 0.1 ≈ 0.1
      new_bias = 2.0 - 0.1 ≈ 1.9     (no decay)
    """
    var lin = Linear[1, 1].make["cpu", INIT=Zero]()
    var w = TileTensor(lin.weight, row_major[1, 1]())
    w[0, 0] = 1.0
    var b = TileTensor(lin.bias, row_major[1]())
    b[0] = 2.0
    var gw = TileTensor(lin.grad_w, row_major[1, 1]())
    gw[0, 0] = 0.5
    var gb = TileTensor(lin.grad_b, row_major[1]())
    gb[0] = 0.1

    var adam = AdamW.make_with_wd["cpu"](
        lin, lr=0.1, weight_decay=0.5,
    )
    adam.step["cpu"](lin)

    var w_after = TileTensor(lin.weight, row_major[1, 1]())
    var b_after = TileTensor(lin.bias, row_major[1]())
    assert_almost_equal(w_after[0, 0], Scalar[DT](0.85), atol=1e-5)
    assert_almost_equal(b_after[0],    Scalar[DT](1.9),  atol=1e-5)

    print("  test_one_step_smoke PASSED (w=" + String(w_after[0, 0])
          + ", b=" + String(b_after[0]) + ")")


def test_init_apply_decay_table() raises:
    """AdamW.make walks Linear and records apply_decay per param.
    Linear[3, 5] → 2 params: weight (apply_decay=True), bias (=False)."""
    var lin = Linear[3, 5].make["cpu", INIT=Zero]()
    var adam = AdamW.make_with_wd["cpu"](lin, weight_decay=0.01)

    assert_equal(len(adam.offsets), 2)
    assert_equal(adam.offsets[0], 0)
    assert_equal(adam.offsets[1], 15)
    assert_equal(len(adam.apply_decay), 2)
    assert_true(adam.apply_decay[0],  "expected weight.apply_decay == True")
    assert_true(not adam.apply_decay[1], "expected bias.apply_decay == False")
    assert_equal(len(adam.m_flat), 20)
    assert_equal(len(adam.v_flat), 20)
    print("  test_init_apply_decay_table PASSED")


def test_no_decay_when_bias_only() raises:
    """If we only feed a bias-style grad through AdamW (apply_decay=False),
    AdamW.step should match Adam's step to fp32 precision (decay is a no-op).

    Verifies the apply_decay flag actually gates the decay term."""
    var lin = Linear[1, 1].make["cpu", INIT=Zero]()
    var b = TileTensor(lin.bias, row_major[1]())
    b[0] = 3.0
    var gb = TileTensor(lin.grad_b, row_major[1]())
    gb[0] = 1.0

    # weight + grad_w stay at 0 → no contribution.
    var adam = AdamW.make_with_wd["cpu"](
        lin, lr=0.1, weight_decay=1.0,  # absurd decay; should be ignored on bias
    )
    adam.step["cpu"](lin)

    var b_after = TileTensor(lin.bias, row_major[1]())
    # Adam-only step (no decay): same hand-calc as Adam test.
    # m = 0.1, v = 0.001, m_hat = 1.0, v_hat = 1.0, sqrt = 1.0
    # update = 0.1 * 1.0 / 1.0 = 0.1; new_b = 3.0 - 0.1 = 2.9.
    assert_almost_equal(b_after[0], Scalar[DT](2.9), atol=1e-5)
    print("  test_no_decay_when_bias_only PASSED (b=" + String(b_after[0]) + ")")


def test_convergence_overfit() raises:
    """AdamW can still drive an overfit task to near-zero loss with a
    small λ. Sanity check that decay doesn't break convergence."""
    comptime IN = 2
    comptime OUT = 1
    comptime BATCH = 1
    comptime TARGET: Scalar[DT] = 5.0
    comptime N_STEPS = 300

    var lin = Linear[IN, OUT].make["cpu", INIT=Zero]()
    var w = TileTensor(lin.weight, row_major[IN, OUT]())
    w[0, 0] =  0.1
    w[1, 0] = -0.2

    var adam = AdamW.make_with_wd["cpu"](
        lin, lr=0.05, weight_decay=1e-3,
    )

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    in_buf[0] = 1.0
    in_buf[1] = 2.0
    var input  = TileTensor(in_buf,  row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())
    var grad_out = TileTensor(go_buf, row_major[BATCH, OUT]())
    var grad_in  = TileTensor(gi_buf, row_major[BATCH, IN]())

    var final_loss: Scalar[DT] = 0.0
    for step_i in range(N_STEPS):
        lin.zero_grad["cpu"]()
        lin.forward["cpu", BATCH](input, output)
        var y = output[0, 0]
        var err = y - TARGET
        final_loss = 0.5 * err * err
        grad_out[0, 0] = err
        lin.backward["cpu", BATCH](grad_out, grad_in)
        adam.step["cpu"](lin)

    assert_true(final_loss < Scalar[DT](1e-2),
        "Expected loss < 1e-2 after 300 steps, got " + String(final_loss))
    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_convergence_overfit PASSED (final loss=" + String(final_loss) + ")")


def main() raises:
    print("=" * 60)
    print("nn2 AdamW unit tests (CPU, Phase 4)")
    print("=" * 60)
    test_one_step_smoke()
    test_init_apply_decay_table()
    test_no_decay_when_bias_only()
    test_convergence_overfit()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
