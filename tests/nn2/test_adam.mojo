"""Adam optimizer CPU tests — Phase 1.

Covers:
  - One-step update with hand-computed expected value (smoke)
  - Init walks param tree and allocates m/v per-param
  - Convergence on a trivial overfitting task: a single Linear layer
    learns to map a known input to a known target via a few hundred
    Adam steps + MSE loss (no Loss class yet — handcoded MSE).
"""

from std.math import abs as fabs, sqrt as fsqrt
from std.memory import alloc
from std.testing import assert_equal, assert_almost_equal, assert_true
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.optimizer import Adam


def test_one_step_smoke() raises:
    """One Adam step on a Linear[1, 1] with known grad. Hand-compute the
    expected weight after one step.

    Setup:
      weight = 1.0, bias = 0.0, grad_w = 0.5, grad_b = 0.0
      lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8
    After step 1:
      m_w = 0.1 * 0.5 = 0.05
      v_w = 0.001 * 0.25 = 0.00025
      bc1 = 1 - 0.9 = 0.1; m_hat = 0.5
      bc2 = 1 - 0.999 = 0.001; v_hat = 0.25; sqrt(v_hat) = 0.5
      update = lr * m_hat / (sqrt(v_hat) + eps) = 0.1 * 0.5 / 0.5 ≈ 0.1
      new_weight ≈ 1.0 - 0.1 ≈ 0.9
    """
    var lin = Linear[1, 1].make["cpu", INIT=Zero]()
    var w = TileTensor(lin.weight, row_major[1, 1]())
    w[0, 0] = 1.0
    # bias is already 0
    var gw = TileTensor(lin.grad_w, row_major[1, 1]())
    gw[0, 0] = 0.5
    # grad_b stays 0

    var adam = Adam.make["cpu"](lin, lr=0.1)
    adam.step["cpu"](lin)

    var w_after = TileTensor(lin.weight, row_major[1, 1]())
    # Approximate equality (eps perturbs it slightly).
    assert_almost_equal(w_after[0, 0], Scalar[DT](0.9), atol=1e-6)

    # Bias had zero grad → should not move (or move only via eps quirks).
    var b_after = TileTensor(lin.bias, row_major[1]())
    assert_almost_equal(b_after[0], 0.0, atol=1e-8)

    print("  test_one_step_smoke PASSED")


def test_init_param_count() raises:
    """Adam.make populates flat m/v lists + offsets table by walking the
    model. Linear[3, 5] has 2 params (weight: 15, bias: 5) → offsets has
    2 entries, m_flat/v_flat have 20 zeros."""
    var lin = Linear[3, 5].make["cpu", INIT=Zero]()
    var adam = Adam.make["cpu"](lin)
    assert_equal(len(adam.offsets), 2)
    assert_equal(adam.offsets[0], 0)     # weight starts at offset 0
    assert_equal(adam.offsets[1], 15)    # bias starts after weight
    assert_equal(len(adam.m_flat), 20)
    assert_equal(len(adam.v_flat), 20)
    for i in range(20):
        assert_equal(adam.m_flat[i], 0.0)
        assert_equal(adam.v_flat[i], 0.0)
    print("  test_init_param_count PASSED")


def test_convergence_overfitting() raises:
    """Train a single Linear[2, 1] to map [[1, 2]] → 5.0 via MSE+Adam.

    MSE: L = 0.5 * (y - target)^2; ∂L/∂y = y - target.
    We hand-roll MSE backward (no Loss class yet).

    Expected: after ~200 steps Adam drives the loss to near-zero.
    """
    comptime IN = 2
    comptime OUT = 1
    comptime BATCH = 1
    comptime TARGET: Scalar[DT] = 5.0
    comptime N_STEPS = 300

    var lin = Linear[IN, OUT].make["cpu", INIT=Zero]()
    # Small random-ish init: w[0,0]=0.1, w[1,0]=-0.2, bias[0]=0.0
    var w = TileTensor(lin.weight, row_major[IN, OUT]())
    w[0, 0] =  0.1
    w[1, 0] = -0.2

    var adam = Adam.make["cpu"](lin, lr=0.05)

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
        grad_out[0, 0] = err   # ∂L/∂y
        lin.backward["cpu", BATCH](grad_out, grad_in)
        adam.step["cpu"](lin)

    # After 300 Adam steps with lr=0.05 the loss should be tiny.
    assert_true(final_loss < Scalar[DT](1e-3),
        "Expected loss < 1e-3 after 300 steps, got " + String(final_loss))

    # Verify forward at the trained weights does map close to TARGET.
    lin.forward["cpu", BATCH](input, output)
    var y_final = output[0, 0]
    assert_almost_equal(y_final, TARGET, atol=0.05)

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_convergence_overfitting PASSED (final loss=" + String(final_loss) + ")")


def main() raises:
    print("=" * 60)
    print("nn2 Adam unit tests (CPU, Phase 1)")
    print("=" * 60)
    test_one_step_smoke()
    test_init_param_count()
    test_convergence_overfitting()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
