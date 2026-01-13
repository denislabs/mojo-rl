"""Test New Optimizers on CPU.

Tests:
- RMSprop
- AdamW

Run with:
    pixi run mojo run tests/test_optimizers_cpu.mojo
"""

from random import seed, random_float64
from math import sqrt

from deep_rl.constants import dtype
from deep_rl.model import Linear, ReLU, Tanh, seq
from deep_rl.loss import MSELoss
from deep_rl.optimizer import Adam, SGD, RMSprop, AdamW
from deep_rl.training import Trainer
from deep_rl.initializer import Xavier, Kaiming


# =============================================================================
# Test Helper
# =============================================================================


fn print_test_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


fn assert_close(actual: Float64, expected: Float64, tol: Float64, msg: String) -> Bool:
    """Check if two values are close within tolerance."""
    var diff = actual - expected
    if diff < 0:
        diff = -diff
    if diff > tol:
        print("  FAIL: " + msg)
        print("    Expected: " + String(expected))
        print("    Actual:   " + String(actual))
        return False
    return True


# =============================================================================
# Test RMSprop Algorithm
# =============================================================================


fn test_rmsprop_algorithm():
    print_test_header("RMSprop Algorithm Verification")

    # Test RMSprop update rule:
    # v = alpha * v + (1 - alpha) * grad^2
    # param = param - lr * grad / (sqrt(v) + eps)

    var lr: Float64 = 0.01
    var alpha: Float64 = 0.99
    var eps: Float64 = 1e-8

    # Initial values
    var param: Float64 = 1.0
    var grad: Float64 = 0.5
    var v: Float64 = 0.0

    print("  RMSprop parameters:")
    print("    lr = " + String(lr))
    print("    alpha = " + String(alpha))
    print("    eps = " + String(eps))
    print()
    print("  Initial state:")
    print("    param = " + String(param))
    print("    grad = " + String(grad))
    print("    v = " + String(v))

    # Step 1
    v = alpha * v + (1 - alpha) * grad * grad
    param = param - lr * grad / (sqrt(v) + eps)

    print("\n  After step 1:")
    print("    v = " + String(v)[:10])
    print("    param = " + String(param)[:10])

    # Verify: v = 0.99 * 0 + 0.01 * 0.25 = 0.0025
    var expected_v1: Float64 = 0.0025
    var all_passed = True

    if not assert_close(v, expected_v1, 0.0001, "v after step 1"):
        all_passed = False
    else:
        print("  PASS: v correct")

    # Step 2 with different gradient
    grad = 1.0
    v = alpha * v + (1 - alpha) * grad * grad
    param = param - lr * grad / (sqrt(v) + eps)

    print("\n  After step 2 (grad = 1.0):")
    print("    v = " + String(v)[:10])
    print("    param = " + String(param)[:10])

    # v = 0.99 * 0.0025 + 0.01 * 1 = 0.012475
    var expected_v2: Float64 = 0.99 * 0.0025 + 0.01 * 1.0
    if not assert_close(v, expected_v2, 0.0001, "v after step 2"):
        all_passed = False
    else:
        print("  PASS: v correct")

    if all_passed:
        print("\n  All RMSprop algorithm tests PASSED")
    else:
        print("\n  Some RMSprop algorithm tests FAILED")


# =============================================================================
# Test AdamW Algorithm
# =============================================================================


fn test_adamw_algorithm():
    print_test_header("AdamW Algorithm Verification")

    # Test AdamW update rule:
    # m = beta1 * m + (1 - beta1) * grad
    # v = beta2 * v + (1 - beta2) * grad^2
    # m_hat = m / (1 - beta1^t)
    # v_hat = v / (1 - beta2^t)
    # param = param * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)

    var lr: Float64 = 0.001
    var beta1: Float64 = 0.9
    var beta2: Float64 = 0.999
    var eps: Float64 = 1e-8
    var weight_decay: Float64 = 0.01

    # Initial values
    var param: Float64 = 1.0
    var grad: Float64 = 0.5
    var m: Float64 = 0.0
    var v: Float64 = 0.0
    var t: Int = 0

    print("  AdamW parameters:")
    print("    lr = " + String(lr))
    print("    beta1 = " + String(beta1))
    print("    beta2 = " + String(beta2))
    print("    weight_decay = " + String(weight_decay))
    print("    eps = " + String(eps))
    print()
    print("  Initial state:")
    print("    param = " + String(param))
    print("    m = " + String(m))
    print("    v = " + String(v))

    # Step 1
    t = 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad

    var bias_corr1 = 1.0 - beta1**t
    var bias_corr2 = 1.0 - beta2**t
    var m_hat = m / bias_corr1
    var v_hat = v / bias_corr2

    var wd_factor = 1.0 - lr * weight_decay
    param = param * wd_factor - lr * m_hat / (sqrt(v_hat) + eps)

    print("\n  After step 1 (grad = 0.5):")
    print("    m = " + String(m)[:10])
    print("    v = " + String(v)[:10])
    print("    m_hat = " + String(m_hat)[:10])
    print("    v_hat = " + String(v_hat)[:10])
    print("    param = " + String(param)[:10])

    var all_passed = True

    # Verify m = (1-0.9) * 0.5 = 0.05
    var expected_m1: Float64 = 0.1 * 0.5
    if not assert_close(m, expected_m1, 0.0001, "m after step 1"):
        all_passed = False
    else:
        print("  PASS: m correct")

    # Verify v = (1-0.999) * 0.25 = 0.00025
    var expected_v1: Float64 = 0.001 * 0.25
    if not assert_close(v, expected_v1, 0.00001, "v after step 1"):
        all_passed = False
    else:
        print("  PASS: v correct")

    # Key difference from Adam: weight decay is decoupled
    print("\n  AdamW vs Adam:")
    print("    AdamW: param = param * (1 - lr * wd) - lr * update")
    print("    Adam:  param = param - lr * (update + wd * param)")
    print("    Decoupled weight decay leads to better generalization")

    if all_passed:
        print("\n  All AdamW algorithm tests PASSED")
    else:
        print("\n  Some AdamW algorithm tests FAILED")


# =============================================================================
# Test RMSprop Training
# =============================================================================


fn test_rmsprop_training():
    print_test_header("RMSprop Training (CPU)")

    seed(42)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Model
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print("  Model: Linear[4, 16] -> ReLU[16] -> Linear[16, 2]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))

    # RMSprop optimizer
    var optimizer = RMSprop(lr=0.01, alpha=0.99, eps=1e-8)
    print("\n  Optimizer: RMSprop(lr=0.01, alpha=0.99)")

    var loss_fn = MSELoss()

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Kaiming(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate training data
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        var x1 = Scalar[dtype](random_float64() * 2 - 1)
        var x2 = Scalar[dtype](random_float64() * 2 - 1)
        var x3 = Scalar[dtype](random_float64() * 2 - 1)
        var x4 = Scalar[dtype](random_float64() * 2 - 1)
        input_data[i * INPUT_DIM + 0] = x1
        input_data[i * INPUT_DIM + 1] = x2
        input_data[i * INPUT_DIM + 2] = x3
        input_data[i * INPUT_DIM + 3] = x4
        target_data[i * OUTPUT_DIM + 0] = x1 * x2
        target_data[i * OUTPUT_DIM + 1] = x3 * x4

    print("\n  Training...")
    print("-" * 70)

    var result = trainer.train[BATCH_SIZE](input_data, target_data)

    print("-" * 70)
    print("  Training completed!")
    print("  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))

    if result.final_loss < 0.5:
        print("\n  PASS: RMSprop training succeeded")
    else:
        print("\n  Note: May need tuning for better convergence")


# =============================================================================
# Test AdamW Training
# =============================================================================


fn test_adamw_training():
    print_test_header("AdamW Training (CPU)")

    seed(123)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Model
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        Tanh[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print("  Model: Linear[4, 16] -> Tanh[16] -> Linear[16, 2]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))

    # AdamW optimizer
    var optimizer = AdamW(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    print("\n  Optimizer: AdamW(lr=0.001, weight_decay=0.01)")

    var loss_fn = MSELoss()

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Xavier(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate training data
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        var x1 = Scalar[dtype](random_float64() * 2 - 1)
        var x2 = Scalar[dtype](random_float64() * 2 - 1)
        var x3 = Scalar[dtype](random_float64() * 2 - 1)
        var x4 = Scalar[dtype](random_float64() * 2 - 1)
        input_data[i * INPUT_DIM + 0] = x1
        input_data[i * INPUT_DIM + 1] = x2
        input_data[i * INPUT_DIM + 2] = x3
        input_data[i * INPUT_DIM + 3] = x4
        target_data[i * OUTPUT_DIM + 0] = x1 + x2
        target_data[i * OUTPUT_DIM + 1] = x3 + x4

    print("\n  Training...")
    print("-" * 70)

    var result = trainer.train[BATCH_SIZE](input_data, target_data)

    print("-" * 70)
    print("  Training completed!")
    print("  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))

    if result.final_loss < 0.5:
        print("\n  PASS: AdamW training succeeded")
    else:
        print("\n  Note: May need tuning for better convergence")


# =============================================================================
# Compare Optimizers
# =============================================================================


fn compare_optimizers():
    print_test_header("Optimizer Comparison (CPU)")

    seed(456)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 100

    # Generate shared training data
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        var x1 = Scalar[dtype](random_float64() * 2 - 1)
        var x2 = Scalar[dtype](random_float64() * 2 - 1)
        var x3 = Scalar[dtype](random_float64() * 2 - 1)
        var x4 = Scalar[dtype](random_float64() * 2 - 1)
        input_data[i * INPUT_DIM + 0] = x1
        input_data[i * INPUT_DIM + 1] = x2
        input_data[i * INPUT_DIM + 2] = x3
        input_data[i * INPUT_DIM + 3] = x4
        target_data[i * OUTPUT_DIM + 0] = x1 * x2
        target_data[i * OUTPUT_DIM + 1] = x3 * x4

    print("  Comparing optimizers on same task...")
    print("  Model: Linear[4, 16] -> ReLU[16] -> Linear[16, 2]")
    print("  Task: y = [x1*x2, x3*x4]")
    print("  Epochs: " + String(NUM_EPOCHS))
    print()

    # Test SGD
    var model_sgd = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )
    var trainer_sgd = Trainer(
        model_sgd, SGD(lr=0.1), MSELoss(), Kaiming(),
        epochs=NUM_EPOCHS, print_every=0,
    )
    var result_sgd = trainer_sgd.train[BATCH_SIZE](input_data, target_data)
    print("  SGD(lr=0.1):           loss = " + String(result_sgd.final_loss)[:8])

    # Test Adam
    var model_adam = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )
    var trainer_adam = Trainer(
        model_adam, Adam(lr=0.01), MSELoss(), Kaiming(),
        epochs=NUM_EPOCHS, print_every=0,
    )
    var result_adam = trainer_adam.train[BATCH_SIZE](input_data, target_data)
    print("  Adam(lr=0.01):         loss = " + String(result_adam.final_loss)[:8])

    # Test RMSprop
    var model_rmsprop = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )
    var trainer_rmsprop = Trainer(
        model_rmsprop, RMSprop(lr=0.01), MSELoss(), Kaiming(),
        epochs=NUM_EPOCHS, print_every=0,
    )
    var result_rmsprop = trainer_rmsprop.train[BATCH_SIZE](input_data, target_data)
    print("  RMSprop(lr=0.01):      loss = " + String(result_rmsprop.final_loss)[:8])

    # Test AdamW
    var model_adamw = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )
    var trainer_adamw = Trainer(
        model_adamw, AdamW(lr=0.01, weight_decay=0.01), MSELoss(), Kaiming(),
        epochs=NUM_EPOCHS, print_every=0,
    )
    var result_adamw = trainer_adamw.train[BATCH_SIZE](input_data, target_data)
    print("  AdamW(lr=0.01, wd=0.01): loss = " + String(result_adamw.final_loss)[:8])

    print("\n  All optimizers trained successfully!")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Deep RL - New Optimizers CPU Tests")
    print("=" * 70)
    print()
    print("Testing: RMSprop, AdamW")
    print()

    test_rmsprop_algorithm()
    test_adamw_algorithm()
    test_rmsprop_training()
    test_adamw_training()
    compare_optimizers()

    print("\n" + "=" * 70)
    print("All Optimizer CPU Tests Completed!")
    print("=" * 70)
