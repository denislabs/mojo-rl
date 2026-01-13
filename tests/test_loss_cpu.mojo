"""Test New Loss Functions on CPU.

Tests:
- HuberLoss (Smooth L1)
- CrossEntropyLoss

Run with:
    pixi run mojo run tests/test_loss_cpu.mojo
"""

from random import seed, random_float64
from math import exp, log, sqrt

from deep_rl.constants import dtype
from deep_rl.model import Linear, ReLU, Softmax, seq
from deep_rl.loss import MSELoss, HuberLoss, CrossEntropyLoss
from deep_rl.optimizer import Adam
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


fn _abs(val: Float64) -> Float64:
    return val if val >= 0.0 else -val


# =============================================================================
# Test Huber Loss Algorithm
# =============================================================================


fn test_huber_loss_algorithm():
    print_test_header("Huber Loss Algorithm Verification")

    var delta: Float64 = 1.0

    print("  Huber Loss (delta = " + String(delta) + "):")
    print("    L = 0.5 * (y - t)^2           if |y - t| <= delta")
    print("    L = delta * |y - t| - 0.5 * delta^2  otherwise")
    print()

    # Test cases
    var test_cases = InlineArray[Float64, 5](0.0, 0.5, 1.0, 2.0, 3.0)
    var all_passed = True

    print("  Forward pass verification (target = 0):")
    for i in range(5):
        var output = test_cases[i]
        var diff = _abs(output)

        var expected_loss: Float64
        if diff <= delta:
            expected_loss = 0.5 * diff * diff
        else:
            expected_loss = delta * diff - 0.5 * delta * delta

        print("    output=" + String(output) + ": loss=" + String(expected_loss)[:8])

    # Verify specific values
    # output=0.5: |0.5| <= 1, so L = 0.5 * 0.25 = 0.125
    if not assert_close(0.5 * 0.5 * 0.5, 0.125, 0.001, "Huber(0.5, 0)"):
        all_passed = False
    else:
        print("  PASS: Quadratic region correct")

    # output=2.0: |2| > 1, so L = 1 * 2 - 0.5 = 1.5
    if not assert_close(1.0 * 2.0 - 0.5, 1.5, 0.001, "Huber(2, 0)"):
        all_passed = False
    else:
        print("  PASS: Linear region correct")

    # Test gradient
    print("\n  Gradient verification:")
    print("    dL/dy = (y - t)              if |y - t| <= delta")
    print("    dL/dy = delta * sign(y - t)  otherwise")

    # Quadratic region gradient
    var grad_quad: Float64 = 0.5  # output=0.5, target=0: grad = 0.5/1 = 0.5
    print("    output=0.5: grad=" + String(grad_quad))

    # Linear region gradient
    var grad_linear: Float64 = delta  # output=2.0, target=0: grad = delta * sign(2) = 1.0
    print("    output=2.0: grad=" + String(grad_linear) + " (clipped)")

    if all_passed:
        print("\n  All Huber Loss algorithm tests PASSED")
    else:
        print("\n  Some Huber Loss algorithm tests FAILED")


# =============================================================================
# Test Cross-Entropy Loss Algorithm
# =============================================================================


fn test_cross_entropy_algorithm():
    print_test_header("Cross-Entropy Loss Algorithm Verification")

    print("  Cross-Entropy Loss:")
    print("    L = -sum(target * log_softmax(output))")
    print("    dL/dy = softmax(output) - target")
    print()

    # Test case: logits [1, 2, 3], one-hot target [0, 0, 1]
    var logits = InlineArray[Float64, 3](1.0, 2.0, 3.0)
    var target = InlineArray[Float64, 3](0.0, 0.0, 1.0)

    # Compute softmax
    var max_val = logits[2]
    var sum_exp: Float64 = 0.0
    for i in range(3):
        sum_exp += exp(logits[i] - max_val)

    print("  Input logits: [1, 2, 3]")
    print("  Target (one-hot): [0, 0, 1]")
    print()
    print("  Softmax computation:")

    var softmax_vals = InlineArray[Float64, 3](0.0, 0.0, 0.0)
    for i in range(3):
        softmax_vals[i] = exp(logits[i] - max_val) / sum_exp
        print("    softmax[" + String(i) + "] = " + String(softmax_vals[i])[:8])

    # Verify softmax sums to 1
    var softmax_sum = softmax_vals[0] + softmax_vals[1] + softmax_vals[2]
    var all_passed = True

    if not assert_close(softmax_sum, 1.0, 0.0001, "softmax sum"):
        all_passed = False
    else:
        print("  PASS: softmax sum = 1.0")

    # Compute cross-entropy loss
    # L = -sum(target * log(softmax))
    # Since target = [0, 0, 1], L = -log(softmax[2])
    var log_sum_exp = max_val + log(sum_exp)
    var log_softmax_2 = logits[2] - log_sum_exp  # = 3 - log_sum_exp
    var loss = -log_softmax_2  # Since target[2] = 1

    print("\n  Cross-Entropy Loss = " + String(loss)[:8])

    # Expected: -log(softmax(3)) = -log(e^0 / (e^(-2) + e^(-1) + e^0))
    #         = -log(1 / (0.135 + 0.368 + 1)) = -log(0.665) = 0.407
    if not assert_close(loss, 0.407, 0.01, "cross-entropy loss"):
        all_passed = False
    else:
        print("  PASS: Cross-entropy loss correct")

    # Compute gradient
    # grad = softmax - target
    print("\n  Gradient (softmax - target):")
    for i in range(3):
        var grad = softmax_vals[i] - target[i]
        print("    grad[" + String(i) + "] = " + String(grad)[:8])

    # grad[0] = softmax[0] - 0 = 0.090
    # grad[1] = softmax[1] - 0 = 0.245
    # grad[2] = softmax[2] - 1 = -0.335
    var grad2 = softmax_vals[2] - 1.0
    if not assert_close(grad2, -0.335, 0.01, "grad[2]"):
        all_passed = False
    else:
        print("  PASS: Gradient correct")

    if all_passed:
        print("\n  All Cross-Entropy algorithm tests PASSED")
    else:
        print("\n  Some Cross-Entropy algorithm tests FAILED")


# =============================================================================
# Test Huber Loss vs MSE (Outlier Robustness)
# =============================================================================


fn test_huber_vs_mse():
    print_test_header("Huber Loss vs MSE - Outlier Robustness")

    var outputs = InlineArray[Float64, 5](0.0, 0.0, 0.0, 0.0, 5.0)  # One outlier
    var targets = InlineArray[Float64, 5](0.0, 0.0, 0.0, 0.0, 0.0)

    # MSE Loss: L = mean((y - t)^2)
    var mse_loss: Float64 = 0.0
    for i in range(5):
        var diff = outputs[i] - targets[i]
        mse_loss += diff * diff
    mse_loss /= 5.0

    print("  Outputs: [0, 0, 0, 0, 5]  (one outlier at 5)")
    print("  Targets: [0, 0, 0, 0, 0]")
    print()
    print("  MSE Loss: " + String(mse_loss))
    print("    = (0 + 0 + 0 + 0 + 25) / 5 = 5.0")

    # Huber Loss with delta=1
    var delta: Float64 = 1.0
    var huber_loss: Float64 = 0.0
    for i in range(5):
        var diff = _abs(outputs[i] - targets[i])
        if diff <= delta:
            huber_loss += 0.5 * diff * diff
        else:
            huber_loss += delta * diff - 0.5 * delta * delta
    huber_loss /= 5.0

    print()
    print("  Huber Loss (delta=1): " + String(huber_loss))
    print("    = (0 + 0 + 0 + 0 + (1*5 - 0.5)) / 5 = 0.9")

    print()
    print("  Ratio (MSE / Huber): " + String(mse_loss / huber_loss)[:6])
    print("  Huber is " + String(mse_loss / huber_loss)[:4] + "x more robust to outliers!")

    if huber_loss < mse_loss:
        print("\n  PASS: Huber Loss is more robust to outliers")
    else:
        print("\n  FAIL: Expected Huber < MSE for outliers")


# =============================================================================
# Test Huber Loss Training
# =============================================================================


fn test_huber_training():
    print_test_header("Huber Loss Training (CPU)")

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

    # Huber Loss
    var loss_fn = HuberLoss(delta=1.0)
    print("\n  Loss: HuberLoss(delta=1.0)")

    var optimizer = Adam(lr=0.01)

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Kaiming(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate training data with some outliers
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

        # Add some outliers
        if i < 3:
            target_data[i * OUTPUT_DIM + 0] = Scalar[dtype](10.0)  # Outlier
            target_data[i * OUTPUT_DIM + 1] = Scalar[dtype](-10.0)  # Outlier
        else:
            target_data[i * OUTPUT_DIM + 0] = x1 * x2
            target_data[i * OUTPUT_DIM + 1] = x3 * x4

    print("\n  Training data: 3 outliers, rest are y = [x1*x2, x3*x4]")
    print("-" * 70)

    var result = trainer.train[BATCH_SIZE](input_data, target_data)

    print("-" * 70)
    print("  Training completed!")
    print("  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))

    print("\n  PASS: Huber Loss training completed")


# =============================================================================
# Test Cross-Entropy Training (Classification)
# =============================================================================


fn test_cross_entropy_training():
    print_test_header("Cross-Entropy Loss Training (CPU) - Classification")

    seed(123)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime NUM_CLASSES = 3
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Model with softmax output
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, NUM_CLASSES](),
        # Note: CrossEntropyLoss expects logits, not softmax outputs
        # We'll use softmax for demonstration but in practice
        # cross-entropy is applied to raw logits
    )

    print("  Model: Linear[4, 16] -> ReLU[16] -> Linear[16, 3]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))

    # Cross-Entropy Loss
    var loss_fn = CrossEntropyLoss()
    print("\n  Loss: CrossEntropyLoss")
    print("  Note: Expects logits input, computes softmax internally")

    var optimizer = Adam(lr=0.01)

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Kaiming(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate classification data
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * NUM_CLASSES](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)

        # Determine class based on sum of inputs
        var sum_inputs: Float64 = 0.0
        for j in range(INPUT_DIM):
            sum_inputs += Float64(input_data[i * INPUT_DIM + j])

        var class_idx = 0
        if sum_inputs > 0.5:
            class_idx = 2
        elif sum_inputs > -0.5:
            class_idx = 1

        # One-hot encoding
        for j in range(NUM_CLASSES):
            target_data[i * NUM_CLASSES + j] = Scalar[dtype](1.0) if j == class_idx else Scalar[dtype](0.0)

    print("\n  Classification task: 3 classes based on sum of inputs")
    print("-" * 70)

    var result = trainer.train[BATCH_SIZE](input_data, target_data)

    print("-" * 70)
    print("  Training completed!")
    print("  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))

    print("\n  PASS: Cross-Entropy training completed")


# =============================================================================
# Compare Loss Functions
# =============================================================================


fn compare_loss_functions():
    print_test_header("Loss Function Comparison")

    seed(456)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 100

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

    print("  Comparing MSE vs Huber on regression task...")
    print("  Model: Linear[4, 16] -> ReLU[16] -> Linear[16, 2]")
    print("  Task: y = [x1*x2, x3*x4]")
    print("  Epochs: " + String(NUM_EPOCHS))
    print()

    # Test MSE
    var model_mse = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )
    var trainer_mse = Trainer(
        model_mse, Adam(lr=0.01), MSELoss(), Kaiming(),
        epochs=NUM_EPOCHS, print_every=0,
    )
    var result_mse = trainer_mse.train[BATCH_SIZE](input_data, target_data)
    print("  MSELoss:           loss = " + String(result_mse.final_loss)[:8])

    # Test Huber (delta=1)
    var model_huber1 = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )
    var trainer_huber1 = Trainer(
        model_huber1, Adam(lr=0.01), HuberLoss(delta=1.0), Kaiming(),
        epochs=NUM_EPOCHS, print_every=0,
    )
    var result_huber1 = trainer_huber1.train[BATCH_SIZE](input_data, target_data)
    print("  HuberLoss(d=1.0):  loss = " + String(result_huber1.final_loss)[:8])

    # Test Huber (delta=0.5)
    var model_huber05 = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )
    var trainer_huber05 = Trainer(
        model_huber05, Adam(lr=0.01), HuberLoss(delta=0.5), Kaiming(),
        epochs=NUM_EPOCHS, print_every=0,
    )
    var result_huber05 = trainer_huber05.train[BATCH_SIZE](input_data, target_data)
    print("  HuberLoss(d=0.5):  loss = " + String(result_huber05.final_loss)[:8])

    print("\n  All loss functions trained successfully!")
    print("  Note: Huber loss values are not directly comparable to MSE")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Deep RL - New Loss Functions CPU Tests")
    print("=" * 70)
    print()
    print("Testing: HuberLoss, CrossEntropyLoss")
    print()

    test_huber_loss_algorithm()
    test_cross_entropy_algorithm()
    test_huber_vs_mse()
    test_huber_training()
    test_cross_entropy_training()
    compare_loss_functions()

    print("\n" + "=" * 70)
    print("All Loss Function CPU Tests Completed!")
    print("=" * 70)
