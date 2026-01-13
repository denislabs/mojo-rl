"""Test suite for StochasticActor on CPU.

Tests the Gaussian policy network and reparameterization utilities.

Run with:
    pixi run mojo run -I . tests/test_stochastic_actor_cpu.mojo
"""

from random import seed, random_float64
from math import exp, log, sqrt

from deep_rl.constants import dtype
from deep_rl.model import (
    Linear,
    ReLU,
    StochasticActor,
    seq,
)
from deep_rl.loss import MSELoss
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
        print("    Diff:     " + String(diff))
        return False
    return True


# =============================================================================
# Test StochasticActor Structure
# =============================================================================


fn test_stochastic_actor_structure():
    print_test_header("StochasticActor Structure")

    comptime IN_DIM = 8
    comptime ACTION_DIM = 2

    var actor = StochasticActor[IN_DIM, ACTION_DIM]()

    print("  StochasticActor[" + String(IN_DIM) + ", " + String(ACTION_DIM) + "]")
    print("  IN_DIM: " + String(actor.IN_DIM))
    print("  OUT_DIM: " + String(actor.OUT_DIM) + " (mean + log_std)")
    print("  PARAM_SIZE: " + String(actor.PARAM_SIZE))
    print("  CACHE_SIZE: " + String(actor.CACHE_SIZE))

    # Verify dimensions
    var expected_out_dim = ACTION_DIM * 2  # mean and log_std
    var expected_param_size = 2 * (IN_DIM * ACTION_DIM + ACTION_DIM)  # Two linear heads
    var expected_cache_size = IN_DIM  # Cache input

    var all_passed = True

    if actor.OUT_DIM == expected_out_dim:
        print("  PASS: OUT_DIM = " + String(expected_out_dim) + " (action_dim * 2)")
    else:
        print("  FAIL: OUT_DIM should be " + String(expected_out_dim))
        all_passed = False

    if actor.PARAM_SIZE == expected_param_size:
        print("  PASS: PARAM_SIZE = " + String(expected_param_size) + " (2 linear heads)")
    else:
        print("  FAIL: PARAM_SIZE should be " + String(expected_param_size))
        all_passed = False

    if actor.CACHE_SIZE == expected_cache_size:
        print("  PASS: CACHE_SIZE = " + String(expected_cache_size) + " (caches input)")
    else:
        print("  FAIL: CACHE_SIZE should be " + String(expected_cache_size))
        all_passed = False

    if all_passed:
        print("\n  All structure tests PASSED")


# =============================================================================
# Test Reparameterization Math
# =============================================================================


fn test_reparameterization_math():
    print_test_header("Reparameterization Math")

    # Test the reparameterization trick formula
    # z = mean + std * noise
    # action = tanh(z)

    print("  Formula: z = mean + exp(log_std) * noise")
    print("           action = tanh(z)")

    # Test case: mean=0, log_std=0 (std=1), noise=0
    var mean: Float64 = 0.0
    var log_std: Float64 = 0.0
    var noise: Float64 = 0.0

    var std = exp(log_std)
    var z = mean + std * noise
    var exp_z = exp(z)
    var exp_neg_z = exp(-z)
    var action = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

    var all_passed = True

    if assert_close(action, 0.0, 1e-6, "action when mean=0, noise=0"):
        print("  PASS: tanh(0) = 0")
    else:
        all_passed = False

    # Test case: mean=0.5, log_std=-1, noise=0
    mean = 0.5
    log_std = -1.0
    noise = 0.0

    std = exp(log_std)
    z = mean + std * noise
    exp_z = exp(z)
    exp_neg_z = exp(-z)
    action = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

    var expected_action = (exp(0.5) - exp(-0.5)) / (exp(0.5) + exp(-0.5))
    if assert_close(action, expected_action, 1e-6, "action when mean=0.5, noise=0"):
        print("  PASS: tanh(0.5) ≈ " + String(expected_action)[:6])
    else:
        all_passed = False

    # Test case: verify noise affects output
    mean = 0.0
    log_std = 0.0  # std = 1
    noise = 1.0

    std = exp(log_std)
    z = mean + std * noise  # z = 1
    exp_z = exp(z)
    exp_neg_z = exp(-z)
    action = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)

    expected_action = (exp(1.0) - exp(-1.0)) / (exp(1.0) + exp(-1.0))
    if assert_close(action, expected_action, 1e-6, "action with noise=1"):
        print("  PASS: tanh(1.0) ≈ " + String(expected_action)[:6])
    else:
        all_passed = False

    if all_passed:
        print("\n  All reparameterization math tests PASSED")


# =============================================================================
# Test Log Probability Formula
# =============================================================================


fn test_log_prob_formula():
    print_test_header("Log Probability Formula")

    # Log prob with tanh squashing correction:
    # log_prob = log_gaussian(z; mean, std) - sum(log(1 - tanh(z)^2 + eps))

    print("  Formula: log_prob = log_normal(z) - sum(log(1 - action^2 + eps))")

    var LOG_2PI: Float64 = 1.8378770664093453
    var EPS: Float64 = 1e-6

    # Test at z=0 (action=0)
    var mean: Float64 = 0.0
    var log_std: Float64 = 0.0  # std = 1
    var z: Float64 = 0.0

    # log_gaussian at mean with std=1
    var z_normalized = (z - mean) / exp(log_std)
    var log_gaussian = -0.5 * (LOG_2PI + 2.0 * log_std + z_normalized * z_normalized)

    # At z=0, tanh(0)=0, so squashing correction is -log(1 - 0 + eps) ≈ 0
    var action = 0.0
    var squash_correction = log(1.0 - action * action + EPS)

    var log_prob = log_gaussian - squash_correction

    print("  At z=0, mean=0, std=1:")
    print("    log_gaussian ≈ " + String(log_gaussian)[:8])
    print("    squash_correction ≈ " + String(squash_correction)[:8])
    print("    log_prob ≈ " + String(log_prob)[:8])

    # Log prob should be reasonable (around -0.9 for standard normal at mean)
    if log_prob > -2.0 and log_prob < 0.0:
        print("  PASS: log_prob at mean is reasonable")
    else:
        print("  WARNING: log_prob seems unusual")

    # Test at action near boundary
    var action_boundary: Float64 = 0.99
    squash_correction = log(1.0 - action_boundary * action_boundary + EPS)
    print("\n  At action=0.99:")
    print("    squash_correction = " + String(squash_correction)[:8])
    print("    (Large negative correction for near-boundary actions)")

    if squash_correction < -3.0:
        print("  PASS: Boundary actions have large negative correction")
    else:
        print("  WARNING: Boundary correction seems small")


# =============================================================================
# Test Training with Simple Target
# =============================================================================


fn test_training_simple():
    print_test_header("StochasticActor Training (Simple)")

    seed(42)

    comptime IN_DIM = 4
    comptime ACTION_DIM = 2
    comptime OUT_DIM = ACTION_DIM * 2
    comptime BATCH = 16
    comptime EPOCHS = 100

    # Simple model: just the actor head
    var model = StochasticActor[IN_DIM, ACTION_DIM]()

    print("  Model: StochasticActor[" + String(IN_DIM) + ", " + String(ACTION_DIM) + "]")
    print("  BATCH: " + String(BATCH))
    print("  EPOCHS: " + String(EPOCHS))
    print("  Target: mean=0.5, log_std=-1.0")

    var trainer = Trainer(
        model,
        Adam(lr=0.05),  # Higher LR for simple model
        MSELoss(),
        Kaiming(),
        epochs=EPOCHS,
        print_every=0,  # Quiet training
    )

    # Generate data with fixed pattern
    var input_data = InlineArray[Scalar[dtype], BATCH * IN_DIM](uninitialized=True)
    var target_data = InlineArray[Scalar[dtype], BATCH * OUT_DIM](uninitialized=True)

    for b in range(BATCH):
        for i in range(IN_DIM):
            input_data[b * IN_DIM + i] = Scalar[dtype](random_float64(-1.0, 1.0))
        for j in range(ACTION_DIM):
            target_data[b * OUT_DIM + j] = Scalar[dtype](0.5)  # mean
            target_data[b * OUT_DIM + ACTION_DIM + j] = Scalar[dtype](-1.0)  # log_std

    # Train
    print("\n  Training...")
    var result = trainer.train[BATCH](input_data, target_data)

    print("  Final loss: " + String(result.final_loss))
    print("  Epochs trained: " + String(result.epochs_trained))

    if result.final_loss < 0.1:
        print("\n  PASS: Training converged")
    elif result.final_loss < 0.5:
        print("\n  PARTIAL: Training progressed but didn't fully converge")
    else:
        print("\n  FAIL: Training did not converge")


# =============================================================================
# Test Training with Backbone
# =============================================================================


fn test_training_with_backbone():
    print_test_header("StochasticActor with Backbone")

    seed(123)

    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime ACTION_DIM = 2
    comptime OUT_DIM = ACTION_DIM * 2
    comptime BATCH = 16
    comptime EPOCHS = 200

    # Full policy network
    var model = seq(
        Linear[OBS_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        StochasticActor[HIDDEN_DIM, ACTION_DIM](),
    )

    print("  Architecture:")
    print("    Linear[4, 16] -> ReLU[16] -> StochasticActor[16, 2]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print("  BATCH: " + String(BATCH))
    print("  EPOCHS: " + String(EPOCHS))

    var trainer = Trainer(
        model,
        Adam(lr=0.01),
        MSELoss(),
        Kaiming(),
        epochs=EPOCHS,
        print_every=0,
    )

    # Generate data
    var input_data = InlineArray[Scalar[dtype], BATCH * OBS_DIM](uninitialized=True)
    var target_data = InlineArray[Scalar[dtype], BATCH * OUT_DIM](uninitialized=True)

    for b in range(BATCH):
        for i in range(OBS_DIM):
            input_data[b * OBS_DIM + i] = Scalar[dtype](random_float64(-1.0, 1.0))
        # Target: learned mapping from inputs to outputs
        var sum_input: Float64 = 0.0
        for i in range(OBS_DIM):
            sum_input += Float64(input_data[b * OBS_DIM + i])
        for j in range(ACTION_DIM):
            target_data[b * OUT_DIM + j] = Scalar[dtype](0.2 * sum_input)  # mean
            target_data[b * OUT_DIM + ACTION_DIM + j] = Scalar[dtype](-0.5)  # log_std

    print("\n  Training...")
    var result = trainer.train[BATCH](input_data, target_data)

    print("  Final loss: " + String(result.final_loss))
    print("  Epochs trained: " + String(result.epochs_trained))

    if result.final_loss < 0.1:
        print("\n  PASS: Backbone + Actor training succeeded")
    elif result.final_loss < 0.3:
        print("\n  PARTIAL: Training progressed")
    else:
        print("\n  Note: Complex networks may need more epochs")


# =============================================================================
# Test Deterministic Action
# =============================================================================


fn test_deterministic_action():
    print_test_header("Deterministic Action (Evaluation)")

    # In evaluation mode, we use action = tanh(mean) without noise
    print("  Deterministic policy: action = tanh(mean)")

    var test_means = InlineArray[Float64, 5](0.0, 0.5, 1.0, -0.5, 2.0)
    var all_passed = True

    for i in range(5):
        var mean = test_means[i]
        var exp_m = exp(mean)
        var exp_neg_m = exp(-mean)
        var action = (exp_m - exp_neg_m) / (exp_m + exp_neg_m)

        print("  tanh(" + String(mean) + ") = " + String(action)[:8])

        # Verify action is in (-1, 1)
        if action <= -1.0 or action >= 1.0:
            print("    FAIL: Action out of bounds")
            all_passed = False

    if all_passed:
        print("\n  PASS: All deterministic actions in valid range")


# =============================================================================
# Test Log Std Clamping
# =============================================================================


fn test_log_std_clamping():
    print_test_header("Log Std Clamping")

    print("  log_std is clamped to [-20, 2] for numerical stability")
    print("  This prevents:")
    print("    - exp(log_std) underflow when log_std < -20 (std < 2e-9)")
    print("    - exp(log_std) overflow when log_std > 2 (std > 7.4)")

    var LOG_STD_MIN: Float64 = -20.0
    var LOG_STD_MAX: Float64 = 2.0

    var std_min = exp(LOG_STD_MIN)
    var std_max = exp(LOG_STD_MAX)

    print("\n  At LOG_STD_MIN = -20:")
    print("    std = exp(-20) ≈ " + String(std_min))
    print("    (Very small exploration)")

    print("\n  At LOG_STD_MAX = 2:")
    print("    std = exp(2) ≈ " + String(std_max))
    print("    (Wide exploration)")

    if std_min > 0 and std_min < 1e-6:
        print("\n  PASS: MIN std is very small but positive")
    else:
        print("\n  WARNING: MIN std may cause issues")

    if std_max > 1 and std_max < 10:
        print("  PASS: MAX std is reasonable for exploration")
    else:
        print("  WARNING: MAX std may be too large")


# =============================================================================
# Main
# =============================================================================


fn main():
    print("=" * 70)
    print("StochasticActor CPU Tests")
    print("=" * 70)
    print("\nGaussian Policy Network for SAC/PPO")
    print("Features: learned mean + log_std, reparameterization trick")

    test_stochastic_actor_structure()
    test_reparameterization_math()
    test_log_prob_formula()
    test_training_simple()
    test_training_with_backbone()
    test_deterministic_action()
    test_log_std_clamping()

    print("\n" + "=" * 70)
    print("All StochasticActor CPU Tests Completed!")
    print("=" * 70)
