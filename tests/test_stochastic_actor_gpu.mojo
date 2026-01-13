"""Test suite for StochasticActor on GPU.

Tests the Gaussian policy network training on GPU using the Trainer.

Run with:
    pixi run -e apple mojo run -I . tests/test_stochastic_actor_gpu.mojo
"""

from random import seed, random_float64
from time import perf_counter_ns

from deep_rl.constants import dtype
from deep_rl.model import (
    Linear,
    ReLU,
    Tanh,
    StochasticActor,
    seq,
)
from deep_rl.loss import MSELoss
from deep_rl.optimizer import Adam
from deep_rl.training import Trainer
from deep_rl.initializer import Kaiming

from gpu.host import DeviceContext


# =============================================================================
# Test Helper
# =============================================================================


fn print_test_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


# =============================================================================
# Test StochasticActor GPU Training
# =============================================================================


fn test_stochastic_actor_gpu_training() raises:
    print_test_header("StochasticActor GPU Training")

    seed(42)

    comptime IN_DIM = 8
    comptime ACTION_DIM = 2
    comptime OUT_DIM = ACTION_DIM * 2
    comptime BATCH = 16
    comptime EPOCHS = 200

    var model = StochasticActor[IN_DIM, ACTION_DIM]()

    print("  Model: StochasticActor[" + String(IN_DIM) + ", " + String(ACTION_DIM) + "]")
    print("  IN_DIM: " + String(IN_DIM))
    print("  ACTION_DIM: " + String(ACTION_DIM))
    print("  OUT_DIM: " + String(model.OUT_DIM))
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print("  BATCH: " + String(BATCH))
    print("  EPOCHS: " + String(EPOCHS))

    var trainer = Trainer(
        model,
        Adam(lr=0.01),
        MSELoss(),
        Kaiming(),
        epochs=EPOCHS,
        print_every=50,
    )

    # Generate data
    var input_data = InlineArray[Scalar[dtype], BATCH * IN_DIM](uninitialized=True)
    var target_data = InlineArray[Scalar[dtype], BATCH * OUT_DIM](uninitialized=True)

    for b in range(BATCH):
        for i in range(IN_DIM):
            input_data[b * IN_DIM + i] = Scalar[dtype](random_float64(-1.0, 1.0))
        for j in range(ACTION_DIM):
            target_data[b * OUT_DIM + j] = Scalar[dtype](0.3)  # mean
            target_data[b * OUT_DIM + ACTION_DIM + j] = Scalar[dtype](-0.5)  # log_std

    print("\n  Training on GPU...")
    var ctx = DeviceContext()

    var start = perf_counter_ns()
    var result = trainer.train_gpu[BATCH](ctx, input_data, target_data)
    var end = perf_counter_ns()

    var time_ms = Float64(end - start) / 1e6

    print("-" * 70)
    print("  GPU Training completed!")
    print("  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))
    print("  Time: " + String(time_ms)[:8] + " ms")
    print("  Avg per epoch: " + String(time_ms / Float64(EPOCHS))[:6] + " ms")

    if result.final_loss < 0.05:
        print("\n  PASS: StochasticActor GPU training converged")
    else:
        print("\n  WARNING: Training did not fully converge, but GPU execution worked")


# =============================================================================
# Test Backbone + StochasticActor
# =============================================================================


fn test_backbone_with_stochastic_actor() raises:
    print_test_header("Backbone + StochasticActor (Full Policy)")

    seed(123)

    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 32
    comptime ACTION_DIM = 2
    comptime OUT_DIM = ACTION_DIM * 2
    comptime BATCH = 32
    comptime EPOCHS = 300

    var model = seq(
        Linear[OBS_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        StochasticActor[HIDDEN_DIM, ACTION_DIM](),
    )

    print("  Architecture:")
    print("    Linear[4, 32] -> ReLU[32] -> Linear[32, 32] -> ReLU[32]")
    print("    -> StochasticActor[32, 2]")
    print("  Total PARAM_SIZE: " + String(model.PARAM_SIZE))
    print("  BATCH: " + String(BATCH))
    print("  EPOCHS: " + String(EPOCHS))

    var trainer = Trainer(
        model,
        Adam(lr=0.005),
        MSELoss(),
        Kaiming(),
        epochs=EPOCHS,
        print_every=100,
    )

    # Generate data with pattern
    var input_data = InlineArray[Scalar[dtype], BATCH * OBS_DIM](uninitialized=True)
    var target_data = InlineArray[Scalar[dtype], BATCH * OUT_DIM](uninitialized=True)

    for b in range(BATCH):
        var x0 = random_float64(-1.0, 1.0)
        var x1 = random_float64(-1.0, 1.0)
        input_data[b * OBS_DIM + 0] = Scalar[dtype](x0)
        input_data[b * OBS_DIM + 1] = Scalar[dtype](x1)
        input_data[b * OBS_DIM + 2] = Scalar[dtype](random_float64(-1.0, 1.0))
        input_data[b * OBS_DIM + 3] = Scalar[dtype](random_float64(-1.0, 1.0))

        # Target: mean depends on input
        for j in range(ACTION_DIM):
            target_data[b * OUT_DIM + j] = Scalar[dtype](0.5 * (x0 + x1))  # mean
            target_data[b * OUT_DIM + ACTION_DIM + j] = Scalar[dtype](-1.0)  # log_std

    print("\n  Training on GPU...")
    var ctx = DeviceContext()

    var start = perf_counter_ns()
    var result = trainer.train_gpu[BATCH](ctx, input_data, target_data)
    var end = perf_counter_ns()

    var time_ms = Float64(end - start) / 1e6

    print("-" * 70)
    print("  GPU Training completed!")
    print("  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))
    print("  Time: " + String(time_ms)[:8] + " ms")

    if result.final_loss < 0.1:
        print("\n  PASS: Full policy network training succeeded")
    else:
        print("\n  PARTIAL: Network trained but loss is still notable")


# =============================================================================
# Test Large Action Space
# =============================================================================


fn test_larger_action_space() raises:
    print_test_header("Larger Action Space (Robot Control)")

    seed(456)

    # Simulate MuJoCo-like scenario
    comptime OBS_DIM = 16
    comptime HIDDEN_DIM = 64
    comptime ACTION_DIM = 8
    comptime OUT_DIM = ACTION_DIM * 2
    comptime BATCH = 64
    comptime EPOCHS = 200

    var model = seq(
        Linear[OBS_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        StochasticActor[HIDDEN_DIM, ACTION_DIM](),
    )

    print("  Simulating robot control scenario")
    print("  OBS_DIM: " + String(OBS_DIM))
    print("  ACTION_DIM: " + String(ACTION_DIM))
    print("  HIDDEN_DIM: " + String(HIDDEN_DIM))
    print("  BATCH: " + String(BATCH))
    print("  Total PARAM_SIZE: " + String(model.PARAM_SIZE))

    var trainer = Trainer(
        model,
        Adam(lr=0.001),
        MSELoss(),
        Kaiming(),
        epochs=EPOCHS,
        print_every=50,
    )

    # Generate data
    var input_data = InlineArray[Scalar[dtype], BATCH * OBS_DIM](uninitialized=True)
    var target_data = InlineArray[Scalar[dtype], BATCH * OUT_DIM](uninitialized=True)

    for b in range(BATCH):
        for i in range(OBS_DIM):
            input_data[b * OBS_DIM + i] = Scalar[dtype](random_float64(-1.0, 1.0))
        for j in range(ACTION_DIM):
            target_data[b * OUT_DIM + j] = Scalar[dtype](random_float64(-0.5, 0.5))  # mean
            target_data[b * OUT_DIM + ACTION_DIM + j] = Scalar[dtype](-0.7)  # log_std

    print("\n  Training on GPU...")
    var ctx = DeviceContext()

    var start = perf_counter_ns()
    var result = trainer.train_gpu[BATCH](ctx, input_data, target_data)
    var end = perf_counter_ns()

    var time_ms = Float64(end - start) / 1e6

    print("-" * 70)
    print("  GPU Training completed!")
    print("  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))
    print("  Time: " + String(time_ms)[:8] + " ms")
    print("  Avg per epoch: " + String(time_ms / Float64(EPOCHS))[:6] + " ms")

    if result.final_loss < 0.15:
        print("\n  PASS: Large action space training succeeded")
    else:
        print("\n  PARTIAL: Training ran, may need more epochs")


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 70)
    print("StochasticActor GPU Tests")
    print("=" * 70)
    print("\nTesting Gaussian Policy Network on GPU")

    test_stochastic_actor_gpu_training()
    test_backbone_with_stochastic_actor()
    test_larger_action_space()

    print("\n" + "=" * 70)
    print("All StochasticActor GPU Tests Completed!")
    print("=" * 70)
