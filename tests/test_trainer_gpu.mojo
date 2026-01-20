"""Test GPU Training with Modular Trainer Design.

This demonstrates:
- Using the Trainer struct with train_gpu() method
- Modular Model trait with Sequential composition (Seq2)
- LossFunction trait (MSELoss) with GPU support
- Optimizer trait (Adam) with GPU support

We train a 2-layer MLP on a simple regression task: y = x1 * x2

Run with:
    pixi run -e apple mojo run test_trainer_gpu.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64

from gpu.host import DeviceContext

from deep_rl.constants import dtype
from deep_rl.model import Linear, ReLU, Tanh, seq
from deep_rl.loss import MSELoss
from deep_rl.optimizer import Adam
from deep_rl.training import Trainer
from deep_rl.initializer import Xavier


# =============================================================================
# Constants
# =============================================================================

comptime BATCH_SIZE = 1024
comptime INPUT_DIM = 2
comptime HIDDEN_DIM = 1024  # Testing with heap-allocated Trainer
comptime OUTPUT_DIM = 1

comptime NUM_EPOCHS = 500
comptime PRINT_EVERY = 50


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("GPU Trainer Test - Modular Design")
    print("=" * 70)
    print()
    print(
        "Network: "
        + String(INPUT_DIM)
        + " -> "
        + String(HIDDEN_DIM)
        + " (Tanh) -> "
        + String(OUTPUT_DIM)
    )
    print("Task: Learn y = x1 * x2 (product function)")
    print("Batch size: " + String(BATCH_SIZE))
    print()

    # =========================================================================
    # Create model using Sequential composition
    # =========================================================================
    # Model: Linear(2 -> 32) -> Tanh -> Linear(32 -> 1)

    # Compose: Seq2[Seq2[Linear, Tanh], Linear]

    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        Tanh[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    # Print model info
    print("Model created:")
    print("  IN_DIM: " + String(model.IN_DIM))
    print("  OUT_DIM: " + String(model.OUT_DIM))
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print("  CACHE_SIZE: " + String(model.CACHE_SIZE))
    print()

    # =========================================================================
    # Create optimizer and loss function
    # =========================================================================

    var optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    var loss_fn = MSELoss()

    # =========================================================================
    # Create Trainer
    # =========================================================================

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Xavier(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    print("Trainer created with Xavier initialization")
    print()

    # =========================================================================
    # Generate training data: y = x1 * x2
    # =========================================================================

    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        var x1 = Scalar[dtype](random_float64() * 2 - 1)  # [-1, 1]
        var x2 = Scalar[dtype](random_float64() * 2 - 1)  # [-1, 1]
        input_data[i * INPUT_DIM + 0] = x1
        input_data[i * INPUT_DIM + 1] = x2
        target_data[i] = x1 * x2

    print("Training data generated: " + String(BATCH_SIZE) + " samples")
    print()

    # =========================================================================
    # Train on GPU
    # =========================================================================

    print("Training on GPU...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    with DeviceContext() as ctx:
        var result = trainer.train_gpu[BATCH_SIZE](
            ctx,
            input_data,
            target_data,
        )

        var end_time = perf_counter_ns()
        var elapsed_ms = Float64(end_time - start_time) / 1e6

        print("-" * 70)
        print()
        print("Training completed!")
        print("  Final loss: " + String(result.final_loss))
        print("  Epochs trained: " + String(result.epochs_trained))
        print("  Total time: " + String(elapsed_ms)[:8] + " ms")
        print(
            "  Average time per epoch: "
            + String(elapsed_ms / Float64(NUM_EPOCHS))[:6]
            + " ms"
        )
        print()

        # =====================================================================
        # Evaluate on test data
        # =====================================================================

        print("=" * 70)
        print("Final Evaluation")
        print("=" * 70)
        print()

        # Generate fresh test data
        for i in range(BATCH_SIZE):
            var x1 = Scalar[dtype](random_float64() * 2 - 1)
            var x2 = Scalar[dtype](random_float64() * 2 - 1)
            input_data[i * INPUT_DIM + 0] = x1
            input_data[i * INPUT_DIM + 1] = x2
            target_data[i] = x1 * x2

        var test_loss = trainer.evaluate[BATCH_SIZE](input_data, target_data)
        print()
        print("Test MSE Loss: " + String(test_loss))

        print()
        print("Sample predictions (x1, x2) -> predicted vs actual:")
        for i in range(5):
            var x1 = Float64(input_data[i * INPUT_DIM + 0])
            var x2 = Float64(input_data[i * INPUT_DIM + 1])
            var actual = Float64(target_data[i])
            print(
                "  ("
                + String(x1)[:6]
                + ", "
                + String(x2)[:6]
                + ") -> actual: "
                + String(actual)[:7]
            )

    print()
    print("=" * 70)
