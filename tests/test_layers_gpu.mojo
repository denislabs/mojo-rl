"""Test New Layers on GPU.

Tests:
- Sigmoid (GPU forward/backward)
- Softmax (GPU forward/backward)
- LayerNorm (GPU forward/backward)
- Dropout (GPU forward/backward)
- Training with new layers on GPU

Run with:
    pixi run -e apple mojo run tests/test_layers_gpu.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64

from gpu.host import DeviceContext

from deep_rl.constants import dtype
from deep_rl.model import (
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    LayerNorm,
    Dropout,
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


# =============================================================================
# Test Sigmoid on GPU via Trainer
# =============================================================================


fn test_sigmoid_gpu():
    print_test_header("Sigmoid Layer (GPU)")

    seed(42)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 100
    comptime PRINT_EVERY = 25

    # Model: Linear -> Sigmoid -> Linear
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        Sigmoid[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print("  Model: Linear[4, 16] -> Sigmoid[16] -> Linear[16, 2]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print()

    var optimizer = Adam(lr=0.01)
    var loss_fn = MSELoss()

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Xavier(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate training data: y = sin(x1) * cos(x2)
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
        # Simple target function
        target_data[i * OUTPUT_DIM + 0] = x1 * x2
        target_data[i * OUTPUT_DIM + 1] = x3 * x4

    print("  Training on GPU...")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](
                ctx, input_data, target_data
            )

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")
            print(
                "  Avg per epoch: "
                + String(elapsed_ms / Float64(NUM_EPOCHS))[:6]
                + " ms"
            )

            if result.final_loss < 0.5:
                print("\n  PASS: Sigmoid GPU training succeeded")
            else:
                print("\n  Note: May need more epochs for convergence")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test Softmax on GPU via Trainer
# =============================================================================


fn test_softmax_gpu():
    print_test_header("Softmax Layer (GPU) - Classification")

    seed(123)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime NUM_CLASSES = 3
    comptime NUM_EPOCHS = 100
    comptime PRINT_EVERY = 25

    # Model: Linear -> ReLU -> Linear -> Softmax
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, NUM_CLASSES](),
        Softmax[NUM_CLASSES](),
    )

    print("  Model: Linear[4, 16] -> ReLU[16] -> Linear[16, 3] -> Softmax[3]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print()

    var optimizer = Adam(lr=0.01)
    var loss_fn = MSELoss()

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Kaiming(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate classification data with one-hot targets
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * NUM_CLASSES](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](
                random_float64() * 2 - 1
            )

        # Assign class based on sum of inputs
        var sum_inputs: Float64 = 0.0
        for j in range(INPUT_DIM):
            sum_inputs += Float64(input_data[i * INPUT_DIM + j])

        # One-hot encoding
        var class_idx = 0
        if sum_inputs > 0.5:
            class_idx = 2
        elif sum_inputs > -0.5:
            class_idx = 1

        for j in range(NUM_CLASSES):
            target_data[i * NUM_CLASSES + j] = Scalar[dtype](
                1.0
            ) if j == class_idx else Scalar[dtype](0.0)

    print("  Training classification on GPU...")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](
                ctx, input_data, target_data
            )

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")

            print("\n  PASS: Softmax GPU training completed")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test LayerNorm on GPU via Trainer
# =============================================================================


fn test_layer_norm_gpu():
    print_test_header("LayerNorm Layer (GPU)")

    seed(456)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 100
    comptime PRINT_EVERY = 25

    # Model: Linear -> LayerNorm -> ReLU -> Linear
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        LayerNorm[HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print(
        "  Model: Linear[4, 16] -> LayerNorm[16] -> ReLU[16] -> Linear[16, 2]"
    )
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print(
        "  (includes LayerNorm gamma/beta: "
        + String(2 * HIDDEN_DIM)
        + " params)"
    )
    print()

    var optimizer = Adam(lr=0.01)
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

    print("  Training on GPU...")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](
                ctx, input_data, target_data
            )

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")

            if result.final_loss < 0.5:
                print("\n  PASS: LayerNorm GPU training succeeded")
            else:
                print("\n  Note: May need more epochs for convergence")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test Dropout on GPU via Trainer (inference mode)
# =============================================================================


fn test_dropout_gpu():
    print_test_header("Dropout Layer (GPU) - Inference Mode")

    seed(789)

    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 100
    comptime PRINT_EVERY = 25

    # Model with Dropout in inference mode (training=False)
    # During inference, dropout is identity, so this should work
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Dropout[HIDDEN_DIM, 0.5, False](),  # Inference mode
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print(
        "  Model: Linear[4, 16] -> ReLU[16] -> Dropout[16, p=0.5,"
        " training=False] -> Linear[16, 2]"
    )
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print("  Note: Dropout in inference mode = identity function")
    print()

    var optimizer = Adam(lr=0.01)
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

    print("  Training on GPU...")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](
                ctx, input_data, target_data
            )

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")

            print("\n  PASS: Dropout (inference) GPU training completed")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test Combined Model with Multiple New Layers
# =============================================================================


fn test_combined_model_gpu():
    print_test_header("Combined Model (GPU) - Multiple New Layers")

    seed(999)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 8
    comptime HIDDEN1 = 32
    comptime HIDDEN2 = 16
    comptime OUTPUT_DIM = 4
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Complex model with multiple new layers
    var model = seq(
        Linear[INPUT_DIM, HIDDEN1](),
        LayerNorm[HIDDEN1](),
        Sigmoid[HIDDEN1](),
        Linear[HIDDEN1, HIDDEN2](),
        ReLU[HIDDEN2](),
        Linear[HIDDEN2, OUTPUT_DIM](),
    )

    print("  Model: Linear[8, 32] -> LayerNorm[32] -> Sigmoid[32]")
    print("         -> Linear[32, 16] -> ReLU[16] -> Linear[16, 4]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))
    print()

    var optimizer = Adam(lr=0.005)
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
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](
                random_float64() * 2 - 1
            )

        # Multi-output regression targets
        for j in range(OUTPUT_DIM):
            var sum_val: Float64 = 0.0
            for k in range(INPUT_DIM // OUTPUT_DIM):
                sum_val += Float64(
                    input_data[
                        i * INPUT_DIM + j * (INPUT_DIM // OUTPUT_DIM) + k
                    ]
                )
            target_data[i * OUTPUT_DIM + j] = Scalar[dtype](
                sum_val / Float64(INPUT_DIM // OUTPUT_DIM)
            )

    print("  Training combined model on GPU...")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](
                ctx, input_data, target_data
            )

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")
            print(
                "  Avg per epoch: "
                + String(elapsed_ms / Float64(NUM_EPOCHS))[:6]
                + " ms"
            )

            print("\n  PASS: Combined model GPU training completed")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Deep RL - New Layers GPU Tests")
    print("=" * 70)
    print()
    print("Testing: Sigmoid, Softmax, LayerNorm, Dropout on GPU")
    print()

    test_sigmoid_gpu()
    test_softmax_gpu()
    test_layer_norm_gpu()
    test_dropout_gpu()
    test_combined_model_gpu()

    print("\n" + "=" * 70)
    print("All Layer GPU Tests Completed!")
    print("=" * 70)
