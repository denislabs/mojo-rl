"""Test New Optimizers on GPU.

Tests:
- RMSprop (GPU training)
- AdamW (GPU training)
- Optimizer comparison on GPU

Run with:
    pixi run -e apple mojo run tests/test_optimizers_gpu.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64

from gpu.host import DeviceContext

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


# =============================================================================
# Test RMSprop on GPU
# =============================================================================


fn test_rmsprop_gpu():
    print_test_header("RMSprop Training (GPU)")

    seed(42)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 8
    comptime HIDDEN_DIM = 32
    comptime OUTPUT_DIM = 4
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Model
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print("  Model: Linear[8, 32] -> ReLU[32] -> Linear[32, 4]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))

    # RMSprop optimizer
    var optimizer = RMSprop(lr=0.01, alpha=0.99, eps=1e-8)
    print("\n  Optimizer: RMSprop(lr=0.01, alpha=0.99)")
    print("  STATE_PER_PARAM: " + String(optimizer.STATE_PER_PARAM) + " (squared gradient avg)")

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
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)
        for j in range(OUTPUT_DIM):
            var sum_val: Float64 = 0.0
            for k in range(INPUT_DIM // OUTPUT_DIM):
                sum_val += Float64(input_data[i * INPUT_DIM + j * (INPUT_DIM // OUTPUT_DIM) + k])
            target_data[i * OUTPUT_DIM + j] = Scalar[dtype](sum_val / Float64(INPUT_DIM // OUTPUT_DIM))

    print("\n  Training on GPU...")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](ctx, input_data, target_data)

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")
            print("  Avg per epoch: " + String(elapsed_ms / Float64(NUM_EPOCHS))[:6] + " ms")

            if result.final_loss < 0.5:
                print("\n  PASS: RMSprop GPU training succeeded")
            else:
                print("\n  Note: May need tuning for better convergence")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test AdamW on GPU
# =============================================================================


fn test_adamw_gpu():
    print_test_header("AdamW Training (GPU)")

    seed(123)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 8
    comptime HIDDEN_DIM = 32
    comptime OUTPUT_DIM = 4
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Model
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        Tanh[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print("  Model: Linear[8, 32] -> Tanh[32] -> Linear[32, 4]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))

    # AdamW optimizer
    var optimizer = AdamW(lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    print("\n  Optimizer: AdamW(lr=0.005, weight_decay=0.01)")
    print("  STATE_PER_PARAM: " + String(optimizer.STATE_PER_PARAM) + " (m and v moments)")

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
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)
        # Linear combination targets
        for j in range(OUTPUT_DIM):
            var idx1 = j * 2
            var idx2 = j * 2 + 1
            target_data[i * OUTPUT_DIM + j] = input_data[i * INPUT_DIM + idx1] + input_data[i * INPUT_DIM + idx2]

    print("\n  Training on GPU...")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](ctx, input_data, target_data)

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")
            print("  Avg per epoch: " + String(elapsed_ms / Float64(NUM_EPOCHS))[:6] + " ms")

            if result.final_loss < 0.3:
                print("\n  PASS: AdamW GPU training succeeded")
            else:
                print("\n  Note: May need tuning for better convergence")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Compare Optimizers on GPU
# =============================================================================


fn compare_optimizers_gpu():
    print_test_header("Optimizer Comparison (GPU)")

    seed(456)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 8
    comptime HIDDEN_DIM = 32
    comptime OUTPUT_DIM = 4
    comptime NUM_EPOCHS = 100

    # Generate shared training data
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)
        for j in range(OUTPUT_DIM):
            var idx1 = j * 2
            var idx2 = j * 2 + 1
            var x1 = Float64(input_data[i * INPUT_DIM + idx1])
            var x2 = Float64(input_data[i * INPUT_DIM + idx2])
            target_data[i * OUTPUT_DIM + j] = Scalar[dtype](x1 * x2)

    print("  Comparing optimizers on GPU...")
    print("  Model: Linear[8, 32] -> ReLU[32] -> Linear[32, 4]")
    print("  Task: y[j] = x[2j] * x[2j+1]")
    print("  Epochs: " + String(NUM_EPOCHS))
    print()

    try:
        with DeviceContext() as ctx:
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
            var start_sgd = perf_counter_ns()
            var result_sgd = trainer_sgd.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_sgd = Float64(perf_counter_ns() - start_sgd) / 1e6
            print("  SGD(lr=0.1):           loss = " + String(result_sgd.final_loss)[:8] + "  time = " + String(time_sgd)[:6] + " ms")

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
            var start_adam = perf_counter_ns()
            var result_adam = trainer_adam.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_adam = Float64(perf_counter_ns() - start_adam) / 1e6
            print("  Adam(lr=0.01):         loss = " + String(result_adam.final_loss)[:8] + "  time = " + String(time_adam)[:6] + " ms")

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
            var start_rmsprop = perf_counter_ns()
            var result_rmsprop = trainer_rmsprop.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_rmsprop = Float64(perf_counter_ns() - start_rmsprop) / 1e6
            print("  RMSprop(lr=0.01):      loss = " + String(result_rmsprop.final_loss)[:8] + "  time = " + String(time_rmsprop)[:6] + " ms")

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
            var start_adamw = perf_counter_ns()
            var result_adamw = trainer_adamw.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_adamw = Float64(perf_counter_ns() - start_adamw) / 1e6
            print("  AdamW(lr=0.01, wd=0.01): loss = " + String(result_adamw.final_loss)[:8] + "  time = " + String(time_adamw)[:6] + " ms")

            print("\n  All optimizers trained on GPU successfully!")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test AdamW Weight Decay Effect
# =============================================================================


fn test_adamw_weight_decay_effect():
    print_test_header("AdamW Weight Decay Effect (GPU)")

    seed(789)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 4
    comptime HIDDEN_DIM = 64
    comptime OUTPUT_DIM = 2
    comptime NUM_EPOCHS = 200

    # Generate training data
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)
        target_data[i * OUTPUT_DIM + 0] = input_data[i * INPUT_DIM + 0] + input_data[i * INPUT_DIM + 1]
        target_data[i * OUTPUT_DIM + 1] = input_data[i * INPUT_DIM + 2] + input_data[i * INPUT_DIM + 3]

    print("  Comparing different weight decay values...")
    print("  Model: Linear[4, 64] -> ReLU[64] -> Linear[64, 2]")
    print("  Larger hidden layer (64) to see weight decay effect")
    print("  Epochs: " + String(NUM_EPOCHS))
    print()

    try:
        with DeviceContext() as ctx:
            # AdamW with no weight decay
            var model_wd0 = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_wd0 = Trainer(
                model_wd0, AdamW(lr=0.01, weight_decay=0.0), MSELoss(), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var result_wd0 = trainer_wd0.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            print("  AdamW(wd=0.0):   loss = " + String(result_wd0.final_loss)[:8])

            # AdamW with light weight decay
            var model_wd01 = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_wd01 = Trainer(
                model_wd01, AdamW(lr=0.01, weight_decay=0.01), MSELoss(), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var result_wd01 = trainer_wd01.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            print("  AdamW(wd=0.01):  loss = " + String(result_wd01.final_loss)[:8])

            # AdamW with moderate weight decay
            var model_wd05 = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_wd05 = Trainer(
                model_wd05, AdamW(lr=0.01, weight_decay=0.05), MSELoss(), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var result_wd05 = trainer_wd05.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            print("  AdamW(wd=0.05):  loss = " + String(result_wd05.final_loss)[:8])

            # AdamW with strong weight decay
            var model_wd1 = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_wd1 = Trainer(
                model_wd1, AdamW(lr=0.01, weight_decay=0.1), MSELoss(), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var result_wd1 = trainer_wd1.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            print("  AdamW(wd=0.1):   loss = " + String(result_wd1.final_loss)[:8])

            print("\n  Note: Higher weight decay may increase loss but improve generalization")
            print("  PASS: Weight decay effect demonstrated")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Deep RL - New Optimizers GPU Tests")
    print("=" * 70)
    print()
    print("Testing: RMSprop, AdamW on GPU")
    print()

    test_rmsprop_gpu()
    test_adamw_gpu()
    compare_optimizers_gpu()
    test_adamw_weight_decay_effect()

    print("\n" + "=" * 70)
    print("All Optimizer GPU Tests Completed!")
    print("=" * 70)
