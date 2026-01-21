"""Benchmark test comparing LinearReLU (fused) vs Linear + ReLU (unfused)."""

from time import perf_counter_ns
from random import seed, random_float64
from math import sin

from gpu.host import DeviceContext

from deep_rl.constants import dtype
from deep_rl.model import Linear, ReLU, LinearReLU, seq
from deep_rl.loss import MSELoss
from deep_rl.optimizer import Adam
from deep_rl.training import Trainer
from deep_rl.initializer import Xavier


# Configuration
comptime BATCH = 1024
comptime IN_DIM = 64
comptime HIDDEN_DIM = 256
comptime OUT_DIM = 64
comptime EPOCHS = 200
comptime PRINT_EVERY = 50


def main():
    seed(42)

    print("=" * 70)
    print("LinearReLU Fusion Benchmark")
    print("=" * 70)
    print()

    print("Network configuration:")
    print("  Input: " + String(IN_DIM))
    print("  Hidden: " + String(HIDDEN_DIM))
    print("  Output: " + String(OUT_DIM))
    print("  Batch size: " + String(BATCH))
    print("  Epochs: " + String(EPOCHS))
    print()

    # Generate training data
    var input_data = InlineArray[Scalar[dtype], BATCH * IN_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH * OUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH):
        var sum_val: Float64 = 0
        for j in range(IN_DIM):
            var val = random_float64() * 2 - 1  # [-1, 1]
            input_data[i * IN_DIM + j] = Scalar[dtype](val)
            sum_val += val
        var sin_val = sin(sum_val * 0.1)
        for j in range(OUT_DIM):
            var noise = random_float64() * 0.2 - 0.1
            target_data[i * OUT_DIM + j] = Scalar[dtype](sin_val + noise)

    print("Training data generated: " + String(BATCH) + " samples")
    print()

    # =========================================================================
    # Test 1: Unfused (Linear + ReLU + Linear + ReLU + Linear)
    # =========================================================================
    print("-" * 70)
    print("Test 1: Unfused (Linear + ReLU + Linear + ReLU + Linear)")
    print("-" * 70)

    var unfused_model = seq(
        Linear[IN_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUT_DIM](),
    )

    print("Unfused model:")
    print("  IN_DIM: " + String(unfused_model.IN_DIM))
    print("  OUT_DIM: " + String(unfused_model.OUT_DIM))
    print("  PARAM_SIZE: " + String(unfused_model.PARAM_SIZE))
    print("  CACHE_SIZE: " + String(unfused_model.CACHE_SIZE))
    print()

    var unfused_trainer = Trainer(
        unfused_model,
        Adam(lr=0.001),
        MSELoss(),
        Xavier(),
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    print("Training unfused model on GPU...")
    var ctx = DeviceContext()
    var start_unfused = perf_counter_ns()
    var unfused_result = unfused_trainer.train_gpu[BATCH](
        ctx, input_data, target_data
    )
    var end_unfused = perf_counter_ns()
    var unfused_time_ms = Float64(end_unfused - start_unfused) / 1_000_000.0

    print()
    print("Unfused Results:")
    print("  Final loss: " + String(unfused_result.final_loss))
    print("  Total time: " + String(unfused_time_ms) + " ms")
    print("  Time per epoch: " + String(unfused_time_ms / EPOCHS) + " ms")
    print()

    # =========================================================================
    # Test 2: Fused (LinearReLU + LinearReLU + Linear)
    # =========================================================================
    print("-" * 70)
    print("Test 2: Fused (LinearReLU + LinearReLU + Linear)")
    print("-" * 70)

    var fused_model = seq(
        LinearReLU[IN_DIM, HIDDEN_DIM](),
        LinearReLU[HIDDEN_DIM, HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUT_DIM](),
    )

    print("Fused model:")
    print("  IN_DIM: " + String(fused_model.IN_DIM))
    print("  OUT_DIM: " + String(fused_model.OUT_DIM))
    print("  PARAM_SIZE: " + String(fused_model.PARAM_SIZE))
    print("  CACHE_SIZE: " + String(fused_model.CACHE_SIZE))
    print()

    var fused_trainer = Trainer(
        fused_model,
        Adam(lr=0.001),
        MSELoss(),
        Xavier(),
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    print("Training fused model on GPU...")
    var start_fused = perf_counter_ns()
    var fused_result = fused_trainer.train_gpu[BATCH](ctx, input_data, target_data)
    var end_fused = perf_counter_ns()
    var fused_time_ms = Float64(end_fused - start_fused) / 1_000_000.0

    print()
    print("Fused Results:")
    print("  Final loss: " + String(fused_result.final_loss))
    print("  Total time: " + String(fused_time_ms) + " ms")
    print("  Time per epoch: " + String(fused_time_ms / EPOCHS) + " ms")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("  Unfused time: " + String(unfused_time_ms) + " ms")
    print("  Fused time:   " + String(fused_time_ms) + " ms")
    var speedup = unfused_time_ms / fused_time_ms
    var improvement = (1.0 - fused_time_ms / unfused_time_ms) * 100
    print("  Speedup:      " + String(speedup) + "x")
    print("  Improvement:  " + String(improvement) + "%")
    print("=" * 70)
