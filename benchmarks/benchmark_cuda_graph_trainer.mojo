"""Benchmark: CUDA Graph capture on Trainer.train_gpu.

Compares direct GPU training vs graph-captured training on a simple
feedforward network using the USE_CUDA_GRAPH comptime parameter.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_cuda_graph_trainer.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.mse import MSELoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming


def main() raises:
    print("=== CUDA Graph Trainer Benchmark ===\n")

    seed(42)
    var ctx = DeviceContext()

    comptime IN_DIM = 4
    comptime HIDDEN = 64
    comptime OUT_DIM = 2
    comptime BATCH = 32
    comptime EPOCHS = 1000

    comptime MODEL = Sequential[
        Linear[IN_DIM, HIDDEN],
        ReLU[HIDDEN],
        Linear[HIDDEN, HIDDEN],
        ReLU[HIDDEN],
        Linear[HIDDEN, OUT_DIM],
    ]
    comptime TRAINER = Trainer[MODEL, Adam[], MSELoss]

    print(
        "  Network: "
        + String(IN_DIM)
        + " → "
        + String(HIDDEN)
        + " → ReLU → "
        + String(HIDDEN)
        + " → ReLU → "
        + String(OUT_DIM)
    )
    print("  Params:", MODEL.PARAM_SIZE)
    print("  Batch:", BATCH)
    print("  Epochs:", EPOCHS)

    # --- Generate synthetic data ---
    var input_data = InlineArray[Scalar[dtype], BATCH * IN_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH * OUT_DIM](
        uninitialized=True
    )
    for b in range(BATCH):
        for i in range(IN_DIM):
            input_data[b * IN_DIM + i] = Scalar[dtype](
                random_float64(-1.0, 1.0)
            )
        var s = Scalar[dtype](0)
        for i in range(IN_DIM):
            s += input_data[b * IN_DIM + i]
        target_data[b * OUT_DIM + 0] = s
        target_data[b * OUT_DIM + 1] = s * Scalar[dtype](0.5)

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](input_data.unsafe_ptr())
    var target_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](target_data.unsafe_ptr())

    # =================================================================
    # Direct GPU training (baseline)
    # =================================================================
    print("\n--- Direct GPU Training ---")

    var state1 = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # Warmup
    _ = TRAINER.train_gpu[BATCH](state1, ctx, input_t, target_t, epochs=10)
    ctx.synchronize()

    var start = perf_counter_ns()
    var result1 = TRAINER.train_gpu[BATCH](
        state1, ctx, input_t, target_t, epochs=EPOCHS
    )
    ctx.synchronize()
    var time_direct = Float64(perf_counter_ns() - start) / 1e6

    print("  Final loss:", result1.final_loss)
    print("  Time:", String(time_direct)[byte=:8], "ms")

    # =================================================================
    # CUDA Graph training
    # =================================================================
    print("\n--- CUDA Graph Training ---")

    var state2 = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # Warmup (discovers the Mojo stream)
    _ = TRAINER.train_gpu[BATCH](state2, ctx, input_t, target_t, epochs=10)
    ctx.synchronize()

    start = perf_counter_ns()
    var result2 = TRAINER.train_gpu[BATCH, USE_CUDA_GRAPH=True](
        state2, ctx, input_t, target_t, epochs=EPOCHS
    )
    ctx.synchronize()
    var time_graph = Float64(perf_counter_ns() - start) / 1e6

    print("  Final loss:", result2.final_loss)
    print("  Time:", String(time_graph)[byte=:8], "ms")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 50)
    print("SUMMARY (" + String(EPOCHS) + " epochs)")
    print("=" * 50)
    print("  Direct:  ", String(time_direct)[byte=:8], "ms")
    print("  Graph:   ", String(time_graph)[byte=:8], "ms")
    if time_graph > 0.0:
        print("  Speedup: ", String(time_direct / time_graph)[byte=:6], "x")
    print("=" * 50)
