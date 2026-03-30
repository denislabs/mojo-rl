"""Benchmark: CUDA Graph capture on Trainer.train_gpu.

Compares direct GPU training vs graph-captured training on a simple
feedforward network. The Trainer's GPU loop has a fixed kernel sequence
per epoch (forward → loss_backward → backward → optimizer_step),
making it an ideal CUDA graph target.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_cuda_graph_trainer.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.mse import MSELoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.training.gpu_network_state import GPUNetworkState
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.cuda import CUDAGraph


def main() raises:
    print("=== CUDA Graph Trainer Benchmark ===\n")

    seed(42)
    var ctx = DeviceContext()

    # --- Network: 4 → 64 → ReLU → 64 → ReLU → 2 ---
    comptime IN_DIM = 4
    comptime HIDDEN = 64
    comptime OUT_DIM = 2
    comptime BATCH = 32

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
        # Simple target: sum of inputs
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
    # Part 1: Direct GPU training (baseline)
    # =================================================================
    print("\n--- Part 1: Direct GPU Training (1000 epochs) ---")

    var state1 = TRAINER.init_state_gpu[Kaiming[]](ctx)
    ctx.synchronize()

    # Warmup
    _ = TRAINER.train_gpu[BATCH](state1, ctx, input_t, target_t, epochs=10)
    ctx.synchronize()

    var start = perf_counter_ns()
    var result1 = TRAINER.train_gpu[BATCH](
        state1, ctx, input_t, target_t, epochs=1000
    )
    ctx.synchronize()
    var time_direct = Float64(perf_counter_ns() - start) / 1e6

    print("  Final loss:", result1.final_loss)
    print("  Time:", String(time_direct)[byte=:8], "ms")
    print("  Per epoch:", String(time_direct / 1000.0)[byte=:8], "ms")

    # =================================================================
    # Part 2: CUDA Graph captured training
    # =================================================================
    print("\n--- Part 2: CUDA Graph Training (1000 epochs) ---")

    var state2 = TRAINER.init_state_gpu[Kaiming[]](ctx)
    ctx.synchronize()

    # Warmup (also discovers the Mojo stream)
    _ = TRAINER.train_gpu[BATCH](state2, ctx, input_t, target_t, epochs=10)
    ctx.synchronize()

    # Capture one epoch
    var graph = CUDAGraph(ctx)
    graph.begin_capture()
    _ = TRAINER.train_gpu[BATCH](state2, ctx, input_t, target_t, epochs=1)
    graph.end_capture()

    print("  Captured graph with", graph.num_nodes(), "nodes")

    # Replay for remaining epochs
    # Warmup the graph replay
    for _ in range(10):
        graph.replay()

    start = perf_counter_ns()
    for _ in range(1000):
        graph.replay()
    var time_graph = Float64(perf_counter_ns() - start) / 1e6

    # Check final loss by running one direct epoch to read it
    var result2 = TRAINER.train_gpu[BATCH](
        state2, ctx, input_t, target_t, epochs=1, print_every=1
    )

    print("  Final loss:", result2.final_loss)
    print("  Time:", String(time_graph)[byte=:8], "ms")
    print("  Per epoch:", String(time_graph / 1000.0)[byte=:8], "ms")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("  Direct:     ", String(time_direct)[byte=:8], "ms (1000 epochs)")
    print("  Graph:      ", String(time_graph)[byte=:8], "ms (1000 replays)")
    if time_graph > 0.0:
        print("  Speedup:    ", String(time_direct / time_graph)[byte=:6], "x")
    print("=" * 50)
