"""Benchmark comparing kernel modes for A2C.

This benchmark compares:
1. Original 1D kernels: Each thread processes one environment
2. Optimized 2D kernels: 2D thread blocks with shared memory reductions
3. Tiled 2D kernels: Loop-based tiling for arbitrarily large HIDDEN_DIM

Run with:
    pixi run -e apple mojo run examples/benchmark_a2c_kernels.mojo

Kernel Selection Guide:
- 1D kernels: Simple, good for small HIDDEN_DIM (64-128), no shared memory limits
- 2D kernels: Fast for HIDDEN_DIM <= ~900 with ENVS_PER_BLOCK=8
- Tiled kernels: Required for HIDDEN_DIM >= 1024, uses HIDDEN_PER_BLOCK tiling

Constraints (Apple Silicon):
- Shared Memory: 32KB limit
  - 2D kernels: ENVS_PER_BLOCK * HIDDEN_DIM * 4 < 32768 bytes
  - Tiled kernels: ENVS_PER_BLOCK * HIDDEN_PER_BLOCK * 4 < 32768 bytes
- Thread Count: ~576 threads per block max for tiled kernels
  - HIDDEN_PER_BLOCK * ENVS_PER_BLOCK <= 576
  - Recommended: HIDDEN_PER_BLOCK=64, ENVS_PER_BLOCK=8 (512 threads)
"""

from deep_agents.gpu import A2CAgent
from envs import CartPoleEnv
from time import perf_counter_ns
from random import seed


fn run_benchmark_1d[
    NUM_ENVS: Int,
    HIDDEN_DIM: Int,
    ROLLOUT_LEN: Int,
](num_updates: Int, label: String) raises -> Tuple[Float64, Float64]:
    """Run benchmark with 1D kernels."""
    seed(42)

    var agent = A2CAgent[HIDDEN_DIM=HIDDEN_DIM]()

    var start = perf_counter_ns()
    var metrics = agent.train[
        CartPoleEnv,
        NUM_ENVS=NUM_ENVS,
        ROLLOUT_LEN=ROLLOUT_LEN,
        USE_2D_KERNELS=False,
        USE_TILED_KERNELS=False,
    ](
        num_updates=num_updates,
        verbose=False,
        environment_name="CartPole",
    )
    var end = perf_counter_ns()

    var elapsed_ms = Float64(end - start) / 1e6
    var total_steps = num_updates * NUM_ENVS * ROLLOUT_LEN
    var throughput = Float64(total_steps) / (Float64(end - start) / 1e9)

    print(label)
    print("  Time:", Int(elapsed_ms), "ms")
    print("  Throughput:", Int(throughput), "steps/sec")
    print("  Episodes:", len(metrics.episodes))
    if len(metrics.episodes) > 0:
        var last_rewards: Float64 = 0
        var count = min(100, len(metrics.episodes))
        for i in range(len(metrics.episodes) - count, len(metrics.episodes)):
            last_rewards += metrics.episodes[i].total_reward
        print("  Last 100 avg reward:", Int(last_rewards / Float64(count)))
    print()

    return (elapsed_ms, throughput)


fn run_benchmark_2d[
    NUM_ENVS: Int,
    HIDDEN_DIM: Int,
    ROLLOUT_LEN: Int,
    ENVS_PER_BLOCK: Int,
](num_updates: Int, label: String) raises -> Tuple[Float64, Float64]:
    """Run benchmark with 2D kernels."""
    seed(42)

    var agent = A2CAgent[HIDDEN_DIM=HIDDEN_DIM]()

    var start = perf_counter_ns()
    var metrics = agent.train[
        CartPoleEnv,
        NUM_ENVS=NUM_ENVS,
        ROLLOUT_LEN=ROLLOUT_LEN,
        ENVS_PER_BLOCK=ENVS_PER_BLOCK,
        USE_2D_KERNELS=True,
        USE_TILED_KERNELS=False,
    ](
        num_updates=num_updates,
        verbose=False,
        environment_name="CartPole",
    )
    var end = perf_counter_ns()

    var elapsed_ms = Float64(end - start) / 1e6
    var total_steps = num_updates * NUM_ENVS * ROLLOUT_LEN
    var throughput = Float64(total_steps) / (Float64(end - start) / 1e9)

    print(label)
    print("  Time:", Int(elapsed_ms), "ms")
    print("  Throughput:", Int(throughput), "steps/sec")
    print("  Episodes:", len(metrics.episodes))
    if len(metrics.episodes) > 0:
        var last_rewards: Float64 = 0
        var count = min(100, len(metrics.episodes))
        for i in range(len(metrics.episodes) - count, len(metrics.episodes)):
            last_rewards += metrics.episodes[i].total_reward
        print("  Last 100 avg reward:", Int(last_rewards / Float64(count)))
    print()

    return (elapsed_ms, throughput)


fn run_benchmark_tiled[
    NUM_ENVS: Int,
    HIDDEN_DIM: Int,
    ROLLOUT_LEN: Int,
    ENVS_PER_BLOCK: Int,
    HIDDEN_PER_BLOCK: Int,
](num_updates: Int, label: String) raises -> Tuple[Float64, Float64]:
    """Run benchmark with tiled 2D kernels."""
    seed(42)

    var agent = A2CAgent[HIDDEN_DIM=HIDDEN_DIM]()

    var start = perf_counter_ns()
    var metrics = agent.train[
        CartPoleEnv,
        NUM_ENVS=NUM_ENVS,
        ROLLOUT_LEN=ROLLOUT_LEN,
        ENVS_PER_BLOCK=ENVS_PER_BLOCK,
        HIDDEN_PER_BLOCK=HIDDEN_PER_BLOCK,
        USE_2D_KERNELS=False,
        USE_TILED_KERNELS=True,
    ](
        num_updates=num_updates,
        verbose=False,
        environment_name="CartPole",
    )
    var end = perf_counter_ns()

    var elapsed_ms = Float64(end - start) / 1e6
    var total_steps = num_updates * NUM_ENVS * ROLLOUT_LEN
    var throughput = Float64(total_steps) / (Float64(end - start) / 1e9)

    print(label)
    print("  Time:", Int(elapsed_ms), "ms")
    print("  Throughput:", Int(throughput), "steps/sec")
    print("  Episodes:", len(metrics.episodes))
    if len(metrics.episodes) > 0:
        var last_rewards: Float64 = 0
        var count = min(100, len(metrics.episodes))
        for i in range(len(metrics.episodes) - count, len(metrics.episodes)):
            last_rewards += metrics.episodes[i].total_reward
        print("  Last 100 avg reward:", Int(last_rewards / Float64(count)))
    print()

    return (elapsed_ms, throughput)


fn main() raises:
    print("=" * 70)
    print("A2C Kernel Benchmark: 1D vs 2D vs Tiled")
    print("=" * 70)
    print()

    # Benchmark parameters
    comptime NUM_UPDATES = 100
    comptime NUM_ENVS = 1024
    comptime HIDDEN_DIM = 1024  # Large HIDDEN_DIM to show tiled benefits
    comptime ROLLOUT_LEN = 128
    comptime ENVS_PER_BLOCK = 8
    comptime HIDDEN_PER_BLOCK = 64  # Tile size for tiled kernels (64 works, larger values may have issues)

    print("Configuration:")
    print("  NUM_ENVS:", NUM_ENVS)
    print("  HIDDEN_DIM:", HIDDEN_DIM)
    print("  ROLLOUT_LEN:", ROLLOUT_LEN)
    print("  NUM_UPDATES:", NUM_UPDATES)
    print("  ENVS_PER_BLOCK:", ENVS_PER_BLOCK)
    print("  HIDDEN_PER_BLOCK:", HIDDEN_PER_BLOCK)
    print()

    # Warmup with 1D kernel
    print("Warmup run...")
    _ = run_benchmark_1d[
        NUM_ENVS=NUM_ENVS,
        HIDDEN_DIM=HIDDEN_DIM,
        ROLLOUT_LEN=ROLLOUT_LEN,
    ](10, "  Warmup (1D)")
    print()

    # Benchmark 1D kernels
    print("-" * 70)
    var result_1d = run_benchmark_1d[
        NUM_ENVS=NUM_ENVS,
        HIDDEN_DIM=HIDDEN_DIM,
        ROLLOUT_LEN=ROLLOUT_LEN,
    ](NUM_UPDATES, "Original 1D Kernels:")

    # NOTE: 2D kernels would hit shared memory limit for HIDDEN_DIM=1024
    # with ENVS_PER_BLOCK=8 (1024 * 8 * 4 = 32KB, at the limit)
    # Skip 2D benchmark for large HIDDEN_DIM

    # Benchmark Tiled kernels
    print("-" * 70)
    var result_tiled = run_benchmark_tiled[
        NUM_ENVS=NUM_ENVS,
        HIDDEN_DIM=HIDDEN_DIM,
        ROLLOUT_LEN=ROLLOUT_LEN,
        ENVS_PER_BLOCK=ENVS_PER_BLOCK,
        HIDDEN_PER_BLOCK=HIDDEN_PER_BLOCK,
    ](NUM_UPDATES, "Tiled 2D Kernels:")

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(
        "Original 1D: ",
        Int(result_1d[0]),
        "ms |",
        Int(result_1d[1]),
        "steps/sec",
    )
    print(
        "Tiled 2D:    ",
        Int(result_tiled[0]),
        "ms |",
        Int(result_tiled[1]),
        "steps/sec",
    )
    print()

    var speedup = result_tiled[1] / result_1d[1]
    if speedup > 1.0:
        print("Tiled kernels are", Int(speedup * 100 - 100), "% faster")
    else:
        print("1D kernels are", Int((1.0 / speedup) * 100 - 100), "% faster")
    print()

    print()
    print("Done!")
