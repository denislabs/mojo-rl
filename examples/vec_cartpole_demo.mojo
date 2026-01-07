"""Demo and benchmark for VecCartPoleEnv.

This example demonstrates the vectorized CartPole environment and compares
its performance against running a single CartPoleEnv multiple times.

Run with:
    pixi run mojo run examples/vec_cartpole_demo.mojo
"""

from time import perf_counter_ns
from random import random_float64, seed
from envs import VecCartPoleEnv, CartPoleEnv


fn run_vectorized_benchmark[
    num_envs: Int
](num_steps: Int) raises -> Tuple[Int, Float64]:
    """Run benchmark using VecCartPoleEnv.

    Parameters:
        num_envs: Number of parallel environments.

    Args:
        num_steps: Total number of environment steps to run.

    Returns:
        Tuple of (total_episodes_completed, elapsed_time_seconds).
    """
    var env = VecCartPoleEnv[num_envs]()
    _ = env.reset_vec()

    var total_episodes = 0
    var start_time = perf_counter_ns()

    for _ in range(num_steps):
        # Generate random actions for all environments
        var actions = SIMD[DType.int32, num_envs]()
        for i in range(num_envs):
            actions[i] = Int32(random_float64() * 2.0)

        var result = env.step_vec(actions)

        # Count completed episodes
        for i in range(num_envs):
            if result.dones[i]:
                total_episodes += 1

    var end_time = perf_counter_ns()
    var elapsed_seconds = Float64(end_time - start_time) / 1_000_000_000.0

    return (total_episodes, elapsed_seconds)


fn run_vectorized_raw_benchmark[
    num_envs: Int
](num_steps: Int) raises -> Tuple[Int, Float64]:
    """Run benchmark using VecCartPoleEnv.step_vec_raw (no observation collection).

    This tests the raw physics throughput without the observation list overhead.

    Parameters:
        num_envs: Number of parallel environments.

    Args:
        num_steps: Total number of environment steps to run.

    Returns:
        Tuple of (total_episodes_completed, elapsed_time_seconds).
    """
    var env = VecCartPoleEnv[num_envs]()
    _ = env.reset_vec()

    var total_episodes = 0
    var start_time = perf_counter_ns()

    for _ in range(num_steps):
        # Generate random actions for all environments
        var actions = SIMD[DType.int32, num_envs]()
        @parameter
        for i in range(num_envs):
            actions[i] = Int32(random_float64() * 2.0)

        var result = env.step_vec_raw(actions)
        var dones = result[1]

        # Count completed episodes using SIMD reduce
        total_episodes += Int(dones.cast[DType.int32]().reduce_add())

    var end_time = perf_counter_ns()
    var elapsed_seconds = Float64(end_time - start_time) / 1_000_000_000.0

    return (total_episodes, elapsed_seconds)


fn run_vectorized_pure_benchmark[
    num_envs: Int
](num_steps: Int) raises -> Tuple[Int, Float64]:
    """Run pure benchmark - fixed actions, no counting overhead.

    This measures the absolute maximum throughput of the physics engine.

    Parameters:
        num_envs: Number of parallel environments.

    Args:
        num_steps: Total number of environment steps to run.

    Returns:
        Tuple of (total_episodes_completed, elapsed_time_seconds).
    """
    var env = VecCartPoleEnv[num_envs]()
    _ = env.reset_vec()

    # Pre-generate a fixed action pattern (alternating left/right)
    var actions = SIMD[DType.int32, num_envs]()
    @parameter
    for i in range(num_envs):
        actions[i] = Int32(i % 2)

    var start_time = perf_counter_ns()

    for _ in range(num_steps):
        _ = env.step_vec_raw(actions)

    var end_time = perf_counter_ns()
    var elapsed_seconds = Float64(end_time - start_time) / 1_000_000_000.0

    # Approximate episodes (can't count without overhead)
    return (0, elapsed_seconds)


fn run_scalar_benchmark(num_envs: Int, num_steps: Int) raises -> Tuple[Int, Float64]:
    """Run benchmark using a single CartPoleEnv (simulating N envs sequentially).

    This runs N sequential steps for each "parallel" step to simulate
    the same amount of work as the vectorized version.

    Args:
        num_envs: Number of environments to simulate.
        num_steps: Total number of environment steps to run.

    Returns:
        Tuple of (total_episodes_completed, elapsed_time_seconds).
    """
    var env = CartPoleEnv()
    _ = env.reset()

    var total_episodes = 0
    var start_time = perf_counter_ns()

    # For each "vectorized" step, we do num_envs sequential steps
    for _ in range(num_steps):
        for _ in range(num_envs):
            # Random action
            var action = Int(random_float64() * 2.0)
            var result = env.step_raw(action)
            var done = result[2]

            if done:
                total_episodes += 1
                _ = env.reset()

    var end_time = perf_counter_ns()
    var elapsed_seconds = Float64(end_time - start_time) / 1_000_000_000.0

    return (total_episodes, elapsed_seconds)


fn print_separator():
    print("=" * 60)


fn main() raises:
    seed()  # Initialize random seed

    print_separator()
    print("VecCartPoleEnv Demo and Benchmark")
    print_separator()
    print()

    # =========================================================================
    # Part 1: Basic Usage Demo
    # =========================================================================
    print("Part 1: Basic Usage")
    print("-" * 40)

    var env = VecCartPoleEnv[8]()
    print("Created VecCartPoleEnv with", env.get_num_envs(), "parallel environments")

    # Reset all environments
    var obs = env.reset_vec()
    print("Reset all environments")
    print("Number of observations:", len(obs))
    print("First observation:", obs[0])

    # Take a step with random actions
    var actions = SIMD[DType.int32, 8](0, 1, 0, 1, 1, 0, 1, 0)
    print("Taking step with actions:", actions)

    var result = env.step_vec(actions)
    print("Rewards:", result.rewards)
    print("Dones:", result.dones)
    print()

    # Run for a few steps to show auto-reset
    print("Running 100 steps to demonstrate auto-reset...")
    var episodes_completed = 0
    for step in range(100):
        # Random actions
        var rand_actions = SIMD[DType.int32, 8]()
        for i in range(8):
            rand_actions[i] = Int32(random_float64() * 2.0)

        var step_result = env.step_vec(rand_actions)

        # Count completed episodes
        for i in range(8):
            if step_result.dones[i]:
                episodes_completed += 1

    print("Episodes completed in 100 steps:", episodes_completed)
    print()

    # =========================================================================
    # Part 2: Performance Benchmark
    # =========================================================================
    print_separator()
    print("Part 2: Performance Benchmark")
    print_separator()
    print()

    comptime NUM_ENVS = 8
    var num_steps = 10000

    print("Configuration:")
    print("  Number of environments:", NUM_ENVS)
    print("  Steps per batch:", num_steps)
    print("  Total env interactions:", NUM_ENVS * num_steps)
    print()

    # Warm-up runs
    print("Warming up...")
    _ = run_vectorized_benchmark[NUM_ENVS](1000)
    _ = run_vectorized_raw_benchmark[NUM_ENVS](1000)
    _ = run_vectorized_pure_benchmark[NUM_ENVS](1000)
    _ = run_scalar_benchmark(NUM_ENVS, 1000)
    print()

    # Benchmark vectorized version (with observation collection)
    print("Running vectorized benchmark (with observations)...")
    var vec_result = run_vectorized_benchmark[NUM_ENVS](num_steps)
    var vec_episodes = vec_result[0]
    var vec_time = vec_result[1]

    # Benchmark vectorized version (raw, no observations)
    print("Running vectorized benchmark (raw, no observations)...")
    var vec_raw_result = run_vectorized_raw_benchmark[NUM_ENVS](num_steps)
    var vec_raw_episodes = vec_raw_result[0]
    var vec_raw_time = vec_raw_result[1]

    # Benchmark vectorized version (pure, no overhead)
    print("Running vectorized benchmark (pure, fixed actions)...")
    var vec_pure_result = run_vectorized_pure_benchmark[NUM_ENVS](num_steps)
    var vec_pure_time = vec_pure_result[1]

    # Benchmark scalar version (sequential)
    print("Running scalar benchmark (sequential)...")
    var scalar_result = run_scalar_benchmark(NUM_ENVS, num_steps)
    var scalar_episodes = scalar_result[0]
    var scalar_time = scalar_result[1]

    # Calculate metrics
    var total_steps = NUM_ENVS * num_steps
    var vec_steps_per_sec = Float64(total_steps) / vec_time
    var vec_raw_steps_per_sec = Float64(total_steps) / vec_raw_time
    var vec_pure_steps_per_sec = Float64(total_steps) / vec_pure_time
    var scalar_steps_per_sec = Float64(total_steps) / scalar_time
    var speedup = scalar_time / vec_time
    var raw_speedup = scalar_time / vec_raw_time
    var pure_speedup = scalar_time / vec_pure_time

    print()
    print("-" * 40)
    print("Results:")
    print("-" * 40)
    print()

    print("Vectorized with observations (step_vec):")
    print("  Time:", vec_time, "seconds")
    print("  Episodes completed:", vec_episodes)
    print("  Steps/second:", Int(vec_steps_per_sec))
    print()

    print("Vectorized raw (step_vec_raw + random actions):")
    print("  Time:", vec_raw_time, "seconds")
    print("  Episodes completed:", vec_raw_episodes)
    print("  Steps/second:", Int(vec_raw_steps_per_sec))
    print()

    print("Vectorized pure (step_vec_raw + fixed actions):")
    print("  Time:", vec_pure_time, "seconds")
    print("  Steps/second:", Int(vec_pure_steps_per_sec))
    print()

    print("Scalar (sequential CartPoleEnv):")
    print("  Time:", scalar_time, "seconds")
    print("  Episodes completed:", scalar_episodes)
    print("  Steps/second:", Int(scalar_steps_per_sec))
    print()

    print("-" * 40)
    print("SPEEDUP vs Scalar:")
    print("  With observations:", speedup, "x")
    print("  Raw:", raw_speedup, "x")
    print("  Pure:", pure_speedup, "x")
    print("-" * 40)

    if pure_speedup > 1.0:
        print("Pure vectorized version is FASTER than scalar!")
    else:
        print("Scalar version is still faster")

    print()
    print_separator()
    print("Demo complete!")
    print_separator()
