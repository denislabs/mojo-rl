"""Benchmark Deep RL Performance.

Measures training step performance to identify optimization opportunities.

Run with:
    pixi run mojo run -I . examples/benchmark_deep_rl.mojo
"""

from std.time import perf_counter_ns
from std.random import random_float64
from mojo_rl.deep_agents import DeepDDPGAgent
from mojo_rl.envs.pendulum import PendulumEnv


fn benchmark_train_step() raises:
    """Benchmark the do_train_step performance."""
    print("=" * 60)
    print("Benchmarking DeepDDPGAgent.do_train_step()")
    print("=" * 60)

    comptime obs_dim = 3
    comptime action_dim = 1
    comptime hidden_dim = 128
    comptime buffer_capacity = 10000
    comptime batch_size = 64

    var agent = DeepDDPGAgent[
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        actor_lr=0.001,
        critic_lr=0.001,
    ](
        gamma=0.99,
        tau=0.005,
        noise_std=0.2,
        action_scale=2.0,
    )

    print("Agent config:")
    print("  obs_dim: " + String(obs_dim))
    print("  action_dim: " + String(action_dim))
    print("  hidden_dim: " + String(hidden_dim))
    print("  batch_size: " + String(batch_size))

    # Fill buffer using List-based API
    print("\nFilling buffer with random transitions...")
    for _ in range(2000):
        var obs = List[Float64]()
        var next_obs = List[Float64]()
        var action = List[Float64]()

        for _ in range(obs_dim):
            obs.append(random_float64() * 2.0 - 1.0)
            next_obs.append(random_float64() * 2.0 - 1.0)
        action.append(random_float64() * 4.0 - 2.0)

        var reward = random_float64() * 2.0 - 1.0
        var done = random_float64() < 0.05
        agent.store_list_transition(obs, action, reward, next_obs, done)

    print("Buffer ready: " + String(agent.is_ready()))

    # Warmup
    print("\nWarming up (10 steps)...")
    for _ in range(10):
        _ = agent.do_train_step()

    # Benchmark do_train_step
    var num_steps = 100
    print("\nBenchmarking " + String(num_steps) + " train steps...")

    var start = perf_counter_ns()
    for _ in range(num_steps):
        _ = agent.do_train_step()
    var end = perf_counter_ns()

    var total_ms = Float64(end - start) / 1_000_000.0
    var per_step_ms = total_ms / Float64(num_steps)
    var steps_per_sec = 1000.0 / per_step_ms

    print("\nResults:")
    print("  Total time: " + String(total_ms)[:8] + " ms")
    print("  Per step: " + String(per_step_ms)[:6] + " ms")
    print("  Steps/sec: " + String(steps_per_sec)[:8])


fn benchmark_episode() raises:
    """Benchmark full episode performance."""
    print("\n" + "=" * 60)
    print("Benchmarking Full Episode")
    print("=" * 60)

    var env = PendulumEnv[DType.float64]()

    comptime obs_dim = 3
    comptime action_dim = 1
    comptime hidden_dim = 128
    comptime buffer_capacity = 10000
    comptime batch_size = 64

    var agent = DeepDDPGAgent[
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        actor_lr=0.001,
        critic_lr=0.001,
    ](
        gamma=0.99,
        tau=0.005,
        noise_std=0.2,
        action_scale=2.0,
    )

    # Pre-fill buffer using List-based API
    print("Pre-filling buffer...")
    for _ in range(1000):
        var obs = List[Float64]()
        var next_obs = List[Float64]()
        var action = List[Float64]()
        for _ in range(obs_dim):
            obs.append(random_float64() * 2.0 - 1.0)
            next_obs.append(random_float64() * 2.0 - 1.0)
        action.append(random_float64() * 4.0 - 2.0)
        agent.store_list_transition(
            obs, action, random_float64(), next_obs, False
        )

    # Benchmark episodes
    var num_episodes = 5
    var max_steps = 200

    print(
        "\nBenchmarking "
        + String(num_episodes)
        + " episodes ("
        + String(max_steps)
        + " steps each)..."
    )

    var total_time_ns: UInt = 0
    var total_steps = 0

    for _ in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var done = False
        var steps = 0

        var ep_start = perf_counter_ns()

        while not done and steps < max_steps:
            var action_list = agent.select_action_list(obs_list)

            var step_result = env.step_obs(action_list)
            var reward = step_result[1]
            done = step_result[2]

            var next_obs_list = step_result[0]
            agent.store_list_transition(
                obs_list, action_list, reward, next_obs_list, done
            )

            if agent.is_ready():
                _ = agent.do_train_step()

            obs_list = next_obs_list
            steps += 1

        var ep_end = perf_counter_ns()
        total_time_ns += ep_end - ep_start
        total_steps += steps

    var total_ms = Float64(total_time_ns) / 1_000_000.0
    var per_step_ms = total_ms / Float64(total_steps)
    var steps_per_sec = 1000.0 / per_step_ms

    print("\nResults:")
    print("  Total steps: " + String(total_steps))
    print("  Total time: " + String(total_ms)[:8] + " ms")
    print("  Per step (with env): " + String(per_step_ms)[:6] + " ms")
    print("  Steps/sec: " + String(steps_per_sec)[:8])

    print("\nNote: Each step includes:")
    print("  - Environment step (physics)")
    print("  - Action selection (actor forward)")
    print("  - Training step (critic + actor forward/backward)")


fn main() raises:
    print("Deep RL Performance Benchmark")
    print("=" * 60)
    print("")

    benchmark_train_step()
    benchmark_episode()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
