"""Benchmark Deep RL Performance.

Measures training step performance to identify optimization opportunities.

Run with:
    pixi run mojo run examples/benchmark_deep_rl.mojo
"""

from time import perf_counter_ns
from deep_agents import DeepDDPGAgent
from envs.pendulum import PendulumEnv


fn benchmark_train_step() raises:
    """Benchmark the train_step performance."""
    print("=" * 60)
    print("Benchmarking DeepDDPGAgent.train_step()")
    print("=" * 60)

    # Create agent with typical dimensions
    comptime obs_dim = 3
    comptime action_dim = 1
    comptime hidden_dim = 128
    comptime buffer_capacity = 10000
    comptime batch_size = 64

    var agent = DeepDDPGAgent[
        obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size
    ](
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.001,
        noise_std=0.2,
        action_scale=2.0,
    )

    print("Agent config:")
    print("  Hidden dim: " + String(hidden_dim))
    print("  Batch size: " + String(batch_size))
    print(
        "  Actor params: "
        + String(
            agent.actor.layer1.num_parameters()
            + agent.actor.layer2.num_parameters()
            + agent.actor.layer3.num_parameters()
        )
    )
    print(
        "  Critic params: "
        + String(
            agent.critic.layer1.num_parameters()
            + agent.critic.layer2.num_parameters()
            + agent.critic.layer3.num_parameters()
        )
    )

    # Fill buffer with random data
    print("\nFilling buffer with random transitions...")
    from random import random_float64

    for _ in range(2000):
        var obs = InlineArray[Float64, obs_dim](fill=0.0)
        var next_obs = InlineArray[Float64, obs_dim](fill=0.0)
        var action = InlineArray[Float64, action_dim](fill=0.0)

        for j in range(obs_dim):
            obs[j] = random_float64() * 2.0 - 1.0
            next_obs[j] = random_float64() * 2.0 - 1.0
        action[0] = random_float64() * 4.0 - 2.0

        var reward = random_float64() * 2.0 - 1.0
        var done = random_float64() < 0.05

        agent.store_transition(obs, action, reward, next_obs, done)

    print("Buffer size: " + String(agent.buffer.len()))

    # Warmup
    print("\nWarming up (10 steps)...")
    for _ in range(10):
        _ = agent.train_step()

    # Benchmark train_step
    var num_steps = 100
    print("\nBenchmarking " + String(num_steps) + " train steps...")

    var start = perf_counter_ns()
    for _ in range(num_steps):
        _ = agent.train_step()
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

    # Create environment and agent
    var env = PendulumEnv[DType.float64]()

    comptime obs_dim = 3
    comptime action_dim = 1
    comptime hidden_dim = 128
    comptime buffer_capacity = 10000
    comptime batch_size = 64

    var agent = DeepDDPGAgent[
        obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size
    ](
        action_scale=2.0,
    )

    # Fill buffer first
    print("Pre-filling buffer...")
    from random import random_float64

    for _ in range(1000):
        var obs = InlineArray[Float64, obs_dim](fill=0.0)
        var next_obs = InlineArray[Float64, obs_dim](fill=0.0)
        var action = InlineArray[Float64, action_dim](fill=0.0)

        for j in range(obs_dim):
            obs[j] = random_float64() * 2.0 - 1.0
            next_obs[j] = random_float64() * 2.0 - 1.0
        action[0] = random_float64() * 4.0 - 2.0

        agent.store_transition(obs, action, random_float64(), next_obs, False)

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

    for ep in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var done = False
        var steps = 0

        var ep_start = perf_counter_ns()

        while not done and steps < max_steps:
            # Convert observation
            var obs = InlineArray[Float64, obs_dim](fill=0.0)
            for i in range(obs_dim):
                if i < len(obs_list):
                    obs[i] = obs_list[i]

            # Select action
            var action = agent.select_action(obs, add_noise=True)

            # Step environment
            var step_result = env.step_continuous(Float64(action[0]))
            var reward = step_result[1]
            done = step_result[2]

            # Get next observation
            var next_obs_list = env.get_obs_list()
            var next_obs = InlineArray[Float64, obs_dim](fill=0.0)
            for i in range(obs_dim):
                if i < len(next_obs_list):
                    next_obs[i] = next_obs_list[i]

            # Store and train
            agent.store_transition(obs, action, reward, next_obs, done)
            _ = agent.train_step()

            obs_list = env.get_obs_list()
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

    # Estimate breakdown
    print("\nNote: Each step includes:")
    print("  - Environment step (physics)")
    print("  - Observation conversion")
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
