"""Benchmark: Native Mojo CartPole vs Gymnasium Python CartPole.

Compares:
1. Training speed (steps per second)
2. Learning performance (final reward)
"""

from time import perf_counter_ns
from envs.gymnasium import (
    CartPoleEnv,
    discretize_cart_pole,
    get_cart_pole_num_states,
)
from envs.cartpole_native import CartPoleNative, discretize_obs_native
from agents.qlearning import QLearningAgent
from random import seed


fn benchmark_gymnasium(
    num_episodes: Int, num_bins: Int
) raises -> Tuple[Float64, Float64, Int]:
    """Benchmark Gymnasium CartPole.

    Returns: (total_time_seconds, avg_eval_reward, total_steps)
    """
    var num_states = get_cart_pole_num_states(num_bins)
    var max_steps = 500

    var env = CartPoleEnv()
    var agent = QLearningAgent(
        num_states=num_states,
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var total_steps = 0
    var start_time = perf_counter_ns()

    # Training
    for episode in range(num_episodes):
        var obs = env.reset()
        var state = discretize_cart_pole(obs, num_bins)

        for _ in range(max_steps):
            var action = agent.select_action(state)
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            var next_state = discretize_cart_pole(next_obs, num_bins)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_steps += 1

            if done:
                break

        agent.decay_epsilon()

    var end_time = perf_counter_ns()
    var total_time = Float64(end_time - start_time) / 1_000_000_000.0

    env.close()

    # Evaluation
    var eval_env = CartPoleEnv()
    var eval_total: Float64 = 0.0

    for _ in range(10):
        var obs = eval_env.reset()
        var state = discretize_cart_pole(obs, num_bins)
        var ep_reward: Float64 = 0.0

        for _ in range(max_steps):
            var action = agent.get_best_action(state)
            var result = eval_env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = discretize_cart_pole(next_obs, num_bins)

            if done:
                break

        eval_total += ep_reward

    eval_env.close()

    return (total_time, eval_total / 10.0, total_steps)


fn benchmark_native(
    num_episodes: Int, num_bins: Int
) -> Tuple[Float64, Float64, Int]:
    """Benchmark Native Mojo CartPole.

    Returns: (total_time_seconds, avg_eval_reward, total_steps)
    """
    var num_states = num_bins * num_bins * num_bins * num_bins
    var max_steps = 500

    var env = CartPoleNative()
    var agent = QLearningAgent(
        num_states=num_states,
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var total_steps = 0
    var start_time = perf_counter_ns()

    # Training
    for episode in range(num_episodes):
        var obs = env.reset()
        var state = discretize_obs_native(obs, num_bins)

        for _ in range(max_steps):
            var action = agent.select_action(state)
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            var next_state = discretize_obs_native(next_obs, num_bins)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_steps += 1

            if done:
                break

        agent.decay_epsilon()

    var end_time = perf_counter_ns()
    var total_time = Float64(end_time - start_time) / 1_000_000_000.0

    # Evaluation
    var eval_env = CartPoleNative()
    var eval_total: Float64 = 0.0

    for _ in range(10):
        var obs = eval_env.reset()
        var state = discretize_obs_native(obs, num_bins)
        var ep_reward: Float64 = 0.0

        for _ in range(max_steps):
            var action = agent.get_best_action(state)
            var result = eval_env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = discretize_obs_native(next_obs, num_bins)

            if done:
                break

        eval_total += ep_reward

    return (total_time, eval_total / 10.0, total_steps)


fn main() raises:
    seed()

    print("=" * 70)
    print("    CartPole Benchmark: Native Mojo vs Gymnasium (Python)")
    print("=" * 70)
    print()

    var num_episodes = 1000
    var num_bins = 10

    print("Configuration:")
    print("  Episodes:", num_episodes)
    print(
        "  State bins:", num_bins, "per dimension ->", num_bins**4, "states"
    )
    print("  Max steps per episode: 500")
    print()

    # Benchmark Native Mojo
    print("Running Native Mojo CartPole...")
    var native_result = benchmark_native(num_episodes, num_bins)
    var native_time = native_result[0]
    var native_reward = native_result[1]
    var native_steps = native_result[2]
    print("  Done!")
    print()

    # Benchmark Gymnasium
    print("Running Gymnasium (Python) CartPole...")
    var gym_result = benchmark_gymnasium(num_episodes, num_bins)
    var gym_time = gym_result[0]
    var gym_reward = gym_result[1]
    var gym_steps = gym_result[2]
    print("  Done!")
    print()

    # Results
    print("=" * 70)
    print("    Results")
    print("=" * 70)
    print()

    var native_sps = Float64(native_steps) / native_time
    var gym_sps = Float64(gym_steps) / gym_time
    var speedup = native_sps / gym_sps

    print("Native Mojo:")
    print("  Training time:", native_time, "seconds")
    print("  Total steps:", native_steps)
    print("  Steps/second:", Int(native_sps))
    print("  Eval reward:", Int(native_reward))
    print()

    print("Gymnasium (Python):")
    print("  Training time:", gym_time, "seconds")
    print("  Total steps:", gym_steps)
    print("  Steps/second:", Int(gym_sps))
    print("  Eval reward:", Int(gym_reward))
    print()

    print("=" * 70)
    print("    Speedup: Native Mojo is", speedup, "x faster!")
    print("=" * 70)
