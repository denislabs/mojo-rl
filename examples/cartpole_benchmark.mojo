"""Benchmark: Native Mojo CartPole vs Gymnasium Python CartPole.

Compares:
1. Training speed (steps per second)
2. Learning performance (final reward)
"""

from time import perf_counter_ns
from envs.gymnasium import GymCartPoleEnv
from envs import CartPoleEnv
from agents.qlearning import QLearningAgent
from random import seed


fn compute_total_steps(steps: List[Int]) -> Int:
    """Sum all episode steps to get total training steps."""
    var total = 0
    for i in range(len(steps)):
        total += steps[i]
    return total


fn benchmark_gymnasium(
    num_episodes: Int, num_bins: Int
) raises -> Tuple[Float64, Float64, Int]:
    """Benchmark Gymnasium CartPole.

    Returns: (total_time_seconds, avg_eval_reward, total_steps)
    """
    var max_steps = 500

    var env = GymCartPoleEnv(num_bins=num_bins)
    var agent = QLearningAgent(
        num_states=env.num_states(),
        num_actions=env.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var start_time = perf_counter_ns()

    # Training using agent.train()
    var metrics = agent.train(
        env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=False,
    )

    var end_time = perf_counter_ns()
    var total_time = Float64(end_time - start_time) / 1_000_000_000.0

    var total_steps = compute_total_steps(metrics.get_steps())

    env.close()

    # Evaluation using agent.evaluate()
    var eval_env = GymCartPoleEnv(num_bins=num_bins)
    var avg_eval_reward = agent.evaluate(eval_env, num_episodes=10)
    eval_env.close()

    return (total_time, avg_eval_reward, total_steps)


fn benchmark_native(
    num_episodes: Int, num_bins: Int
) raises -> Tuple[Float64, Float64, Int]:
    """Benchmark Native Mojo CartPole.

    Returns: (total_time_seconds, avg_eval_reward, total_steps)
    """
    var max_steps = 500

    var env = CartPoleEnv(num_bins=num_bins)
    var agent = QLearningAgent(
        num_states=CartPoleEnv.get_num_states(num_bins),
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var start_time = perf_counter_ns()

    # Training using agent.train()
    var metrics = agent.train(
        env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=False,
    )

    var end_time = perf_counter_ns()
    var total_time = Float64(end_time - start_time) / 1_000_000_000.0

    var total_steps = compute_total_steps(metrics.get_steps())

    # Evaluation using agent.evaluate()
    var avg_eval_reward = agent.evaluate(env, num_episodes=10)

    return (total_time, avg_eval_reward, total_steps)


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
