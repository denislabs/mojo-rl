"""Comparison of Uniform Replay vs Prioritized Experience Replay.

This demo trains two Q-Learning agents on CliffWalking:
1. QLearningReplayAgent - Uses uniform random sampling
2. QLearningPERAgent - Uses prioritized sampling (TD error based)

CliffWalking is ideal for demonstrating PER because:
- Falling off the cliff gives -100 reward (high TD error)
- These rare but important transitions should be replayed more
- PER prioritizes these high-error transitions

Expected result: PER should learn faster, especially early in training,
because it focuses on surprising/important transitions.
"""

from envs import CliffWalkingEnv
from agents import QLearningReplayAgent, QLearningPERAgent


fn run_comparison():
    """Run comparison between uniform replay and PER."""
    print("=" * 70)
    print("Prioritized Experience Replay vs Uniform Replay Comparison")
    print("Environment: CliffWalking (4x12 grid)")
    print("=" * 70)

    # Training parameters
    var num_episodes = 500
    var num_runs = 3  # Average over multiple runs for stability
    var max_steps = 200

    # Shared hyperparameters
    var buffer_size = 5000
    var batch_size = 32
    var min_buffer_size = 100
    var learning_rate = 0.1
    var discount_factor = 0.99
    var epsilon = 1.0
    var epsilon_decay = 0.995
    var epsilon_min = 0.01

    # PER-specific hyperparameters
    var alpha = 0.6  # Priority exponent
    var beta_start = 0.4  # Initial IS correction

    print("\nHyperparameters:")
    print("  Episodes:", num_episodes)
    print("  Buffer size:", buffer_size)
    print("  Batch size:", batch_size)
    print("  Learning rate:", learning_rate)
    print("  Discount factor:", discount_factor)
    print("  PER alpha:", alpha)
    print("  PER beta_start:", beta_start)
    print()

    # Track results
    var uniform_rewards = List[Float64]()
    var per_rewards = List[Float64]()
    var uniform_final_eval = List[Float64]()
    var per_final_eval = List[Float64]()

    for run in range(num_runs):
        print("Run", run + 1, "/", num_runs)

        # Create environments
        var env_uniform = CliffWalkingEnv()
        var env_per = CliffWalkingEnv()

        var num_states = env_uniform.num_states()
        var num_actions = env_uniform.num_actions()

        # Create agents
        var uniform_agent = QLearningReplayAgent(
            num_states=num_states,
            num_actions=num_actions,
            buffer_size=buffer_size,
            batch_size=batch_size,
            min_buffer_size=min_buffer_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
        )

        var per_agent = QLearningPERAgent(
            num_states=num_states,
            num_actions=num_actions,
            buffer_size=buffer_size,
            batch_size=batch_size,
            min_buffer_size=min_buffer_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            alpha=alpha,
            beta_start=beta_start,
        )

        # Train uniform replay agent
        print("  Training Uniform Replay agent...")
        var uniform_metrics = uniform_agent.train(
            env_uniform,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps,
            verbose=False,
            environment_name="CliffWalking",
        )

        # Train PER agent
        print("  Training PER agent...")
        var per_metrics = per_agent.train(
            env_per,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps,
            verbose=False,
            environment_name="CliffWalking",
        )

        # Record mean rewards
        uniform_rewards.append(uniform_metrics.mean_reward())
        per_rewards.append(per_metrics.mean_reward())

        # Evaluate final policies
        var uniform_eval = uniform_agent.evaluate(env_uniform, num_episodes=20)
        var per_eval = per_agent.evaluate(env_per, num_episodes=20)
        uniform_final_eval.append(uniform_eval)
        per_final_eval.append(per_eval)

        print("    Uniform mean reward:", uniform_metrics.mean_reward())
        print("    PER mean reward:", per_metrics.mean_reward())
        print("    Uniform final eval:", uniform_eval)
        print("    PER final eval:", per_eval)

    # Compute averages
    var avg_uniform_reward: Float64 = 0.0
    var avg_per_reward: Float64 = 0.0
    var avg_uniform_eval: Float64 = 0.0
    var avg_per_eval: Float64 = 0.0

    for i in range(num_runs):
        avg_uniform_reward += uniform_rewards[i]
        avg_per_reward += per_rewards[i]
        avg_uniform_eval += uniform_final_eval[i]
        avg_per_eval += per_final_eval[i]

    avg_uniform_reward /= Float64(num_runs)
    avg_per_reward /= Float64(num_runs)
    avg_uniform_eval /= Float64(num_runs)
    avg_per_eval /= Float64(num_runs)

    # Print results
    print()
    print("=" * 70)
    print("Results (averaged over", num_runs, "runs)")
    print("=" * 70)
    print()
    print("Mean Training Reward:")
    print("  Uniform Replay:", avg_uniform_reward)
    print("  PER:           ", avg_per_reward)
    var improvement = (avg_per_reward - avg_uniform_reward) / (
        -avg_uniform_reward
    ) * 100.0 if avg_uniform_reward < 0 else 0.0
    print("  Improvement:    ", improvement, "%")
    print()
    print("Final Policy Evaluation (20 episodes, greedy):")
    print("  Uniform Replay:", avg_uniform_eval)
    print("  PER:           ", avg_per_eval)
    print()

    if avg_per_eval > avg_uniform_eval:
        print("PER learned a better policy!")
    elif avg_per_eval < avg_uniform_eval:
        print("Uniform replay learned a better policy (unusual).")
    else:
        print("Both methods performed equally.")

    print()
    print("Note: CliffWalking optimal path has reward of -13.")
    print("      (11 steps right + 1 step down from start to goal)")
    print()


fn demonstrate_per_sampling():
    """Demonstrate how PER samples differently than uniform."""
    print()
    print("=" * 70)
    print("PER Sampling Behavior Demonstration")
    print("=" * 70)
    print()

    from core.replay_buffer import PrioritizedReplayBuffer

    var buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

    # Add transitions with different TD errors
    # Simulate: normal steps (low TD error) and cliff falls (high TD error)

    print("Adding 10 'normal' transitions (TD error ~ 0.5)...")
    for i in range(10):
        buffer.push(i, 0, -1.0, i + 1, False)  # Normal -1 step

    print("Adding 2 'cliff fall' transitions (TD error ~ 100)...")
    buffer.push(10, 1, -100.0, 0, False)  # Cliff fall
    buffer.push(11, 1, -100.0, 0, False)  # Cliff fall

    # Update priorities based on TD errors
    # Normal transitions: TD error ~ 0.5
    for i in range(10):
        buffer.update_priority(i, 0.5)
    # Cliff falls: TD error ~ 100
    buffer.update_priority(10, 100.0)
    buffer.update_priority(11, 100.0)

    # Sample and count how often each type is selected
    var normal_count = 0
    var cliff_count = 0
    var num_samples = 1000

    print()
    print("Sampling", num_samples, "times...")

    for _ in range(num_samples):
        var result = buffer.sample(batch_size=1)
        var indices = result[0].copy()
        if len(indices) > 0:
            if indices[0] >= 10:
                cliff_count += 1
            else:
                normal_count += 1

    print()
    print("Sampling distribution:")
    print("  Normal transitions (10 in buffer):", normal_count, "samples")
    print("  Cliff falls (2 in buffer):        ", cliff_count, "samples")
    print()

    var expected_normal_uniform = Float64(num_samples) * 10.0 / 12.0
    var expected_cliff_uniform = Float64(num_samples) * 2.0 / 12.0

    print("Uniform sampling would give approximately:")
    print("  Normal:", Int(expected_normal_uniform), "(", 10.0 / 12.0 * 100.0, "%)")
    print("  Cliff: ", Int(expected_cliff_uniform), "(", 2.0 / 12.0 * 100.0, "%)")
    print()

    var actual_cliff_pct = Float64(cliff_count) / Float64(num_samples) * 100.0
    print("PER gave cliff falls", actual_cliff_pct, "% of samples")
    print("This is", actual_cliff_pct / (2.0 / 12.0 * 100.0), "x more than uniform!")
    print()


fn main():
    """Run PER comparison demo."""
    demonstrate_per_sampling()
    run_comparison()
