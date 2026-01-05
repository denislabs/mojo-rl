"""CartPole POC: Q-Learning with discretized observations.

This demonstrates using Gymnasium's CartPole environment with tabular Q-Learning
by discretizing the continuous observation space.
"""

from envs.gymnasium_cartpole import CartPoleEnv, discretize_obs, get_num_states
from agents.qlearning import QLearningAgent
from random import seed, random_float64


fn train_cartpole() raises:
    """Train Q-Learning agent on CartPole using discretized observations."""
    print("=" * 60)
    print("CartPole Q-Learning POC (Gymnasium Integration)")
    print("=" * 60)

    # Hyperparameters
    var num_bins = 10  # Bins per dimension -> 10^4 = 10000 states
    var num_episodes = 1000
    var max_steps = 500

    var num_states = get_num_states(num_bins)
    var num_actions = 2

    print("State space:", num_states, "discretized states")
    print("Action space:", num_actions, "actions")
    print()

    # Initialize environment (no rendering during training for speed)
    var env = CartPoleEnv()
    var agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Training metrics
    var total_rewards = List[Float64]()
    var episode_lengths = List[Int]()
    var best_reward: Float64 = 0.0
    var solved_count = 0

    print("Training for", num_episodes, "episodes...")
    print()

    for episode in range(num_episodes):
        var obs = env.reset()
        var state = discretize_obs(obs, num_bins)
        var episode_reward: Float64 = 0.0
        var steps = 0

        for _ in range(max_steps):
            # Select action using epsilon-greedy
            var action = agent.select_action(state)

            # Take step in environment
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            var next_state = discretize_obs(next_obs, num_bins)

            # Q-Learning update
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # Decay exploration
        agent.decay_epsilon()

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if episode_reward > best_reward:
            best_reward = episode_reward

        # CartPole is "solved" if avg reward over 100 episodes >= 475
        if episode_reward >= 475:
            solved_count += 1

        # Progress logging
        if (episode + 1) % 100 == 0:
            var avg_reward: Float64 = 0.0
            var avg_length: Float64 = 0.0
            var start_idx = max(0, len(total_rewards) - 100)
            var count = len(total_rewards) - start_idx

            for i in range(start_idx, len(total_rewards)):
                avg_reward += total_rewards[i]
                avg_length += Float64(episode_lengths[i])

            avg_reward /= Float64(count)
            avg_length /= Float64(count)

            print(
                "Episode", episode + 1,
                "| Avg Reward (100):", Int(avg_reward),
                "| Avg Length:", Int(avg_length),
                "| Epsilon:", agent.get_epsilon(),
                "| Best:", Int(best_reward),
            )

    env.close()

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Final evaluation (greedy policy, no rendering)
    print()
    print("Evaluating with greedy policy (10 episodes, no render)...")

    var eval_env = CartPoleEnv()
    var eval_rewards = List[Float64]()

    for _ in range(10):
        var obs = eval_env.reset()
        var state = discretize_obs(obs, num_bins)
        var ep_reward: Float64 = 0.0

        for _ in range(max_steps):
            var action = agent.get_best_action(state)
            var result = eval_env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = discretize_obs(next_obs, num_bins)

            if done:
                break

        eval_rewards.append(ep_reward)

    eval_env.close()

    var eval_avg: Float64 = 0.0
    for i in range(len(eval_rewards)):
        eval_avg += eval_rewards[i]
    eval_avg /= Float64(len(eval_rewards))

    print("Evaluation Avg Reward:", Int(eval_avg))
    print()

    if eval_avg >= 475:
        print("SUCCESS! CartPole solved (avg reward >= 475)")
    else:
        print("Not solved yet. Try more episodes or tune hyperparameters.")

    # Visual demo with rendering
    print()
    print("=" * 60)
    print("Visual Demo - Watch the trained agent!")
    print("=" * 60)
    print("Opening window with 3 visual episodes...")
    print("(Close the window or wait for episodes to finish)")
    print()

    # Create environment with human rendering
    var render_env = CartPoleEnv(render_mode="human")

    for ep in range(3):
        var obs = render_env.reset()
        var state = discretize_obs(obs, num_bins)
        var ep_reward: Float64 = 0.0

        for _ in range(max_steps):
            # Greedy action selection
            var action = agent.get_best_action(state)
            var result = render_env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            ep_reward += reward
            state = discretize_obs(next_obs, num_bins)

            if done:
                break

        print("Visual Episode", ep + 1, "| Reward:", Int(ep_reward))

    render_env.close()
    print()
    print("Demo complete!")


fn main() raises:
    seed()
    train_cartpole()
