"""CartPole with Linear Function Approximation.

Demonstrates linear Q-learning with polynomial features on CartPole.
CartPole is more amenable to polynomial features than MountainCar because:
1. The value function is smoother
2. Dense rewards (+1 per step) provide better learning signal
3. The optimal policy is simpler (keep pole balanced)

This example shows that linear function approximation works correctly
when the environment's value function can be approximated by the features.
"""

from core.linear_fa import PolynomialFeatures, make_cartpole_poly_features
from agents.linear_qlearning import LinearQLearningAgent, LinearSARSAAgent
from envs.cartpole_native import CartPoleNative


fn train_cartpole_linear(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    degree: Int = 2,
    learning_rate: Float64 = 0.5,
    epsilon_decay: Float64 = 0.995,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train linear Q-learning with polynomial features on CartPole.

    Args:
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        degree: Maximum polynomial degree.
        learning_rate: Learning rate.
        epsilon_decay: Epsilon decay rate.
        verbose: Print progress.

    Returns:
        List of episode rewards (steps survived).
    """
    # Create polynomial feature extractor with normalization
    var features = make_cartpole_poly_features(degree=degree)

    if verbose:
        print("Training Linear Q-Learning on CartPole")
        print("  Polynomial degree:", degree)
        print("  Number of features:", features.get_num_features())
        print("  Learning rate:", learning_rate)
        print("")

    # Create agent
    var agent = LinearQLearningAgent(
        num_features=features.get_num_features(),
        num_actions=2,
        learning_rate=learning_rate,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        init_std=0.0,  # Zero init
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var solved_count = 0

    for episode in range(num_episodes):
        var obs = env.reset()
        var phi = features.get_features_simd4(obs)
        var episode_reward: Float64 = 0.0

        for _ in range(max_steps):
            # Select action
            var action = agent.select_action(phi)

            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next features
            var next_phi = features.get_features_simd4(next_obs)

            # Update agent
            agent.update(phi, action, reward, next_phi, done)

            episode_reward += reward
            phi = next_phi^

            if done:
                break

        # Track if episode reached 195+ steps (solved criterion)
        if episode_reward >= 195:
            solved_count += 1

        # Decay exploration
        agent.decay_epsilon()

        episode_rewards.append(episode_reward)

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            var avg_reward: Float64 = 0.0
            var count = min(100, len(episode_rewards))
            for i in range(count):
                avg_reward += episode_rewards[len(episode_rewards) - count + i]
            avg_reward /= Float64(count)

            print(
                "Episode",
                episode + 1,
                "| Avg (100):",
                Int(avg_reward),
                "| Solved (>=195):",
                solved_count,
                "| Epsilon:",
                agent.get_epsilon(),
            )

    return episode_rewards^


fn main() raises:
    """Run CartPole linear function approximation example."""
    print("=" * 60)
    print("CartPole - Linear Function Approximation")
    print("=" * 60)
    print("")
    print("Goal: Balance pole for as long as possible")
    print("Reward: +1 per step (max 500)")
    print("Solved: Average reward >= 195 over 100 episodes")
    print("")

    # Train with polynomial features
    print("-" * 60)
    print("Training with Polynomial Features (degree=2)")
    print("-" * 60)
    _ = train_cartpole_linear(
        num_episodes=500,
        max_steps=500,
        degree=2,
        learning_rate=0.5,
        epsilon_decay=0.995,
        verbose=True,
    )

    print("")
    print("=" * 60)
    print("Training Complete!")
    print("")
    print("Note: CartPole with polynomial features shows that linear")
    print("function approximation works when the value function can be")
    print("well-approximated by the chosen features.")
    print("=" * 60)
