"""MountainCar with Linear Function Approximation.

Demonstrates training on MountainCar using linear Q-learning with
polynomial and RBF features. This showcases the flexibility of
arbitrary feature representations compared to tile coding.

IMPORTANT NOTE: MountainCar is extremely challenging for linear function
approximation with global features (polynomial, RBF). The value function
is highly non-linear and requires counter-intuitive behavior (going backward
to build momentum). Tile coding works much better because it provides
localized features.

Feature types demonstrated:
1. Polynomial features - x, y, x², xy, y², etc.
2. RBF features - Gaussian radial basis functions centered on a grid

For MountainCar specifically, use tile coding (see mountain_car_tiled.mojo).
This example is better suited for environments with smoother value functions.

Linear function approximation:
    Q(s, a) = w[a]^T * φ(s)

where φ(s) is the feature vector extracted from state s.
"""

from core.linear_fa import (
    PolynomialFeatures,
    RBFFeatures,
    make_grid_rbf_centers,
    make_mountain_car_poly_features,
)
from agents.linear_qlearning import LinearQLearningAgent, LinearSARSALambdaAgent
from envs.mountain_car_native import MountainCarNative


fn train_mountain_car_poly(
    num_episodes: Int = 1000,
    max_steps: Int = 200,
    degree: Int = 3,
    learning_rate: Float64 = 0.01,
    epsilon_decay: Float64 = 0.995,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train linear Q-learning with polynomial features on MountainCar.

    Polynomial features create combinations of state variables:
    For degree=3: [1, x, y, x², xy, y², x³, x²y, xy², y³]

    Args:
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        degree: Maximum polynomial degree.
        learning_rate: Learning rate (smaller than tabular).
        epsilon_decay: Epsilon decay rate.
        verbose: Print progress.

    Returns:
        List of episode rewards.
    """
    # Create polynomial feature extractor
    var features = make_mountain_car_poly_features(degree=degree)

    if verbose:
        print("Training Linear Q-Learning with Polynomial Features")
        print("  Polynomial degree:", degree)
        print("  Number of features:", features.get_num_features())
        print("  Learning rate:", learning_rate)
        print("")

    # Create agent
    var agent = LinearQLearningAgent(
        num_features=features.get_num_features(),
        num_actions=3,
        learning_rate=learning_rate,
        discount_factor=1.0,  # No discounting for episodic task
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        init_std=0.001,
    )

    # Create environment
    var env = MountainCarNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var successes = 0

    for episode in range(num_episodes):
        var obs = env.reset()
        var phi = features.get_features_simd2(obs)
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
            var next_phi = features.get_features_simd2(next_obs)

            # Update agent
            agent.update(phi, action, reward, next_phi, done)

            episode_reward += reward
            phi = next_phi^

            if done:
                if next_obs[0] >= 0.5:
                    successes += 1
                break

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
                "| Successes:",
                successes,
                "| Epsilon:",
                agent.get_epsilon(),
            )

    return episode_rewards^


fn train_mountain_car_rbf(
    num_episodes: Int = 1000,
    max_steps: Int = 200,
    num_centers_per_dim: Int = 10,
    sigma: Float64 = 0.1,
    learning_rate: Float64 = 0.1,
    epsilon_decay: Float64 = 0.995,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train linear Q-learning with RBF features on MountainCar.

    RBF features provide localized activations based on distance to centers:
        φ_i(s) = exp(-||s - c_i||² / (2σ²))

    Args:
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        num_centers_per_dim: Number of RBF centers per dimension.
        sigma: RBF width parameter.
        learning_rate: Learning rate.
        epsilon_decay: Epsilon decay rate.
        verbose: Print progress.

    Returns:
        List of episode rewards.
    """
    # MountainCar state bounds
    var state_low = List[Float64]()
    state_low.append(-1.2)  # position min
    state_low.append(-0.07)  # velocity min

    var state_high = List[Float64]()
    state_high.append(0.6)  # position max
    state_high.append(0.07)  # velocity max

    # Create RBF centers on a grid
    var centers = make_grid_rbf_centers(
        state_low=state_low^,
        state_high=state_high^,
        num_centers_per_dim=num_centers_per_dim,
    )

    # Create RBF feature extractor
    var features = RBFFeatures(centers=centers^, sigma=sigma)

    if verbose:
        print("Training Linear Q-Learning with RBF Features")
        print("  Centers per dim:", num_centers_per_dim)
        print("  Total features:", features.get_num_features())
        print("  Sigma:", sigma)
        print("  Learning rate:", learning_rate)
        print("")

    # Create agent
    var agent = LinearQLearningAgent(
        num_features=features.get_num_features(),
        num_actions=3,
        learning_rate=learning_rate,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        init_std=0.0,  # Zero init for RBF
    )

    # Create environment
    var env = MountainCarNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var successes = 0

    for episode in range(num_episodes):
        var obs = env.reset()
        var phi = features.get_features_simd2(obs)
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
            var next_phi = features.get_features_simd2(next_obs)

            # Update agent
            agent.update(phi, action, reward, next_phi, done)

            episode_reward += reward
            phi = next_phi^

            if done:
                if next_obs[0] >= 0.5:
                    successes += 1
                break

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
                "| Successes:",
                successes,
                "| Epsilon:",
                agent.get_epsilon(),
            )

    return episode_rewards^


fn train_mountain_car_linear_sarsa_lambda(
    num_episodes: Int = 1000,
    max_steps: Int = 200,
    degree: Int = 3,
    learning_rate: Float64 = 0.01,
    lambda_: Float64 = 0.9,
    epsilon_decay: Float64 = 0.995,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train SARSA(λ) with polynomial features on MountainCar.

    Combines linear function approximation with eligibility traces
    for faster credit assignment in sparse reward problems.

    Args:
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        degree: Maximum polynomial degree.
        learning_rate: Learning rate.
        lambda_: Eligibility trace decay.
        epsilon_decay: Epsilon decay rate.
        verbose: Print progress.

    Returns:
        List of episode rewards.
    """
    # Create polynomial feature extractor
    var features = make_mountain_car_poly_features(degree=degree)

    if verbose:
        print("Training Linear SARSA(λ) with Polynomial Features")
        print("  Polynomial degree:", degree)
        print("  Number of features:", features.get_num_features())
        print("  Lambda:", lambda_)
        print("  Learning rate:", learning_rate)
        print("")

    # Create SARSA(λ) agent
    var agent = LinearSARSALambdaAgent(
        num_features=features.get_num_features(),
        num_actions=3,
        learning_rate=learning_rate,
        discount_factor=1.0,
        lambda_=lambda_,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        init_std=0.001,
    )

    # Create environment
    var env = MountainCarNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var successes = 0

    for episode in range(num_episodes):
        # Reset environment and traces
        var obs = env.reset()
        agent.reset()  # Reset eligibility traces

        var phi = features.get_features_simd2(obs)
        var action = agent.select_action(phi)
        var episode_reward: Float64 = 0.0

        for _ in range(max_steps):
            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next features and action (on-policy)
            var next_phi = features.get_features_simd2(next_obs)
            var next_action = agent.select_action(next_phi)

            # Update with eligibility traces
            agent.update(phi, action, reward, next_phi, next_action, done)

            episode_reward += reward
            phi = next_phi^
            action = next_action

            if done:
                if next_obs[0] >= 0.5:
                    successes += 1
                break

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
                "| Successes:",
                successes,
                "| Epsilon:",
                agent.get_epsilon(),
            )

    return episode_rewards^


fn main() raises:
    """Run MountainCar linear function approximation comparison."""
    print("=" * 60)
    print("MountainCar - Linear Function Approximation")
    print("=" * 60)
    print("")
    print("Comparing different feature representations:")
    print("  1. Polynomial features (x, y, x², xy, y², ...)")
    print("  2. RBF features (Gaussian radial basis functions)")
    print("  3. SARSA(λ) with polynomial features")
    print("")
    print("Goal: Reach position >= 0.5 (flag on the right hill)")
    print("Reward: -1 per step (minimize steps to reach goal)")
    print("")

    # Method 1: Polynomial features with normalized state (best for linear FA)
    print("-" * 60)
    print("Method 1: Polynomial Features (degree=3, normalized)")
    print("-" * 60)
    _ = train_mountain_car_poly(
        num_episodes=2000,
        max_steps=200,
        degree=3,
        learning_rate=0.5,  # High LR works now with feature normalization
        epsilon_decay=0.998,  # Slower decay for more exploration
        verbose=True,
    )
    print("")

    # Method 2: RBF features
    print("-" * 60)
    print("Method 2: RBF Features (10x10 grid)")
    print("-" * 60)
    _ = train_mountain_car_rbf(
        num_episodes=2000,
        max_steps=200,
        num_centers_per_dim=10,
        sigma=0.15,  # Tighter RBFs for better locality
        learning_rate=0.5,
        epsilon_decay=0.998,
        verbose=True,
    )
    print("")

    # Method 3: SARSA(λ) with polynomial (eligibility traces help)
    print("-" * 60)
    print("Method 3: SARSA(λ) with Polynomial Features")
    print("-" * 60)
    _ = train_mountain_car_linear_sarsa_lambda(
        num_episodes=2000,
        max_steps=200,
        degree=3,
        learning_rate=0.5,
        lambda_=0.9,
        epsilon_decay=0.998,
        verbose=True,
    )

    print("")
    print("=" * 60)
    print("Comparison Complete!")
    print("")
    print("Notes:")
    print("- Polynomial features are compact but may not capture locality")
    print("- RBF features provide smooth local generalization")
    print("- SARSA(λ) with traces can speed up learning in sparse reward tasks")
    print("- Tile coding (see mountain_car_tiled.mojo) often works best for RL")
    print("=" * 60)
