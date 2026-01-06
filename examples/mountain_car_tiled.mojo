"""MountainCar with Tile Coding - Function Approximation Example.

Demonstrates training on MountainCar using tile-coded Q-learning.
MountainCar is a challenging sparse reward problem where the agent
must learn to build momentum by swinging back and forth.

Key challenges:
1. Sparse reward (-1 every step until goal)
2. Requires building momentum (can't go straight to goal)
3. Only 2D state space but continuous
"""

from core.tile_coding import TileCoding
from agents.tiled_qlearning import TiledQLearningAgent, TiledSARSALambdaAgent
from envs.mountain_car_native import (
    MountainCarNative,
    make_mountain_car_tile_coding,
)
from envs.mountain_car_renderer import MountainCarRenderer


fn train_mountain_car_tiled(
    num_episodes: Int = 1000,
    max_steps: Int = 200,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    learning_rate: Float64 = 0.5,
    epsilon_decay: Float64 = 0.99,
    verbose: Bool = True,
) -> List[Float64]:
    """Train Q-learning agent with tile coding on MountainCar.

    Args:
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode (MountainCar default is 200).
        num_tilings: Number of tilings (more = finer resolution).
        tiles_per_dim: Tiles per dimension (8^2 = 64 tiles per tiling).
        learning_rate: Learning rate (higher for MountainCar due to sparse reward).
        epsilon_decay: Epsilon decay rate.
        verbose: Print progress.

    Returns:
        List of episode rewards (negative, closer to 0 is better).
    """
    # Create tile coding for MountainCar
    var tc = make_mountain_car_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create agent with optimistic initialization for exploration
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=3,
        learning_rate=learning_rate,
        discount_factor=1.0,  # No discounting for episodic task
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        init_value=0.0,  # Optimistic init helps exploration
    )

    # Create environment
    var env = MountainCarNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var successes = 0

    if verbose:
        print("Training Tiled Q-Learning on MountainCar")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Learning rate:", learning_rate)
        print("")

    for episode in range(num_episodes):
        var obs = env.reset()
        var tiles = tc.get_tiles_simd2(obs)
        var episode_reward: Float64 = 0.0

        for _ in range(max_steps):
            # Select action
            var action = agent.select_action(tiles)

            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next tiles
            var next_tiles = tc.get_tiles_simd2(next_obs)

            # Update agent
            agent.update(tiles, action, reward, next_tiles, done)

            episode_reward += reward
            tiles = next_tiles^

            if done:
                # Check if we reached the goal (position >= 0.5)
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


fn train_mountain_car_sarsa_lambda(
    num_episodes: Int = 1000,
    max_steps: Int = 200,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    learning_rate: Float64 = 0.5,
    lambda_: Float64 = 0.9,
    epsilon_decay: Float64 = 0.99,
    verbose: Bool = True,
) -> List[Float64]:
    """Train SARSA(lambda) agent with tile coding on MountainCar.

    SARSA(lambda) with eligibility traces often performs well on MountainCar
    due to faster credit assignment.

    Args:
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        num_tilings: Number of tilings.
        tiles_per_dim: Tiles per dimension.
        learning_rate: Learning rate.
        lambda_: Eligibility trace decay (0.9 typical).
        epsilon_decay: Epsilon decay rate.
        verbose: Print progress.

    Returns:
        List of episode rewards.
    """
    # Create tile coding
    var tc = make_mountain_car_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create SARSA(lambda) agent
    var agent = TiledSARSALambdaAgent(
        tile_coding=tc,
        num_actions=3,
        learning_rate=learning_rate,
        discount_factor=1.0,
        lambda_=lambda_,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
    )

    # Create environment
    var env = MountainCarNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var successes = 0

    if verbose:
        print("Training Tiled SARSA(lambda) on MountainCar")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Lambda:", lambda_)
        print("")

    for episode in range(num_episodes):
        # Reset environment and traces
        var obs = env.reset()
        agent.reset()  # Reset eligibility traces

        var tiles = tc.get_tiles_simd2(obs)
        var action = agent.select_action(tiles)
        var episode_reward: Float64 = 0.0

        for _ in range(max_steps):
            # Take action
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next tiles and action (SARSA is on-policy)
            var next_tiles = tc.get_tiles_simd2(next_obs)
            var next_action = agent.select_action(next_tiles)

            # Update agent with eligibility traces
            agent.update(tiles, action, reward, next_tiles, next_action, done)

            episode_reward += reward
            tiles = next_tiles^
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
    """Run MountainCar tile coding example with training and visualization."""
    print("=" * 60)
    print("MountainCar with Tile Coding - Function Approximation")
    print("=" * 60)
    print("")
    print("Goal: Reach position >= 0.5 (flag on the right hill)")
    print("Reward: -1 per step (minimize steps to reach goal)")
    print("Best possible: ~-100 steps (perfect policy)")
    print("")

    # Create tile coding
    var tc = make_mountain_car_tile_coding(num_tilings=8, tiles_per_dim=8)

    # Create agent
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=3,
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    var env = MountainCarNative()

    # Training phase
    print("-" * 60)
    print("Training Tiled Q-Learning on MountainCar")
    print("  Tilings:", tc.get_num_tilings())
    print("  Total tiles:", tc.get_num_tiles())
    print("")

    var successes = 0
    for episode in range(500):
        var obs = env.reset()
        var tiles = tc.get_tiles_simd2(obs)
        var episode_reward: Float64 = 0.0

        for _ in range(200):
            var action = agent.select_action(tiles)
            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            var next_tiles = tc.get_tiles_simd2(next_obs)
            agent.update(tiles, action, reward, next_tiles, done)

            episode_reward += reward
            tiles = next_tiles^

            if done:
                if next_obs[0] >= 0.5:
                    successes += 1
                break

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(
                "Episode", episode + 1,
                "| Successes:", successes,
                "| Epsilon:", agent.get_epsilon(),
            )

    print("")
    print("=" * 60)
    print("Training Complete!")
    print("Total successes:", successes, "/ 500")
    print("")
    print("Now showing learned policy with visualization...")
    print("Close the window when done watching.")
    print("=" * 60)
    print("")

    # Visualization phase - show trained agent
    var renderer = MountainCarRenderer()

    for demo_episode in range(5):
        var obs = env.reset()
        var total_reward: Float64 = 0.0

        print("Demo episode", demo_episode + 1)

        for step in range(200):
            var tiles = tc.get_tiles_simd2(obs)
            var action = agent.get_best_action(tiles)  # Greedy policy

            var result = env.step(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            total_reward += reward

            # Render
            renderer.render(next_obs[0], next_obs[1], step + 1, total_reward)

            obs = next_obs

            if done:
                if next_obs[0] >= 0.5:
                    print("  Goal reached in", step + 1, "steps!")
                else:
                    print("  Timeout after", step + 1, "steps")
                break

    renderer.close()
    print("")
    print("Demo complete!")
