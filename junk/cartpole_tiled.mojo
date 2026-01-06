"""CartPole with Tile Coding - Linear Function Approximation Example.

Demonstrates training on CartPole using tile-coded Q-learning,
which provides smooth generalization over the continuous state space.

This is much more efficient than naive discretization because:
1. Multiple tilings provide generalization between nearby states
2. Learning in one state automatically updates nearby states
3. Memory usage is controlled (8 tilings * 8^4 = 32,768 tiles vs 10^4 discrete states)
"""

from core.tile_coding import TileCoding, make_cartpole_tile_coding
from agents.tiled_qlearning import TiledQLearningAgent, TiledSARSALambdaAgent
from envs import CartPoleNative


fn train_cartpole_tiled(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    learning_rate: Float64 = 0.1,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train Q-learning agent with tile coding on CartPole.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode (CartPole default is 500)
        num_tilings: Number of tilings (more = finer resolution)
        tiles_per_dim: Tiles per state dimension (8^4 = 4096 tiles per tiling)
        learning_rate: Learning rate α
        verbose: Print progress

    Returns:
        List of episode rewards
    """
    # Create tile coding for CartPole
    var tc = make_cartpole_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create agent
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=learning_rate,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0

    if verbose:
        print("Training Tiled Q-Learning on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Learning rate:", learning_rate)
        print("")

    for episode in range(num_episodes):
        var obs = env.reset_obs()
        var tiles = tc.get_tiles_simd4(obs)
        var episode_reward: Float64 = 0.0

        for step in range(max_steps):
            # Select action
            var action = agent.select_action(tiles)

            # Take action
            var result = env.step_raw(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next tiles
            var next_tiles = tc.get_tiles_simd4(next_obs)

            # Update agent
            agent.update(tiles, action, reward, next_tiles, done)

            episode_reward += reward
            tiles = next_tiles^

            if done:
                break

        # Decay exploration
        agent.decay_epsilon()

        episode_rewards.append(episode_reward)

        # Update running average
        if episode == 0:
            running_reward = episode_reward
        else:
            running_reward = 0.99 * running_reward + 0.01 * episode_reward

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            var avg_reward: Float64 = 0.0
            var count = min(100, len(episode_rewards))
            for i in range(count):
                avg_reward += episode_rewards[len(episode_rewards) - count + i]
            avg_reward /= Float64(count)

            print(
                "Episode", episode + 1,
                "| Avg (100):", Int(avg_reward),
                "| Running:", Int(running_reward),
                "| Epsilon:", agent.get_epsilon(),
            )

    return episode_rewards^


fn train_cartpole_sarsa_lambda(
    num_episodes: Int = 1000,
    max_steps: Int = 500,
    num_tilings: Int = 8,
    tiles_per_dim: Int = 8,
    learning_rate: Float64 = 0.1,
    lambda_: Float64 = 0.9,
    verbose: Bool = True,
) raises -> List[Float64]:
    """Train SARSA(λ) agent with tile coding on CartPole.

    SARSA(λ) often learns faster than Q-learning on CartPole due to
    eligibility traces providing faster credit assignment.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        num_tilings: Number of tilings
        tiles_per_dim: Tiles per state dimension
        learning_rate: Learning rate α
        lambda_: Eligibility trace decay (0.9 is typical)
        verbose: Print progress

    Returns:
        List of episode rewards
    """
    # Create tile coding
    var tc = make_cartpole_tile_coding(
        num_tilings=num_tilings,
        tiles_per_dim=tiles_per_dim,
    )

    # Create SARSA(λ) agent
    var agent = TiledSARSALambdaAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=learning_rate,
        discount_factor=0.99,
        lambda_=lambda_,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Create environment
    var env = CartPoleNative()

    # Training loop
    var episode_rewards = List[Float64]()
    var running_reward: Float64 = 0.0

    if verbose:
        print("Training Tiled SARSA(λ) on CartPole")
        print("  Tilings:", num_tilings)
        print("  Tiles per dim:", tiles_per_dim)
        print("  Total tiles:", tc.get_num_tiles())
        print("  Lambda:", lambda_)
        print("")

    for episode in range(num_episodes):
        # Reset environment and traces
        var obs = env.reset_obs()
        agent.reset()  # Reset eligibility traces

        var tiles = tc.get_tiles_simd4(obs)
        var action = agent.select_action(tiles)
        var episode_reward: Float64 = 0.0

        for step in range(max_steps):
            # Take action
            var result = env.step_raw(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            # Get next tiles and action (SARSA is on-policy)
            var next_tiles = tc.get_tiles_simd4(next_obs)
            var next_action = agent.select_action(next_tiles)

            # Update agent with eligibility traces
            agent.update(tiles, action, reward, next_tiles, next_action, done)

            episode_reward += reward
            tiles = next_tiles^
            action = next_action

            if done:
                break

        # Decay exploration
        agent.decay_epsilon()

        episode_rewards.append(episode_reward)

        # Update running average
        if episode == 0:
            running_reward = episode_reward
        else:
            running_reward = 0.99 * running_reward + 0.01 * episode_reward

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            var avg_reward: Float64 = 0.0
            var count = min(100, len(episode_rewards))
            for i in range(count):
                avg_reward += episode_rewards[len(episode_rewards) - count + i]
            avg_reward /= Float64(count)

            print(
                "Episode", episode + 1,
                "| Avg (100):", Int(avg_reward),
                "| Running:", Int(running_reward),
                "| Epsilon:", agent.get_epsilon(),
            )

    return episode_rewards^


fn evaluate_agent(
    agent: TiledQLearningAgent,
    tc: TileCoding,
    num_episodes: Int = 100,
) raises -> Float64:
    """Evaluate trained agent (greedy policy, no exploration).

    Args:
        agent: Trained TiledQLearningAgent
        tc: TileCoding used during training
        num_episodes: Number of evaluation episodes

    Returns:
        Average reward over evaluation episodes
    """
    var env = CartPoleNative()
    var total_reward: Float64 = 0.0

    for _ in range(num_episodes):
        var obs = env.reset_obs()
        var episode_reward: Float64 = 0.0

        for _ in range(500):
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.get_best_action(tiles)  # Greedy

            var result = env.step_raw(action)
            var next_obs = result[0]
            var reward = result[1]
            var done = result[2]

            episode_reward += reward
            obs = next_obs

            if done:
                break

        total_reward += episode_reward

    return total_reward / Float64(num_episodes)


fn main() raises:
    """Run CartPole tile coding example."""
    print("=" * 60)
    print("CartPole with Tile Coding - Function Approximation")
    print("=" * 60)
    print("")

    # Train Q-learning
    print("-" * 60)
    var q_rewards = train_cartpole_tiled(
        num_episodes=500,
        num_tilings=8,
        tiles_per_dim=8,
        learning_rate=0.1,
        verbose=True,
    )

    # Summary
    print("")
    print("=" * 60)
    print("Training Complete!")
    print("")

    # Calculate final average
    var q_final: Float64 = 0.0
    var last_n = 100

    for i in range(last_n):
        q_final += q_rewards[len(q_rewards) - last_n + i]

    q_final /= Float64(last_n)

    print("Q-Learning final avg (100 episodes):", Int(q_final))
    print("")
    print("CartPole solved when avg reward >= 475 over 100 episodes")
    print("=" * 60)
