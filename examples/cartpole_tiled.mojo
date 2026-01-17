"""CartPole with Tile Coding - Linear Function Approximation Example.

Demonstrates training on CartPole using tile-coded Q-learning,
which provides smooth generalization over the continuous state space.

This is much more efficient than naive discretization because:
1. Multiple tilings provide generalization between nearby states
2. Learning in one state automatically updates nearby states
3. Memory usage is controlled (8 tilings * 8^4 = 32,768 tiles vs 10^4 discrete states)
"""

from core.tile_coding import TileCoding
from agents.tiled_qlearning import TiledQLearningAgent, TiledSARSALambdaAgent
from envs import CartPoleEnv


fn main() raises:
    """Run CartPole tile coding example."""
    print("=" * 60)
    print("CartPole with Tile Coding - Function Approximation")
    print("=" * 60)
    print("")

    # Train Q-learning
    print("-" * 60)
    # Create tile coding
    var tc = CartPoleEnv.make_tile_coding(
        num_tilings=8,
        tiles_per_dim=8,
    )

    # Create Tiled Q-learning agent
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Create environment
    var env = CartPoleEnv[DType.float64]()

    var metrics = agent.train(
        env,
        tc,
        num_episodes=10_000,
        max_steps_per_episode=500,
        verbose=True,
    )

    print("")
    print("=" * 60)
    print("Training Complete!")
    print("")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())
    print("Std reward:", metrics.std_reward())
    print("")
    print("CartPole solved when avg reward >= 475 over 100 episodes")

    var eval_reward = agent.evaluate(env, tc, num_episodes=10)
    print("  Average evaluation reward:", Int(eval_reward))
    print()

    if eval_reward >= 475:
        print("SUCCESS: CartPole solved!")
    else:
        print(
            "Training complete. Consider increasing episodes or tuning"
            " hyperparameters."
        )
    print("=" * 60)

    var _ = agent.evaluate(env, tc, num_episodes=3, render=True)

    env.close()
    print()
    print("Demo complete!")
