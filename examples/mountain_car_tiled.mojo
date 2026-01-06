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
from envs.mountain_car import MountainCarEnv


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
    var tc = MountainCarEnv.make_tile_coding(num_tilings=8, tiles_per_dim=8)

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

    var env = MountainCarEnv()

    # Training phase
    print("-" * 60)
    print("Training Tiled Q-Learning on MountainCar")
    print("  Tilings:", tc.get_num_tilings())
    print("  Total tiles:", tc.get_num_tiles())
    print("")

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
    print("Now showing learned policy with visualization...")
    print("Close the window when done watching.")
    print("=" * 60)
    print("")

    # Visualization phase - show trained agent using integrated render()
    var _ = agent.evaluate(env, tc, num_episodes=3, render=True)

    env.close()
    print("")
    print("Demo complete!")
