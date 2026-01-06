"""Native CartPole Demo with integrated SDL2 Rendering.

Trains a Q-Learning agent on the native Mojo CartPole using the
training function, then visualizes the trained agent.
"""

from envs import CartPoleEnv, CartPoleAction
from agents.qlearning import QLearningAgent
from random import seed


fn main() raises:
    seed()

    print("=" * 60)
    print("Native Mojo CartPole with Integrated SDL2 Rendering")
    print("=" * 60)
    print()

    # Hyperparameters
    var num_bins = 10
    var num_episodes = 2000  # More episodes for better policy
    var max_steps = 500

    var num_states = CartPoleEnv.get_num_states(num_bins)
    var num_actions = 2

    print("Configuration:")
    print("  State space:", num_states, "discretized states")
    print("  Episodes:", num_episodes)
    print()

    # Initialize environment and agent
    var env = CartPoleEnv(num_bins=num_bins)
    var agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,  # Slower decay for better exploration
        epsilon_min=0.01,
    )

    print("Training (pure Mojo, no rendering)...")
    var metrics = agent.train(
        env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
    )

    print()
    print("Training complete! Best reward:", Int(metrics.max_reward()))
    print()

    # Evaluation without rendering
    print("Evaluating (10 episodes, no render)...")
    var eval_reward = agent.evaluate(env, num_episodes=10)
    print("  Average evaluation reward:", Int(eval_reward))
    print()

    # Visual demo with integrated SDL2 renderer
    print("=" * 60)
    print("Visual Demo - Watch the trained agent!")
    print("=" * 60)
    print("Rendering 3 episodes with integrated SDL2...")
    print("(Close window to exit)")
    print()

    var _ = agent.evaluate(env, num_episodes=3, render=True)

    env.close()
    print()
    print("Demo complete!")
