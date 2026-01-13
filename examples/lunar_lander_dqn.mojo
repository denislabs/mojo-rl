"""Train Deep DQN on LunarLander.

Run with: pixi run mojo run examples/lunar_lander_dqn.mojo

This trains a Deep Q-Network on the native Mojo LunarLander environment.
DQN uses:
- Neural network Q-function (2-layer MLP with ReLU)
- Target network with soft updates
- Experience replay
- Epsilon-greedy exploration with decay

Expected performance:
- Episode 100: ~-200 avg reward (still learning)
- Episode 300: ~-50 avg reward (improving)
- Episode 500: ~100+ avg reward (solving)

LunarLander is solved when average reward > 200 over 100 episodes.
"""

from random import seed

from envs.lunar_lander import LunarLanderEnv
from deep_agents import DQNAgent


fn main() raises:
    print("=" * 60)
    print("Deep DQN on LunarLander")
    print("=" * 60)
    print()

    # Seed for reproducibility
    seed(42)

    # Create environment
    var env = LunarLanderEnv(continuous=False, enable_wind=False)

    # Create DQN agent with tuned hyperparameters
    # LunarLander: 8D observations, 4 discrete actions
    #
    # Best hyperparameters found for LunarLander (Double DQN):
    # - hidden_dim=128: Good capacity for LunarLander
    # - lr=5e-4: Stable learning rate
    # - gamma=0.99: Standard discount factor
    # - epsilon_decay=0.997: Slow decay for good exploration
    # - tau=0.005: Standard target update rate
    # - Double DQN enabled (default) for reduced overestimation
    var agent = DQNAgent[
        obs_dim=8,
        num_actions=4,
        hidden_dim=128,
        buffer_capacity=20000,
        batch_size=64,
    ](
        gamma=0.99,  # Standard discount
        tau=0.005,  # Standard target update
        lr=0.0005,  # Stable learning rate (5e-4)
        epsilon=1.0,  # Start with full exploration
        epsilon_min=0.01,  # Low minimum
        epsilon_decay=0.997,  # Slow decay for exploration
    )

    # Train
    var metrics = agent.train(
        env,
        num_episodes=600,  # Sufficient for convergence
        max_steps_per_episode=1000,
        warmup_steps=5000,  # Warmup for diverse experiences
        train_every=4,  # Standard DQN training frequency
        verbose=True,
        print_every=25,
        environment_name="LunarLander",
    )

    print()
    print("=" * 60)
    print("Evaluation (no exploration)")
    print("=" * 60)

    # Evaluate without rendering first
    var eval_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=1000,
        verbose=True,
        render=False,
    )
    print("Average evaluation reward: " + String(eval_reward)[:10])

    # Final evaluation with rendering
    print()
    print("Running 3 rendered episodes...")
    _ = agent.evaluate(
        env,
        num_episodes=3,
        max_steps_per_episode=1000,
        verbose=True,
        render=True,
    )

    env.close()
    print()
    print("Training complete!")
