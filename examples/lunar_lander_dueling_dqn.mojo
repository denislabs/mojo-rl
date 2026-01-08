"""Train Deep Dueling DQN on LunarLander.

Run with: pixi run mojo run examples/lunar_lander_dueling_dqn.mojo

This trains a Dueling Deep Q-Network on the native Mojo LunarLander environment.

Dueling DQN separates the Q-network into two streams:
- Value stream V(s): How good is this state?
- Advantage stream A(s,a): How much better is action a than others?

Final Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A))

This architecture helps the network learn state values without needing to
learn the effect of each action at every state - particularly useful when
actions don't always affect the outcome (e.g., when the lander is high up,
the exact action matters less than when close to the ground).

Reference: Wang et al. "Dueling Network Architectures for Deep RL" (2016)

Expected performance:
- Episode 100: ~-200 avg reward (still learning)
- Episode 300: ~-50 avg reward (improving)
- Episode 500: ~100+ avg reward (solving)

LunarLander is solved when average reward > 200 over 100 episodes.
"""

from random import seed

from envs.lunar_lander import LunarLanderEnv
from deep_agents.cpu import DeepDuelingDQNAgent


fn main() raises:
    print("=" * 60)
    print("Deep Dueling DQN on LunarLander")
    print("=" * 60)
    print()

    # Seed for reproducibility
    seed(42)

    # Create environment
    var env = LunarLanderEnv(continuous=False, enable_wind=False)

    # Create Dueling DQN agent with tuned hyperparameters
    # LunarLander: 8D observations, 4 discrete actions
    #
    # Architecture:
    # - Shared: 8 -> 128 (relu) -> 128 (relu)
    # - Value stream: 128 -> 64 (relu) -> 1
    # - Advantage stream: 128 -> 64 (relu) -> 4
    # - Q(s,a) = V(s) + (A(s,a) - mean(A))
    #
    # Using Double DQN (default) for reduced overestimation
    var agent = DeepDuelingDQNAgent[
        obs_dim=8,
        num_actions=4,
        hidden_dim=128,  # Shared feature layers
        stream_hidden_dim=64,  # Value/Advantage stream hidden
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
    _ = agent.train(
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
