"""Deep SAC Demo on Pendulum Environment.

This example demonstrates training a Deep SAC agent on the native Pendulum
environment. SAC (Soft Actor-Critic) is a maximum entropy RL algorithm that
learns a stochastic policy for continuous control.

Run with:
    pixi run mojo run examples/deep_sac_demo.mojo
"""

from deep_agents import DeepSACAgent
from envs import PendulumEnv


fn main() raises:
    print("=" * 60)
    print("Deep SAC Demo on Pendulum")
    print("=" * 60)

    # Create environment
    var env = PendulumEnv()

    # Create Deep SAC agent
    # Pendulum: obs_dim=3 (cos θ, sin θ, θ_dot), action_dim=1 (torque in [-2, 2])
    var agent = DeepSACAgent[
        obs_dim=3,
        action_dim=1,
        hidden_dim=128,
        buffer_capacity=50000,
        batch_size=64,
    ](
        gamma=0.99,
        tau=0.005,
        actor_lr=0.0003,
        critic_lr=0.0003,
        action_scale=2.0,  # Pendulum actions in [-2, 2]
        alpha=0.2,  # Entropy coefficient
        auto_alpha=True,  # Automatically tune alpha
        target_entropy=-1.0,  # Target entropy (-action_dim)
    )

    # Print agent info
    agent.print_info()
    print("-" * 60)

    # Train the agent
    print("\nTraining Deep SAC agent...")
    _ = agent.train(
        env,
        num_episodes=100,
        max_steps_per_episode=200,
        warmup_steps=500,
        train_every=1,
        verbose=True,
        print_every=10,
        environment_name="Pendulum",
    )

    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    var eval_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps_per_episode=200,
        verbose=True,
    )
    print("\nMean evaluation reward: " + String(eval_reward)[:10])

    print("\n" + "=" * 60)
    print("Deep SAC Demo Complete!")
    print("=" * 60)
