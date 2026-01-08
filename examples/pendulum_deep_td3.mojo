"""Deep TD3 on Pendulum - Simple Training Example.

This demonstrates using the DeepTD3Agent's train() and evaluate()
functions for easy training on continuous control environments.

TD3 improves over DDPG with:
1. Twin critics (Q1, Q2) - uses min(Q1, Q2) to reduce overestimation
2. Delayed policy updates - actor updates every N critic updates
3. Target policy smoothing - adds clipped noise to target actions

Run with:
    pixi run mojo run examples/pendulum_deep_td3.mojo
"""

from deep_agents import DeepTD3Agent
from envs.pendulum import PendulumEnv


fn main() raises:
    print("=" * 60)
    print("Deep TD3 on Pendulum")
    print("=" * 60)
    print("")

    # Create environment
    var env = PendulumEnv()

    # Create Deep TD3 agent with compile-time dimensions
    # obs_dim=3: [cos(θ), sin(θ), θ_dot]
    # action_dim=1: torque
    var agent = DeepTD3Agent[
        obs_dim=3,
        action_dim=1,
        hidden_dim=128,
        buffer_capacity=50000,
        batch_size=64,
    ](
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.001,
        noise_std=0.2,
        noise_decay=0.995,
        action_scale=2.0,  # Pendulum actions in [-2, 2]
        # TD3-specific parameters:
        policy_delay=2,  # Update actor every 2 critic updates
        target_noise_std=0.2,  # Noise added to target actions
        target_noise_clip=0.5,  # Clip range for target noise
    )

    # Train the agent
    _ = agent.train(
        env,
        num_episodes=100,
        max_steps_per_episode=200,
        warmup_steps=1000,
        verbose=True,
        print_every=10,
        environment_name="Pendulum",
    )

    print("")

    # Evaluate the trained policy
    print("=" * 60)
    print("Evaluating trained policy (no noise)")
    print("=" * 60)
    var eval_reward = agent.evaluate(env, num_episodes=10, verbose=True)
    print("")
    print("Average evaluation reward: " + String(eval_reward)[:10])

    # Performance check
    if eval_reward > -200.0:
        print("Excellent! Near-optimal performance.")
    elif eval_reward > -500.0:
        print("Good! Agent learned effective control.")
    elif eval_reward > -1000.0:
        print("Moderate. Agent shows learning.")
    else:
        print("Needs more training.")

    # Render one episode
    var _ = agent.evaluate(env, num_episodes=1, verbose=True, render=True)

    print("")
    print("=" * 60)
    print("Done!")
    print("=" * 60)
