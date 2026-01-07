"""TD3 training on Pendulum environment with linear function approximation.

This example demonstrates:
1. Native Mojo Pendulum environment with continuous actions
2. TD3 algorithm with polynomial feature extraction
3. Twin Q-networks to reduce overestimation
4. Delayed policy updates for stable learning
5. Target policy smoothing to prevent Q-function exploitation
6. Using agent.train() and agent.evaluate() methods

TD3 improvements over DDPG:
- Twin critics: min(Q1, Q2) reduces overestimation bias
- Delayed updates: actor updates every 2 critic updates
- Smoothed targets: noise added to target actions

The goal is to swing up the pendulum and balance it at the top.
Pendulum starts from a random position and must learn to apply
torque to reach the upright position.

Run with:
    pixi run mojo run examples/pendulum_td3.mojo

Requirements:
    - SDL2 for rendering (optional): brew install sdl2 sdl2_ttf
"""

from envs import PendulumEnv
from core import ContinuousReplayBuffer, PolynomialFeatures, TrainingMetrics
from agents import TD3Agent


fn main() raises:
    print("\n" + "=" * 60)
    print("    TD3 on Pendulum - Linear Function Approximation")
    print("=" * 60 + "\n")

    # Create environment
    var env = PendulumEnv()

    # Create polynomial feature extractor (degree 2)
    # Observation: [cos(θ), sin(θ), θ_dot]
    var features = PendulumEnv.make_poly_features(degree=2)
    print("Feature dimensionality:", features.get_num_features())

    # Create replay buffer
    var buffer = ContinuousReplayBuffer(
        capacity=100000,
        feature_dim=features.get_num_features(),
    )

    # Create TD3 agent
    var agent = TD3Agent(
        num_state_features=features.get_num_features(),
        action_scale=2.0,  # Pendulum torque range: [-2, 2]
        actor_lr=0.001,
        critic_lr=0.002,
        discount_factor=0.99,
        tau=0.005,
        noise_std=0.1,  # Exploration noise
        policy_delay=2,  # Update actor every 2 critic updates
        target_noise_std=0.2,  # Target policy smoothing noise
        target_noise_clip=0.5,  # Clip target noise
        init_std=0.1,
    )

    # Training parameters
    var num_episodes = 300
    var max_steps = 200
    var batch_size = 64
    var min_buffer_size = 1000
    var warmup_episodes = 10
    var print_every = 20

    # Train the agent using agent.train() method
    print("\nStarting training...")
    _ = agent.train(
        env,
        features,
        buffer,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        batch_size=batch_size,
        min_buffer_size=min_buffer_size,
        warmup_episodes=warmup_episodes,
        verbose=True,
        print_every=print_every,
        environment_name="Pendulum",
    )

    # Evaluate the trained agent using agent.evaluate() method
    print("\nEvaluating trained agent...")
    var eval_reward = agent.evaluate(
        env,
        features,
        num_episodes=10,
        max_steps_per_episode=200,
    )
    print("Evaluation average reward:", String(eval_reward)[:8])

    # Demo with rendering (if SDL2 available)
    print("\nRunning visual demo (close window to exit)...")
    print("Watch the pendulum swing up and balance!")

    # Run a few episodes with rendering
    var _ = agent.evaluate(
        env,
        features,
        num_episodes=3,
        max_steps_per_episode=200,
        render=True,
    )

    env.close()
    print("\nDemo complete!")
