"""DDPG training on Pendulum environment with linear function approximation.

This example demonstrates:
1. Native Mojo Pendulum environment with continuous actions
2. DDPG algorithm with polynomial feature extraction
3. Experience replay buffer for off-policy learning
4. Target networks for stable training
5. Gaussian exploration noise
6. Using agent.train() and agent.evaluate() methods

The goal is to swing up the pendulum and balance it at the top.
Pendulum starts from a random position and must learn to apply
torque to reach the upright position.

Run with:
    pixi run mojo run examples/pendulum_ddpg.mojo

Requirements:
    - SDL2 for rendering (optional): brew install sdl2 sdl2_ttf
"""

from envs import PendulumEnv
from core import ContinuousReplayBuffer, PolynomialFeatures, TrainingMetrics
from agents import DDPGAgent


fn main() raises:
    print("\n" + "=" * 60)
    print("    DDPG on Pendulum - Linear Function Approximation")
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

    # Create DDPG agent
    var agent = DDPGAgent(
        num_state_features=features.get_num_features(),
        action_scale=2.0,  # Pendulum torque range: [-2, 2]
        actor_lr=0.001,
        critic_lr=0.002,
        discount_factor=0.99,
        tau=0.005,
        noise_std=0.2,
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
    for demo_ep in range(3):
        var obs = env.reset_obs()
        var demo_reward: Float64 = 0.0

        for step in range(200):
            var state_features = features.get_features_simd4(obs)
            var action = agent.select_action(state_features)

            var result = env.step_continuous(action)
            obs = result[0]
            demo_reward += result[1]

            env.render()

            if result[2]:
                break

        print("Demo episode", demo_ep + 1, "reward:", String(demo_reward)[:8])

    env.close()
    print("\nDemo complete!")
