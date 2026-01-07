"""Solve Pendulum - Continuous Control with DDPG/TD3/SAC.

Pendulum is a continuous control task where a frictionless pendulum starts
from a random position and must be swung up and balanced at the top.

State: [cos(theta), sin(theta), theta_dot] (3D continuous)
Action: torque in [-2.0, 2.0] (continuous!)

The pendulum hangs downward initially (theta = pi) and must learn to
apply torque to swing up and balance at the top (theta = 0).

Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
  - Maximum reward per step: 0 (balanced at top, no velocity, no torque)
  - Episode reward typically ranges from -2000 (random) to -200 (good)

Key challenge: CONTINUOUS ACTION SPACE
Unlike discrete environments, Pendulum requires algorithms that can
output continuous actions (like DDPG, TD3, or SAC).

Best algorithms for Pendulum:
1. DDPG: Deep Deterministic Policy Gradient - deterministic actor
2. TD3: Twin Delayed DDPG - reduced overestimation with twin critics
3. SAC: Soft Actor-Critic - maximum entropy RL

Run with:
    pixi run mojo run examples/solve_pendulum.mojo

Requires SDL2 for visualization: brew install sdl2 sdl2_ttf
"""

from envs import PendulumEnv
from core import ContinuousReplayBuffer, PolynomialFeatures, TrainingMetrics
from agents import DDPGAgent, TD3Agent


fn main() raises:
    print("=" * 60)
    print("    Solving Pendulum - Continuous Control")
    print("=" * 60)
    print("")
    print("Environment: Pendulum-v1")
    print("State: 3D continuous [cos(theta), sin(theta), theta_dot]")
    print("Action: CONTINUOUS torque in [-2.0, 2.0]")
    print("Goal: Swing up and balance at the top")
    print("Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)")
    print("")
    print("Good performance: avg reward > -300")
    print("Excellent performance: avg reward > -200")
    print("")

    # Create polynomial feature extractor (degree 3 is more expressive)
    var features = PendulumEnv.make_poly_features(degree=3)
    print("Feature configuration:")
    print("  Polynomial degree: 3")
    print("  Feature dimensionality:", features.get_num_features())
    print("")

    # Training parameters
    var num_episodes = 1_000
    var max_steps = 200
    var batch_size = 128
    var min_buffer_size = 2000
    var warmup_episodes = 20

    # ========================================================================
    # Algorithm 1: DDPG (Deep Deterministic Policy Gradient)
    # ========================================================================
    print("-" * 60)
    print("Algorithm 1: DDPG")
    print("-" * 60)
    print("Deterministic actor with Gaussian exploration noise.")
    print("")

    var env_ddpg = PendulumEnv()
    var buffer_ddpg = ContinuousReplayBuffer(
        capacity=100000,
        feature_dim=features.get_num_features(),
    )
    # Use optimized hyperparameters with noise decay and reward scaling
    var agent_ddpg = DDPGAgent(
        num_state_features=features.get_num_features(),
        action_scale=2.0,  # Pendulum torque range: [-2, 2]
        actor_lr=0.0003,  # Lower actor LR for stability
        critic_lr=0.001,
        discount_factor=0.99,
        tau=0.005,
        noise_std=0.3,  # Start with higher noise
        noise_std_min=0.05,
        noise_decay=0.995,
        reward_scale=0.1,  # Scale rewards for stability
        updates_per_step=2,  # More updates per step
        init_std=0.1,
    )

    var metrics_ddpg = agent_ddpg.train(
        env_ddpg,
        features,
        buffer_ddpg,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        batch_size=batch_size,
        min_buffer_size=min_buffer_size,
        warmup_episodes=warmup_episodes,
        verbose=True,
        print_every=30,
        environment_name="Pendulum",
    )

    print("")
    print("DDPG results:")
    print("  Mean reward:", String(metrics_ddpg.mean_reward())[:8])
    print("  Max reward:", String(metrics_ddpg.max_reward())[:8])
    print("")

    # ========================================================================
    # Algorithm 2: TD3 (Twin Delayed DDPG)
    # ========================================================================
    print("-" * 60)
    print("Algorithm 2: TD3 (Twin Delayed DDPG)")
    print("-" * 60)
    print("Improved DDPG with twin critics and delayed policy updates.")
    print("")

    var env_td3 = PendulumEnv()
    var buffer_td3 = ContinuousReplayBuffer(
        capacity=100000,
        feature_dim=features.get_num_features(),
    )
    # Use optimized hyperparameters with noise decay and reward scaling
    var agent_td3 = TD3Agent(
        num_state_features=features.get_num_features(),
        action_scale=2.0,
        actor_lr=0.0003,  # Lower actor LR for stability
        critic_lr=0.001,
        discount_factor=0.99,
        tau=0.005,
        noise_std=0.3,  # Start with higher noise
        noise_std_min=0.05,
        noise_decay=0.995,
        reward_scale=0.1,  # Scale rewards for stability
        updates_per_step=2,  # More updates per step
        policy_delay=2,  # Update policy every 2 critic updates
        target_noise_std=0.2,
        target_noise_clip=0.5,
        init_std=0.1,
    )

    var metrics_td3 = agent_td3.train(
        env_td3,
        features,
        buffer_td3,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        batch_size=batch_size,
        min_buffer_size=min_buffer_size,
        warmup_episodes=warmup_episodes,
        verbose=True,
        print_every=30,
        environment_name="Pendulum",
    )

    print("")
    print("TD3 results:")
    print("  Mean reward:", String(metrics_td3.mean_reward())[:8])
    print("  Max reward:", String(metrics_td3.max_reward())[:8])
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Algorithm | Mean Reward | Max Reward")
    print("-" * 60)
    print(
        "DDPG      |",
        String(metrics_ddpg.mean_reward())[:8],
        "   |",
        String(metrics_ddpg.max_reward())[:8],
    )
    print(
        "TD3       |",
        String(metrics_td3.mean_reward())[:8],
        "   |",
        String(metrics_td3.max_reward())[:8],
    )
    print("")
    print("Good: avg reward > -300 | Excellent: avg reward > -200")
    print("")

    # ========================================================================
    # Evaluation
    # ========================================================================
    print("-" * 60)
    print("Evaluation (deterministic policy):")
    print("-" * 60)

    var eval_ddpg = agent_ddpg.evaluate(
        env_ddpg, features, num_episodes=10, max_steps_per_episode=200
    )
    var eval_td3 = agent_td3.evaluate(
        env_td3, features, num_episodes=10, max_steps_per_episode=200
    )

    print("DDPG eval avg reward:", String(eval_ddpg)[:8])
    print("TD3 eval avg reward:", String(eval_td3)[:8])
    print("")

    # ========================================================================
    # Visual Demo
    # ========================================================================
    print("-" * 60)
    print("Visual Demo - Watch the trained agent!")
    print("-" * 60)
    print("Using TD3 agent (deterministic policy) with SDL2 rendering.")
    print("Watch the pendulum swing up and balance!")
    print("Close the window when done watching.")
    print("")

    # Run visual demo using SIMD API for performance
    var _ = agent_td3.evaluate(
        env_td3,
        features,
        num_episodes=1,
        max_steps_per_episode=200,
        render=True,
    )

    env_td3.close()

    print("")
    print("Demo complete!")
    print("")
    print("Note: Pendulum is never 'solved' (no terminal state).")
    print("A good policy keeps the pendulum balanced at the top")
    print("with minimal oscillation and energy expenditure.")
    print("=" * 60)
