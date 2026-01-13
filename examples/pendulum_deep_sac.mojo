"""Deep SAC training on Pendulum environment.

This example demonstrates:
1. Native Mojo Pendulum environment with continuous actions
2. Deep SAC algorithm with neural network function approximation
3. Stochastic Gaussian policy for better exploration
4. Twin Q-networks to reduce overestimation
5. Automatic entropy temperature tuning
6. Maximum entropy RL objective

Deep SAC key features vs linear SAC:
- Neural network function approximation instead of polynomial features
- Direct mapping from observations to actions
- Better scalability to high-dimensional problems
- More expressive policy representation

The goal is to swing up the pendulum and balance it at the top.
Pendulum starts from a random position and must learn to apply
torque to reach the upright position.

Run with:
    pixi run mojo run examples/pendulum_deep_sac.mojo

Requirements:
    - SDL2 for rendering (optional): brew install sdl2 sdl2_ttf
"""

from envs import PendulumEnv
from deep_agents.sac import DeepSACAgent


fn main() raises:
    print("\n" + "=" * 60)
    print("    Deep SAC on Pendulum - Neural Network")
    print("=" * 60 + "\n")

    # Create environment
    var env = PendulumEnv()

    # Pendulum observation: [cos(theta), sin(theta), theta_dot]
    # Pendulum action: torque in [-2, 2]
    print("Observation dim: 3")
    print("Action dim: 1")
    print("Action range: [-2, 2]")

    # Create Deep SAC agent
    # obs_dim=3, action_dim=1, hidden_dim=64, buffer_capacity=50000, batch_size=64
    var agent = DeepSACAgent[3, 1, 64, 50000, 64](
        gamma=0.99,
        tau=0.005,
        actor_lr=0.0003,
        critic_lr=0.0003,
        action_scale=2.0,  # Pendulum torque range: [-2, 2]
        alpha=0.1,  # Initial entropy coefficient
        auto_alpha=False,  # Disable auto tuning for stability
        alpha_lr=0.0001,
        target_entropy=-1.0,  # Target entropy (approx -dim(action_space))
    )

    # Training parameters
    var num_episodes = 200
    var max_steps = 200
    var warmup_steps = 1000
    var print_every = 20

    print("\nStarting training...")
    print("Episodes:", num_episodes)
    print("Warmup steps:", warmup_steps)
    print("")

    # Train the agent
    var metrics = agent.train(
        env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        warmup_steps=warmup_steps,
        train_every=1,
        verbose=True,
        print_every=print_every,
        environment_name="Pendulum",
    )

    # Evaluate the trained agent
    print("\n" + "-" * 40)
    print("Evaluating trained agent (deterministic policy)...")
    var eval_reward = agent.evaluate(
        env,
        num_episodes=10,
        max_steps=200,
        render=False,
    )
    print("Evaluation average reward:", String(eval_reward)[:8])

    # Print final statistics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("Total training steps:", agent.get_train_steps())
    print("Final alpha:", String(agent.get_alpha())[:6])
    print("Final evaluation reward:", String(eval_reward)[:8])
    print("")
