"""Deep SAC training on BipedalWalker environment.

This example demonstrates:
1. Native Mojo BipedalWalker environment with 24D observations
2. Deep SAC algorithm with 4D continuous action space
3. Twin Q-networks to reduce overestimation
4. Automatic entropy temperature tuning
5. Stochastic Gaussian policy for exploration

The goal is to train the bipedal walker to walk forward without falling.
The walker must learn to coordinate four motor joints (two hips, two knees)
to achieve stable locomotion.

Observation space: 24D
- Hull angle and angular velocity (2)
- Horizontal and vertical velocity (2)
- Joint angles and speeds (8)
- Leg ground contacts (2)
- 10-beam lidar readings (10)

Action space: 4D continuous [-1, 1]
- hip1, knee1, hip2, knee2 torques

Run with:
    pixi run mojo run examples/bipedal_walker_sac.mojo

Requirements:
    - SDL2 for rendering: brew install sdl2 sdl2_ttf
"""

from random import seed

from envs import BipedalWalkerEnv
from deep_agents.sac import DeepSACAgent


fn main() raises:
    print("\n" + "=" * 60)
    print("    Deep SAC on BipedalWalker - Neural Network")
    print("=" * 60 + "\n")

    # Seed for reproducibility
    seed(42)

    # Create environment (normal mode)
    var env = BipedalWalkerEnv(hardcore=False)

    # BipedalWalker observation: 24D (hull state, joint states, lidar)
    # BipedalWalker action: 4D continuous (hip1, knee1, hip2, knee2)
    print("Observation dim: 24")
    print("Action dim: 4")
    print("Action range: [-1, 1]")

    # Create Deep SAC agent
    # obs_dim=24, action_dim=4, hidden_dim=128, buffer_capacity=10000, batch_size=64
    #
    # NOTE: BipedalWalker has 24D observations. The default ReplayBuffer uses
    # stack-allocated InlineArrays which can cause stack overflow with large
    # observation spaces and buffer sizes. We use smaller buffer (10k) and
    # hidden_dim (128) to avoid this. For production training, consider:
    # 1. Using HeapReplayBuffer (requires custom agent implementation)
    # 2. Running with increased stack size: ulimit -s unlimited
    var agent = DeepSACAgent[24, 4, 128, 10000, 64](
        gamma=0.99,
        tau=0.005,
        actor_lr=0.0003,
        critic_lr=0.0003,
        action_scale=1.0,  # BipedalWalker action range: [-1, 1]
        alpha=0.2,  # Initial entropy coefficient
        auto_alpha=True,  # Automatically tune alpha
        alpha_lr=0.0003,
        target_entropy=-4.0,  # Target entropy (approx -dim(action_space))
    )

    # Training parameters
    var num_episodes = 500
    var max_steps = 1600
    var warmup_steps = 1000  # Reduced to match smaller buffer
    var print_every = 25

    print("\nStarting training...")
    print("Episodes:", num_episodes)
    print("Max steps per episode:", max_steps)
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
        environment_name="BipedalWalker",
    )

    # Evaluate the trained agent
    print("\n" + "-" * 40)
    print("Evaluating trained agent (deterministic policy)...")
    var eval_reward = agent.evaluate(
        env,
        num_episodes=5,
        max_steps=1600,
        render=True,  # Render evaluation episodes
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

    # Cleanup
    env.close()
