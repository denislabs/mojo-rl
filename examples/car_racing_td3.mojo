"""
CarRacing TD3 Training

Trains a TD3 (Twin Delayed DDPG) agent on the CarRacing environment.

TD3 is well-suited for continuous control tasks like CarRacing because:
- Deterministic policy with exploration noise
- Twin Q-networks reduce overestimation bias
- Delayed policy updates improve stability
- Target policy smoothing adds regularization

Run with:
    pixi run mojo run examples/car_racing_td3.mojo
"""

from envs.car_racing import CarRacingEnv
from deep_agents.td3 import DeepTD3Agent


fn main() raises:
    print("=" * 60)
    print("CarRacing TD3 Training")
    print("=" * 60)
    print("")
    print("Environment: CarRacing (continuous control)")
    print("Algorithm: TD3 (Twin Delayed DDPG)")
    print("Observation dim: 13")
    print("Action dim: 3 (steering, gas, brake)")
    print("")

    # Create environment
    var env = CarRacingEnv(continuous=True)

    # Create TD3 agent
    # Parameters: obs_dim=13, action_dim=3, hidden_dim=128, buffer=10000, batch=64
    # NOTE: Using smaller buffer (10k) and hidden_dim (128) to avoid stack overflow
    # with stack-allocated InlineArrays in the ReplayBuffer.
    var agent = DeepTD3Agent[13, 3, 128, 10000, 64](
        actor_lr=0.0003,
        critic_lr=0.0003,
        gamma=0.99,
        tau=0.005,
        noise_std=0.2,  # Higher initial noise for exploration
        noise_decay=0.999,
        noise_std_min=0.05,
        target_noise_std=0.2,
        target_noise_clip=0.5,
        policy_delay=2,
        action_scale=1.0,
    )

    print("Agent created with:")
    print("  - Hidden dim: 128")
    print("  - Buffer size: 10000")
    print("  - Actor LR: 0.0003")
    print("  - Critic LR: 0.0003")
    print("  - Gamma: 0.99")
    print("  - Tau: 0.005")
    print("  - Exploration noise: 0.2 -> 0.05")
    print("  - Target noise: 0.2 (clipped to 0.5)")
    print("  - Policy delay: 2")
    print("")
    print("Starting training...")
    print("")

    # Train the agent
    var metrics = agent.train(
        env,
        num_episodes=300,
        max_steps_per_episode=1000,
        warmup_steps=1000,  # Reduced to match smaller buffer
        train_every=1,
        verbose=True,
        print_every=10,
        environment_name="CarRacing",
    )

    # Print final results
    print("")
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    metrics.print_summary()

    env.close()
