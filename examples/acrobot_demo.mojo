"""Demo of the native Mojo Acrobot environment.

Run with: pixi run mojo run examples/acrobot_demo.mojo
"""

from envs import AcrobotEnv, AcrobotAction
from random import random_float64


fn main() raises:
    print("=== Acrobot Environment Demo ===\n")

    # Create environment
    var env = AcrobotEnv(num_bins=6)

    print("Environment created successfully!")
    print("Observation dimension:", env.obs_dim())
    print("Number of actions:", env.num_actions())
    print("Number of discrete states:", env.num_states())
    print()

    # Run a few episodes
    var num_episodes = 3
    for episode in range(num_episodes):
        _ = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0

        print("Episode", episode + 1)

        while not env.is_done():
            # Random action
            var action_idx = Int(random_float64() * 3.0)
            if action_idx > 2:
                action_idx = 2
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            var reward = result[1]
            total_reward += reward
            steps += 1

            # Print first few steps
            if steps <= 3:
                _ = env.get_obs()
                print(
                    "  Step",
                    steps,
                    "| Action:",
                    action_idx,
                    "| Reward:",
                    reward,
                )

        print(
            "  Done! Total reward:",
            total_reward,
            "| Steps:",
            steps,
        )
        print()

    # Test raw observation API
    print("Testing raw observation API...")
    var obs = env.reset_obs()
    print("Initial observation (6D):")
    print(
        "  cos(θ1):",
        obs[0],
        "sin(θ1):",
        obs[1],
    )
    print(
        "  cos(θ2):",
        obs[2],
        "sin(θ2):",
        obs[3],
    )
    print(
        "  θ1_dot:",
        obs[4],
        "θ2_dot:",
        obs[5],
    )
    print()

    # Test step_raw
    var result = env.step_raw(2)  # Apply positive torque
    var new_obs = result[0]
    var reward = result[1]
    var done = result[2]
    print("After applying +1 torque:")
    print(
        "  cos(θ1):",
        new_obs[0],
        "sin(θ1):",
        new_obs[1],
    )
    print("  Reward:", reward, "Done:", done)
    print()

    # Test tile coding factory
    print("Testing tile coding factory...")
    _ = AcrobotEnv.make_tile_coding(num_tilings=8, tiles_per_dim=8)
    print("Tile coding created with", 8, "tilings")
    print()

    # Test polynomial features factory
    print("Testing polynomial features factory...")
    _ = AcrobotEnv.make_poly_features(degree=2)
    print("Polynomial features created with degree 2")
    print()

    print("=== Acrobot Demo Complete ===")
