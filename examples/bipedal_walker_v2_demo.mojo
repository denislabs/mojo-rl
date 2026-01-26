"""Demo of BipedalWalker v2 GPU environment (CPU single-env mode).

Run with: pixi run mojo run examples/bipedal_walker_v2_demo.mojo

Features:
- Uses physics_gpu module for physics simulation
- Motor-enabled revolute joints for leg control
- Edge terrain collision
- 24D observation (hull state + joint states + lidar)
- 4D continuous action (hip and knee torques)

This demo runs the CPU single-environment mode for visualization.
For batch GPU training, see bipedal_walker_v2_gpu_demo.mojo.
"""

from envs.bipedal_walker import BipedalWalkerV2, BipedalWalkerState, BipedalWalkerAction
from random import random_float64, seed
from time import sleep


fn main() raises:
    print("=== BipedalWalker v2 Demo (CPU Mode) ===")
    print("Using physics_gpu module with motor-enabled joints")
    print()

    # Seed random for reproducibility
    seed(42)

    # Create environment
    var env = BipedalWalkerV2[DType.float32](seed=42)

    # Run episodes
    var num_episodes = 3
    for episode in range(num_episodes):
        var state = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0
        var done = False

        print("Episode", episode + 1, "starting...")

        while not done and steps < 500:
            # Random action with slight bias
            var hip1 = Float32((random_float64() * 2.0 - 1.0) * 0.5)
            var knee1 = Float32((random_float64() * 2.0 - 1.0) * 0.5)
            var hip2 = Float32((random_float64() * 2.0 - 1.0) * 0.5)
            var knee2 = Float32((random_float64() * 2.0 - 1.0) * 0.5)

            var action = BipedalWalkerAction[DType.float32](hip1, knee1, hip2, knee2)
            var result = env.step(action)
            state = result[0]
            var reward = Float64(result[1])
            done = result[2]

            total_reward += reward
            steps += 1

            # Print progress every 100 steps
            if steps % 100 == 0:
                print("  Step", steps, "| Reward:", Int(total_reward))
                print("    Hull angle:", Float64(state.hull_angle))
                print("    Vel x:", Float64(state.vel_x), "y:", Float64(state.vel_y))
                print("    Leg contacts: L=", state.leg1_contact, "R=", state.leg2_contact)

        print(
            "Episode",
            episode + 1,
            "| Total Reward:",
            Int(total_reward),
            "| Steps:",
            steps,
        )
        print()

    # Test observation dimensions
    print("=== Observation Test ===")
    var obs = env.get_obs_list()
    print("Observation dimension:", len(obs))
    print("First 4 values (hull state):")
    print("  hull_angle:", Float64(obs[0]))
    print("  hull_angular_velocity:", Float64(obs[1]))
    print("  vel_x:", Float64(obs[2]))
    print("  vel_y:", Float64(obs[3]))

    print()
    print("Leg 1 state (indices 4-8):")
    print("  hip1_angle:", Float64(obs[4]))
    print("  hip1_speed:", Float64(obs[5]))
    print("  knee1_angle:", Float64(obs[6]))
    print("  knee1_speed:", Float64(obs[7]))
    print("  leg1_contact:", Float64(obs[8]))

    print()
    print("Lidar values (indices 14-23):")
    for i in range(10):
        print("  lidar[", i, "]:", Float64(obs[14 + i]))

    print()
    print("=== Demo Complete ===")
