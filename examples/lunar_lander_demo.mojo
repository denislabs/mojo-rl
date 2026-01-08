"""Demo of the native Mojo LunarLander environment with SDL2 rendering.

Run with: pixi run mojo run examples/lunar_lander_demo.mojo

Requires SDL2: brew install sdl2 sdl2_ttf

Features:
- Pure Mojo physics engine (no Box2D/Python dependency)
- SDL2 rendering showing lander, terrain, and helipad
- Simple heuristic controller (alternates with random actions)
- Watch the lander try to land on the helipad!

Actions: 0=nothing, 1=left engine, 2=main engine, 3=right engine
"""

from envs.lunar_lander import LunarLanderEnv
from random import random_float64, seed
from time import sleep


fn main() raises:
    print("=== LunarLander Native Rendering Demo ===")
    print("Pure Mojo implementation with custom physics engine")
    print("Close the window or press Ctrl+C to exit\n")

    # Seed random for reproducibility
    seed(42)

    # Create environment (discrete actions, no wind for clearer demo)
    var env = LunarLanderEnv(continuous=False, enable_wind=False)

    # Run episodes with rendering
    var num_episodes = 5
    for episode in range(num_episodes):
        var state = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0
        var done = False

        # Alternate between heuristic and random policies
        var use_heuristic = (episode % 2) == 0
        var policy_name = "Heuristic" if use_heuristic else "Random"
        print("Episode", episode + 1, "(", policy_name, "policy) starting...")

        while not done and steps < 1000:
            var action: Int

            if use_heuristic:
                # Simple heuristic policy
                action = _heuristic_action(state.x, state.y, state.vx, state.vy, state.angle, state.angular_velocity)
            else:
                # Random action
                action = Int(random_float64() * 4.0)
                if action > 3:
                    action = 3

            var result = env.step_discrete(action)
            # Copy the new state
            state = result[0].copy()
            var reward = result[1]
            done = result[2]

            total_reward += reward
            steps += 1

            # Render the frame
            env.render()

        var status = "SUCCESS!" if total_reward > 100 else "CRASHED"
        print(
            "Episode",
            episode + 1,
            "-",
            status,
            "| Reward:",
            Int(total_reward),
            "| Steps:",
            steps,
        )
        print()

        # Small pause between episodes
        sleep(0.5)

    # Clean up
    env.close()
    print("=== Demo Complete ===")


fn _heuristic_action(x: Float64, y: Float64, vx: Float64, vy: Float64, angle: Float64, angular_vel: Float64) -> Int:
    """Simple heuristic policy for LunarLander.

    Actions: 0=nop, 1=left engine, 2=main engine, 3=right engine
    """
    # If falling too fast, fire main engine
    if vy < -0.5:
        return 2  # Main engine

    # If tilted too much, correct with side engines
    if angle > 0.2:
        return 1  # Left engine to rotate right
    if angle < -0.2:
        return 3  # Right engine to rotate left

    # If rotating too fast, dampen
    if angular_vel > 0.2:
        return 1  # Left engine
    if angular_vel < -0.2:
        return 3  # Right engine

    # If drifting horizontally, correct
    if x > 0.2 and vx > 0:
        return 1  # Left engine
    if x < -0.2 and vx < 0:
        return 3  # Right engine

    # If still high up and not falling fast, do nothing
    if y > 0.5:
        return 0  # No action

    # Near the ground, fire main engine to slow descent
    if vy < -0.1:
        return 2  # Main engine

    return 0  # Default: no action
