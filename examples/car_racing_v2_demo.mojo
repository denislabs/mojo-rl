"""CarRacing V2 Demo - Test rendering and basic functionality.

This demo creates a CarRacingV2 environment and runs it with keyboard controls
or random actions to test the rendering system.

Usage:
    pixi run mojo run examples/car_racing_v2_demo.mojo
"""

from time import sleep
from random import random_float64

from envs.car_racing import CarRacingV2, CarRacingV2Action
from envs.car_racing.constants import CRConstants
from render import RendererBase


fn main() raises:
    print("=== CarRacing V2 Demo ===")
    print("Testing GPU-accelerated CarRacing environment with rendering")
    print()

    # Create environment
    var env = CarRacingV2(max_steps=500)
    print("Environment created")
    print("  Observation dim:", env.obs_dim())
    print("  Action dim:", env.action_dim())
    print("  Track tiles:", env.track.track_length)
    print()

    # Create renderer
    var renderer = RendererBase(
        CRConstants.WINDOW_W,
        CRConstants.WINDOW_H,
        CRConstants.FPS,
        "CarRacing V2 Demo",
    )
    print("Renderer created:", CRConstants.WINDOW_W, "x", CRConstants.WINDOW_H)
    print()

    # Reset environment
    var state = env.reset()
    print("Environment reset")
    print("  Initial position: (", state.x, ",", state.y, ")")
    print("  Initial angle:", state.angle)
    print()

    print("Running demo with random actions...")
    print("Press Ctrl+C to stop")
    print()

    var total_reward: Float64 = 0.0
    var step = 0
    var done = False

    while not done and step < 500:
        # Generate random action (with slight gas bias to move forward)
        var steering = Float64(random_float64()) * 2.0 - 1.0  # [-1, 1]
        var gas = Float64(random_float64()) * 0.8  # [0, 0.8] - mostly forward
        var brake = Float64(random_float64()) * 0.2  # [0, 0.2] - occasional brake

        # Remap to [-1, 1] range expected by environment
        gas = gas * 2.0 - 1.0  # Now [-1, 0.6]
        brake = brake * 2.0 - 1.0  # Now [-1, -0.6]

        var action = CarRacingV2Action[DType.float64](steering, gas, brake)

        # Step environment
        var result = env.step(action)
        state = result[0]
        var reward = result[1]
        done = result[2]

        total_reward += reward
        step += 1

        # Render
        env.render(renderer)

        # Print progress every 50 steps
        if step % 50 == 0:
            print(
                "Step",
                step,
                "| Reward:",
                reward,
                "| Total:",
                total_reward,
                "| Tiles:",
                env.tiles_visited,
                "/",
                env.track.track_length,
            )

        # Small delay for visualization (50 FPS)
        sleep(0.02)

    print()
    print("=== Demo Complete ===")
    print("Total steps:", step)
    print("Total reward:", total_reward)
    print("Tiles visited:", env.tiles_visited, "/", env.track.track_length)

    # Keep window open for a moment
    print()
    print("Closing in 2 seconds...")
    sleep(2.0)
