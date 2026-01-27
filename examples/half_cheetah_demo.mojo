"""HalfCheetah Demo - Test rendering and basic functionality.

This demo creates a HalfCheetahPlanar environment and runs it with random actions
to test the rendering system.

Usage:
    pixi run mojo run examples/half_cheetah_demo.mojo
"""

from time import sleep
from random import random_float64

from envs.half_cheetah import HalfCheetahPlanarV2, HalfCheetahPlanarAction
from envs.half_cheetah.constants import HCConstants
from render import RendererBase


fn main() raises:
    print("=== HalfCheetah Planar Demo ===")
    print("Testing HalfCheetah environment with rendering")
    print()

    # Create environment
    var env = HalfCheetahPlanarV2[DType.float64]()
    print("Environment created")
    print("  Observation dim:", HCConstants.OBS_DIM_VAL)
    print("  Action dim:", HCConstants.ACTION_DIM_VAL)
    print("  Max steps:", HCConstants.MAX_STEPS)
    print()

    # Create renderer (wider window to see horizontal movement)
    var renderer = RendererBase(
        800,  # width
        400,  # height
        30,  # fps
        "HalfCheetah Planar Demo",
    )
    print("Renderer created: 800 x 400")
    print()

    # Reset environment
    var state = env.reset()
    print("Environment reset")
    print("  Initial torso height:", state.torso_z)
    print("  Initial torso angle:", state.torso_angle)
    print()

    print("Running demo with random actions...")
    print("Press Ctrl+C to stop or close the window")
    print()

    var total_reward: Float64 = 0.0
    var step = 0
    var done = False

    while not done and step < 500:
        # Generate random actions for all 6 joints
        # Actions are in [-1, 1] range
        var bthigh = Scalar[DType.float64](random_float64() * 2.0 - 1.0)
        var bshin = Scalar[DType.float64](random_float64() * 2.0 - 1.0)
        var bfoot = Scalar[DType.float64](random_float64() * 2.0 - 1.0)
        var fthigh = Scalar[DType.float64](random_float64() * 2.0 - 1.0)
        var fshin = Scalar[DType.float64](random_float64() * 2.0 - 1.0)
        var ffoot = Scalar[DType.float64](random_float64() * 2.0 - 1.0)

        var action = HalfCheetahPlanarAction[DType.float64](
            bthigh, bshin, bfoot, fthigh, fshin, ffoot
        )

        # Step environment
        var result = env.step(action)
        state = result[0]
        var reward = result[1]
        done = result[2]

        total_reward += Float64(reward)
        step += 1

        # Render
        env.render(renderer)

        # Check if window was closed
        if renderer.get_should_quit():
            print("Window closed by user")
            break

        # Print progress every 50 steps
        if step % 50 == 0:
            print(
                "Step",
                step,
                "| Reward:",
                String(Float64(reward))[:7],
                "| Total:",
                String(total_reward)[:8],
                "| X:",
                String(Float64(state.torso_z))[:6],
                "| Vel:",
                String(Float64(state.vel_x))[:6],
            )

        # Small delay for visualization
        sleep(0.02)

    print()
    print("=== Demo Complete ===")
    print("Total steps:", step)
    print("Total reward:", total_reward)

    # Keep window open for a moment
    print()
    print("Closing in 2 seconds...")
    sleep(2.0)

    # Clean up
    renderer.close()
