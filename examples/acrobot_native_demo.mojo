"""Demo of the native Mojo Acrobot environment with SDL2 rendering.

Run with: pixi run mojo run examples/acrobot_native_demo.mojo

Requires SDL2: brew install sdl2 sdl2_ttf
"""

from envs import AcrobotEnv
from random import random_float64
from time import sleep
from render import RendererBase


fn main() raises:
    print("=== Acrobot Native Rendering Demo ===")
    print("Close the window or press Ctrl+C to exit\n")

    # Create environment and renderer
    var env = AcrobotEnv(num_bins=6)
    var renderer = RendererBase(500, 500, 15, "Acrobot")

    # Run episodes with rendering
    var num_episodes = 5
    for episode in range(num_episodes):
        _ = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0

        print("Episode", episode + 1, "- Watch the window!")

        while not env.is_done():
            # Random action (could be replaced with trained policy)
            var action_idx = Int(random_float64() * 3.0)
            if action_idx > 2:
                action_idx = 2
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            total_reward += result[1]
            steps += 1

            # Render at approximately 15 FPS (matching Gymnasium)
            env.render(renderer)

            # Add small delay to see the animation
            sleep(0.02)

        print(
            "Episode",
            episode + 1,
            "finished | Reward:",
            total_reward,
            "| Steps:",
            steps,
        )
        print()

    # Clean up
    renderer.close()
    print("=== Demo Complete ===")
