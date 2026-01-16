"""Demo of the native Mojo BipedalWalker environment with SDL2 rendering.

Run with: pixi run mojo run examples/bipedal_walker_demo.mojo

Requires SDL2: brew install sdl2 sdl2_ttf

Features:
- Pure Mojo physics engine (no Box2D/Python dependency)
- SDL2 rendering showing walker, terrain, and scrolling viewport
- Random and simple heuristic policies
- Watch the bipedal walker try to walk forward!

Actions: 4D continuous [hip1, knee1, hip2, knee2] in [-1, 1]
"""

from envs.bipedal_walker import BipedalWalkerEnv, BipedalWalkerState
from random import random_float64, seed
from time import sleep
from render import RendererBase


fn main() raises:
    print("=== BipedalWalker Native Rendering Demo ===")
    print("Pure Mojo implementation with custom physics engine")
    print("Close the window or press Ctrl+C to exit\n")

    # Seed random for reproducibility
    seed(42)

    # Create renderer (shared across environments)
    var renderer = RendererBase(600, 400, 50, "BipedalWalker")

    # Create environment (normal mode for first demo)
    var env = BipedalWalkerEnv(hardcore=False)

    # Run episodes with rendering
    var num_episodes = 3
    for episode in range(num_episodes):
        var state = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0
        var done = False

        var mode_name = "Normal"
        print("Episode", episode + 1, "(", mode_name, "mode) starting...")

        while not done and steps < 1600:
            # Random action with slight forward bias
            var hip1 = (random_float64() * 2.0 - 1.0) * 0.5
            var knee1 = (random_float64() * 2.0 - 1.0) * 0.5
            var hip2 = (random_float64() * 2.0 - 1.0) * 0.5
            var knee2 = (random_float64() * 2.0 - 1.0) * 0.5

            var result = env.step_continuous_4d(hip1, knee1, hip2, knee2)
            state = result[0].copy()
            var reward = result[1]
            done = result[2]

            total_reward += reward
            steps += 1

            # Render the frame
            env.render(renderer)

            # Print progress every 100 steps
            if steps % 100 == 0:
                print("  Step", steps, "| Reward so far:", Int(total_reward))

        var status = "SUCCESS!" if total_reward > 200 else "DONE"
        print(
            "Episode",
            episode + 1,
            "-",
            status,
            "| Total Reward:",
            Int(total_reward),
            "| Steps:",
            steps,
        )
        print()

        # Small pause between episodes
        sleep(0.5)

    print("=== Now trying Hardcore mode ===\n")

    # Create hardcore environment
    var env_hardcore = BipedalWalkerEnv(hardcore=True)

    for episode in range(2):
        var state = env_hardcore.reset()
        var total_reward: Float64 = 0.0
        var steps = 0
        var done = False

        print("Hardcore Episode", episode + 1, "starting...")

        while not done and steps < 2000:
            # Random action
            var hip1 = (random_float64() * 2.0 - 1.0) * 0.5
            var knee1 = (random_float64() * 2.0 - 1.0) * 0.5
            var hip2 = (random_float64() * 2.0 - 1.0) * 0.5
            var knee2 = (random_float64() * 2.0 - 1.0) * 0.5

            var result = env_hardcore.step_continuous_4d(hip1, knee1, hip2, knee2)
            state = result[0].copy()
            var reward = result[1]
            done = result[2]

            total_reward += reward
            steps += 1

            # Render the frame
            env_hardcore.render(renderer)

        print(
            "Hardcore Episode",
            episode + 1,
            "| Total Reward:",
            Int(total_reward),
            "| Steps:",
            steps,
        )
        print()

        sleep(0.5)

    # Clean up
    renderer.close()
    print("=== Demo Complete ===")
