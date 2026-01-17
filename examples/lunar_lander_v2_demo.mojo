"""Demo script for LunarLanderV2 with rendering.

This demonstrates the GPU physics engine with SDL2 visualization.
Use arrow keys or let it run randomly:
- UP: Main engine
- LEFT: Left engine
- RIGHT: Right engine
- ESC: Quit
"""

from time import sleep
from random import random_float64

from envs.lunar_lander_v2 import LunarLanderV2
from render import RendererBase


fn main() raises:
    print("LunarLander V2 Demo")
    print("==================")
    print("Using GPU physics engine with SDL2 rendering")
    print()
    print("Controls:")
    print("  Main engine fires automatically when falling fast")
    print("  Random side thrusters for orientation")
    print("  Watch the lander try to land!")
    print()
    print("Press Ctrl+C to quit")
    print()

    # Create environment (single env for demo)
    var env = LunarLanderV2[BATCH=1](seed=42, enable_wind=False)

    # Create renderer
    var renderer = RendererBase(
        width=600,
        height=400,
        title="LunarLander V2 - GPU Physics Demo",
    )

    # Initialize display
    if not renderer.init_display():
        print("Failed to initialize SDL2 display")
        return

    # Reset environment
    env.reset_all()

    # Main loop
    var episode = 0
    var total_reward: Float64 = 0.0
    var step = 0
    var max_steps = 1000
    var frame_delay_ms = 20  # 50 FPS

    print("Starting episode", episode + 1)

    while True:
        # Simple policy: fire main engine when falling, random side thrusters
        var obs = env.get_observation(0)
        var vy = Float64(obs[3])  # Vertical velocity
        var angle = Float64(obs[4])  # Angle

        var action = 0  # Default: no action

        # Fire main engine if falling too fast or too low
        if vy < -0.1:
            action = 2  # Main engine

        # Random side thrusters for orientation correction
        if random_float64() < 0.1:
            if angle > 0.1:
                action = 1  # Left engine to correct right tilt
            elif angle < -0.1:
                action = 3  # Right engine to correct left tilt

        # Step environment
        var result = env.step(0, action)
        var reward = Float64(result[0])
        var done = result[1]
        total_reward += reward
        step += 1

        # Render
        env.render(0, renderer)

        # Check for quit (would need SDL event handling for proper input)
        # For now, just use step limit

        if done or step >= max_steps:
            print("Episode", episode + 1, "finished after", step, "steps")
            print("  Total reward:", total_reward)

            if Float64(obs[1]) < 0.1 and Float64(obs[3]) > -0.5:
                print("  Result: LANDED!")
            else:
                print("  Result: Crashed or out of bounds")

            # Reset for next episode
            episode += 1
            if episode >= 5:
                print()
                print("Demo complete! Ran 5 episodes.")
                break

            print()
            print("Starting episode", episode + 1)
            env.reset_all()
            total_reward = 0.0
            step = 0

        # Frame delay for visualization
        sleep(Float64(frame_delay_ms) / 1000.0)

    # Cleanup
    renderer.close()
    print("Done!")
