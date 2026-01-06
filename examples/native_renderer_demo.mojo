"""Demo of native SDL2 rendering integrated into environments.

This demonstrates the native rendering capabilities for CartPole
and MountainCar with integrated SDL2 rendering (no separate renderer needed).

Requirements: brew install sdl2 sdl2_ttf
"""

from envs.cartpole_native import CartPoleNative
from envs.mountain_car_native import MountainCarNative


fn demo_cartpole_native() raises:
    """Demo CartPole with integrated native SDL2 renderer."""
    print("=== CartPole Native Demo ===")
    print("Watch the random policy (close window to continue)")
    print("")

    var env = CartPoleNative()
    var _ = env.reset()

    for step in range(200):
        # Random action (0 or 1)
        var action = step % 2

        var result = env.step(action)
        var done = result[2]

        # Render using integrated render() method
        env.render()

        if done:
            print("Episode ended at step", step + 1)
            break

    env.close()
    print("CartPole demo complete!")
    print("")


fn demo_mountaincar_native() raises:
    """Demo MountainCar with integrated native SDL2 renderer."""
    print("=== MountainCar Native Demo ===")
    print("Watch the oscillating policy (close window to continue)")
    print("")

    var env = MountainCarNative()
    var obs = env.reset()

    for step in range(200):
        # Simple oscillating policy based on velocity
        var action: Int
        if obs[1] < 0:
            action = 0  # Push left when going left
        else:
            action = 2  # Push right when going right

        var result = env.step(action)
        var next_obs = result[0]
        var done = result[2]

        # Render using integrated render() method
        env.render()

        obs = next_obs

        if done:
            if next_obs[0] >= 0.5:
                print("Goal reached at step", step + 1, "!")
            else:
                print("Episode ended at step", step + 1)
            break

    env.close()
    print("MountainCar demo complete!")
    print("")


fn main():
    """Run native renderer demos."""
    print("=" * 60)
    print("Native SDL2 Environment Demo")
    print("Rendering integrated into environment via render() method")
    print("No separate renderer object needed!")
    print("=" * 60)
    print("")

    try:
        demo_cartpole_native()
        demo_mountaincar_native()
    except e:
        print("Error:", e)

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)
