"""Half Cheetah GC Demo - Test the HalfCheetahGC environment.

This demo runs the Half Cheetah environment with random actions
and optionally renders the visualization.

Usage:
    pixi run mojo run examples/half_cheetah_gc_demo.mojo
"""

from envs.half_cheetah_gc import HalfCheetahGC, HalfCheetahGCAction
from random import random_float64


fn main() raises:
    print("=" * 60)
    print("HalfCheetahGC Environment Demo")
    print("=" * 60)

    # Create environment
    var env = HalfCheetahGC()
    print("\nEnvironment created successfully!")
    print("  Observation dim:", env.obs_dim())
    print("  Action dim:", env.action_dim())
    print("  Max steps:", env.get_max_steps())

    # Reset environment
    var state = env.reset()
    print("\nInitial state:")
    print("  z_position:", state.z_position)
    print("  y_angle:", state.y_angle)
    print("  x_velocity:", state.x_velocity)

    # Try to initialize renderer
    var use_render = False
    try:
        if env.init_renderer():
            use_render = True
            print("\n3D Renderer initialized successfully!")
            print("Press window close button to exit.")
    except e:
        print("\nCould not initialize renderer:", e)
        print("Running without visualization.")

    # Run a few episodes
    var num_episodes = 3 if not use_render else 1
    var steps_per_episode = 100 if not use_render else 1000

    for episode in range(num_episodes):
        state = env.reset()
        var total_reward: Float64 = 0.0

        print("\n--- Episode", episode + 1, "---")

        for step in range(steps_per_episode):
            # Random action in [-1, 1]
            var action = HalfCheetahGCAction(
                bthigh=random_float64() * 2.0 - 1.0,
                bshin=random_float64() * 2.0 - 1.0,
                bfoot=random_float64() * 2.0 - 1.0,
                fthigh=random_float64() * 2.0 - 1.0,
                fshin=random_float64() * 2.0 - 1.0,
                ffoot=random_float64() * 2.0 - 1.0,
            )

            # Step environment
            var result = env.step(action)
            state = result[0]
            var reward = Float64(result[1])
            var done = result[2]

            total_reward += reward

            # Render if available
            if use_render:
                try:
                    env.render_frame()
                    if env.check_renderer_quit():
                        print("User closed window.")
                        env.close_renderer()
                        env.close()
                        return
                    env.renderer_delay(10)  # ~100 FPS
                except:
                    pass

            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                print(
                    "  Step",
                    step + 1,
                    ": x_pos =",
                    env.get_x_position(),
                    ", x_vel =",
                    state.x_velocity,
                    ", reward =",
                    reward,
                )

            if done:
                break

        print("Episode", episode + 1, "finished:")
        print("  Total reward:", total_reward)
        print("  Final x position:", env.get_x_position())
        print("  Steps taken:", env.get_current_step())

    # Cleanup
    if use_render:
        try:
            env.close_renderer()
        except:
            pass
    env.close()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
