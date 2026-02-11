"""Half Cheetah Demo - Test the HalfCheetah environment.

This demo runs the Half Cheetah environment with random actions
and optionally renders the visualization.

Usage:
    pixi run mojo run examples/half_cheetah_demo.mojo
"""

from envs.half_cheetah import HalfCheetah
from core import ContAction, ObsState
from random import random_float64


fn main() raises:
    print("=" * 60)
    print("HalfCheetah Environment Demo")
    print("=" * 60)

    # Create environment
    var env = HalfCheetah()
    print("\nEnvironment created successfully!")
    print("  Observation dim:", env.obs_dim())
    print("  Action dim:", env.action_dim())
    print("  Max steps:", env.get_max_steps())

    # Reset environment
    var state = env.reset()
    print("\nInitial state:")
    # obs[0]=rootz, obs[1]=rooty(pitch), obs[8]=rootx_vel
    print("  z_position:", state[0])
    print("  y_angle:", state[1])
    print("  x_velocity:", state[8])

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
            var action = ContAction[6]()
            for ai in range(6):
                action[ai] = random_float64() * 2.0 - 1.0

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
                    state[8],
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
