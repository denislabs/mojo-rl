"""HopperGC Demo - Test the HopperGC environment with random actions.

This demo shows:
- Creating a HopperGC environment (using Generalized Coordinates physics)
- Running episodes with random actions
- Optional rendering with the 3D renderer
"""

from random import random_float64

from envs.hopper_gc import HopperGC


fn run_episode(
    mut env: HopperGC, render: Bool = False, max_steps: Int = 500
) raises -> Float64:
    """Run a single episode with random actions.

    Returns:
        Total reward accumulated during the episode.
    """
    _ = env.reset_obs_list()
    var total_reward: Float64 = 0.0
    var done = False
    var step = 0

    if render:
        _ = env.init_renderer()

    while not done and step < max_steps:
        # Generate random actions in [-1, 1]
        var action = List[Scalar[DType.float64]]()
        action.append(random_float64() * 2.0 - 1.0)  # thigh
        action.append(random_float64() * 2.0 - 1.0)  # leg
        action.append(random_float64() * 2.0 - 1.0)  # foot

        # Step environment
        var result = env.step_continuous_vec(action)
        var reward = result[1]
        done = result[2]

        total_reward += Float64(reward)
        step += 1

        if render:
            env.render_frame()
            if env.check_renderer_quit():
                break
            env.renderer_delay(16)  # ~60 FPS

    if render:
        env.close_renderer()

    return total_reward


fn main() raises:
    print("=== HopperGC Environment Demo ===")
    print()
    print(
        "This environment uses Generalized Coordinates physics (MuJoCo-style)"
    )
    print("with SemiImplicitEulerIntegrator for energy-conserving simulation.")
    print()

    # Create environment
    var env = HopperGC()

    print("Environment created:")
    print("  Observation dim:", env.obs_dim())
    print("  Action dim:", env.action_dim())
    print("  Action range: [", env.action_low(), ",", env.action_high(), "]")
    print()

    # Run a few episodes without rendering
    print("Running 3 episodes without rendering...")
    for ep in range(3):
        var reward = run_episode(env, render=False, max_steps=200)
        print("  Episode", ep + 1, "reward:", reward)
    print()

    # Run one episode with rendering
    print("Running 1 episode with rendering (close window to stop)...")
    var reward = run_episode(env, render=True, max_steps=10000)
    print("  Episode reward:", reward)
    print()

    print("Demo complete!")
