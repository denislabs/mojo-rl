"""Debug HalfCheetah V2 physics to see what's happening."""

from envs.half_cheetah import HalfCheetahPlanarV2, HalfCheetahPlanarAction, HCConstants

fn main() raises:
    print("=" * 60)
    print("Debugging HalfCheetah V2 Physics")
    print("=" * 60)

    var env = HalfCheetahPlanarV2()
    var state = env.reset()

    print("\nInitial state:")
    print("  Torso height (z):", state.torso_z)
    print("  Torso angle:", state.torso_angle)
    print("  Velocity X:", state.vel_x)
    print("  Velocity Z:", state.vel_z)

    # Run a few steps with zero action
    print("\n--- Running 10 steps with ZERO action ---")
    for step in range(10):
        var action = HalfCheetahPlanarAction[DType.float64](0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        var result = env._step_cpu_continuous(action)
        var reward = Float64(result[0])

        print("Step", step + 1, ":")
        print("  Reward:", reward)
        print("  Torso z:", env.cached_state.torso_z)
        print("  Torso angle:", env.cached_state.torso_angle)
        print("  Vel X:", env.cached_state.vel_x)
        print("  Vel Z:", env.cached_state.vel_z)
        print("  Done:", result[1])

    # Reset and run with some action
    _ = env.reset()
    print("\n--- Running 10 steps with action [0.5, 0.0, 0.0, -0.5, 0.0, 0.0] ---")
    for step in range(10):
        var action = HalfCheetahPlanarAction[DType.float64](0.5, 0.0, 0.0, -0.5, 0.0, 0.0)
        var result = env._step_cpu_continuous(action)
        var reward = Float64(result[0])

        print("Step", step + 1, ":")
        print("  Reward:", reward)
        print("  Torso z:", env.cached_state.torso_z)
        print("  Vel X:", env.cached_state.vel_x)

    print("\n" + "=" * 60)
    print("CPU Physics Debug Complete")
    print("=" * 60)
