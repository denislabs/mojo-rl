"""
Compare LunarLanderV2 (works) vs LunarLanderV2GPU (CPU path)

Both use the same physics engine, so they should produce identical results.
This test will help identify any differences.

Run with:
    pixi run -e apple mojo run tests/test_v2_vs_v2gpu_cpu.mojo
"""

from math import sqrt

from envs.lunar_lander_v2 import LunarLanderV2
from envs.lunar_lander_v2_gpu import LunarLanderV2GPU


fn abs_f64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


fn print_separator():
    print("=" * 70)


fn test_cpu_comparison() raises:
    """Compare LunarLanderV2 and LunarLanderV2GPU CPU paths."""
    print_separator()
    print("LunarLanderV2 vs LunarLanderV2GPU (CPU) Comparison")
    print_separator()

    # Create both environments
    var env_v2 = LunarLanderV2[BATCH=1](seed=42, enable_wind=False)
    var env_v2gpu = LunarLanderV2GPU[DType.float32](enable_wind=False)

    # Reset both
    env_v2.reset_all()
    _ = env_v2gpu.reset()

    # Set identical initial conditions
    var init_x: Float64 = 10.0
    var init_y: Float64 = 8.0

    # V2
    env_v2.physics.set_body_position(0, 0, init_x, init_y)
    env_v2.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_angle(0, 0, 0.0)
    # Also set legs
    env_v2.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)

    # V2GPU
    env_v2gpu.physics.set_body_position(0, 0, init_x, init_y)
    env_v2gpu.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2gpu.physics.set_body_angle(0, 0, 0.0)
    env_v2gpu.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    env_v2gpu.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)
    env_v2gpu._update_cached_state()

    print("Initial state: (", init_x, ",", init_y, ") at rest")
    print()

    # Compare raw velocities
    print("Comparing raw body velocities (not normalized obs):")
    print()
    print("Step | Action | V2 vy     | V2GPU vy  | diff      | V2 y      | V2GPU y")
    print("-" * 75)

    var actions = List[Int]()
    actions.append(0)
    actions.append(0)
    actions.append(0)
    actions.append(2)
    actions.append(2)
    actions.append(2)
    actions.append(2)
    actions.append(2)
    actions.append(1)
    actions.append(1)
    actions.append(3)
    actions.append(3)
    actions.append(0)
    actions.append(0)
    actions.append(0)

    for step in range(len(actions)):
        var action = actions[step]

        # Get raw velocities before step
        var v2_vy_before = Float64(env_v2.physics.get_body_vy(0, 0))
        var v2gpu_vy_before = Float64(env_v2gpu.physics.get_body_vy(0, 0))

        # Step both environments
        _ = env_v2.step(0, action)
        var result = env_v2gpu.step_obs(action)
        _ = result[0].copy()

        # Get raw velocities after step
        var v2_vy = Float64(env_v2.physics.get_body_vy(0, 0))
        var v2gpu_vy = Float64(env_v2gpu.physics.get_body_vy(0, 0))
        var v2_y = Float64(env_v2.physics.get_body_y(0, 0))
        var v2gpu_y = Float64(env_v2gpu.physics.get_body_y(0, 0))

        var diff = abs_f64(v2_vy - v2gpu_vy)

        var action_str: String
        if action == 0:
            action_str = "nop"
        elif action == 1:
            action_str = "left"
        elif action == 2:
            action_str = "main"
        else:
            action_str = "right"

        print(step, "    |", action_str, "  |", v2_vy, "|", v2gpu_vy, "|", diff, "|", v2_y, "|", v2gpu_y)

    print_separator()


fn test_free_fall_comparison() raises:
    """Compare free fall only."""
    print_separator()
    print("Free Fall Comparison (no actions)")
    print_separator()

    var env_v2 = LunarLanderV2[BATCH=1](seed=42, enable_wind=False)
    var env_v2gpu = LunarLanderV2GPU[DType.float32](enable_wind=False)

    env_v2.reset_all()
    _ = env_v2gpu.reset()

    var init_x: Float64 = 10.0
    var init_y: Float64 = 8.0

    # Set identical initial states
    env_v2.physics.set_body_position(0, 0, init_x, init_y)
    env_v2.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_angle(0, 0, 0.0)
    env_v2.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)

    env_v2gpu.physics.set_body_position(0, 0, init_x, init_y)
    env_v2gpu.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2gpu.physics.set_body_angle(0, 0, 0.0)
    env_v2gpu.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    env_v2gpu.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)
    env_v2gpu._update_cached_state()

    print("5 steps of free fall (action=0):")
    print()
    print("Step | V2 vy     | V2GPU vy  | V2 dvy    | V2GPU dvy | diff")
    print("-" * 65)

    var v2_vy_prev: Float64 = 0.0
    var v2gpu_vy_prev: Float64 = 0.0

    for step in range(5):
        _ = env_v2.step(0, 0)  # nop
        var result = env_v2gpu.step_obs(0)
        _ = result[0].copy()

        var v2_vy = Float64(env_v2.physics.get_body_vy(0, 0))
        var v2gpu_vy = Float64(env_v2gpu.physics.get_body_vy(0, 0))

        var v2_dvy = v2_vy - v2_vy_prev
        var v2gpu_dvy = v2gpu_vy - v2gpu_vy_prev

        var diff = abs_f64(v2_dvy - v2gpu_dvy)

        print(step, "    |", v2_vy, "|", v2gpu_vy, "|", v2_dvy, "|", v2gpu_dvy, "|", diff)

        v2_vy_prev = v2_vy
        v2gpu_vy_prev = v2gpu_vy

    print()
    print("Expected dvy per step (gravity only): -0.2")

    print_separator()


fn test_engine_comparison() raises:
    """Compare engine effects."""
    print_separator()
    print("Main Engine Comparison")
    print_separator()

    var env_v2 = LunarLanderV2[BATCH=1](seed=42, enable_wind=False)
    var env_v2gpu = LunarLanderV2GPU[DType.float32](enable_wind=False)

    env_v2.reset_all()
    _ = env_v2gpu.reset()

    var init_x: Float64 = 10.0
    var init_y: Float64 = 8.0

    # Set identical initial states
    env_v2.physics.set_body_position(0, 0, init_x, init_y)
    env_v2.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_angle(0, 0, 0.0)
    env_v2.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)

    env_v2gpu.physics.set_body_position(0, 0, init_x, init_y)
    env_v2gpu.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2gpu.physics.set_body_angle(0, 0, 0.0)
    env_v2gpu.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    env_v2gpu.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)
    env_v2gpu._update_cached_state()

    print("5 steps with main engine (action=2):")
    print()
    print("Step | V2 vy     | V2GPU vy  | V2 dvy    | V2GPU dvy | diff")
    print("-" * 65)

    var v2_vy_prev: Float64 = 0.0
    var v2gpu_vy_prev: Float64 = 0.0

    for step in range(5):
        _ = env_v2.step(0, 2)  # main engine
        var result = env_v2gpu.step_obs(2)
        _ = result[0].copy()

        var v2_vy = Float64(env_v2.physics.get_body_vy(0, 0))
        var v2gpu_vy = Float64(env_v2gpu.physics.get_body_vy(0, 0))

        var v2_dvy = v2_vy - v2_vy_prev
        var v2gpu_dvy = v2gpu_vy - v2gpu_vy_prev

        var diff = abs_f64(v2_dvy - v2gpu_dvy)

        print(step, "    |", v2_vy, "|", v2gpu_vy, "|", v2_dvy, "|", v2gpu_dvy, "|", diff)

        v2_vy_prev = v2_vy
        v2gpu_vy_prev = v2gpu_vy

    print()
    print("Expected dvy per step (engine + gravity): ~0.15")

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    LunarLanderV2 vs LunarLanderV2GPU (CPU) COMPARISON")
    print("=" * 70)
    print()

    test_free_fall_comparison()
    print()

    test_engine_comparison()
    print()

    test_cpu_comparison()
    print()

    print("=" * 70)
    print("    Comparison tests completed!")
    print("=" * 70)
