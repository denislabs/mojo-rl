"""
Engine Force Debug Test

This test analyzes the exact force/impulse values to understand
the CPU vs GPU physics differences.

Run with:
    pixi run -e apple mojo run tests/test_engine_force_debug.mojo
"""

from math import sin, cos, sqrt
from gpu.host import DeviceContext

from envs.lunar_lander_v2_gpu import LunarLanderV2GPU, STATE_SIZE_VAL, OBS_OFFSET, BODIES_OFFSET, FORCES_OFFSET

# Constants from LunarLanderV2GPU
comptime SCALE: Float64 = 30.0
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SIDE_ENGINE_POWER: Float64 = 0.6
comptime LANDER_MASS: Float64 = 5.0
comptime LANDER_INERTIA: Float64 = 3.33
comptime DT: Float64 = 0.02
comptime FPS: Int = 50
comptime H_UNITS: Float64 = 400.0 / SCALE  # 13.33
comptime gpu_dtype = DType.float32


fn print_separator():
    print("=" * 70)


fn test_cpu_engine_analysis() raises:
    """Analyze CPU engine force application in detail."""
    print_separator()
    print("CPU Engine Force Analysis")
    print_separator()

    var cpu_env = LunarLanderV2GPU[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set known initial state: angle=0, at rest
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    print("Initial state:")
    print("  Position: (10.0, 8.0)")
    print("  Velocity: (0, 0, 0)")
    print("  Angle: 0")
    print()

    # Calculate expected impulse values for main engine
    var angle: Float64 = 0.0
    var tip_x = sin(angle)
    var tip_y = cos(angle)

    var main_y_offset: Float64 = 4.0 / SCALE

    # Without dispersion (assume 0)
    var ox = tip_x * main_y_offset  # 0 * 0.133 = 0
    var oy = -tip_y * main_y_offset  # -1 * 0.133 = -0.133

    var impulse_x = -ox * MAIN_ENGINE_POWER  # 0
    var impulse_y = -oy * MAIN_ENGINE_POWER  # 0.133 * 13 = 1.73

    var dv_from_engine_x = impulse_x / LANDER_MASS
    var dv_from_engine_y = impulse_y / LANDER_MASS  # 1.73 / 5 = 0.346

    print("Expected main engine impulse (no dispersion, angle=0):")
    print("  tip = (", tip_x, ", ", tip_y, ")")
    print("  main_y_offset =", main_y_offset)
    print("  oy =", oy)
    print("  impulse_y =", impulse_y)
    print("  dv_y from engine =", dv_from_engine_y)
    print()

    # Expected gravity contribution
    var dv_from_gravity_y: Float64 = -10.0 * DT  # -0.2

    print("Expected gravity contribution:")
    print("  dv_y from gravity =", dv_from_gravity_y)
    print()

    # Expected net change
    var expected_dv_y = dv_from_engine_y + dv_from_gravity_y  # 0.346 - 0.2 = 0.146

    print("Expected net velocity change:")
    print("  dv_y = engine + gravity =", dv_from_engine_y, "+", dv_from_gravity_y, "=", expected_dv_y)
    print()

    # Convert to normalized observation
    var vy_norm_factor: Float64 = (H_UNITS / 2.0) / Float64(FPS)  # 6.67 / 50 = 0.133
    var expected_dvy_norm = expected_dv_y * vy_norm_factor

    print("Observation normalization:")
    print("  vy_norm_factor =", vy_norm_factor)
    print("  Expected dvy_norm =", expected_dvy_norm)
    print()

    # Now actually run a step and observe
    var obs_before = cpu_env.get_obs_list()
    print("Actual before step:")
    print("  vy_norm =", obs_before[3])

    # Take step with main engine
    var result = cpu_env.step_obs(2)
    var obs_after = result[0].copy()

    print("Actual after step:")
    print("  vy_norm =", obs_after[3])

    var actual_dvy_norm = Float64(obs_after[3]) - Float64(obs_before[3])
    print()
    print("Actual dvy_norm =", actual_dvy_norm)
    print("Expected dvy_norm =", expected_dvy_norm)
    print("Ratio (actual/expected) =", actual_dvy_norm / expected_dvy_norm if expected_dvy_norm != 0 else 0.0)

    print_separator()


fn test_force_vs_impulse() raises:
    """Test the difference between force-based and impulse-based physics."""
    print_separator()
    print("Force vs Impulse Physics Comparison")
    print_separator()

    # Impulse-based (CPU):
    # dv = impulse / mass
    # impulse = offset * POWER
    # dv = (offset * POWER) / mass

    var offset: Float64 = 4.0 / SCALE  # 0.133
    var impulse = offset * MAIN_ENGINE_POWER  # 1.73
    var dv_impulse = impulse / LANDER_MASS  # 0.346

    print("Impulse-based (CPU):")
    print("  offset =", offset)
    print("  impulse = offset * POWER =", impulse)
    print("  dv = impulse / mass =", dv_impulse)
    print()

    # Force-based (GPU):
    # force = impulse / dt
    # dv = (force / mass) * dt
    # dv = (impulse / dt / mass) * dt
    # dv = impulse / mass (same!)

    var force = impulse / DT  # 86.5
    var dv_force = (force / LANDER_MASS) * DT  # (86.5 / 5) * 0.02 = 0.346

    print("Force-based (GPU):")
    print("  force = impulse / dt =", force)
    print("  dv = (force / mass) * dt =", dv_force)
    print()

    print("Both methods should give the same dv:")
    print("  dv_impulse =", dv_impulse)
    print("  dv_force =", dv_force)
    print("  Difference =", dv_impulse - dv_force)

    print_separator()


fn test_gravity_integration() raises:
    """Test gravity integration to verify physics step."""
    print_separator()
    print("Gravity Integration Test (Free Fall)")
    print_separator()

    var cpu_env = LunarLanderV2GPU[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set known initial state: at rest, no angle
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    var obs_before = cpu_env.get_obs_list()
    print("Before step (free fall, no action):")
    print("  vy_norm =", obs_before[3])

    # Take step with no action (free fall)
    var result = cpu_env.step_obs(0)  # nop
    var obs_after = result[0].copy()

    print("After step:")
    print("  vy_norm =", obs_after[3])

    var actual_dvy_norm = Float64(obs_after[3]) - Float64(obs_before[3])

    # Expected: only gravity
    var expected_dvy = -10.0 * DT  # -0.2
    var vy_norm_factor = (H_UNITS / 2.0) / Float64(FPS)
    var expected_dvy_norm = expected_dvy * vy_norm_factor  # -0.2 * 0.133 = -0.0267

    print()
    print("Actual dvy_norm =", actual_dvy_norm)
    print("Expected dvy_norm (gravity only) =", expected_dvy_norm)
    print("Ratio (actual/expected) =", actual_dvy_norm / expected_dvy_norm if expected_dvy_norm != 0 else 0.0)

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    ENGINE FORCE DEBUG TEST")
    print("=" * 70)
    print()

    test_force_vs_impulse()
    print()

    test_gravity_integration()
    print()

    test_cpu_engine_analysis()
    print()

    print("=" * 70)
    print("    Debug tests completed!")
    print("=" * 70)
