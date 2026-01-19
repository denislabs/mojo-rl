"""
Gravity Debug Test - Compare CPU vs GPU free fall

This test compares free fall behavior between CPU and GPU
to see if gravity is applied consistently.

Run with:
    pixi run -e apple mojo run tests/test_gravity_debug.mojo
"""

from math import sqrt
from gpu.host import DeviceContext

from envs.lunar_lander_v2_gpu import (
    LunarLanderV2GPU,
    STATE_SIZE_VAL,
    OBS_OFFSET,
    BODIES_OFFSET,
)

comptime gpu_dtype = DType.float32
comptime H_UNITS: Float64 = 400.0 / 30.0
comptime FPS: Int = 50


fn print_separator():
    print("=" * 70)


fn test_cpu_free_fall_detailed() raises:
    """Detailed CPU free fall analysis."""
    print_separator()
    print("CPU Free Fall Analysis")
    print_separator()

    var cpu_env = LunarLanderV2GPU[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set known initial state
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)

    # Also set legs to same velocity to avoid joint effects
    cpu_env.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)

    cpu_env._update_cached_state()

    print("Initial state: all bodies at rest")
    print()

    # Get raw velocity from physics state
    var vy_raw_before = Float64(cpu_env.physics.get_body_vy(0, 0))
    print("Raw vy before step:", vy_raw_before)

    # Take one step with no action
    var result = cpu_env.step_obs(0)
    _ = result[0].copy()

    var vy_raw_after = Float64(cpu_env.physics.get_body_vy(0, 0))
    print("Raw vy after step:", vy_raw_after)

    var dvy_raw = vy_raw_after - vy_raw_before
    print("Raw dvy:", dvy_raw)

    # Expected: gravity * dt = -10 * 0.02 = -0.2
    var expected_dvy: Float64 = -10.0 * 0.02
    print("Expected dvy (gravity * dt):", expected_dvy)
    print("Ratio (actual/expected):", dvy_raw / expected_dvy if expected_dvy != 0 else 0.0)

    print_separator()


fn test_cpu_vs_gpu_free_fall() raises:
    """Compare CPU and GPU free fall."""
    print_separator()
    print("CPU vs GPU Free Fall Comparison (5 steps)")
    print_separator()

    # CPU
    var cpu_env = LunarLanderV2GPU[DType.float32](enable_wind=False)
    _ = cpu_env.reset()
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    print("Step | CPU vy_raw | GPU vy_norm | CPU dvy | GPU dvy")
    print("-" * 60)

    with DeviceContext() as ctx:
        comptime BATCH_SIZE = 1
        comptime STATE_SIZE = STATE_SIZE_VAL

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
        var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        # Reset GPU env
        LunarLanderV2GPU[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Read initial GPU state
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var cpu_vy_prev = Float64(cpu_env.physics.get_body_vy(0, 0))
        var gpu_vy_prev = Float64(host_states[OBS_OFFSET + 3])

        for step in range(5):
            # CPU step
            var result = cpu_env.step_obs(0)  # nop
            _ = result[0].copy()
            var cpu_vy = Float64(cpu_env.physics.get_body_vy(0, 0))
            var cpu_dvy = cpu_vy - cpu_vy_prev

            # GPU step
            host_actions[0] = Scalar[gpu_dtype](0)
            ctx.enqueue_copy(actions_buf, host_actions)
            ctx.synchronize()

            LunarLanderV2GPU[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
                ctx, states_buf, actions_buf, rewards_buf, dones_buf
            )
            ctx.synchronize()

            ctx.enqueue_copy(host_states, states_buf)
            ctx.synchronize()

            var gpu_vy_norm = Float64(host_states[OBS_OFFSET + 3])
            var gpu_dvy = gpu_vy_norm - gpu_vy_prev

            print(step, "    |", cpu_vy, "|", gpu_vy_norm, "|", cpu_dvy, "|", gpu_dvy)

            cpu_vy_prev = cpu_vy
            gpu_vy_prev = gpu_vy_norm

    print_separator()


fn test_engine_vs_gravity() raises:
    """Test engine effect vs gravity."""
    print_separator()
    print("Engine vs Gravity Analysis")
    print_separator()

    var cpu_env = LunarLanderV2GPU[DType.float32](enable_wind=False)
    _ = cpu_env.reset()
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    # Test 1: Free fall
    var vy_before_ff = Float64(cpu_env.physics.get_body_vy(0, 0))
    var result_ff = cpu_env.step_obs(0)
    _ = result_ff[0].copy()
    var vy_after_ff = Float64(cpu_env.physics.get_body_vy(0, 0))
    var dvy_freefall = vy_after_ff - vy_before_ff

    print("Free fall (1 step):")
    print("  dvy =", dvy_freefall)
    print()

    # Reset for engine test
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)

    # Test 2: Main engine
    var vy_before_eng = Float64(cpu_env.physics.get_body_vy(0, 0))
    var result_eng = cpu_env.step_obs(2)  # main engine
    _ = result_eng[0].copy()
    var vy_after_eng = Float64(cpu_env.physics.get_body_vy(0, 0))
    var dvy_engine = vy_after_eng - vy_before_eng

    print("Main engine (1 step):")
    print("  dvy =", dvy_engine)
    print()

    # Compute engine-only effect
    var engine_only = dvy_engine - dvy_freefall
    print("Engine effect (engine - gravity):")
    print("  engine_only =", engine_only)

    # Expected engine effect
    var expected_engine: Float64 = (4.0 / 30.0) * 13.0 / 5.0  # offset * power / mass
    print("  expected_engine =", expected_engine)
    print("  ratio =", engine_only / expected_engine if expected_engine != 0 else 0.0)

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    GRAVITY DEBUG TEST")
    print("=" * 70)
    print()

    test_cpu_free_fall_detailed()
    print()

    test_engine_vs_gravity()
    print()

    test_cpu_vs_gpu_free_fall()
    print()

    print("=" * 70)
    print("    Debug tests completed!")
    print("=" * 70)
