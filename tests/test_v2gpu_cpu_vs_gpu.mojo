"""
Compare LunarLanderV2GPU CPU path vs GPU path

The CPU path works correctly (verified against LunarLanderV2).
This test identifies differences in the GPU path.

Run with:
    pixi run -e apple mojo run tests/test_v2gpu_cpu_vs_gpu.mojo
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


fn abs_f64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


fn print_separator():
    print("=" * 70)


fn test_free_fall_cpu_vs_gpu() raises:
    """Compare free fall between CPU and GPU paths."""
    print_separator()
    print("LunarLanderV2GPU: CPU vs GPU Free Fall (5 steps)")
    print_separator()

    # CPU environment
    var cpu_env = LunarLanderV2GPU[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set CPU to known state
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    print("Both start at (10, 8) with velocity 0")
    print()
    print("Step | CPU vy_raw | GPU vy_raw* | CPU dvy   | GPU dvy*  | diff")
    print("-" * 70)
    print("* GPU vy_raw estimated from normalized obs / norm_factor")
    print()

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

        # Normalization factor: vy_norm = vy * (H_UNITS / 2) / FPS
        var norm_factor: Float64 = (H_UNITS / 2.0) / Float64(FPS)  # ~0.133

        var cpu_vy_prev: Float64 = 0.0
        var gpu_vy_norm_prev: Float64 = 0.0

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
            var gpu_dvy_norm = gpu_vy_norm - gpu_vy_norm_prev

            # Convert GPU norm to raw (inverse of normalization)
            var gpu_vy_raw = gpu_vy_norm / norm_factor
            var gpu_dvy_raw = gpu_dvy_norm / norm_factor

            var diff = abs_f64(cpu_dvy - gpu_dvy_raw)

            print(step, "    |", cpu_vy, "|", gpu_vy_raw, "|", cpu_dvy, "|", gpu_dvy_raw, "|", diff)

            cpu_vy_prev = cpu_vy
            gpu_vy_norm_prev = gpu_vy_norm

    print()
    print("Expected dvy per step (gravity only): -0.2")
    print_separator()


fn test_main_engine_cpu_vs_gpu() raises:
    """Compare main engine between CPU and GPU paths."""
    print_separator()
    print("LunarLanderV2GPU: CPU vs GPU Main Engine (5 steps)")
    print_separator()

    var cpu_env = LunarLanderV2GPU[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 1, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_velocity(0, 2, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    print("Both start at (10, 8) with velocity 0")
    print()
    print("Step | CPU vy_raw | GPU vy_raw* | CPU dvy   | GPU dvy*  | diff")
    print("-" * 70)

    with DeviceContext() as ctx:
        comptime BATCH_SIZE = 1
        comptime STATE_SIZE = STATE_SIZE_VAL

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
        var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        LunarLanderV2GPU[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        var norm_factor: Float64 = (H_UNITS / 2.0) / Float64(FPS)

        var cpu_vy_prev: Float64 = 0.0
        var gpu_vy_norm_prev: Float64 = 0.0

        for step in range(5):
            # CPU step with main engine
            var result = cpu_env.step_obs(2)
            _ = result[0].copy()
            var cpu_vy = Float64(cpu_env.physics.get_body_vy(0, 0))
            var cpu_dvy = cpu_vy - cpu_vy_prev

            # GPU step with main engine
            host_actions[0] = Scalar[gpu_dtype](2)
            ctx.enqueue_copy(actions_buf, host_actions)
            ctx.synchronize()

            LunarLanderV2GPU[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
                ctx, states_buf, actions_buf, rewards_buf, dones_buf
            )
            ctx.synchronize()

            ctx.enqueue_copy(host_states, states_buf)
            ctx.synchronize()

            var gpu_vy_norm = Float64(host_states[OBS_OFFSET + 3])
            var gpu_dvy_norm = gpu_vy_norm - gpu_vy_norm_prev

            var gpu_vy_raw = gpu_vy_norm / norm_factor
            var gpu_dvy_raw = gpu_dvy_norm / norm_factor

            var diff = abs_f64(cpu_dvy - gpu_dvy_raw)

            print(step, "    |", cpu_vy, "|", gpu_vy_raw, "|", cpu_dvy, "|", gpu_dvy_raw, "|", diff)

            cpu_vy_prev = cpu_vy
            gpu_vy_norm_prev = gpu_vy_norm

    print()
    print("Expected dvy per step (engine + gravity): ~0.15")
    print_separator()


fn test_read_raw_gpu_velocity() raises:
    """Read the raw body velocity from GPU state (not normalized obs)."""
    print_separator()
    print("Direct GPU Body Velocity Read (not normalized obs)")
    print_separator()

    with DeviceContext() as ctx:
        comptime BATCH_SIZE = 1
        comptime STATE_SIZE = STATE_SIZE_VAL

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
        var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        LunarLanderV2GPU[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        # The body state is at BODIES_OFFSET
        # Body 0 (lander) has: x, y, angle, vx, vy, omega, inv_mass, inv_inertia, ...
        # IDX_VX = 3, IDX_VY = 4
        var lander_offset = BODIES_OFFSET
        print("Body state layout at BODIES_OFFSET =", BODIES_OFFSET)
        print()
        print("Initial GPU lander body state:")
        print("  x =", host_states[lander_offset + 0])
        print("  y =", host_states[lander_offset + 1])
        print("  angle =", host_states[lander_offset + 2])
        print("  vx =", host_states[lander_offset + 3])
        print("  vy =", host_states[lander_offset + 4])
        print("  omega =", host_states[lander_offset + 5])
        print()

        print("Obs at OBS_OFFSET =", OBS_OFFSET)
        print("  obs[0] (x_norm) =", host_states[OBS_OFFSET + 0])
        print("  obs[1] (y_norm) =", host_states[OBS_OFFSET + 1])
        print("  obs[2] (vx_norm) =", host_states[OBS_OFFSET + 2])
        print("  obs[3] (vy_norm) =", host_states[OBS_OFFSET + 3])
        print()

        # Take a step with no action
        host_actions[0] = Scalar[gpu_dtype](0)
        ctx.enqueue_copy(actions_buf, host_actions)
        ctx.synchronize()

        LunarLanderV2GPU[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf
        )
        ctx.synchronize()

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        print("After 1 step (nop):")
        print("  Body vy =", host_states[lander_offset + 4])
        print("  Obs vy_norm =", host_states[OBS_OFFSET + 3])

        # Expected: vy should be ~ -0.2 (gravity * dt)
        var expected_vy: Float64 = -10.0 * 0.02
        print()
        print("Expected body vy (gravity * dt) =", expected_vy)

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    LunarLanderV2GPU: CPU vs GPU PATH COMPARISON")
    print("=" * 70)
    print()

    test_read_raw_gpu_velocity()
    print()

    test_free_fall_cpu_vs_gpu()
    print()

    test_main_engine_cpu_vs_gpu()
    print()

    print("=" * 70)
    print("    Comparison tests completed!")
    print("=" * 70)
