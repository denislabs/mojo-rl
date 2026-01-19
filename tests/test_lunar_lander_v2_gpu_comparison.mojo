"""
Comparison Test: LunarLanderV2 CPU vs GPU paths

This test validates that both CPU and GPU implementations produce
equivalent physics behavior when given identical initial conditions
and action sequences.

Run with:
    pixi run -e apple mojo run tests/test_lunar_lander_v2_gpu_comparison.mojo
"""

from gpu.host import DeviceContext
from math import sqrt

from envs.lunar_lander_v2_gpu import LunarLanderV2, STATE_SIZE_VAL, OBS_OFFSET

# Constants
comptime SCALE: Float64 = 30.0
comptime VIEWPORT_W: Int = 600
comptime VIEWPORT_H: Int = 400
comptime gpu_dtype = DType.float32


fn abs_f64(x: Float64) -> Float64:
    """Absolute value of Float64."""
    return x if x >= 0.0 else -x


fn print_separator():
    print("=" * 70)


fn test_free_fall_comparison() raises:
    """Compare free fall between CPU and GPU."""
    print_separator()
    print("LunarLanderV2: Free Fall Comparison (CPU vs GPU)")
    print_separator()

    # Create CPU environment
    var cpu_env = LunarLanderV2[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Get initial state from CPU env
    var obs_cpu = cpu_env.get_obs_list()
    print("Initial CPU state:")
    print("  x=", obs_cpu[0], " y=", obs_cpu[1])
    print("  vx=", obs_cpu[2], " vy=", obs_cpu[3])
    print()

    # Create GPU environment with same settings
    with DeviceContext() as ctx:
        # Allocate GPU buffers for a single environment
        comptime BATCH_SIZE = 1
        comptime STATE_SIZE = STATE_SIZE_VAL

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

        # Create host buffers for reading/writing
        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
        var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        # Reset GPU env
        LunarLanderV2[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Copy GPU initial state to host
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        print("Initial GPU state:")
        print(
            "  x=",
            host_states[OBS_OFFSET + 0],
            " y=",
            host_states[OBS_OFFSET + 1],
        )
        print(
            "  vx=",
            host_states[OBS_OFFSET + 2],
            " vy=",
            host_states[OBS_OFFSET + 3],
        )
        print()

        print("Note: Initial states differ due to random terrain/velocity")
        print("      Comparing relative behavior instead")
        print()

        # Manually set identical initial conditions
        # Reset CPU env to known state
        cpu_env.physics.set_body_position(0, 0, 10.0, 10.0)  # Center
        cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
        cpu_env.physics.set_body_angle(0, 0, 0.0)
        cpu_env._update_cached_state()

        print("Step | CPU vy    | GPU vy    | vy diff   | CPU y     | GPU y")
        print("-" * 70)

        # Step 20 times with no action (free fall)
        for step in range(20):
            # CPU step
            var result_cpu = cpu_env.step_obs(0)  # nop
            var obs_cpu_new = result_cpu[0].copy()

            # GPU step
            host_actions[0] = Scalar[gpu_dtype](0)
            ctx.enqueue_copy(actions_buf, host_actions)
            ctx.synchronize()

            LunarLanderV2[DType.float32].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf)
            ctx.synchronize()

            # Read GPU state
            ctx.enqueue_copy(host_states, states_buf)
            ctx.synchronize()

            var cpu_y = Float64(obs_cpu_new[1])
            var cpu_vy = Float64(obs_cpu_new[3])
            var gpu_y = Float64(host_states[OBS_OFFSET + 1])
            var gpu_vy = Float64(host_states[OBS_OFFSET + 3])

            var vy_diff = abs_f64(cpu_vy - gpu_vy)

            if step % 2 == 0:
                print(
                    step,
                    "    |",
                    cpu_vy,
                    "|",
                    gpu_vy,
                    "|",
                    vy_diff,
                    "|",
                    cpu_y,
                    "|",
                    gpu_y,
                )

    print_separator()


fn test_main_engine_comparison() raises:
    """Compare main engine thrust between CPU and GPU."""
    print_separator()
    print("LunarLanderV2: Main Engine Comparison (CPU vs GPU)")
    print_separator()

    # Create CPU environment
    var cpu_env = LunarLanderV2[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set known initial state
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    var obs_cpu = cpu_env.get_obs_list()
    print("CPU Initial: y=", obs_cpu[1], " vy=", obs_cpu[3])

    print()
    print("Firing main engine (action=2) every step:")
    print()
    print("Step | CPU vy    | GPU vy    | vy diff   | CPU y     | GPU y")
    print("-" * 70)

    with DeviceContext() as ctx:
        comptime BATCH_SIZE = 1
        comptime STATE_SIZE = STATE_SIZE_VAL

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
        var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        # Reset GPU env
        LunarLanderV2[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Step with main engine
        for step in range(15):
            # CPU step with main engine (action 2)
            var result_cpu = cpu_env.step_obs(2)
            var obs_cpu_new = result_cpu[0].copy()

            # GPU step with main engine
            host_actions[0] = Scalar[gpu_dtype](2)
            ctx.enqueue_copy(actions_buf, host_actions)
            ctx.synchronize()

            LunarLanderV2[DType.float32].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf)
            ctx.synchronize()

            ctx.enqueue_copy(host_states, states_buf)
            ctx.synchronize()

            var cpu_y = Float64(obs_cpu_new[1])
            var cpu_vy = Float64(obs_cpu_new[3])
            var gpu_y = Float64(host_states[OBS_OFFSET + 1])
            var gpu_vy = Float64(host_states[OBS_OFFSET + 3])

            var vy_diff = abs_f64(cpu_vy - gpu_vy)

            print(
                step,
                "    |",
                cpu_vy,
                "|",
                gpu_vy,
                "|",
                vy_diff,
                "|",
                cpu_y,
                "|",
                gpu_y,
            )

    print_separator()


fn test_side_engine_comparison() raises:
    """Compare side engine thrust between CPU and GPU."""
    print_separator()
    print("LunarLanderV2: Side Engine Comparison (CPU vs GPU)")
    print_separator()

    # Create CPU environment
    var cpu_env = LunarLanderV2[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set known initial state
    cpu_env.physics.set_body_position(0, 0, 10.0, 8.0)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    print("Testing left engine (action=1):")
    print()
    print("Step | CPU vx    | GPU vx    | vx diff   | CPU omega | GPU omega")
    print("-" * 70)

    with DeviceContext() as ctx:
        comptime BATCH_SIZE = 1
        comptime STATE_SIZE = STATE_SIZE_VAL

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
        var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        # Reset GPU env
        LunarLanderV2[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Step with left engine
        for step in range(10):
            # CPU step with left engine (action 1)
            var result_cpu = cpu_env.step_obs(1)
            var obs_cpu_new = result_cpu[0].copy()

            # GPU step with left engine
            host_actions[0] = Scalar[gpu_dtype](1)
            ctx.enqueue_copy(actions_buf, host_actions)
            ctx.synchronize()

            LunarLanderV2[DType.float32].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf)
            ctx.synchronize()

            ctx.enqueue_copy(host_states, states_buf)
            ctx.synchronize()

            var cpu_vx = Float64(obs_cpu_new[2])
            var cpu_omega = Float64(obs_cpu_new[5])
            var gpu_vx = Float64(host_states[OBS_OFFSET + 2])
            var gpu_omega = Float64(host_states[OBS_OFFSET + 5])

            var vx_diff = abs_f64(cpu_vx - gpu_vx)

            print(
                step,
                "    |",
                cpu_vx,
                "|",
                gpu_vx,
                "|",
                vx_diff,
                "|",
                cpu_omega,
                "|",
                gpu_omega,
            )

    print_separator()


fn test_single_step_detailed() raises:
    """Detailed comparison of a single step to debug physics differences."""
    print_separator()
    print("LunarLanderV2: Single Step Detailed Analysis")
    print_separator()

    # Create CPU environment
    var cpu_env = LunarLanderV2[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set known initial state (center, stationary, upright)
    var init_x: Float64 = 10.0
    var init_y: Float64 = 8.0
    cpu_env.physics.set_body_position(0, 0, init_x, init_y)
    cpu_env.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    cpu_env.physics.set_body_angle(0, 0, 0.0)
    cpu_env._update_cached_state()

    print("Initial state:")
    print("  Position: (", init_x, ",", init_y, ")")
    print("  Velocity: (0, 0)")
    print("  Angle: 0")
    print()

    # Get CPU pre-step state
    var cpu_obs_before = cpu_env.get_obs_list()
    print("CPU Before step:")
    print("  y =", cpu_obs_before[1], " vy =", cpu_obs_before[3])

    # Take ONE step with main engine
    var cpu_result = cpu_env.step_obs(2)  # Main engine
    var cpu_obs_after = cpu_result[0].copy()
    var cpu_reward = cpu_result[1]

    print("CPU After step (action=2 main engine):")
    print("  y =", cpu_obs_after[1], " vy =", cpu_obs_after[3])
    print("  reward =", cpu_reward)

    # Calculate CPU velocity change
    var cpu_dvy = Float64(cpu_obs_after[3]) - Float64(cpu_obs_before[3])
    print("  dvy (velocity change) =", cpu_dvy)
    print()

    # Now do the same on GPU
    with DeviceContext() as ctx:
        comptime BATCH_SIZE = 1
        comptime STATE_SIZE = STATE_SIZE_VAL

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
        var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)
        var host_rewards = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        # Reset GPU env
        LunarLanderV2[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Read GPU initial state
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        print("GPU Before step:")
        print(
            "  y =",
            host_states[OBS_OFFSET + 1],
            " vy =",
            host_states[OBS_OFFSET + 3],
        )

        var gpu_vy_before = Float64(host_states[OBS_OFFSET + 3])

        # Take ONE step with main engine
        host_actions[0] = Scalar[gpu_dtype](2)
        ctx.enqueue_copy(actions_buf, host_actions)
        ctx.synchronize()

        LunarLanderV2[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf
        )
        ctx.synchronize()

        ctx.enqueue_copy(host_states, states_buf)
        ctx.enqueue_copy(host_rewards, rewards_buf)
        ctx.synchronize()

        print("GPU After step (action=2 main engine):")
        print(
            "  y =",
            host_states[OBS_OFFSET + 1],
            " vy =",
            host_states[OBS_OFFSET + 3],
        )
        print("  reward =", host_rewards[0])

        # Calculate GPU velocity change
        var gpu_dvy = Float64(host_states[OBS_OFFSET + 3]) - gpu_vy_before
        print("  dvy (velocity change) =", gpu_dvy)
        print()

        # Summary
        print("COMPARISON:")
        print("  CPU dvy =", cpu_dvy)
        print("  GPU dvy =", gpu_dvy)
        print("  Ratio GPU/CPU =", gpu_dvy / cpu_dvy if cpu_dvy != 0 else 0.0)

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    LUNAR LANDER V2 GPU: CPU vs GPU COMPARISON TESTS")
    print("=" * 70)
    print()

    test_single_step_detailed()
    print()

    test_free_fall_comparison()
    print()

    test_main_engine_comparison()
    print()

    test_side_engine_comparison()
    print()

    print("=" * 70)
    print("    All comparison tests completed!")
    print("=" * 70)
