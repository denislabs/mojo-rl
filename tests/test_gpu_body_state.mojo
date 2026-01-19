"""
Direct GPU Body State Test

Read raw body velocities from GPU state buffer before and after steps.

Run with:
    pixi run -e apple mojo run tests/test_gpu_body_state.mojo
"""

from gpu.host import DeviceContext

from envs.lunar_lander_v2_gpu import (
    LunarLanderV2GPU,
    STATE_SIZE_VAL,
    OBS_OFFSET,
    BODIES_OFFSET,
    FORCES_OFFSET,
)

comptime gpu_dtype = DType.float32

# Body state indices (from physics_gpu/state.mojo)
comptime IDX_X = 0
comptime IDX_Y = 1
comptime IDX_ANGLE = 2
comptime IDX_VX = 3
comptime IDX_VY = 4
comptime IDX_OMEGA = 5
comptime IDX_INV_MASS = 6
comptime IDX_INV_INERTIA = 7
comptime BODY_STATE_SIZE = 13


fn print_separator():
    print("=" * 70)


fn test_gpu_single_step_free_fall() raises:
    """Test a single GPU step in free fall."""
    print_separator()
    print("GPU Single Step Free Fall (reading raw body state)")
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

        # Reset
        LunarLanderV2GPU[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Read initial state
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var vy_before = Float64(host_states[BODIES_OFFSET + IDX_VY])
        print("Initial body state:")
        print("  x =", host_states[BODIES_OFFSET + IDX_X])
        print("  y =", host_states[BODIES_OFFSET + IDX_Y])
        print("  vy =", host_states[BODIES_OFFSET + IDX_VY])
        print("  inv_mass =", host_states[BODIES_OFFSET + IDX_INV_MASS])
        print()

        # Take one step with no action
        host_actions[0] = Scalar[gpu_dtype](0)
        ctx.enqueue_copy(actions_buf, host_actions)
        ctx.synchronize()

        LunarLanderV2GPU[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf
        )
        ctx.synchronize()

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var vy_after = Float64(host_states[BODIES_OFFSET + IDX_VY])
        print("After 1 step (nop):")
        print("  vy =", host_states[BODIES_OFFSET + IDX_VY])
        print()

        var dvy = vy_after - vy_before
        print("dvy =", dvy)
        print("Expected dvy (gravity * dt = -10 * 0.02) = -0.2")
        print("Ratio =", dvy / -0.2 if dvy != 0 else 0.0)

    print_separator()


fn test_gpu_single_step_main_engine() raises:
    """Test a single GPU step with main engine."""
    print_separator()
    print("GPU Single Step Main Engine (reading raw body state)")
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

        # Reset
        LunarLanderV2GPU[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Read initial state
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var vy_before = Float64(host_states[BODIES_OFFSET + IDX_VY])
        print("Initial body state:")
        print("  x =", host_states[BODIES_OFFSET + IDX_X])
        print("  y =", host_states[BODIES_OFFSET + IDX_Y])
        print("  vy =", host_states[BODIES_OFFSET + IDX_VY])
        print()

        # Check forces before step
        print("Forces before step:")
        print("  fx =", host_states[FORCES_OFFSET + 0])
        print("  fy =", host_states[FORCES_OFFSET + 1])
        print("  torque =", host_states[FORCES_OFFSET + 2])
        print()

        # Take one step with main engine
        host_actions[0] = Scalar[gpu_dtype](2)  # main engine
        ctx.enqueue_copy(actions_buf, host_actions)
        ctx.synchronize()

        LunarLanderV2GPU[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf
        )
        ctx.synchronize()

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var vy_after = Float64(host_states[BODIES_OFFSET + IDX_VY])
        print("After 1 step (main engine):")
        print("  vy =", host_states[BODIES_OFFSET + IDX_VY])
        print()

        # Note: forces are cleared after the step
        print("Forces after step (should be 0):")
        print("  fx =", host_states[FORCES_OFFSET + 0])
        print("  fy =", host_states[FORCES_OFFSET + 1])
        print("  torque =", host_states[FORCES_OFFSET + 2])
        print()

        var dvy = vy_after - vy_before
        print("dvy =", dvy)

        # Expected: engine effect + gravity
        # Engine: impulse/mass = (4/30 * 13) / 5 = 0.347
        # Gravity: -10 * 0.02 = -0.2
        # Net: 0.147
        print("Expected dvy (engine + gravity) = 0.147")
        print("Ratio =", dvy / 0.147 if dvy != 0 else 0.0)

    print_separator()


fn test_multiple_steps() raises:
    """Test multiple GPU steps."""
    print_separator()
    print("GPU Multiple Steps (raw body vy)")
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

        # Reset
        LunarLanderV2GPU[DType.float32].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        print("Step | Action | vy_before | vy_after  | dvy       | expected")
        print("-" * 65)

        var actions_list = List[Int]()
        actions_list.append(0)  # nop
        actions_list.append(0)  # nop
        actions_list.append(2)  # main
        actions_list.append(2)  # main
        actions_list.append(0)  # nop

        for step in range(len(actions_list)):
            var action = actions_list[step]

            # Read state before
            ctx.enqueue_copy(host_states, states_buf)
            ctx.synchronize()
            var vy_before = Float64(host_states[BODIES_OFFSET + IDX_VY])

            # Take step
            host_actions[0] = Scalar[gpu_dtype](action)
            ctx.enqueue_copy(actions_buf, host_actions)
            ctx.synchronize()

            LunarLanderV2GPU[DType.float32].step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
                ctx, states_buf, actions_buf, rewards_buf, dones_buf
            )
            ctx.synchronize()

            # Read state after
            ctx.enqueue_copy(host_states, states_buf)
            ctx.synchronize()
            var vy_after = Float64(host_states[BODIES_OFFSET + IDX_VY])

            var dvy = vy_after - vy_before

            var action_str: String
            var expected: Float64
            if action == 0:
                action_str = "nop"
                expected = -0.2
            elif action == 2:
                action_str = "main"
                expected = 0.147
            else:
                action_str = "???"
                expected = 0.0

            print(step, "    |", action_str, " |", vy_before, "|", vy_after, "|", dvy, "|", expected)

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    GPU BODY STATE TEST")
    print("=" * 70)
    print()

    test_gpu_single_step_free_fall()
    print()

    test_gpu_single_step_main_engine()
    print()

    test_multiple_steps()
    print()

    print("=" * 70)
    print("    Tests completed!")
    print("=" * 70)
