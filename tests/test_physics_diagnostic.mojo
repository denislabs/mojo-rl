"""Diagnostic test to identify the source of CPU/GPU physics drift.

This test isolates each physics component:
1. Free-fall (gravity integration)
2. Engine impulses (linear and angular)
3. Contact response (landing physics)
4. Angular dynamics (rotation integration)

Run with:
    pixi run -e apple mojo run tests/test_physics_diagnostic.mojo
"""

from math import cos, sin, sqrt
from random import seed, random_float64


fn abs_f32(x: Float32) -> Float32:
    """Absolute value for Float32."""
    return x if x >= 0 else -x


fn abs_scalar(x: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Absolute value for Scalar."""
    return x if x >= 0 else -x

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu import LunarLanderGPU, gpu_dtype
from envs.lunar_lander_gpu_v2 import (
    LunarLanderGPUv2,
    FULL_STATE_SIZE as V2_STATE_SIZE,
)
from envs.lunar_lander_gpu_v3 import LunarLanderGPUv3

# Test parameters
comptime NUM_STEPS: Int = 20
comptime TOLERANCE: Float32 = 0.01  # 1% relative tolerance


fn format_float(val: Float32, width: Int = 10) -> String:
    """Format float to fixed width string."""
    var s = String(val)
    if len(s) > width:
        return String(s[:width])
    return s


fn print_header(title: String):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


fn print_comparison_row(
    name: String,
    cpu: Float32,
    gpu_v1: Float32,
    gpu_v2: Float32,
):
    """Print a row comparing CPU, GPU v1, and GPU v2 values."""
    var diff_v1 = abs_f32(cpu - gpu_v1)
    var diff_v2 = abs_f32(cpu - gpu_v2)

    var status_v1 = "OK" if diff_v1 < 0.1 else "DRIFT"
    var status_v2 = "OK" if diff_v2 < 0.1 else "DRIFT"

    print(
        "  " + name + ": CPU=" + format_float(cpu, 8)
        + " | V1=" + format_float(gpu_v1, 8) + " (" + status_v1 + ")"
        + " | V2=" + format_float(gpu_v2, 8) + " (" + status_v2 + ")"
    )


# =============================================================================
# GPU Step Helpers
# =============================================================================


fn create_gpu_state_v1(
    x_obs: Float32,
    y_obs: Float32,
    vx_obs: Float32,
    vy_obs: Float32,
    angle: Float32,
    angular_vel_obs: Float32,
    left_contact: Float32,
    right_contact: Float32,
) -> List[Scalar[DType.float32]]:
    """Create GPU v1 state (8 values)."""
    var state = List[Scalar[DType.float32]]()
    state.append(x_obs)
    state.append(y_obs)
    state.append(vx_obs)
    state.append(vy_obs)
    state.append(angle)
    state.append(angular_vel_obs)
    state.append(left_contact)
    state.append(right_contact)
    return state^


fn create_gpu_state_v2(
    x_obs: Float32,
    y_obs: Float32,
    vx_obs: Float32,
    vy_obs: Float32,
    angle: Float32,
    angular_vel_obs: Float32,
    left_contact: Float32,
    right_contact: Float32,
) -> List[Scalar[DType.float32]]:
    """Create GPU v2 state (12 values with hidden state)."""
    var state = List[Scalar[DType.float32]]()
    # Observable state
    state.append(x_obs)
    state.append(y_obs)
    state.append(vx_obs)
    state.append(vy_obs)
    state.append(angle)
    state.append(angular_vel_obs)
    state.append(left_contact)
    state.append(right_contact)
    # Hidden state (prev velocities, contact cache)
    state.append(vx_obs)  # prev_vx
    state.append(vy_obs)  # prev_vy
    state.append(angular_vel_obs)  # prev_angular_vel
    state.append(Scalar[DType.float32](0.0))  # contact_impulse
    return state^


fn run_gpu_v1_step(
    ctx: DeviceContext,
    initial_state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int = 0,
) raises -> List[Float32]:
    """Run a single GPU v1 physics step."""
    comptime BATCH_SIZE = 1
    comptime STATE_SIZE = 8

    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

    var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
    var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

    for i in range(8):
        host_states[i] = initial_state[i]
    host_actions[0] = Scalar[gpu_dtype](action)

    ctx.enqueue_copy(states_buf, host_states)
    ctx.enqueue_copy(actions_buf, host_actions)
    ctx.synchronize()

    LunarLanderGPU.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, UInt64(step_seed)
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_states, states_buf)
    ctx.synchronize()

    var result = List[Float32]()
    for i in range(8):
        result.append(Float32(host_states[i]))
    return result^


fn run_gpu_v2_step(
    ctx: DeviceContext,
    initial_state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int = 0,
) raises -> List[Float32]:
    """Run a single GPU v2 physics step."""
    comptime BATCH_SIZE = 1
    comptime STATE_SIZE = V2_STATE_SIZE  # 12

    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

    var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
    var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

    for i in range(len(initial_state)):
        host_states[i] = initial_state[i]
    host_actions[0] = Scalar[gpu_dtype](action)

    ctx.enqueue_copy(states_buf, host_states)
    ctx.enqueue_copy(actions_buf, host_actions)
    ctx.synchronize()

    LunarLanderGPUv2.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, UInt64(step_seed)
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_states, states_buf)
    ctx.synchronize()

    var result = List[Float32]()
    for i in range(8):  # Only return observable state
        result.append(Float32(host_states[i]))
    return result^


# =============================================================================
# Test 1: Free-Fall (Gravity Integration)
# =============================================================================


fn test_freefall(ctx: DeviceContext) raises -> Tuple[Float32, Float32]:
    """Test gravity integration without engines or contact.

    Returns max drift for (GPU v1, GPU v2).
    """
    print_header("TEST 1: FREE-FALL (Gravity Integration)")
    print("Starting from top of screen, no engines, measuring y and vy drift")
    print("")

    # Create CPU environment with no wind
    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Set initial state: high up, no velocity, level
    # We'll manually track states since we can't set CPU env state directly
    # Instead, we reset and use the initial state
    var cpu_obs = cpu_env.get_obs_list()

    # For GPU, start with same normalized state
    var gpu_v1_state = create_gpu_state_v1(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )
    var gpu_v2_state = create_gpu_state_v2(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )

    print("Initial state:")
    print("  y_obs=" + format_float(cpu_obs[1]) + " vy_obs=" + format_float(cpu_obs[3]))
    print("")

    var max_drift_v1: Float32 = 0.0
    var max_drift_v2: Float32 = 0.0

    print("Step | CPU vy    | V1 vy     | V2 vy     | V1 diff   | V2 diff")
    print("-" * 65)

    for step in range(NUM_STEPS):
        # Step all with no-op action (action=0)
        var cpu_result = cpu_env.step_discrete(0)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        var gpu_v1_next = run_gpu_v1_step(ctx, gpu_v1_state, 0, step)
        var gpu_v2_next = run_gpu_v2_step(ctx, gpu_v2_state, 0, step)

        # Compare vy (most affected by gravity)
        var diff_v1 = abs_f32(cpu_next[3] - gpu_v1_next[3])
        var diff_v2 = abs_f32(cpu_next[3] - gpu_v2_next[3])

        if diff_v1 > max_drift_v1:
            max_drift_v1 = diff_v1
        if diff_v2 > max_drift_v2:
            max_drift_v2 = diff_v2

        if step < 5 or step % 5 == 0:
            print(
                String(step) + "    | "
                + format_float(cpu_next[3], 9) + " | "
                + format_float(gpu_v1_next[3], 9) + " | "
                + format_float(gpu_v2_next[3], 9) + " | "
                + format_float(diff_v1, 9) + " | "
                + format_float(diff_v2, 9)
            )

        if cpu_done:
            print("CPU terminated at step " + String(step))
            break

        # Update states for next iteration
        gpu_v1_state.clear()
        for i in range(8):
            gpu_v1_state.append(gpu_v1_next[i])

        gpu_v2_state.clear()
        for i in range(8):
            gpu_v2_state.append(gpu_v2_next[i])
        # Hidden state for v2
        gpu_v2_state.append(gpu_v2_next[2])  # prev_vx
        gpu_v2_state.append(gpu_v2_next[3])  # prev_vy
        gpu_v2_state.append(gpu_v2_next[5])  # prev_angular_vel
        gpu_v2_state.append(Scalar[DType.float32](0.0))

    print("")
    print("FREE-FALL RESULT:")
    print("  Max vy drift V1: " + format_float(max_drift_v1))
    print("  Max vy drift V2: " + format_float(max_drift_v2))

    return (max_drift_v1, max_drift_v2)


# =============================================================================
# Test 2: Main Engine Impulse
# =============================================================================


fn test_main_engine(ctx: DeviceContext) raises -> Tuple[Float32, Float32]:
    """Test main engine impulse application.

    Returns max drift for (GPU v1, GPU v2).
    """
    print_header("TEST 2: MAIN ENGINE (Linear Impulse)")
    print("Starting from high position, firing main engine continuously")
    print("")

    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()
    var cpu_obs = cpu_env.get_obs_list()

    var gpu_v1_state = create_gpu_state_v1(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )
    var gpu_v2_state = create_gpu_state_v2(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )

    var max_drift_v1: Float32 = 0.0
    var max_drift_v2: Float32 = 0.0

    print("Step | CPU vy    | V1 vy     | V2 vy     | CPU angle | V1 angle  | V2 angle")
    print("-" * 80)

    for step in range(NUM_STEPS):
        # Fire main engine (action=2)
        # Use fixed seed to eliminate random dispersion differences
        var cpu_result = cpu_env.step_discrete(2)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        var gpu_v1_next = run_gpu_v1_step(ctx, gpu_v1_state, 2, 12345)  # Fixed seed
        var gpu_v2_next = run_gpu_v2_step(ctx, gpu_v2_state, 2, 12345)

        # Compare vy and angle
        var diff_vy_v1 = abs_f32(cpu_next[3] - gpu_v1_next[3])
        var diff_vy_v2 = abs_f32(cpu_next[3] - gpu_v2_next[3])
        var diff_angle_v1 = abs_f32(cpu_next[4] - gpu_v1_next[4])
        var diff_angle_v2 = abs_f32(cpu_next[4] - gpu_v2_next[4])

        var total_diff_v1 = diff_vy_v1 + diff_angle_v1
        var total_diff_v2 = diff_vy_v2 + diff_angle_v2

        if total_diff_v1 > max_drift_v1:
            max_drift_v1 = total_diff_v1
        if total_diff_v2 > max_drift_v2:
            max_drift_v2 = total_diff_v2

        if step < 5 or step % 5 == 0:
            print(
                String(step) + "    | "
                + format_float(cpu_next[3], 9) + " | "
                + format_float(gpu_v1_next[3], 9) + " | "
                + format_float(gpu_v2_next[3], 9) + " | "
                + format_float(cpu_next[4], 9) + " | "
                + format_float(gpu_v1_next[4], 9) + " | "
                + format_float(gpu_v2_next[4], 9)
            )

        if cpu_done:
            print("CPU terminated at step " + String(step))
            break

        # Update states
        gpu_v1_state.clear()
        for i in range(8):
            gpu_v1_state.append(gpu_v1_next[i])

        gpu_v2_state.clear()
        for i in range(8):
            gpu_v2_state.append(gpu_v2_next[i])
        gpu_v2_state.append(gpu_v2_next[2])
        gpu_v2_state.append(gpu_v2_next[3])
        gpu_v2_state.append(gpu_v2_next[5])
        gpu_v2_state.append(Scalar[DType.float32](0.0))

    print("")
    print("MAIN ENGINE RESULT:")
    print("  Max total drift V1: " + format_float(max_drift_v1))
    print("  Max total drift V2: " + format_float(max_drift_v2))

    return (max_drift_v1, max_drift_v2)


# =============================================================================
# Test 3: Side Engine (Angular Impulse)
# =============================================================================


fn test_side_engine(ctx: DeviceContext) raises -> Tuple[Float32, Float32]:
    """Test side engine angular impulse application.

    Returns max drift for (GPU v1, GPU v2).
    """
    print_header("TEST 3: SIDE ENGINE (Angular Impulse)")
    print("Alternating left/right engines to test torque generation")
    print("")

    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()
    var cpu_obs = cpu_env.get_obs_list()

    var gpu_v1_state = create_gpu_state_v1(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )
    var gpu_v2_state = create_gpu_state_v2(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )

    var max_drift_v1: Float32 = 0.0
    var max_drift_v2: Float32 = 0.0

    print("Step | Action | CPU ang_v | V1 ang_v  | V2 ang_v  | V1 diff   | V2 diff")
    print("-" * 75)

    for step in range(NUM_STEPS):
        # Alternate left (1) and right (3) engines
        var action = 1 if step % 2 == 0 else 3

        var cpu_result = cpu_env.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        var gpu_v1_next = run_gpu_v1_step(ctx, gpu_v1_state, action, 12345)
        var gpu_v2_next = run_gpu_v2_step(ctx, gpu_v2_state, action, 12345)

        # Compare angular velocity
        var diff_v1 = abs_f32(cpu_next[5] - gpu_v1_next[5])
        var diff_v2 = abs_f32(cpu_next[5] - gpu_v2_next[5])

        if diff_v1 > max_drift_v1:
            max_drift_v1 = diff_v1
        if diff_v2 > max_drift_v2:
            max_drift_v2 = diff_v2

        var action_str = "left " if action == 1 else "right"

        if step < 5 or step % 5 == 0:
            print(
                String(step) + "    | " + action_str + "  | "
                + format_float(cpu_next[5], 9) + " | "
                + format_float(gpu_v1_next[5], 9) + " | "
                + format_float(gpu_v2_next[5], 9) + " | "
                + format_float(diff_v1, 9) + " | "
                + format_float(diff_v2, 9)
            )

        if cpu_done:
            print("CPU terminated at step " + String(step))
            break

        # Update states
        gpu_v1_state.clear()
        for i in range(8):
            gpu_v1_state.append(gpu_v1_next[i])

        gpu_v2_state.clear()
        for i in range(8):
            gpu_v2_state.append(gpu_v2_next[i])
        gpu_v2_state.append(gpu_v2_next[2])
        gpu_v2_state.append(gpu_v2_next[3])
        gpu_v2_state.append(gpu_v2_next[5])
        gpu_v2_state.append(Scalar[DType.float32](0.0))

    print("")
    print("SIDE ENGINE RESULT:")
    print("  Max angular_vel drift V1: " + format_float(max_drift_v1))
    print("  Max angular_vel drift V2: " + format_float(max_drift_v2))

    return (max_drift_v1, max_drift_v2)


# =============================================================================
# Test 4: Contact Response
# =============================================================================


fn test_contact(ctx: DeviceContext) raises -> Tuple[Float32, Float32]:
    """Test contact physics when landing.

    This is the most complex test - we run until contact and then
    observe the settling behavior.

    Returns max drift for (GPU v1, GPU v2).
    """
    print_header("TEST 4: CONTACT RESPONSE (Landing Physics)")
    print("Letting lander fall and observing contact behavior")
    print("This tests: penetration correction, velocity response, friction")
    print("")

    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()
    var cpu_obs = cpu_env.get_obs_list()

    var gpu_v1_state = create_gpu_state_v1(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )
    var gpu_v2_state = create_gpu_state_v2(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )

    var max_drift_v1: Float32 = 0.0
    var max_drift_v2: Float32 = 0.0
    var contact_step = -1

    print("Step | CPU y     | V1 y      | V2 y      | CPU leg_L | V1 leg_L  | V2 leg_L")
    print("-" * 80)

    # Let it fall with main engine to slow descent
    for step in range(100):  # Longer to allow landing
        # Use main engine occasionally to control descent
        var action = 2 if step % 3 == 0 else 0

        var cpu_result = cpu_env.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        var gpu_v1_next = run_gpu_v1_step(ctx, gpu_v1_state, action, step)
        var gpu_v2_next = run_gpu_v2_step(ctx, gpu_v2_state, action, step)

        # Track contact
        var cpu_left_contact = cpu_next[6]
        var v1_left_contact = gpu_v1_next[6]
        var v2_left_contact = gpu_v2_next[6]

        if contact_step < 0 and cpu_left_contact > 0.5:
            contact_step = step
            print("\n*** CONTACT DETECTED at step " + String(step) + " ***\n")

        # Compare y position and contacts
        var diff_y_v1 = abs_f32(cpu_next[1] - gpu_v1_next[1])
        var diff_y_v2 = abs_f32(cpu_next[1] - gpu_v2_next[1])
        var diff_contact_v1 = abs_f32(cpu_left_contact - v1_left_contact)
        var diff_contact_v2 = abs_f32(cpu_left_contact - v2_left_contact)

        var total_diff_v1 = diff_y_v1 + diff_contact_v1
        var total_diff_v2 = diff_y_v2 + diff_contact_v2

        if total_diff_v1 > max_drift_v1:
            max_drift_v1 = total_diff_v1
        if total_diff_v2 > max_drift_v2:
            max_drift_v2 = total_diff_v2

        # Print more frequently around contact
        var should_print = step < 5 or step % 10 == 0
        if contact_step >= 0 and step - contact_step < 10:
            should_print = True

        if should_print:
            print(
                String(step) + "    | "
                + format_float(cpu_next[1], 9) + " | "
                + format_float(gpu_v1_next[1], 9) + " | "
                + format_float(gpu_v2_next[1], 9) + " | "
                + format_float(cpu_left_contact, 9) + " | "
                + format_float(v1_left_contact, 9) + " | "
                + format_float(v2_left_contact, 9)
            )

        if cpu_done:
            print("\nCPU terminated at step " + String(step))
            break

        # Update states
        gpu_v1_state.clear()
        for i in range(8):
            gpu_v1_state.append(gpu_v1_next[i])

        gpu_v2_state.clear()
        for i in range(8):
            gpu_v2_state.append(gpu_v2_next[i])
        gpu_v2_state.append(gpu_v2_next[2])
        gpu_v2_state.append(gpu_v2_next[3])
        gpu_v2_state.append(gpu_v2_next[5])
        gpu_v2_state.append(Scalar[DType.float32](0.0))

    print("")
    print("CONTACT RESULT:")
    print("  Max total drift V1: " + format_float(max_drift_v1))
    print("  Max total drift V2: " + format_float(max_drift_v2))

    return (max_drift_v1, max_drift_v2)


# =============================================================================
# Test 5: Controlled Landing Sequence
# =============================================================================


fn test_landing_sequence(ctx: DeviceContext) raises -> Tuple[Float32, Float32]:
    """Test a realistic landing sequence with control inputs.

    This simulates what a trained policy would do.

    Returns max drift for (GPU v1, GPU v2).
    """
    print_header("TEST 5: CONTROLLED LANDING SEQUENCE")
    print("Simulating a controlled descent with policy-like actions")
    print("")

    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()
    var cpu_obs = cpu_env.get_obs_list()

    var gpu_v1_state = create_gpu_state_v1(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )
    var gpu_v2_state = create_gpu_state_v2(
        cpu_obs[0], cpu_obs[1], cpu_obs[2], cpu_obs[3],
        cpu_obs[4], cpu_obs[5], cpu_obs[6], cpu_obs[7]
    )

    var max_drift_v1: Float32 = 0.0
    var max_drift_v2: Float32 = 0.0

    # Track state divergence over time
    var divergence_v1 = List[Float32]()
    var divergence_v2 = List[Float32]()

    print("Using simple control policy: main engine if falling, side engines for angle")
    print("")

    for step in range(150):
        # Simple control policy based on CPU state
        var action = 0  # default: no-op

        var vy = cpu_obs[3]
        var angle = cpu_obs[4]
        var angular_vel = cpu_obs[5]

        # Control logic
        if vy < -0.1:
            action = 2  # Main engine if falling
        elif angle > 0.1 or angular_vel > 0.2:
            action = 1  # Left engine to correct right tilt
        elif angle < -0.1 or angular_vel < -0.2:
            action = 3  # Right engine to correct left tilt

        var cpu_result = cpu_env.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        var gpu_v1_next = run_gpu_v1_step(ctx, gpu_v1_state, action, step + 1000)
        var gpu_v2_next = run_gpu_v2_step(ctx, gpu_v2_state, action, step + 1000)

        # Compute total state divergence
        var total_diff_v1: Float32 = 0.0
        var total_diff_v2: Float32 = 0.0
        for i in range(8):
            total_diff_v1 += abs_f32(cpu_next[i] - gpu_v1_next[i])
            total_diff_v2 += abs_f32(cpu_next[i] - gpu_v2_next[i])

        divergence_v1.append(total_diff_v1)
        divergence_v2.append(total_diff_v2)

        if total_diff_v1 > max_drift_v1:
            max_drift_v1 = total_diff_v1
        if total_diff_v2 > max_drift_v2:
            max_drift_v2 = total_diff_v2

        if step < 5 or step % 20 == 0:
            var action_str = "nop  "
            if action == 1:
                action_str = "left "
            elif action == 2:
                action_str = "main "
            elif action == 3:
                action_str = "right"
            print(
                "Step " + String(step) + " [" + action_str + "]: "
                + "V1 div=" + format_float(total_diff_v1, 8)
                + " V2 div=" + format_float(total_diff_v2, 8)
            )

        if cpu_done:
            print("\nCPU terminated at step " + String(step))
            break

        # Update states
        cpu_obs.clear()
        for i in range(8):
            cpu_obs.append(cpu_next[i])

        gpu_v1_state.clear()
        for i in range(8):
            gpu_v1_state.append(gpu_v1_next[i])

        gpu_v2_state.clear()
        for i in range(8):
            gpu_v2_state.append(gpu_v2_next[i])
        gpu_v2_state.append(gpu_v2_next[2])
        gpu_v2_state.append(gpu_v2_next[3])
        gpu_v2_state.append(gpu_v2_next[5])
        gpu_v2_state.append(Scalar[DType.float32](0.0))

    print("")
    print("CONTROLLED LANDING RESULT:")
    print("  Max total divergence V1: " + format_float(max_drift_v1))
    print("  Max total divergence V2: " + format_float(max_drift_v2))

    # Analyze divergence trend
    if len(divergence_v1) > 10:
        var early_v1: Float32 = 0.0
        var late_v1: Float32 = 0.0
        var early_v2: Float32 = 0.0
        var late_v2: Float32 = 0.0
        for i in range(5):
            early_v1 += divergence_v1[i]
            early_v2 += divergence_v2[i]
            late_v1 += divergence_v1[len(divergence_v1) - 5 + i]
            late_v2 += divergence_v2[len(divergence_v2) - 5 + i]
        early_v1 /= 5
        late_v1 /= 5
        early_v2 /= 5
        late_v2 /= 5

        print("")
        print("  Divergence trend:")
        print("    V1: early=" + format_float(early_v1) + " late=" + format_float(late_v1)
              + " growth=" + format_float(late_v1 / early_v1 if early_v1 > 0.001 else 0.0) + "x")
        print("    V2: early=" + format_float(early_v2) + " late=" + format_float(late_v2)
              + " growth=" + format_float(late_v2 / early_v2 if early_v2 > 0.001 else 0.0) + "x")

    return (max_drift_v1, max_drift_v2)


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 70)
    print("LUNAR LANDER CPU/GPU PHYSICS DIAGNOSTIC TEST")
    print("=" * 70)
    print("")
    print("This test identifies which physics components cause CPU/GPU drift.")
    print("Lower drift = better match. Target: < 0.1 for good transfer.")
    print("")

    seed(42)
    var ctx = DeviceContext()

    # Run all tests
    var freefall_drift = test_freefall(ctx)
    var main_engine_drift = test_main_engine(ctx)
    var side_engine_drift = test_side_engine(ctx)
    var contact_drift = test_contact(ctx)
    var landing_drift = test_landing_sequence(ctx)

    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    print("")
    print("Component              | GPU V1 Drift | GPU V2 Drift | Primary Issue")
    print("-" * 70)

    fn drift_status(d: Float32) -> String:
        if d < 0.1:
            return "OK"
        elif d < 0.5:
            return "MODERATE"
        else:
            return "HIGH"

    print(
        "Free-fall (gravity)    | "
        + format_float(freefall_drift[0], 12) + " | "
        + format_float(freefall_drift[1], 12) + " | "
        + drift_status(freefall_drift[0]) + "/" + drift_status(freefall_drift[1])
    )
    print(
        "Main engine (impulse)  | "
        + format_float(main_engine_drift[0], 12) + " | "
        + format_float(main_engine_drift[1], 12) + " | "
        + drift_status(main_engine_drift[0]) + "/" + drift_status(main_engine_drift[1])
    )
    print(
        "Side engine (torque)   | "
        + format_float(side_engine_drift[0], 12) + " | "
        + format_float(side_engine_drift[1], 12) + " | "
        + drift_status(side_engine_drift[0]) + "/" + drift_status(side_engine_drift[1])
    )
    print(
        "Contact (landing)      | "
        + format_float(contact_drift[0], 12) + " | "
        + format_float(contact_drift[1], 12) + " | "
        + drift_status(contact_drift[0]) + "/" + drift_status(contact_drift[1])
    )
    print(
        "Full landing sequence  | "
        + format_float(landing_drift[0], 12) + " | "
        + format_float(landing_drift[1], 12) + " | "
        + drift_status(landing_drift[0]) + "/" + drift_status(landing_drift[1])
    )

    print("")
    print("RECOMMENDATIONS:")
    print("-" * 70)

    # Identify worst component
    var max_v1 = freefall_drift[0]
    var max_v2 = freefall_drift[1]
    var worst_component_v1 = "free-fall"
    var worst_component_v2 = "free-fall"

    if main_engine_drift[0] > max_v1:
        max_v1 = main_engine_drift[0]
        worst_component_v1 = "main engine"
    if main_engine_drift[1] > max_v2:
        max_v2 = main_engine_drift[1]
        worst_component_v2 = "main engine"

    if side_engine_drift[0] > max_v1:
        max_v1 = side_engine_drift[0]
        worst_component_v1 = "side engine"
    if side_engine_drift[1] > max_v2:
        max_v2 = side_engine_drift[1]
        worst_component_v2 = "side engine"

    if contact_drift[0] > max_v1:
        max_v1 = contact_drift[0]
        worst_component_v1 = "contact"
    if contact_drift[1] > max_v2:
        max_v2 = contact_drift[1]
        worst_component_v2 = "contact"

    print("V1 worst component: " + worst_component_v1 + " (drift=" + format_float(max_v1) + ")")
    print("V2 worst component: " + worst_component_v2 + " (drift=" + format_float(max_v2) + ")")
    print("")

    if worst_component_v2 == "contact":
        print("-> Focus on improving CONTACT PHYSICS:")
        print("   - Use impulse-based velocity response instead of damping")
        print("   - Increase contact solver iterations (try 6-8)")
        print("   - Match Box2D's Baumgarte stabilization parameters")
    elif worst_component_v2 == "main engine" or worst_component_v2 == "side engine":
        print("-> Focus on improving ENGINE PHYSICS:")
        print("   - Verify torque calculation matches CPU exactly")
        print("   - Check integration order (apply impulse then integrate)")
        print("   - Consider random dispersion synchronization")
    elif worst_component_v2 == "free-fall":
        print("-> Focus on improving GRAVITY INTEGRATION:")
        print("   - Match Box2D's semi-implicit Euler exactly")
        print("   - Check if sub-stepping introduces drift")

    print("")
    print("=" * 70)
