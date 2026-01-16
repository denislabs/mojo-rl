"""Test that CPU and GPU LunarLander physics produce similar results.

This test verifies that given the same initial state and action,
the CPU (Box2D-based) and GPU (simplified) physics produce approximately
the same next state.

It also tests ACCUMULATED drift over many steps to ensure policies
trained on GPU can transfer to CPU evaluation.

Run with:
    pixi run -e apple mojo run tests/test_cpu_gpu_physics_match.mojo
"""

from math import cos, sin, sqrt
from random import seed, random_float64

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu import LunarLanderGPU, gpu_dtype

# Constants for comparison
comptime TOLERANCE: Float64 = 0.15  # Allow 15% relative error for physics differences
comptime ABS_TOLERANCE: Float64 = 0.5  # Absolute tolerance for small values

# Extended test parameters
comptime LONG_EPISODE_STEPS: Int = 200  # Steps for accumulated drift test
comptime DRIFT_CHECK_INTERVAL: Int = 25  # Check drift every N steps


fn abs_val(x: Float64) -> Float64:
    """Absolute value helper."""
    return x if x >= 0 else -x


fn compare_states(
    cpu_state: List[Float64],
    gpu_state: List[Float64],
    step_num: Int,
    action: Int,
) -> Tuple[Bool, String]:
    """Compare CPU and GPU states, return (passed, message)."""
    var state_names = List[String]()
    state_names.append("x")
    state_names.append("y")
    state_names.append("vx")
    state_names.append("vy")
    state_names.append("angle")
    state_names.append("angular_vel")
    state_names.append("left_contact")
    state_names.append("right_contact")

    var max_diff: Float64 = 0.0
    var max_diff_name = String("")
    var all_passed = True
    var details = String("")

    for i in range(8):
        var cpu_val = cpu_state[i]
        var gpu_val = gpu_state[i]
        var diff = abs_val(cpu_val - gpu_val)

        # Calculate relative error (avoid division by zero)
        var rel_error: Float64 = 0.0
        var max_abs = abs_val(cpu_val) if abs_val(cpu_val) > abs_val(gpu_val) else abs_val(gpu_val)
        if max_abs > 0.01:
            rel_error = diff / max_abs
        else:
            rel_error = diff  # Use absolute error for small values

        if diff > max_diff:
            max_diff = diff
            max_diff_name = state_names[i]

        # Check if within tolerance
        var passed = (rel_error < TOLERANCE) or (diff < ABS_TOLERANCE)
        if not passed:
            all_passed = False
            details += "  " + state_names[i] + ": CPU=" + String(cpu_val) + " GPU=" + String(gpu_val) + " diff=" + String(diff) + "\n"

    var msg = String("Step ") + String(step_num) + " action=" + String(action) + " max_diff=" + String(max_diff) + " (" + max_diff_name + ")"
    if not all_passed:
        msg += "\n" + details

    return (all_passed, msg)


fn run_gpu_single_step(
    ctx: DeviceContext,
    initial_state: List[Float64],
    action: Int,
    step_seed: Int = 0,
) raises -> List[Float64]:
    """Run a single GPU physics step and return the resulting state."""
    comptime BATCH_SIZE = 1
    comptime STATE_SIZE = 8

    # Allocate GPU buffers
    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

    # Create host buffers for data transfer
    var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
    var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)
    var host_rewards = ctx.enqueue_create_host_buffer[gpu_dtype](1)
    var host_dones = ctx.enqueue_create_host_buffer[gpu_dtype](1)

    # Copy initial state to host buffer
    for i in range(8):
        host_states[i] = Scalar[gpu_dtype](initial_state[i])

    # Set action
    host_actions[0] = Scalar[gpu_dtype](action)

    # Initialize rewards and dones
    host_rewards[0] = Scalar[gpu_dtype](0.0)
    host_dones[0] = Scalar[gpu_dtype](0.0)

    # Copy to GPU
    ctx.enqueue_copy(states_buf, host_states)
    ctx.enqueue_copy(actions_buf, host_actions)
    ctx.enqueue_copy(rewards_buf, host_rewards)
    ctx.enqueue_copy(dones_buf, host_dones)
    ctx.synchronize()

    # Run GPU physics step with seed for random dispersion
    LunarLanderGPU.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, UInt64(step_seed)
    )
    ctx.synchronize()

    # Copy result back to host
    ctx.enqueue_copy(host_states, states_buf)
    ctx.synchronize()

    # Convert to Float64 List
    var result = List[Float64]()
    for i in range(8):
        result.append(Float64(host_states[i]))

    return result^


fn test_single_action(
    mut cpu_env: LunarLanderEnv,
    ctx: DeviceContext,
    action: Int,
    step_num: Int,
) raises -> Tuple[Bool, String, List[Float64], List[Float64]]:
    """Test a single action on both CPU and GPU, starting from CPU's current state."""

    # Get CPU state before step
    var cpu_state_before = cpu_env.get_obs_list()

    # Run CPU step
    var cpu_result = cpu_env.step_discrete(action)
    var cpu_state_after = cpu_result[0].to_list()

    # Run GPU step with same initial state (pass step_num for random seed)
    var gpu_state_after = run_gpu_single_step(ctx, cpu_state_before, action, step_num)

    # Compare
    var comparison = compare_states(cpu_state_after, gpu_state_after, step_num, action)

    return (comparison[0], comparison[1], cpu_state_after^, gpu_state_after^)


fn main() raises:
    print("=" * 60)
    print("CPU vs GPU LunarLander Physics Comparison Test")
    print("=" * 60)
    print("")

    # Initialize
    seed(42)
    var ctx = DeviceContext()
    var cpu_env = LunarLanderEnv(enable_wind=True)

    # Reset CPU environment
    _ = cpu_env.reset()

    print("Testing physics match for each action type...")
    print("Tolerance: " + String(TOLERANCE * 100) + "% relative or " + String(ABS_TOLERANCE) + " absolute")
    print("")

    var total_tests = 0
    var passed_tests = 0

    # Test sequence: run several steps with different actions
    var actions = List[Int]()
    # Test each action multiple times
    actions.append(0)  # nop
    actions.append(2)  # main engine
    actions.append(2)  # main engine
    actions.append(1)  # left engine
    actions.append(3)  # right engine
    actions.append(0)  # nop
    actions.append(2)  # main engine
    actions.append(1)  # left engine
    actions.append(1)  # left engine
    actions.append(3)  # right engine
    actions.append(3)  # right engine
    actions.append(2)  # main engine
    actions.append(0)  # nop
    actions.append(2)  # main engine
    actions.append(2)  # main engine

    print("Action sequence: ", end="")
    for i in range(len(actions)):
        print(String(actions[i]) + " ", end="")
    print("\n")

    # Run tests
    for i in range(len(actions)):
        var action = actions[i]
        var result = test_single_action(cpu_env, ctx, action, i)
        var passed = result[0]
        var msg = result[1]

        total_tests += 1
        if passed:
            passed_tests += 1
            print("[PASS] " + msg)
        else:
            print("[FAIL] " + msg)

    print("")
    print("=" * 60)
    print("Results: " + String(passed_tests) + "/" + String(total_tests) + " tests passed")

    if passed_tests == total_tests:
        print("SUCCESS: GPU physics matches CPU physics within tolerance!")
    else:
        print("WARNING: Some physics differences exceed tolerance")
        print("This may affect generalization between GPU training and CPU evaluation")

    print("=" * 60)

    # Detailed state comparison for first few steps
    print("\nDetailed comparison (first 5 steps with main engine):")
    print("-" * 60)

    # Reset and run detailed comparison
    _ = cpu_env.reset()
    var cpu_state = cpu_env.get_obs_list()

    for step in range(5):
        var action = 2  # main engine
        var cpu_result = cpu_env.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var gpu_next = run_gpu_single_step(ctx, cpu_state, action, step + 100)

        print("\nStep " + String(step) + " (main engine):")
        print("  State    |    CPU     |    GPU     |   Diff")
        print("  " + "-" * 45)

        var state_names = List[String]()
        state_names.append("x      ")
        state_names.append("y      ")
        state_names.append("vx     ")
        state_names.append("vy     ")
        state_names.append("angle  ")
        state_names.append("ang_vel")
        state_names.append("left_c ")
        state_names.append("right_c")

        for i in range(8):
            var diff = abs_val(cpu_next[i] - gpu_next[i])
            print("  " + state_names[i] + " | " + String(cpu_next[i])[:10] + " | " + String(gpu_next[i])[:10] + " | " + String(diff)[:8])

        # Update cpu_state for next iteration (copy for use in next step)
        cpu_state.clear()
        for i in range(8):
            cpu_state.append(cpu_next[i])

    # =========================================================================
    # EXTENDED TEST: Accumulated drift over many steps
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXTENDED TEST: Accumulated Drift Over " + String(LONG_EPISODE_STEPS) + " Steps")
    print("=" * 60)
    print("")
    print("Running CPU and GPU in parallel with SAME actions...")
    print("This simulates what happens during evaluation.")
    print("")

    # Reset both environments
    seed(123)  # Different seed for variety
    var cpu_env2 = LunarLanderEnv(enable_wind=False)  # Match GPU (no wind)
    _ = cpu_env2.reset()

    # Get initial CPU state and use it for GPU too
    var cpu_obs = cpu_env2.get_obs_list()
    var gpu_obs = List[Float64]()
    for j in range(8):
        gpu_obs.append(cpu_obs[j])

    var state_names2 = List[String]()
    state_names2.append("x      ")
    state_names2.append("y      ")
    state_names2.append("vx     ")
    state_names2.append("vy     ")
    state_names2.append("angle  ")
    state_names2.append("ang_vel")
    state_names2.append("left_c ")
    state_names2.append("right_c")

    print("Initial state (both same):")
    print("  x=" + String(cpu_obs[0])[:7] + " y=" + String(cpu_obs[1])[:7] +
          " vx=" + String(cpu_obs[2])[:7] + " vy=" + String(cpu_obs[3])[:7])
    print("")

    var max_accumulated_diff = List[Float64]()
    for j in range(8):
        max_accumulated_diff.append(0.0)

    var episode_done = False
    var final_step = 0

    print("Step  | Action | Max Diff | Worst State | CPU Reward | GPU Reward")
    print("-" * 70)

    for step in range(LONG_EPISODE_STEPS):
        if episode_done:
            break

        final_step = step

        # Choose action based on simple heuristic (like a basic policy)
        # Fire main engine if falling, side engines to correct angle
        var action = 0  # default nop
        var vy_val = cpu_obs[3]
        var angle_val = cpu_obs[4]
        var angvel_val = cpu_obs[5]

        # Simple control logic
        if vy_val < -0.1:  # Falling
            action = 2  # Main engine
        elif angle_val > 0.1 or angvel_val > 0.2:
            action = 1  # Left engine to correct
        elif angle_val < -0.1 or angvel_val < -0.2:
            action = 3  # Right engine to correct
        else:
            # Random action for variety
            var rand_val = random_float64()
            if rand_val < 0.4:
                action = 2  # Main engine more often
            elif rand_val < 0.6:
                action = 1
            elif rand_val < 0.8:
                action = 3
            # else nop

        # Step CPU
        var cpu_result = cpu_env2.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_reward = cpu_result[1]
        var cpu_done = cpu_result[2]

        # Step GPU with same action from current GPU state (pass step for random seed)
        var gpu_next = run_gpu_single_step(ctx, gpu_obs, action, step + 200)

        # Compute GPU reward approximation (simplified shaping)
        var gpu_x = gpu_next[0]
        var gpu_y = gpu_next[1]
        var gpu_vx = gpu_next[2]
        var gpu_vy = gpu_next[3]
        var gpu_angle = gpu_next[4]
        var gpu_dist = sqrt(gpu_x * gpu_x + gpu_y * gpu_y)
        var gpu_speed = sqrt(gpu_vx * gpu_vx + gpu_vy * gpu_vy)
        var gpu_angle_abs = gpu_angle if gpu_angle >= 0 else -gpu_angle

        # Track differences
        var step_max_diff: Float64 = 0.0
        var worst_state_idx = 0
        for j in range(8):
            var diff = abs_val(cpu_next[j] - gpu_next[j])
            if diff > max_accumulated_diff[j]:
                max_accumulated_diff[j] = diff
            if diff > step_max_diff:
                step_max_diff = diff
                worst_state_idx = j

        # Print at intervals
        if step % DRIFT_CHECK_INTERVAL == 0 or cpu_done or step < 5:
            var action_name = "nop  "
            if action == 1:
                action_name = "left "
            elif action == 2:
                action_name = "main "
            elif action == 3:
                action_name = "right"

            print(
                String(step)[:5] + "  | " + action_name + "  | " +
                String(step_max_diff)[:8] + " | " + state_names2[worst_state_idx] +
                "  | " + String(cpu_reward)[:10] + " | (approx)"
            )

        # Update states for next step (each uses its own state)
        cpu_obs.clear()
        gpu_obs.clear()
        for j in range(8):
            cpu_obs.append(cpu_next[j])
            gpu_obs.append(gpu_next[j])

        if cpu_done:
            episode_done = True
            print("\nCPU episode terminated at step " + String(step))

    print("")
    print("-" * 70)
    print("ACCUMULATED DRIFT SUMMARY (max difference seen over " + String(final_step + 1) + " steps):")
    print("-" * 70)

    var any_large_drift = False
    for j in range(8):
        var drift = max_accumulated_diff[j]
        var status = "OK"
        if drift > 1.0:
            status = "LARGE!"
            any_large_drift = True
        elif drift > 0.5:
            status = "MODERATE"
        print("  " + state_names2[j] + ": " + String(drift)[:10] + "  " + status)

    print("")
    if any_large_drift:
        print("WARNING: Large accumulated drift detected!")
        print("This may cause policies trained on GPU to fail on CPU evaluation.")
    else:
        print("Accumulated drift is within acceptable bounds.")

    # Final state comparison
    print("")
    print("Final state comparison at step " + String(final_step) + ":")
    print("  State    |    CPU     |    GPU     |   Diff")
    print("  " + "-" * 45)
    for j in range(8):
        var diff = abs_val(cpu_obs[j] - gpu_obs[j])
        print("  " + state_names2[j] + " | " + String(cpu_obs[j])[:10] + " | " + String(gpu_obs[j])[:10] + " | " + String(diff)[:8])

    print("\n" + "=" * 60)
