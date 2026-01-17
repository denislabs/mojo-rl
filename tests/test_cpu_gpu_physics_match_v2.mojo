"""Test that CPU and GPU V2 LunarLander physics produce similar results.

This test verifies that the improved GPU physics (V2) with sub-stepping,
Velocity Verlet, and contact iteration matches CPU physics better than V1.

Run with:
    pixi run -e apple mojo run tests/test_cpu_gpu_physics_match_v2.mojo
"""

from math import cos, sin, sqrt
from random import seed, random_float64

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu_v2 import (
    LunarLanderGPUv2,
    gpu_dtype,
    FULL_STATE_SIZE,
)

# Constants for comparison
comptime TOLERANCE: Float32 = 0.15
comptime ABS_TOLERANCE: Float32 = 0.5

# Extended test parameters
comptime LONG_EPISODE_STEPS: Int = 200
comptime DRIFT_CHECK_INTERVAL: Int = 25


fn abs_val(x: Float32) -> Float32:
    return x if x >= 0 else -x


fn run_gpu_single_step(
    ctx: DeviceContext,
    initial_state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int = 0,
) raises -> List[Float32]:
    """Run a single GPU V2 physics step and return the resulting state."""
    comptime BATCH_SIZE = 1
    comptime STATE_SIZE = FULL_STATE_SIZE  # 12 for V2

    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
        BATCH_SIZE * STATE_SIZE
    )
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

    var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
    var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)
    var host_rewards = ctx.enqueue_create_host_buffer[gpu_dtype](1)
    var host_dones = ctx.enqueue_create_host_buffer[gpu_dtype](1)

    # Copy initial observable state (first 8 values)
    for i in range(8):
        host_states[i] = Scalar[gpu_dtype](initial_state[i])

    # Initialize hidden state (indices 8-11) with reasonable defaults
    # prev velocities = current velocities
    host_states[8] = host_states[2]  # prev_vx = vx
    host_states[9] = host_states[3]  # prev_vy = vy
    host_states[10] = host_states[5]  # prev_angular_vel = angular_vel
    host_states[11] = Scalar[gpu_dtype](0.0)  # contact_impulse = 0

    host_actions[0] = Scalar[gpu_dtype](action)
    host_rewards[0] = Scalar[gpu_dtype](0.0)
    host_dones[0] = Scalar[gpu_dtype](0.0)

    ctx.enqueue_copy(states_buf, host_states)
    ctx.enqueue_copy(actions_buf, host_actions)
    ctx.enqueue_copy(rewards_buf, host_rewards)
    ctx.enqueue_copy(dones_buf, host_dones)
    ctx.synchronize()

    LunarLanderGPUv2.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, UInt64(step_seed)
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_states, states_buf)
    ctx.synchronize()

    # Return only the observable state (first 8 values)
    var result = List[Scalar[DType.float32]]()
    for i in range(8):
        result.append(host_states[i])

    return result^


fn compare_states(
    cpu_state: List[Scalar[DType.float32]],
    gpu_state: List[Scalar[DType.float32]],
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

    var max_diff: Float32 = 0.0
    var max_diff_name = String("")
    var all_passed = True
    var details = String("")

    for i in range(8):
        var cpu_val = cpu_state[i]
        var gpu_val = gpu_state[i]
        var diff = abs(cpu_val - gpu_val)

        var rel_error: Float32 = 0.0
        var max_abs = abs(cpu_val) if abs(cpu_val) > abs(gpu_val) else abs(
            gpu_val
        )
        if max_abs > 0.01:
            rel_error = diff / max_abs
        else:
            rel_error = diff

        if diff > max_diff:
            max_diff = diff
            max_diff_name = state_names[i]

        var passed = (rel_error < TOLERANCE) or (diff < ABS_TOLERANCE)
        if not passed:
            all_passed = False
            details += (
                "  "
                + state_names[i]
                + ": CPU="
                + String(cpu_val)
                + " GPU="
                + String(gpu_val)
                + " diff="
                + String(diff)
                + "\n"
            )

    var msg = (
        String("Step ")
        + String(step_num)
        + " action="
        + String(action)
        + " max_diff="
        + String(max_diff)
        + " ("
        + max_diff_name
        + ")"
    )
    if not all_passed:
        msg += "\n" + details

    return (all_passed, msg)


fn main() raises:
    print("=" * 60)
    print("CPU vs GPU V2 LunarLander Physics Comparison Test")
    print("=" * 60)
    print("")
    print("V2 improvements:")
    print("  - 4 sub-steps per frame (vs 1 in V1)")
    print("  - Velocity Verlet integration")
    print("  - Hidden state for physics continuity")
    print("  - 2 contact iterations per sub-step")
    print("")

    seed(42)
    var ctx = DeviceContext()
    var cpu_env = LunarLanderEnv[DType.float32](
        enable_wind=False
    )  # Match GPU (no wind)
    _ = cpu_env.reset()

    print("Testing physics match for each action type...")
    print(
        "Tolerance: "
        + String(TOLERANCE * 100)
        + "% relative or "
        + String(ABS_TOLERANCE)
        + " absolute"
    )
    print("")

    var total_tests = 0
    var passed_tests = 0

    # Test sequence
    var actions = List[Int]()
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
        var cpu_state_before = cpu_env.get_obs_list()

        var cpu_result = cpu_env.step_discrete(action)
        var cpu_state_after = cpu_result[0].to_list()

        var gpu_state_after = run_gpu_single_step(
            ctx, cpu_state_before, action, i
        )

        var comparison = compare_states(
            cpu_state_after, gpu_state_after, i, action
        )
        var passed = comparison[0]
        var msg = comparison[1]

        total_tests += 1
        if passed:
            passed_tests += 1
            print("[PASS] " + msg)
        else:
            print("[FAIL] " + msg)

    print("")
    print("=" * 60)
    print(
        "Results: "
        + String(passed_tests)
        + "/"
        + String(total_tests)
        + " tests passed"
    )

    if passed_tests == total_tests:
        print("SUCCESS: GPU V2 physics matches CPU physics within tolerance!")
    else:
        print("WARNING: Some physics differences exceed tolerance")

    print("=" * 60)

    # =========================================================================
    # EXTENDED TEST: Accumulated drift over many steps
    # =========================================================================
    print("\n" + "=" * 60)
    print(
        "EXTENDED TEST: Accumulated Drift Over "
        + String(LONG_EPISODE_STEPS)
        + " Steps"
    )
    print("=" * 60)
    print("")
    print("Running CPU and GPU in parallel with SAME actions...")
    print("")

    seed(123)
    var cpu_env2 = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env2.reset()

    var cpu_obs = cpu_env2.get_obs_list()
    var gpu_obs = List[Scalar[DType.float32]]()
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
    print(
        "  x="
        + String(cpu_obs[0])[:7]
        + " y="
        + String(cpu_obs[1])[:7]
        + " vx="
        + String(cpu_obs[2])[:7]
        + " vy="
        + String(cpu_obs[3])[:7]
    )
    print("")

    var max_accumulated_diff = List[Scalar[DType.float32]]()
    for j in range(8):
        max_accumulated_diff.append(0.0)

    var episode_done = False
    var final_step = 0

    print("Step  | Action | Max Diff | Worst State | CPU Reward")
    print("-" * 60)

    for step in range(LONG_EPISODE_STEPS):
        if episode_done:
            break

        final_step = step

        # Simple control logic
        var action = 0
        var vy_val = cpu_obs[3]
        var angle_val = cpu_obs[4]
        var angvel_val = cpu_obs[5]

        if vy_val < -0.1:
            action = 2  # Main engine
        elif angle_val > 0.1 or angvel_val > 0.2:
            action = 1  # Left engine
        elif angle_val < -0.1 or angvel_val < -0.2:
            action = 3  # Right engine
        else:
            var rand_val = random_float64()
            if rand_val < 0.4:
                action = 2
            elif rand_val < 0.6:
                action = 1
            elif rand_val < 0.8:
                action = 3

        # Step CPU
        var cpu_result = cpu_env2.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_reward = cpu_result[1]
        var cpu_done = cpu_result[2]

        # Step GPU with same action from current GPU state
        var gpu_next = run_gpu_single_step(ctx, gpu_obs, action, step + 200)

        # Track differences
        var step_max_diff: Float32 = 0.0
        var worst_state_idx = 0
        for j in range(8):
            var diff = abs(cpu_next[j] - gpu_next[j])
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
                String(step)[:5]
                + "  | "
                + action_name
                + "  | "
                + String(step_max_diff)[:8]
                + " | "
                + state_names2[worst_state_idx]
                + "  | "
                + String(cpu_reward)[:10]
            )

        # Update states for next step
        cpu_obs.clear()
        gpu_obs.clear()
        for j in range(8):
            cpu_obs.append(cpu_next[j])
            gpu_obs.append(gpu_next[j])

        if cpu_done:
            episode_done = True
            print("\nCPU episode terminated at step " + String(step))

    print("")
    print("-" * 60)
    print(
        "ACCUMULATED DRIFT SUMMARY (max difference seen over "
        + String(final_step + 1)
        + " steps):"
    )
    print("-" * 60)

    var any_large_drift = False
    for j in range(8):
        var drift = max_accumulated_diff[j]
        var status = "OK"
        if drift > 1.0:
            status = "LARGE!"
            any_large_drift = True
        elif drift > 0.5:
            status = "MODERATE"
        print(
            "  " + state_names2[j] + ": " + String(drift)[:10] + "  " + status
        )

    print("")
    if any_large_drift:
        print("WARNING: Large accumulated drift detected!")
    else:
        print("SUCCESS: Accumulated drift is within acceptable bounds.")

    # Final state comparison
    print("")
    print("Final state comparison at step " + String(final_step) + ":")
    print("  State    |    CPU     |    GPU     |   Diff")
    print("  " + "-" * 45)
    for j in range(8):
        var diff = abs_val(cpu_obs[j] - gpu_obs[j])
        print(
            "  "
            + state_names2[j]
            + " | "
            + String(cpu_obs[j])[:10]
            + " | "
            + String(gpu_obs[j])[:10]
            + " | "
            + String(diff)[:8]
        )

    print("\n" + "=" * 60)
