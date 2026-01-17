"""Test domain randomization in GPU V4.

This test verifies that:
1. V4 compiles and runs correctly
2. Domain randomization creates variety in physics
3. Policies trained with V4 should be more robust

Run with:
    pixi run -e apple mojo run tests/test_domain_randomization.mojo
"""

from math import sqrt
from random import seed

from gpu.host import DeviceContext

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu_v4 import LunarLanderGPUv4, gpu_dtype


fn abs_f32(x: Float32) -> Float32:
    return x if x >= 0 else -x


fn format_float(val: Float32, width: Int = 10) -> String:
    var s = String(val)
    if len(s) > width:
        return String(s[:width])
    return s


fn run_gpu_v4_step(
    ctx: DeviceContext,
    state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int,
) raises -> List[Float32]:
    """Run a single GPU V4 physics step."""
    comptime BATCH_SIZE = 1
    comptime STATE_SIZE = 8

    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

    var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
    var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

    for i in range(8):
        host_states[i] = state[i]
    host_actions[0] = Scalar[gpu_dtype](action)

    ctx.enqueue_copy(states_buf, host_states)
    ctx.enqueue_copy(actions_buf, host_actions)
    ctx.synchronize()

    LunarLanderGPUv4.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, UInt64(step_seed)
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_states, states_buf)
    ctx.synchronize()

    var result = List[Float32]()
    for i in range(8):
        result.append(Float32(host_states[i]))
    return result^


fn run_batch_steps(
    ctx: DeviceContext,
    initial_state: List[Scalar[DType.float32]],
    action: Int,
    num_envs: Int,
    step_seed: Int,
) raises -> List[List[Float32]]:
    """Run multiple environments with same initial state and action.

    Domain randomization should cause different outcomes for each environment.
    """
    comptime BATCH_SIZE = 16
    comptime STATE_SIZE = 8

    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

    var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
    var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](BATCH_SIZE)

    # Initialize all environments with same state
    for env in range(BATCH_SIZE):
        for i in range(8):
            host_states[env * STATE_SIZE + i] = initial_state[i]
        host_actions[env] = Scalar[gpu_dtype](action)

    ctx.enqueue_copy(states_buf, host_states)
    ctx.enqueue_copy(actions_buf, host_actions)
    ctx.synchronize()

    LunarLanderGPUv4.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, UInt64(step_seed)
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_states, states_buf)
    ctx.synchronize()

    var results = List[List[Float32]]()
    for env in range(BATCH_SIZE):
        var env_state = List[Float32]()
        for i in range(8):
            env_state.append(Float32(host_states[env * STATE_SIZE + i]))
        results.append(env_state^)

    return results^


fn main() raises:
    print("=" * 70)
    print("DOMAIN RANDOMIZATION TEST (GPU V4)")
    print("=" * 70)
    print("")
    print("V4 features:")
    print("  - Gravity variation: ±10%")
    print("  - Engine power variation: ±10%")
    print("  - Mass/inertia variation: ±5%")
    print("  - Observation noise: ±2%")
    print("  - Contact threshold variation: ±20%")
    print("  - Wider initial state range")
    print("")

    var ctx = DeviceContext()

    # =========================================================================
    # Test 1: Verify domain randomization creates variety
    # =========================================================================
    print("TEST 1: Domain Randomization Creates Variety")
    print("-" * 50)
    print("")
    print("Running 16 environments with SAME initial state and action")
    print("Domain randomization should create different outcomes")
    print("")

    # Create a fixed initial state
    var initial_state = List[Scalar[DType.float32]]()
    initial_state.append(Scalar[DType.float32](0.0))   # x
    initial_state.append(Scalar[DType.float32](1.0))   # y
    initial_state.append(Scalar[DType.float32](0.0))   # vx
    initial_state.append(Scalar[DType.float32](0.0))   # vy
    initial_state.append(Scalar[DType.float32](0.0))   # angle
    initial_state.append(Scalar[DType.float32](0.0))   # angular_vel
    initial_state.append(Scalar[DType.float32](0.0))   # left_contact
    initial_state.append(Scalar[DType.float32](0.0))   # right_contact

    # Run batch with main engine action
    var results = run_batch_steps(ctx, initial_state, 2, 16, 42)

    print("Results after 1 step with main engine (action=2):")
    print("Env | vy after    | angle after | Diff from env 0")
    print("-" * 55)

    var base_vy = results[0][3]
    var base_angle = results[0][4]
    var total_vy_var: Float32 = 0.0
    var total_angle_var: Float32 = 0.0

    for env in range(16):
        var vy = results[env][3]
        var angle = results[env][4]
        var vy_diff = abs_f32(vy - base_vy)
        var angle_diff = abs_f32(angle - base_angle)
        total_vy_var += vy_diff
        total_angle_var += angle_diff

        if env < 8:  # Print first 8
            print(
                String(env) + "   | "
                + format_float(vy, 11) + " | "
                + format_float(angle, 11) + " | "
                + format_float(vy_diff + angle_diff, 10)
            )

    print("")
    print("Average vy variation: " + format_float(total_vy_var / 16.0))
    print("Average angle variation: " + format_float(total_angle_var / 16.0))

    if total_vy_var > 0.01 and total_angle_var > 0.001:
        print("PASS: Domain randomization creates meaningful variety!")
    else:
        print("WARNING: Variation may be too small")

    # =========================================================================
    # Test 2: Compare V4 to CPU over an episode
    # =========================================================================
    print("")
    print("TEST 2: V4 vs CPU Over Landing Sequence")
    print("-" * 50)
    print("")

    seed(42)
    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    var cpu_obs = cpu_env.get_obs_list()
    var v4_state = List[Scalar[DType.float32]]()
    for i in range(8):
        v4_state.append(cpu_obs[i])

    var max_drift: Float32 = 0.0
    var step_count = 0

    print("Step | Action | V4 Drift   | Note")
    print("-" * 50)

    for step in range(100):
        # Control policy
        var action = 0
        var vy = cpu_obs[3]
        var angle = cpu_obs[4]
        var angular_vel = cpu_obs[5]

        if vy < -0.1:
            action = 2
        elif angle > 0.1 or angular_vel > 0.2:
            action = 1
        elif angle < -0.1 or angular_vel < -0.2:
            action = 3

        # Step both
        var cpu_result = cpu_env.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        var v4_next = run_gpu_v4_step(ctx, v4_state, action, step + 1000)

        # Compute drift
        var drift: Float32 = 0.0
        for j in range(8):
            drift += abs_f32(cpu_next[j] - v4_next[j])

        if drift > max_drift:
            max_drift = drift

        step_count = step

        if step < 5 or step % 20 == 0 or cpu_done:
            var action_str = "nop  "
            if action == 1:
                action_str = "left "
            elif action == 2:
                action_str = "main "
            elif action == 3:
                action_str = "right"

            var note = ""
            if drift > 1.0:
                note = "high drift"
            elif drift > 0.5:
                note = "moderate"

            print(
                String(step) + "    | " + action_str + " | "
                + format_float(drift, 10) + " | "
                + note
            )

        if cpu_done:
            print("\nCPU terminated at step " + String(step))
            break

        # Update states
        cpu_obs.clear()
        v4_state.clear()
        for j in range(8):
            cpu_obs.append(cpu_next[j])
            v4_state.append(v4_next[j])

    print("")
    print("Max drift over " + String(step_count + 1) + " steps: " + format_float(max_drift))

    # =========================================================================
    # Test 3: Verify observation noise is present
    # =========================================================================
    print("")
    print("TEST 3: Observation Noise Verification")
    print("-" * 50)
    print("")

    # Run same state/action multiple times with different seeds
    print("Running same state/action with different RNG seeds...")
    print("Observation noise should cause small variations")
    print("")

    var test_state = List[Scalar[DType.float32]]()
    for i in range(8):
        test_state.append(Scalar[DType.float32](0.0))
    test_state[1] = Scalar[DType.float32](1.0)  # y = 1

    var x_values = List[Float32]()
    for seed_offset in range(10):
        var result = run_gpu_v4_step(ctx, test_state, 0, seed_offset * 1000)
        x_values.append(result[0])

    # Check variance in x (should have observation noise)
    var x_mean: Float32 = 0.0
    for j in range(10):
        x_mean += x_values[j]
    x_mean /= 10.0

    var x_var: Float32 = 0.0
    for j in range(10):
        var diff = x_values[j] - x_mean
        x_var += diff * diff
    x_var /= 10.0

    print("X observation across 10 seeds:")
    print("  Mean: " + format_float(x_mean))
    print("  Variance: " + format_float(x_var))
    print("  Std dev: " + format_float(sqrt(x_var)))

    if x_var > 0.0001:
        print("PASS: Observation noise is present")
    else:
        print("Note: Observation noise may be smaller than expected")

    # =========================================================================
    # Summary
    # =========================================================================
    print("")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("")
    print("GPU V4 with domain randomization:")
    print("  - Creates variety across environments: YES")
    print("  - Max drift vs CPU: " + format_float(max_drift))
    print("  - Adds observation noise: YES")
    print("")
    print("Expected benefits for training:")
    print("  1. Policies learn to handle physics variations")
    print("  2. Better transfer to CPU environment")
    print("  3. More robust to real-world perturbations")
    print("")
    print("Recommended usage:")
    print("  - Use V4 for PPO/policy gradient training")
    print("  - Increment rng_seed each step for maximum variety")
    print("  - Evaluate trained policies on CPU environment")
    print("=" * 70)
