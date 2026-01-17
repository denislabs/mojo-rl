"""Compare all GPU versions (V1, V2, V3) against CPU.

Run with:
    pixi run -e apple mojo run tests/test_all_versions_compare.mojo
"""

from math import sqrt
from random import seed

from gpu.host import DeviceContext

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu import LunarLanderGPU, gpu_dtype
from envs.lunar_lander_gpu_v2 import LunarLanderGPUv2, FULL_STATE_SIZE as V2_STATE_SIZE
from envs.lunar_lander_gpu_v3 import LunarLanderGPUv3


fn abs_f32(x: Float32) -> Float32:
    return x if x >= 0 else -x


fn format_float(val: Float32, width: Int = 10) -> String:
    var s = String(val)
    if len(s) > width:
        return String(s[:width])
    return s


fn run_gpu_v1_step(
    ctx: DeviceContext,
    state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int,
) raises -> List[Float32]:
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
    state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int,
) raises -> List[Float32]:
    comptime BATCH_SIZE = 1
    comptime STATE_SIZE = V2_STATE_SIZE  # 12

    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)

    var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE)
    var host_actions = ctx.enqueue_create_host_buffer[gpu_dtype](1)

    # Copy observable state
    for i in range(8):
        host_states[i] = state[i]
    # Initialize hidden state
    host_states[8] = state[2]  # prev_vx
    host_states[9] = state[3]  # prev_vy
    host_states[10] = state[5]  # prev_angular_vel
    host_states[11] = Scalar[gpu_dtype](0.0)  # contact_impulse

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
    for i in range(8):
        result.append(Float32(host_states[i]))
    return result^


fn run_gpu_v3_step(
    ctx: DeviceContext,
    state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int,
) raises -> List[Float32]:
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

    LunarLanderGPUv3.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, UInt64(step_seed)
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_states, states_buf)
    ctx.synchronize()

    var result = List[Float32]()
    for i in range(8):
        result.append(Float32(host_states[i]))
    return result^


fn main() raises:
    print("=" * 80)
    print("LUNAR LANDER GPU VERSIONS COMPARISON")
    print("=" * 80)
    print("")

    seed(42)
    var ctx = DeviceContext()
    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    # Get initial state
    var cpu_obs = cpu_env.get_obs_list()

    # Initialize all GPU states from same CPU state
    var v1_state = List[Scalar[DType.float32]]()
    var v2_state = List[Scalar[DType.float32]]()
    var v3_state = List[Scalar[DType.float32]]()
    for i in range(8):
        v1_state.append(cpu_obs[i])
        v2_state.append(cpu_obs[i])
        v3_state.append(cpu_obs[i])

    # Track max drift per version
    var max_drift_v1: Float32 = 0.0
    var max_drift_v2: Float32 = 0.0
    var max_drift_v3: Float32 = 0.0

    # Track per-step drift
    var drift_history_v1 = List[Float32]()
    var drift_history_v2 = List[Float32]()
    var drift_history_v3 = List[Float32]()

    print("Running controlled landing sequence...")
    print("Using same actions for CPU and all GPU versions")
    print("")

    print("Step | Action | V1 Drift   | V2 Drift   | V3 Drift   | Winner")
    print("-" * 70)

    for step in range(150):
        # Control policy based on CPU state
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

        # Step CPU
        var cpu_result = cpu_env.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        # Step all GPU versions
        var v1_next = run_gpu_v1_step(ctx, v1_state, action, step + 1000)
        var v2_next = run_gpu_v2_step(ctx, v2_state, action, step + 1000)
        var v3_next = run_gpu_v3_step(ctx, v3_state, action, step + 1000)

        # Compute total drift for each version
        var drift_v1: Float32 = 0.0
        var drift_v2: Float32 = 0.0
        var drift_v3: Float32 = 0.0

        for j in range(8):
            drift_v1 += abs_f32(cpu_next[j] - v1_next[j])
            drift_v2 += abs_f32(cpu_next[j] - v2_next[j])
            drift_v3 += abs_f32(cpu_next[j] - v3_next[j])

        drift_history_v1.append(drift_v1)
        drift_history_v2.append(drift_v2)
        drift_history_v3.append(drift_v3)

        if drift_v1 > max_drift_v1:
            max_drift_v1 = drift_v1
        if drift_v2 > max_drift_v2:
            max_drift_v2 = drift_v2
        if drift_v3 > max_drift_v3:
            max_drift_v3 = drift_v3

        # Determine winner
        var winner = "V1"
        var min_drift = drift_v1
        if drift_v2 < min_drift:
            min_drift = drift_v2
            winner = "V2"
        if drift_v3 < min_drift:
            min_drift = drift_v3
            winner = "V3"

        if step < 5 or step % 20 == 0 or cpu_done:
            var action_str = "nop  "
            if action == 1:
                action_str = "left "
            elif action == 2:
                action_str = "main "
            elif action == 3:
                action_str = "right"

            print(
                String(step) + "    | " + action_str + " | "
                + format_float(drift_v1, 10) + " | "
                + format_float(drift_v2, 10) + " | "
                + format_float(drift_v3, 10) + " | "
                + winner
            )

        if cpu_done:
            print("\nCPU terminated at step " + String(step))
            break

        # Update states for next step
        cpu_obs.clear()
        v1_state.clear()
        v2_state.clear()
        v3_state.clear()
        for j in range(8):
            cpu_obs.append(cpu_next[j])
            v1_state.append(v1_next[j])
            v2_state.append(v2_next[j])
            v3_state.append(v3_next[j])

    # Summary
    print("")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("")

    print("Max total drift over episode:")
    print("  GPU V1: " + format_float(max_drift_v1))
    print("  GPU V2: " + format_float(max_drift_v2))
    print("  GPU V3: " + format_float(max_drift_v3))

    # Calculate improvement
    var v3_vs_v1_improvement = (max_drift_v1 - max_drift_v3) / max_drift_v1 * 100.0
    var v3_vs_v2_improvement = (max_drift_v2 - max_drift_v3) / max_drift_v2 * 100.0

    print("")
    print("V3 improvement over V1: " + format_float(v3_vs_v1_improvement, 5) + "%")
    print("V3 improvement over V2: " + format_float(v3_vs_v2_improvement, 5) + "%")

    # Analyze drift growth rate
    if len(drift_history_v1) > 20:
        var early_v1: Float32 = 0.0
        var late_v1: Float32 = 0.0
        var early_v2: Float32 = 0.0
        var late_v2: Float32 = 0.0
        var early_v3: Float32 = 0.0
        var late_v3: Float32 = 0.0

        for i in range(5):
            early_v1 += drift_history_v1[i]
            early_v2 += drift_history_v2[i]
            early_v3 += drift_history_v3[i]
            late_v1 += drift_history_v1[len(drift_history_v1) - 5 + i]
            late_v2 += drift_history_v2[len(drift_history_v2) - 5 + i]
            late_v3 += drift_history_v3[len(drift_history_v3) - 5 + i]

        early_v1 /= 5
        early_v2 /= 5
        early_v3 /= 5
        late_v1 /= 5
        late_v2 /= 5
        late_v3 /= 5

        var growth_v1 = late_v1 / early_v1 if early_v1 > 0.001 else 0.0
        var growth_v2 = late_v2 / early_v2 if early_v2 > 0.001 else 0.0
        var growth_v3 = late_v3 / early_v3 if early_v3 > 0.001 else 0.0

        print("")
        print("Drift growth factor (late/early):")
        print("  GPU V1: " + format_float(growth_v1) + "x")
        print("  GPU V2: " + format_float(growth_v2) + "x")
        print("  GPU V3: " + format_float(growth_v3) + "x")

    print("")
    print("RECOMMENDATION:")
    print("-" * 50)

    var best_version = "V1"
    var best_drift = max_drift_v1
    if max_drift_v2 < best_drift:
        best_drift = max_drift_v2
        best_version = "V2"
    if max_drift_v3 < best_drift:
        best_drift = max_drift_v3
        best_version = "V3"

    print("Best GPU version: " + best_version + " (max drift: " + format_float(best_drift) + ")")

    if best_drift < 1.0:
        print("Status: GOOD - Policies should transfer well to CPU")
    elif best_drift < 2.0:
        print("Status: MODERATE - Consider domain randomization during training")
    else:
        print("Status: HIGH - Further physics improvements needed")

    print("=" * 80)
