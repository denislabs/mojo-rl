"""Compare CPU vs GPU V3 physics.

Run with:
    pixi run -e apple mojo run tests/test_physics_v3_compare.mojo
"""

from math import sqrt
from random import seed

from gpu.host import DeviceContext

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu_v3 import LunarLanderGPUv3, gpu_dtype


fn abs_f32(x: Float32) -> Float32:
    return x if x >= 0 else -x


fn format_float(val: Float32, width: Int = 10) -> String:
    var s = String(val)
    if len(s) > width:
        return String(s[:width])
    return s


fn create_gpu_state(
    obs: List[Scalar[DType.float32]],
) -> List[Scalar[DType.float32]]:
    """Create GPU V3 state (8 values)."""
    var state = List[Scalar[DType.float32]]()
    for i in range(8):
        state.append(obs[i])
    return state^


fn run_gpu_step(
    ctx: DeviceContext,
    initial_state: List[Scalar[DType.float32]],
    action: Int,
    step_seed: Int = 0,
) raises -> List[Float32]:
    """Run a single GPU V3 physics step."""
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
    print("=" * 70)
    print("CPU vs GPU V3 LunarLander Physics Comparison")
    print("=" * 70)
    print("")
    print("V3 improvements:")
    print("  - Leg tip position accounts for full leg length")
    print("  - Impulse-based contact response (not velocity damping)")
    print("  - Semi-implicit Euler matching Box2D order")
    print("")

    seed(42)
    var ctx = DeviceContext()
    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    var cpu_obs = cpu_env.get_obs_list()
    var gpu_state = create_gpu_state(cpu_obs)

    # Test controlled landing sequence
    print("CONTROLLED LANDING SEQUENCE")
    print("-" * 70)
    print("")

    var state_names = List[String]()
    state_names.append("x      ")
    state_names.append("y      ")
    state_names.append("vx     ")
    state_names.append("vy     ")
    state_names.append("angle  ")
    state_names.append("ang_vel")
    state_names.append("left_c ")
    state_names.append("right_c")

    var max_drift = List[Float32]()
    for j in range(8):
        max_drift.append(0.0)

    print("Step | Action | Total Diff | Worst State")
    print("-" * 50)

    for step in range(150):
        # Simple control policy
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

        var gpu_next = run_gpu_step(ctx, gpu_state, action, step + 1000)

        # Compute differences
        var total_diff: Float32 = 0.0
        var worst_idx = 0
        var worst_diff: Float32 = 0.0

        for j in range(8):
            var diff = abs_f32(cpu_next[j] - gpu_next[j])
            total_diff += diff
            if diff > max_drift[j]:
                max_drift[j] = diff
            if diff > worst_diff:
                worst_diff = diff
                worst_idx = j

        if step < 5 or step % 20 == 0 or cpu_done:
            var action_str = "nop  "
            if action == 1:
                action_str = "left "
            elif action == 2:
                action_str = "main "
            elif action == 3:
                action_str = "right"
            print(
                String(step) + "    | " + action_str + "  | "
                + format_float(total_diff, 10) + " | "
                + state_names[worst_idx]
            )

        if cpu_done:
            print("\nCPU terminated at step " + String(step))
            break

        # Update for next step
        cpu_obs.clear()
        gpu_state.clear()
        for j in range(8):
            cpu_obs.append(cpu_next[j])
            gpu_state.append(gpu_next[j])

    print("")
    print("MAX DRIFT PER STATE COMPONENT:")
    print("-" * 40)
    var total_max: Float32 = 0.0
    for j in range(8):
        var status = "OK" if max_drift[j] < 0.1 else ("MODERATE" if max_drift[j] < 0.5 else "HIGH")
        print("  " + state_names[j] + ": " + format_float(max_drift[j], 10) + " " + status)
        total_max += max_drift[j]
    print("")
    print("Total max drift: " + format_float(total_max))

    if total_max < 1.0:
        print("GOOD: Total drift under 1.0 - policies should transfer well")
    elif total_max < 3.0:
        print("MODERATE: Total drift under 3.0 - may need domain randomization")
    else:
        print("HIGH: Total drift over 3.0 - needs more work to match physics")

    print("=" * 70)
