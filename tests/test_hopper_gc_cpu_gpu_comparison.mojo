"""Test to compare CPU and GPU HopperGC implementations.

This test identifies differences between CPU and GPU physics by:
1. Starting from identical initial states (no noise)
2. Applying the same actions
3. Comparing resulting states step by step
"""

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from deep_rl import dtype as gpu_dtype

from envs.hopper_gc import HopperGC
from physics3d.gpu.constants import (
    state_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
)


fn math_abs(x: Float64) -> Float64:
    if x < 0:
        return -x
    return x


fn print_state_comparison(
    cpu_qpos: InlineArray[Scalar[DType.float64], 6],
    cpu_qvel: InlineArray[Scalar[DType.float64], 6],
    gpu_qpos: List[Scalar[gpu_dtype]],
    gpu_qvel: List[Scalar[gpu_dtype]],
    step: Int,
) raises:
    """Print comparison of CPU and GPU states."""
    print("\n=== Step", step, "===")
    print("qpos comparison:")
    print("  idx     CPU             GPU             diff")
    var max_qpos_diff: Float64 = 0.0
    for i in range(6):
        var cpu_val = Float64(cpu_qpos[i])
        var gpu_val = Float64(gpu_qpos[i])
        var diff = math_abs(cpu_val - gpu_val)
        if diff > max_qpos_diff:
            max_qpos_diff = diff
        print("  [", i, "]", cpu_val, "  ", gpu_val, "  ", diff)

    print("\nqvel comparison:")
    print("  idx     CPU             GPU             diff")
    var max_qvel_diff: Float64 = 0.0
    for i in range(6):
        var cpu_val = Float64(cpu_qvel[i])
        var gpu_val = Float64(gpu_qvel[i])
        var diff = math_abs(cpu_val - gpu_val)
        if diff > max_qvel_diff:
            max_qvel_diff = diff
        print("  [", i, "]", cpu_val, "  ", gpu_val, "  ", diff)

    print("\nMax qpos diff:", max_qpos_diff)
    print("Max qvel diff:", max_qvel_diff)


fn initialize_gpu_state_from_cpu[
    STATE_SIZE: Int,
](
    mut state_host: List[Scalar[gpu_dtype]],
    cpu_qpos: InlineArray[Scalar[DType.float64], 6],
    cpu_qvel: InlineArray[Scalar[DType.float64], 6],
):
    """Initialize GPU state buffer from CPU state (no noise)."""
    comptime QPOS_OFF = qpos_offset[6, 6]()
    comptime QVEL_OFF = qvel_offset[6, 6]()
    comptime QACC_OFF = qacc_offset[6, 6]()
    comptime QFRC_OFF = qfrc_offset[6, 6]()

    # Copy qpos
    for i in range(6):
        state_host[QPOS_OFF + i] = Scalar[gpu_dtype](cpu_qpos[i])

    # Copy qvel
    for i in range(6):
        state_host[QVEL_OFF + i] = Scalar[gpu_dtype](cpu_qvel[i])

    # Zero qacc and qfrc
    for i in range(6):
        state_host[QACC_OFF + i] = Scalar[gpu_dtype](0.0)
        state_host[QFRC_OFF + i] = Scalar[gpu_dtype](0.0)


fn extract_gpu_qpos_qvel[
    STATE_SIZE: Int,
](
    state_host: List[Scalar[gpu_dtype]],
    mut out_qpos: List[Scalar[gpu_dtype]],
    mut out_qvel: List[Scalar[gpu_dtype]],
):
    """Extract qpos and qvel from GPU state buffer."""
    comptime QPOS_OFF = qpos_offset[6, 6]()
    comptime QVEL_OFF = qvel_offset[6, 6]()

    out_qpos.clear()
    out_qvel.clear()

    for i in range(6):
        out_qpos.append(state_host[QPOS_OFF + i])
        out_qvel.append(state_host[QVEL_OFF + i])


fn main() raises:
    print("=" * 60)
    print("HopperGC CPU vs GPU Comparison Test")
    print("=" * 60)

    # Create CPU environment
    var env = HopperGC[DType.float64](
        torque_limit=200.0,
        min_height=0.7,
        max_pitch=0.2,
        max_steps=1000,
        timestep=0.002,
        friction=0.5,
    )

    # Reset CPU environment
    _ = env.reset()

    # Get initial CPU state
    var cpu_qpos = env.get_qpos()
    var cpu_qvel = env.get_qvel()

    print("\nInitial CPU state:")
    print(
        "qpos:",
        cpu_qpos[0],
        cpu_qpos[1],
        cpu_qpos[2],
        cpu_qpos[3],
        cpu_qpos[4],
        cpu_qpos[5],
    )
    print(
        "qvel:",
        cpu_qvel[0],
        cpu_qvel[1],
        cpu_qvel[2],
        cpu_qvel[3],
        cpu_qvel[4],
        cpu_qvel[5],
    )

    # Initialize GPU
    var ctx = DeviceContext()

    comptime STATE_SIZE = HopperGC[DType.float64].STATE_SIZE
    comptime BATCH_SIZE = 1
    comptime OBS_DIM = HopperGC[DType.float64].OBS_DIM
    comptime ACTION_DIM = HopperGC[DType.float64].ACTION_DIM

    # Create GPU buffers
    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
        BATCH_SIZE * STATE_SIZE
    )
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](
        BATCH_SIZE * ACTION_DIM
    )
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * OBS_DIM)

    # Initialize GPU state from CPU (ensures identical starting point)
    var state_host = List[Scalar[gpu_dtype]](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(Scalar[gpu_dtype](0.0))

    initialize_gpu_state_from_cpu[STATE_SIZE](state_host, cpu_qpos, cpu_qvel)

    # Copy to GPU
    ctx.enqueue_copy(states_buf, state_host.unsafe_ptr())
    ctx.synchronize()

    # Verify initial GPU state
    ctx.enqueue_copy(state_host.unsafe_ptr(), states_buf)
    ctx.synchronize()

    var gpu_qpos = List[Scalar[gpu_dtype]](capacity=6)
    var gpu_qvel = List[Scalar[gpu_dtype]](capacity=6)
    extract_gpu_qpos_qvel[STATE_SIZE](state_host, gpu_qpos, gpu_qvel)

    print("\nInitial GPU state (after copy from CPU):")
    print(
        "qpos:",
        gpu_qpos[0],
        gpu_qpos[1],
        gpu_qpos[2],
        gpu_qpos[3],
        gpu_qpos[4],
        gpu_qpos[5],
    )
    print(
        "qvel:",
        gpu_qvel[0],
        gpu_qvel[1],
        gpu_qvel[2],
        gpu_qvel[3],
        gpu_qvel[4],
        gpu_qvel[5],
    )

    # Test with specific actions
    var test_actions = List[
        Tuple[
            Scalar[DType.float64], Scalar[DType.float64], Scalar[DType.float64]
        ]
    ]()
    # Various action sequences to test
    test_actions.append((0.0, 0.0, 0.0))  # No action
    test_actions.append((0.5, 0.3, -0.2))  # Mixed actions
    test_actions.append((1.0, 1.0, 1.0))  # Max positive
    test_actions.append((-1.0, -1.0, -1.0))  # Max negative
    test_actions.append((0.0, 0.0, 0.0))  # No action again

    print("\n" + "=" * 60)
    print("Running step-by-step comparison")
    print("=" * 60)

    for step in range(len(test_actions)):
        var action = test_actions[step]

        print("\n--- Action:", action[0], action[1], action[2], "---")

        # Prepare CPU action
        var cpu_action_list = List[Scalar[DType.float64]]()
        cpu_action_list.append(action[0])
        cpu_action_list.append(action[1])
        cpu_action_list.append(action[2])

        # Prepare GPU action
        var action_host = List[Scalar[gpu_dtype]](capacity=ACTION_DIM)
        action_host.append(Scalar[gpu_dtype](action[0]))
        action_host.append(Scalar[gpu_dtype](action[1]))
        action_host.append(Scalar[gpu_dtype](action[2]))
        ctx.enqueue_copy(actions_buf, action_host.unsafe_ptr())
        ctx.synchronize()

        # Step CPU
        var cpu_result = env.step_continuous_vec(cpu_action_list)
        cpu_qpos = env.get_qpos()
        cpu_qvel = env.get_qvel()

        # Step GPU
        HopperGC[DType.float64].step_kernel_gpu[
            BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
        )
        ctx.synchronize()

        # Extract GPU state
        ctx.enqueue_copy(state_host.unsafe_ptr(), states_buf)
        ctx.synchronize()

        extract_gpu_qpos_qvel[STATE_SIZE](state_host, gpu_qpos, gpu_qvel)

        # Compare
        print_state_comparison(cpu_qpos, cpu_qvel, gpu_qpos, gpu_qvel, step + 1)

    # Run many more steps with zero action to see drift
    print("\n" + "=" * 60)
    print("Running 100 steps with zero action to observe drift")
    print("=" * 60)

    var action_host_zero = List[Scalar[gpu_dtype]](capacity=ACTION_DIM)
    action_host_zero.append(Scalar[gpu_dtype](0.0))
    action_host_zero.append(Scalar[gpu_dtype](0.0))
    action_host_zero.append(Scalar[gpu_dtype](0.0))
    ctx.enqueue_copy(actions_buf, action_host_zero.unsafe_ptr())
    ctx.synchronize()

    var cpu_action_zero = List[Scalar[DType.float64]]()
    cpu_action_zero.append(0.0)
    cpu_action_zero.append(0.0)
    cpu_action_zero.append(0.0)

    for step in range(100):
        # Step CPU
        _ = env.step_continuous_vec(cpu_action_zero)

        # Step GPU
        HopperGC[DType.float64].step_kernel_gpu[
            BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
        )
        ctx.synchronize()

        # Print every 20 steps
        if (step + 1) % 20 == 0:
            cpu_qpos = env.get_qpos()
            cpu_qvel = env.get_qvel()

            ctx.enqueue_copy(state_host.unsafe_ptr(), states_buf)
            ctx.synchronize()

            extract_gpu_qpos_qvel[STATE_SIZE](state_host, gpu_qpos, gpu_qvel)

            print(
                "\n--- After", step + 6, "total steps ---"
            )  # +5 from initial + 1 for 1-indexing
            var max_qpos_diff: Float64 = 0.0
            var max_qvel_diff: Float64 = 0.0
            for i in range(6):
                var qpos_diff = math_abs(
                    Float64(cpu_qpos[i]) - Float64(gpu_qpos[i])
                )
                var qvel_diff = math_abs(
                    Float64(cpu_qvel[i]) - Float64(gpu_qvel[i])
                )
                if qpos_diff > max_qpos_diff:
                    max_qpos_diff = qpos_diff
                if qvel_diff > max_qvel_diff:
                    max_qvel_diff = qvel_diff

            print("CPU qpos[1] (z):", cpu_qpos[1], " GPU qpos[1]:", gpu_qpos[1])
            print(
                "CPU qvel[0] (vx):", cpu_qvel[0], " GPU qvel[0]:", gpu_qvel[0]
            )
            print("Max qpos diff:", max_qpos_diff)
            print("Max qvel diff:", max_qvel_diff)

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
    print("\nRemaining differences (expected due to precision):")
    print("1. GPU uses float32, CPU uses float64 (causes ~1e-6 to 1e-8 drift)")
    print("2. Both CPU and GPU now use RESET_NOISE_SCALE=0.005")
    print("3. GPU reset does not run forward kinematics (CPU does)")
