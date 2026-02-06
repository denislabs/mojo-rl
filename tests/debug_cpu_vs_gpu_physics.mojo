"""Diagnostic: Compare CPU vs GPU HalfCheetahGC physics step-by-step.

This script runs both CPU and GPU envs from identical initial states
with identical actions, printing observations and rewards to pinpoint
where divergence occurs.

Run with:
    pixi run -e apple mojo run tests/debug_cpu_vs_gpu_physics.mojo
"""

from random import seed
from collections import InlineArray

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from envs.half_cheetah_gc import HalfCheetahGC, HalfCheetahGCConstants
from deep_rl import dtype as gpu_dtype
from deep_rl.constants import TPB
from physics3d.gpu.constants import (
    gc_qpos_offset,
    gc_qvel_offset,
    gc_metadata_offset,
    gc_state_size,
    GC_META_IDX_STEP_COUNT,
    GC_META_IDX_PREV_X,
)

comptime C = HalfCheetahGCConstants[DType.float32]
comptime OBS_DIM = C.OBS_DIM  # 17
comptime ACTION_DIM = C.ACTION_DIM  # 6
comptime dtype = DType.float32


fn print_obs(name: String, obs: List[Scalar[dtype]]):
    print(name + ":")
    var labels = List[String]()
    labels.append("  z_pos   ")
    labels.append("  y_angle ")
    labels.append("  bthigh  ")
    labels.append("  bshin   ")
    labels.append("  bfoot   ")
    labels.append("  fthigh  ")
    labels.append("  fshin   ")
    labels.append("  ffoot   ")
    labels.append("  x_vel   ")
    labels.append("  z_vel   ")
    labels.append("  y_angvel")
    labels.append("  bthigh_v")
    labels.append("  bshin_v ")
    labels.append("  bfoot_v ")
    labels.append("  fthigh_v")
    labels.append("  fshin_v ")
    labels.append("  ffoot_v ")
    for i in range(len(obs)):
        print(labels[i] + " = " + String(Float64(obs[i])))


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DIAGNOSTIC: CPU vs GPU HalfCheetahGC Physics Comparison")
    print("=" * 70)

    # Fixed actions for testing
    var test_actions = List[List[Scalar[dtype]]]()

    # Action set 0: all zeros
    var a0 = List[Scalar[dtype]]()
    for _ in range(ACTION_DIM):
        a0.append(Scalar[dtype](0.0))
    test_actions.append(a0^)

    # Action set 1: small positive
    var a1 = List[Scalar[dtype]]()
    for _ in range(ACTION_DIM):
        a1.append(Scalar[dtype](0.3))
    test_actions.append(a1^)

    # Action set 2: mixed
    var a2 = List[Scalar[dtype]]()
    a2.append(Scalar[dtype](0.5))
    a2.append(Scalar[dtype](-0.3))
    a2.append(Scalar[dtype](0.1))
    a2.append(Scalar[dtype](-0.5))
    a2.append(Scalar[dtype](0.3))
    a2.append(Scalar[dtype](-0.1))
    test_actions.append(a2^)

    # =====================================================
    # Part 1: CPU environment
    # =====================================================
    print()
    print("=" * 70)
    print("PART 1: CPU ENVIRONMENT (HalfCheetahGC[float32])")
    print("=" * 70)

    var cpu_env = HalfCheetahGC[dtype]()
    var cpu_obs = cpu_env.reset_obs_list()

    print()
    print_obs("Initial CPU obs (after reset)", cpu_obs)

    # Print raw qpos/qvel
    print()
    print("Raw CPU qpos after reset:")
    for i in range(10):
        print(
            "  qpos["
            + String(i)
            + "] = "
            + String(Float64(cpu_env.data.qpos[i]))
        )
    print("Raw CPU qvel after reset:")
    for i in range(10):
        print(
            "  qvel["
            + String(i)
            + "] = "
            + String(Float64(cpu_env.data.qvel[i]))
        )

    # Step with each action set
    for step in range(len(test_actions)):
        print()
        print("-" * 40)
        print("CPU Step", step, "with actions:", end="")
        for j in range(ACTION_DIM):
            print(" " + String(Float64(test_actions[step][j]))[:6], end="")
        print()

        # Reset for each action set test
        cpu_obs = cpu_env.reset_obs_list()
        var result = cpu_env.step_continuous_vec(test_actions[step])
        var obs_after = result[0].copy()
        var reward = result[1]
        var done = result[2]

        print("  Reward:", Float64(reward))
        print("  Done:", done)
        print_obs("  Obs after step", obs_after)

        # Print raw qpos/qvel after step
        print("  Raw qpos after step:")
        for i in range(10):
            print(
                "    qpos["
                + String(i)
                + "] = "
                + String(Float64(cpu_env.data.qpos[i]))
            )
        print("  Raw qvel after step:")
        for i in range(10):
            print(
                "    qvel["
                + String(i)
                + "] = "
                + String(Float64(cpu_env.data.qvel[i]))
            )

    # =====================================================
    # Part 2: GPU environment (single env, no noise)
    # =====================================================
    print()
    print("=" * 70)
    print("PART 2: GPU ENVIRONMENT (single env, no reset noise)")
    print("=" * 70)

    with DeviceContext() as ctx:
        comptime STATE_SIZE = HalfCheetahGC[dtype].STATE_SIZE
        comptime MODEL_SIZE = HalfCheetahGC[dtype].MODEL_SIZE
        comptime N_ENVS = 1

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](OBS_DIM)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](1)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](1)
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](ACTION_DIM)

        # Initialize state buffer to zero
        ctx.enqueue_memset(states_buf, 0)
        ctx.synchronize()

        # Manually set initial state (matching CPU reset - NO noise)
        comptime QPOS_OFF = gc_qpos_offset[
            HalfCheetahGC[dtype].NQ, HalfCheetahGC[dtype].NV
        ]()
        comptime QVEL_OFF = gc_qvel_offset[
            HalfCheetahGC[dtype].NQ, HalfCheetahGC[dtype].NV
        ]()
        comptime META_OFF = gc_metadata_offset[
            HalfCheetahGC[dtype].NQ,
            HalfCheetahGC[dtype].NV,
            HalfCheetahGC[dtype].NUM_BODIES,
            HalfCheetahGC[dtype].MAX_CONTACTS,
        ]()

        # Set initial qpos on host
        var state_host = List[Scalar[gpu_dtype]](capacity=STATE_SIZE)
        for _ in range(STATE_SIZE):
            state_host.append(Scalar[gpu_dtype](0.0))

        # rootx=0, rootz=0.7, rest=0 (matching CPU reset)
        state_host[QPOS_OFF + 0] = Scalar[gpu_dtype](0.0)  # rootx
        state_host[QPOS_OFF + 1] = C.INITIAL_Z  # rootz = 0.7
        for i in range(2, 10):
            state_host[QPOS_OFF + i] = Scalar[gpu_dtype](0.0)

        # qvel all zero
        for i in range(10):
            state_host[QVEL_OFF + i] = Scalar[gpu_dtype](0.0)

        # step_count = 0, prev_x = 0
        state_host[META_OFF + GC_META_IDX_STEP_COUNT] = Scalar[gpu_dtype](0.0)
        state_host[META_OFF + GC_META_IDX_PREV_X] = Scalar[gpu_dtype](0.0)

        # Copy to GPU
        ctx.enqueue_copy(states_buf, state_host.unsafe_ptr())
        ctx.synchronize()

        # Print initial state from GPU
        print()
        print("Initial GPU state (from buffer):")
        var read_state = List[Scalar[gpu_dtype]](capacity=STATE_SIZE)
        for _ in range(STATE_SIZE):
            read_state.append(Scalar[gpu_dtype](0.0))
        ctx.enqueue_copy(read_state.unsafe_ptr(), states_buf)
        ctx.synchronize()

        print("  GPU qpos:")
        for i in range(10):
            print(
                "    qpos["
                + String(i)
                + "] = "
                + String(Float64(read_state[QPOS_OFF + i]))
            )
        print("  GPU qvel:")
        for i in range(10):
            print(
                "    qvel["
                + String(i)
                + "] = "
                + String(Float64(read_state[QVEL_OFF + i]))
            )

        # Step with each action set
        for step in range(len(test_actions)):
            print()
            print("-" * 40)
            print("GPU Step", step, "with actions:", end="")
            for j in range(ACTION_DIM):
                print(" " + String(Float64(test_actions[step][j]))[:6], end="")
            print()

            # Reset state (matching CPU)
            for i in range(STATE_SIZE):
                state_host[i] = Scalar[gpu_dtype](0.0)
            state_host[QPOS_OFF + 0] = Scalar[gpu_dtype](0.0)
            state_host[QPOS_OFF + 1] = C.INITIAL_Z
            state_host[META_OFF + GC_META_IDX_STEP_COUNT] = Scalar[gpu_dtype](
                0.0
            )
            state_host[META_OFF + GC_META_IDX_PREV_X] = Scalar[gpu_dtype](0.0)
            ctx.enqueue_copy(states_buf, state_host.unsafe_ptr())
            ctx.synchronize()

            # Copy actions to GPU
            var actions_host = List[Scalar[gpu_dtype]](capacity=ACTION_DIM)
            for j in range(ACTION_DIM):
                actions_host.append(test_actions[step][j])
            ctx.enqueue_copy(actions_buf, actions_host.unsafe_ptr())
            ctx.synchronize()

            # Step the GPU env
            HalfCheetahGC[dtype].step_kernel_gpu[
                N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                obs_buf,
                UInt64(0),  # rng_seed
            )
            ctx.synchronize()

            # Read back results
            var gpu_obs_host = List[Scalar[gpu_dtype]](capacity=OBS_DIM)
            for _ in range(OBS_DIM):
                gpu_obs_host.append(Scalar[gpu_dtype](0.0))
            ctx.enqueue_copy(gpu_obs_host.unsafe_ptr(), obs_buf)

            var gpu_reward_host = List[Scalar[gpu_dtype]](capacity=1)
            gpu_reward_host.append(Scalar[gpu_dtype](0.0))
            ctx.enqueue_copy(gpu_reward_host.unsafe_ptr(), rewards_buf)

            var gpu_done_host = List[Scalar[gpu_dtype]](capacity=1)
            gpu_done_host.append(Scalar[gpu_dtype](0.0))
            ctx.enqueue_copy(gpu_done_host.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            print("  Reward:", Float64(gpu_reward_host[0]))
            print("  Done:", Float64(gpu_done_host[0]) > 0.5)
            print_obs("  Obs after step", gpu_obs_host)

            # Read raw qpos/qvel from state buffer
            ctx.enqueue_copy(read_state.unsafe_ptr(), states_buf)
            ctx.synchronize()
            print("  Raw qpos after step:")
            for i in range(10):
                print(
                    "    qpos["
                    + String(i)
                    + "] = "
                    + String(Float64(read_state[QPOS_OFF + i]))
                )
            print("  Raw qvel after step:")
            for i in range(10):
                print(
                    "    qvel["
                    + String(i)
                    + "] = "
                    + String(Float64(read_state[QVEL_OFF + i]))
                )

    # =====================================================
    # Part 3: Multi-step comparison (10 steps with constant action)
    # =====================================================
    print()
    print("=" * 70)
    print(
        "PART 3: Multi-step trajectory (10 steps, action=[0.5, -0.3, 0.1, -0.5,"
        " 0.3, -0.1])"
    )
    print("=" * 70)

    var multi_action = List[Scalar[dtype]]()
    multi_action.append(Scalar[dtype](0.5))
    multi_action.append(Scalar[dtype](-0.3))
    multi_action.append(Scalar[dtype](0.1))
    multi_action.append(Scalar[dtype](-0.5))
    multi_action.append(Scalar[dtype](0.3))
    multi_action.append(Scalar[dtype](-0.1))

    # CPU multi-step
    print()
    print("CPU trajectory:")
    var cpu_env2 = HalfCheetahGC[dtype]()
    _ = cpu_env2.reset_obs_list()
    var cpu_total_reward: Float64 = 0.0

    for step in range(10):
        var result = cpu_env2.step_continuous_vec(multi_action)
        var reward = Float64(result[1])
        cpu_total_reward += reward
        var rootx = Float64(cpu_env2.data.qpos[0])
        var rootz = Float64(cpu_env2.data.qpos[1])
        var x_vel = Float64(cpu_env2.data.qvel[0])
        print(
            "  Step",
            String(step).rjust(2),
            "| reward:",
            String(reward)[:10].ljust(10),
            "| rootx:",
            String(rootx)[:10].ljust(10),
            "| rootz:",
            String(rootz)[:10].ljust(10),
            "| x_vel:",
            String(x_vel)[:10].ljust(10),
        )
    print("  CPU total reward (10 steps):", cpu_total_reward)

    # GPU multi-step
    print()
    print("GPU trajectory:")
    with DeviceContext() as ctx:
        comptime STATE_SIZE = HalfCheetahGC[dtype].STATE_SIZE
        comptime N_ENVS = 1

        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](OBS_DIM)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](1)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](1)
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](ACTION_DIM)

        # Initialize state (matching CPU reset - NO noise)
        comptime QPOS_OFF = gc_qpos_offset[
            HalfCheetahGC[dtype].NQ, HalfCheetahGC[dtype].NV
        ]()
        comptime QVEL_OFF = gc_qvel_offset[
            HalfCheetahGC[dtype].NQ, HalfCheetahGC[dtype].NV
        ]()
        comptime META_OFF = gc_metadata_offset[
            HalfCheetahGC[dtype].NQ,
            HalfCheetahGC[dtype].NV,
            HalfCheetahGC[dtype].NUM_BODIES,
            HalfCheetahGC[dtype].MAX_CONTACTS,
        ]()

        var state_host = List[Scalar[gpu_dtype]](capacity=STATE_SIZE)
        for _ in range(STATE_SIZE):
            state_host.append(Scalar[gpu_dtype](0.0))
        state_host[QPOS_OFF + 1] = C.INITIAL_Z  # rootz = 0.7
        ctx.enqueue_copy(states_buf, state_host.unsafe_ptr())
        ctx.synchronize()

        # Copy actions
        var actions_host = List[Scalar[gpu_dtype]](capacity=ACTION_DIM)
        for j in range(ACTION_DIM):
            actions_host.append(multi_action[j])
        ctx.enqueue_copy(actions_buf, actions_host.unsafe_ptr())
        ctx.synchronize()

        var gpu_total_reward: Float64 = 0.0

        for step in range(10):
            HalfCheetahGC[dtype].step_kernel_gpu[
                N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                obs_buf,
                UInt64(step),
            )
            ctx.synchronize()

            # Read results
            var gpu_reward_host = List[Scalar[gpu_dtype]](capacity=1)
            gpu_reward_host.append(Scalar[gpu_dtype](0.0))
            ctx.enqueue_copy(gpu_reward_host.unsafe_ptr(), rewards_buf)
            ctx.synchronize()
            var reward = Float64(gpu_reward_host[0])
            gpu_total_reward += reward

            # Read state
            ctx.enqueue_copy(state_host.unsafe_ptr(), states_buf)
            ctx.synchronize()
            var rootx = Float64(state_host[QPOS_OFF + 0])
            var rootz = Float64(state_host[QPOS_OFF + 1])
            var x_vel = Float64(state_host[QVEL_OFF + 0])
            print(
                "  Step",
                String(step).rjust(2),
                "| reward:",
                String(reward)[:10].ljust(10),
                "| rootx:",
                String(rootx)[:10].ljust(10),
                "| rootz:",
                String(rootz)[:10].ljust(10),
                "| x_vel:",
                String(x_vel)[:10].ljust(10),
            )
        print("  GPU total reward (10 steps):", gpu_total_reward)

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(
        "If CPU and GPU produce different trajectories from identical"
        " initial state"
    )
    print("and actions, the physics engines diverge.")
    print("If they match, the issue is in how the evaluation pipeline feeds")
    print("observations to the network or applies actions.")
