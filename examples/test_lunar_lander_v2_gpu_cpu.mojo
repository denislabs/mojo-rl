"""Test LunarLanderV2 environment using CPU kernels.

This tests the GPU environment implementation using the CPU fallback kernels
to verify the physics and state layout are correct.

Usage:
    pixi run mojo run examples/test_lunar_lander_v2_gpu_cpu.mojo
"""

from time import perf_counter_ns
from random import seed

from layout import Layout, LayoutTensor

from envs.lunar_lander import LunarLanderV2
from physics2d import dtype


comptime BATCH_SIZE: Int = 8
comptime STATE_SIZE: Int = LunarLanderV2.STATE_SIZE
comptime OBS_DIM: Int = LunarLanderV2.OBS_DIM
comptime NUM_ACTIONS: Int = LunarLanderV2.NUM_ACTIONS


fn main() raises:
    seed(42)

    print("=" * 70)
    print("LunarLanderV2 CPU Test")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Batch size:", BATCH_SIZE)
    print("  State size:", STATE_SIZE)
    print("  Observation dim:", OBS_DIM)
    print("  Number of actions:", NUM_ACTIONS)
    print()

    # Allocate CPU buffers
    var states_data = List[Scalar[dtype]](capacity=BATCH_SIZE * STATE_SIZE)
    for _ in range(BATCH_SIZE * STATE_SIZE):
        states_data.append(Scalar[dtype](0))

    var actions_data = List[Scalar[dtype]](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        actions_data.append(Scalar[dtype](0))

    var rewards_data = List[Scalar[dtype]](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        rewards_data.append(Scalar[dtype](0))

    var dones_data = List[Scalar[dtype]](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        dones_data.append(Scalar[dtype](0))

    # Create LayoutTensors
    var states = LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
    ](states_data.unsafe_ptr())

    var actions = LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ](actions_data.unsafe_ptr())

    var rewards = LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ](rewards_data.unsafe_ptr())

    var dones = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin](
        dones_data.unsafe_ptr()
    )

    # Reset all environments
    print("1. Resetting all environments...")
    LunarLanderV2.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

    # Print initial states
    print()
    print("Initial observations for first 3 environments:")
    for env in range(3):
        print("  Env", env, ":", end="")
        for i in range(OBS_DIM):
            var obs_val = Float64(rebind[Scalar[dtype]](states[env, i]))
            print(" ", obs_val, end="")
        print()

    # Run some steps with random actions
    print()
    print("2. Running 100 steps with random actions...")

    var total_reward: Float64 = 0.0
    var episode_count: Int = 0
    var step_count: Int = 0
    var rng = UInt32(42)

    for step in range(100):
        # Generate random actions
        for i in range(BATCH_SIZE):
            rng = rng ^ (rng << 13)
            rng = rng ^ (rng >> 17)
            rng = rng ^ (rng << 5)
            var action = Int(rng % NUM_ACTIONS)
            actions_data[i] = Scalar[dtype](action)

        # Step all environments
        LunarLanderV2.step_kernel[BATCH_SIZE, STATE_SIZE](
            states,
            LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin](
                actions_data.unsafe_ptr()
            ),
            rewards,
            dones,
            Scalar[DType.uint64](rng),
        )

        # Accumulate rewards and count episodes
        for i in range(BATCH_SIZE):
            total_reward += Float64(rewards_data[i])
            if Float64(dones_data[i]) > 0.5:
                episode_count += 1
                # Reset done environments manually by calling reset on specific env
                # In practice, we'd use selective_reset_kernel

        step_count += 1

    print("  Total steps:", step_count * BATCH_SIZE)
    print("  Total reward:", total_reward)
    print("  Episodes completed:", episode_count)
    print(
        "  Average reward per step:",
        total_reward / Float64(step_count * BATCH_SIZE),
    )

    # Print final states
    print()
    print("Final observations for first 3 environments:")
    for env in range(3):
        print("  Env", env, ":", end="")
        for i in range(OBS_DIM):
            var obs_val = Float64(rebind[Scalar[dtype]](states[env, i]))
            print(" ", obs_val, end="")
        print()

    print()
    print("Test completed successfully!")
    print()
    print("Note: This tests the CPU implementation. The GPU version uses")
    print("the same logic but with GPU-optimized physics methods.")
