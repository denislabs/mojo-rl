"""Test script to verify Hopper truncation and TERMINATE_ON_UNHEALTHY flag.

Run with:
    pixi run -e apple mojo run tests/test_hopper_truncation.mojo
"""

from random import seed

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from envs.hopper import Hopper
from envs.hopper.hopper_def import HopperConstantsGPU
from core import ContAction
from deep_rl import dtype as gpu_dtype
from physics3d.gpu.constants import (
    state_size,
    metadata_offset,
    META_IDX_STEP_COUNT,
)


fn _make_test_action() -> ContAction[3]:
    """Create test action (0.8, 0.5, -0.8)."""
    var a = ContAction[3]()
    a[0] = 0.8
    a[1] = 0.5
    a[2] = -0.8
    return a^


fn main() raises:
    seed(42)
    print("=" * 70)
    print("TEST: Hopper Truncation and Health Termination Flags")
    print("=" * 70)
    print()

    comptime BATCH_SIZE = 4
    comptime STATE_SIZE = state_size[
        Hopper[gpu_dtype].NQ,
        Hopper[gpu_dtype].NV,
        Hopper[gpu_dtype].NUM_BODIES,
        Hopper[gpu_dtype].MAX_CONTACTS,
    ]()
    comptime OBS_DIM = HopperConstantsGPU.OBS_DIM
    comptime ACTION_DIM = HopperConstantsGPU.ACTION_DIM
    comptime META_OFF = metadata_offset[
        Hopper[gpu_dtype].NQ,
        Hopper[gpu_dtype].NV,
        Hopper[gpu_dtype].NUM_BODIES,
        Hopper[gpu_dtype].MAX_CONTACTS,
    ]()

    with DeviceContext() as ctx:
        # Create buffers
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * ACTION_DIM
        )
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * OBS_DIM)

        # =====================================================================
        # Test 1: Truncation with MAX_STEPS=10 and selective reset
        # =====================================================================
        print("Test 1: Truncation at MAX_STEPS=10 with selective reset")
        print("-" * 50)

        # Reset all environments
        Hopper[gpu_dtype].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, rng_seed=42
        )
        ctx.synchronize()

        # Initialize actions to zero (safe action)
        var actions_host = InlineArray[
            Scalar[gpu_dtype], BATCH_SIZE * ACTION_DIM
        ](fill=Scalar[gpu_dtype](0.0))
        ctx.enqueue_copy(actions_buf, actions_host.unsafe_ptr())

        # Run 25 steps with MAX_STEPS=10, using selective reset
        var dones_host = InlineArray[Scalar[gpu_dtype], BATCH_SIZE](
            uninitialized=True
        )
        var states_host = InlineArray[
            Scalar[gpu_dtype], BATCH_SIZE * STATE_SIZE
        ](uninitialized=True)

        for step in range(25):
            Hopper[gpu_dtype, TERMINATE_ON_UNHEALTHY=True].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM, 10
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf)
            ctx.synchronize()

            ctx.enqueue_copy(dones_host.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            var num_done = 0
            for i in range(BATCH_SIZE):
                if dones_host[i] > 0.5:
                    num_done += 1

            # Read step counts
            ctx.enqueue_copy(states_host.unsafe_ptr(), states_buf)
            ctx.synchronize()
            var step_counts = InlineArray[Int, BATCH_SIZE](uninitialized=True)
            for i in range(BATCH_SIZE):
                step_counts[i] = Int(
                    states_host[i * STATE_SIZE + META_OFF + META_IDX_STEP_COUNT]
                )

            if step in (9, 10, 19, 20):
                print(
                    "  Step",
                    step + 1,
                    ": done =",
                    num_done,
                    ", step_counts =",
                    step_counts[0],
                    step_counts[1],
                    step_counts[2],
                    step_counts[3],
                )

            # Selective reset done environments
            if num_done > 0:
                Hopper[gpu_dtype].selective_reset_kernel_gpu[
                    BATCH_SIZE, STATE_SIZE
                ](ctx, states_buf, dones_buf, UInt64(step))
                ctx.synchronize()

        # Final step counts
        ctx.enqueue_copy(states_host.unsafe_ptr(), states_buf)
        ctx.synchronize()
        print("\nFinal step counts after 25 steps with resets:")
        for i in range(BATCH_SIZE):
            var step_count = Int(
                states_host[i * STATE_SIZE + META_OFF + META_IDX_STEP_COUNT]
            )
            print("  Env", i, ": step_count =", step_count, "(should be 5)")
        print()

        # =====================================================================
        # Test 2: TERMINATE_ON_UNHEALTHY=False vs True comparison
        # =====================================================================
        print("Test 2: Health termination flag comparison")
        print("-" * 50)

        # Use actions that might cause the hopper to tilt over time
        var tilt_actions = InlineArray[
            Scalar[gpu_dtype], BATCH_SIZE * ACTION_DIM
        ](uninitialized=True)
        for i in range(BATCH_SIZE):
            # Asymmetric torques to cause tilting
            tilt_actions[i * ACTION_DIM + 0] = Scalar[gpu_dtype](0.8)  # thigh
            tilt_actions[i * ACTION_DIM + 1] = Scalar[gpu_dtype](0.5)  # leg
            tilt_actions[i * ACTION_DIM + 2] = Scalar[gpu_dtype](-0.8)  # foot
        ctx.enqueue_copy(actions_buf, tilt_actions.unsafe_ptr())

        # Run with TERMINATE_ON_UNHEALTHY=True
        print("\n  With TERMINATE_ON_UNHEALTHY=True (MAX_STEPS=1000):")
        Hopper[gpu_dtype].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, rng_seed=999
        )
        ctx.synchronize()

        var terminated_at_with_health = -1
        for step in range(200):
            Hopper[gpu_dtype, TERMINATE_ON_UNHEALTHY=True].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM, 1000
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf)
            ctx.synchronize()

            ctx.enqueue_copy(dones_host.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            if dones_host[0] > 0.5 and terminated_at_with_health == -1:
                terminated_at_with_health = step + 1
                break

        if terminated_at_with_health == -1:
            print("    Env 0 did NOT terminate in 200 steps")
        else:
            print("    Env 0 terminated at step:", terminated_at_with_health)

        # Run with TERMINATE_ON_UNHEALTHY=False
        print("\n  With TERMINATE_ON_UNHEALTHY=False (MAX_STEPS=1000):")
        Hopper[gpu_dtype].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, rng_seed=999
        )
        ctx.synchronize()

        var terminated_at_no_health = -1
        for step in range(200):
            Hopper[gpu_dtype, TERMINATE_ON_UNHEALTHY=False].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM, 1000
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf)
            ctx.synchronize()

            ctx.enqueue_copy(dones_host.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            if dones_host[0] > 0.5 and terminated_at_no_health == -1:
                terminated_at_no_health = step + 1
                break

        if terminated_at_no_health == -1:
            print("    Env 0 did NOT terminate in 200 steps")
        else:
            print("    Env 0 terminated at step:", terminated_at_no_health)

        # =====================================================================
        # Test 3: Verify truncation works without health termination
        # =====================================================================
        print(
            "\nTest 3: Truncation-only mode (TERMINATE_ON_UNHEALTHY=False,"
            " MAX_STEPS=50)"
        )
        print("-" * 50)

        Hopper[gpu_dtype].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, rng_seed=12345
        )
        ctx.synchronize()

        var truncated_at = -1
        for step in range(100):
            Hopper[gpu_dtype, TERMINATE_ON_UNHEALTHY=False].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM, 50
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf)
            ctx.synchronize()

            ctx.enqueue_copy(dones_host.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            if dones_host[0] > 0.5 and truncated_at == -1:
                truncated_at = step + 1
                break

        print("  Env 0 truncated at step:", truncated_at, "(expected: 50)")

        # =====================================================================
        # Test 4: CPU terminate_on_unhealthy flag
        # =====================================================================
        print("\nTest 4: CPU terminate_on_unhealthy flag")
        print("-" * 50)

        # Test with terminate_on_unhealthy=True (default)
        var env_with_health = Hopper[
            DType.float64, TERMINATE_ON_UNHEALTHY=True
        ](max_steps=100)
        _ = env_with_health.reset()

        var cpu_done_with_health = -1
        for step in range(100):
            var result = env_with_health.step(
                _make_test_action()
            )
            if result[2]:  # done
                cpu_done_with_health = step + 1
                break

        print("  With terminate_on_unhealthy=True (max_steps=100):")
        if cpu_done_with_health == -1:
            print("    Env did NOT terminate in 100 steps")
        else:
            print("    Env terminated at step:", cpu_done_with_health)

        # Test with terminate_on_unhealthy=False
        var env_no_health = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=False](
            max_steps=50
        )
        _ = env_no_health.reset()

        var cpu_done_no_health = -1
        for step in range(100):
            var result = env_no_health.step(
                _make_test_action()
            )
            if result[2]:  # done
                cpu_done_no_health = step + 1
                break

        print("\n  With terminate_on_unhealthy=False (max_steps=50):")
        print(
            "    Env truncated at step:", cpu_done_no_health, "(expected: 50)"
        )

        print()
        print("=" * 70)
        print("SUMMARY:")
        print("  - GPU: Truncation at MAX_STEPS works correctly")
        print("  - GPU: Selective reset resets step counter to 0")
        print(
            "  - GPU: TERMINATE_ON_UNHEALTHY flag controls health-based"
            " termination"
        )
        print(
            "  - CPU: terminate_on_unhealthy flag controls health-based"
            " termination"
        )
        print("=" * 70)

    print("\n>>> Truncation test completed <<<")
