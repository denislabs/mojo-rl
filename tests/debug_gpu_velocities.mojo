"""Debug GPU physics to check for velocity explosion."""

from random import seed, random_float64
from gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor, Layout

from envs.half_cheetah import HalfCheetahPlanarV2, HCConstants
from physics2d.constants import dtype, BODY_STATE_SIZE, IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY, IDX_OMEGA


fn main() raises:
    seed(42)
    print("=" * 70)
    print("GPU Velocity Diagnostic - PPO Configuration")
    print("=" * 70)
    print()
    print("Settings:")
    print("  MAX_TORQUE:", HCConstants.MAX_TORQUE)
    print("  DT:", HCConstants.DT)
    print("  FRAME_SKIP:", HCConstants.FRAME_SKIP)
    print("  VELOCITY_ITERATIONS:", HCConstants.VELOCITY_ITERATIONS)
    print("  TERMINATE_WHEN_UNHEALTHY:", HCConstants.TERMINATE_WHEN_UNHEALTHY)
    print()

    # Test with PPO batch size (512 environments)
    print("=== Testing with BATCH=512 (PPO config) ===")
    with DeviceContext() as ctx:
        comptime BATCH = 512
        comptime STATE_SIZE = HCConstants.STATE_SIZE_VAL
        comptime OBS_DIM = HCConstants.OBS_DIM_VAL
        comptime ACTION_DIM = 6

        var states_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
        var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH)

        # Reset all environments
        HalfCheetahPlanarV2[dtype].reset_kernel_gpu[BATCH, STATE_SIZE](
            ctx, states_buf, rng_seed=42
        )
        ctx.synchronize()

        # Host buffers
        var host_rewards = List[Scalar[dtype]](capacity=BATCH)
        for _ in range(BATCH):
            host_rewards.append(Scalar[dtype](0))
        var host_actions = List[Scalar[dtype]](capacity=BATCH * ACTION_DIM)
        for _ in range(BATCH * ACTION_DIM):
            host_actions.append(Scalar[dtype](0))

        # Run 256 steps (one rollout) and track rewards
        var total_rewards = List[Float64](capacity=BATCH)
        for _ in range(BATCH):
            total_rewards.append(0.0)

        print("Running 256 steps (one rollout) across 512 environments...")
        for step in range(256):
            # Random actions for all environments
            for i in range(BATCH * ACTION_DIM):
                host_actions[i] = Scalar[dtype](random_float64(-1.0, 1.0))
            actions_buf.enqueue_copy_from(host_actions.unsafe_ptr())

            # Step all environments
            HalfCheetahPlanarV2[dtype].step_kernel_gpu[
                BATCH, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf, rng_seed=UInt64(step))
            ctx.synchronize()

            # Read rewards
            rewards_buf.enqueue_copy_to(host_rewards.unsafe_ptr())
            ctx.synchronize()

            # Accumulate
            for i in range(BATCH):
                total_rewards[i] += Float64(host_rewards[i])

        # Compute statistics
        var sum_rewards: Float64 = 0.0
        var min_reward: Float64 = total_rewards[0]
        var max_reward: Float64 = total_rewards[0]
        for i in range(BATCH):
            sum_rewards += total_rewards[i]
            if total_rewards[i] < min_reward:
                min_reward = total_rewards[i]
            if total_rewards[i] > max_reward:
                max_reward = total_rewards[i]

        print()
        print("Results after 256 steps per environment:")
        print("  Average reward:", sum_rewards / Float64(BATCH))
        print("  Min reward:", min_reward)
        print("  Max reward:", max_reward)
        print()

        if min_reward < -1000:
            print("WARNING: Rewards are too negative - physics may be broken!")
        else:
            print("OK: Rewards look reasonable")

    # Also test single environment
    print()
    print("=== Testing with BATCH=1 for comparison ===")
    with DeviceContext() as ctx:
        comptime BATCH = 1
        comptime STATE_SIZE = HCConstants.STATE_SIZE_VAL
        comptime OBS_DIM = HCConstants.OBS_DIM_VAL
        comptime ACTION_DIM = 6

        var states_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
        var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH)

        # Reset
        HalfCheetahPlanarV2[dtype].reset_kernel_gpu[BATCH, STATE_SIZE](
            ctx, states_buf, rng_seed=42
        )
        ctx.synchronize()

        # Host buffers
        var host_state = List[Scalar[dtype]](capacity=STATE_SIZE)
        for _ in range(STATE_SIZE):
            host_state.append(Scalar[dtype](0))
        var host_actions = List[Scalar[dtype]](capacity=ACTION_DIM)
        for _ in range(ACTION_DIM):
            host_actions.append(Scalar[dtype](0))
        var host_rewards = List[Scalar[dtype]](capacity=1)
        host_rewards.append(Scalar[dtype](0))
        var host_dones = List[Scalar[dtype]](capacity=1)
        host_dones.append(Scalar[dtype](0))

        print("Running 100 steps with random actions...")
        print("-" * 70)
        print("Step | Reward | TorsoX | TorsoY | TorsoVX | TorsoVY | TorsoOmega")
        print("-" * 70)

        var total_reward: Float64 = 0.0
        var max_vx: Float64 = 0.0
        var max_vy: Float64 = 0.0
        var max_omega: Float64 = 0.0

        for step in range(100):
            # Random actions
            for i in range(ACTION_DIM):
                host_actions[i] = Scalar[dtype](random_float64(-1.0, 1.0))
            actions_buf.enqueue_copy_from(host_actions.unsafe_ptr())

            # Step
            HalfCheetahPlanarV2[dtype].step_kernel_gpu[
                BATCH, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf, rng_seed=UInt64(step))
            ctx.synchronize()

            # Read results
            rewards_buf.enqueue_copy_to(host_rewards.unsafe_ptr())
            dones_buf.enqueue_copy_to(host_dones.unsafe_ptr())
            states_buf.enqueue_copy_to(host_state.unsafe_ptr())
            ctx.synchronize()

            var reward = Float64(host_rewards[0])
            var done = Float64(host_dones[0])
            total_reward += reward

            var torso_off = HCConstants.BODIES_OFFSET
            var x = Float64(host_state[torso_off + IDX_X])
            var y = Float64(host_state[torso_off + IDX_Y])
            var vx = Float64(host_state[torso_off + IDX_VX])
            var vy = Float64(host_state[torso_off + IDX_VY])
            var omega = Float64(host_state[torso_off + IDX_OMEGA])

            # Track max velocities
            if abs(vx) > max_vx:
                max_vx = abs(vx)
            if abs(vy) > max_vy:
                max_vy = abs(vy)
            if abs(omega) > max_omega:
                max_omega = abs(omega)

            # Print every 10 steps
            if step % 10 == 0 or done > 0.5:
                print(step + 1, "|", reward, "|", x, "|", y, "|", vx, "|", vy, "|", omega)

            if done > 0.5:
                print(">>> TERMINATED <<<")
                break

        print("-" * 70)
        print()
        print("Summary:")
        print("  Total reward:", total_reward)
        print("  Max |VX|:", max_vx)
        print("  Max |VY|:", max_vy)
        print("  Max |Omega|:", max_omega)
        print()

        if max_vx > 10.0 or max_vy > 10.0 or max_omega > 50.0:
            print("WARNING: Velocities seem high - physics may be unstable!")
        else:
            print("OK: Velocities are reasonable")

        if abs(total_reward) > 100:
            print("WARNING: Total reward seems too large!")
        else:
            print("OK: Rewards are reasonable")

    print()
    print("=" * 70)
