"""Test native Pong environment — GPU path."""

from envs.arcade_games.pong import PongEnv
from std.gpu.host import DeviceContext

comptime dtype = DType.float32
comptime BATCH_SIZE = 64
comptime STATE_SIZE = PongEnv[DType.float64].STATE_SIZE
comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM
comptime NUM_ACTIONS = PongEnv[DType.float64].NUM_ACTIONS


fn main() raises:
    print("=== Testing PongEnv (GPU) ===")
    print("BATCH_SIZE:", BATCH_SIZE)
    print("STATE_SIZE:", STATE_SIZE)
    print("OBS_DIM:", OBS_DIM)
    print("NUM_ACTIONS:", NUM_ACTIONS)

    var ctx = DeviceContext()
    print("GPU device ready")

    # Allocate buffers
    var states = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * STATE_SIZE)
    var actions = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var rewards = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var dones = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var terminated = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var obs = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OBS_DIM)

    # Reset all environments
    print("\n--- Reset ---")
    PongEnv[DType.float64].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states
    )
    ctx.synchronize()
    print("Reset complete.")

    # Read back states to verify
    var host_states = ctx.enqueue_create_host_buffer[dtype](
        BATCH_SIZE * STATE_SIZE
    )
    ctx.enqueue_copy(host_states, states)
    ctx.synchronize()

    # Print first env state
    print("\nEnv 0 state after reset:")
    for j in range(STATE_SIZE):
        print("  [", j, "]", host_states[j])

    # Set actions: all NOOP for first test
    var host_actions = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE)
    for j in range(BATCH_SIZE):
        host_actions[j] = 0.0  # NOOP
    ctx.enqueue_copy(actions, host_actions)

    # Run 100 steps
    print("\n--- Step 100 times (NOOP) ---")
    for step in range(100):
        PongEnv[DType.float64].step_kernel_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM](
            ctx, states, actions, rewards, dones, terminated, obs, rng_seed=UInt64(step)
        )
    ctx.synchronize()

    # Read back results
    var host_rewards = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE)
    var host_dones = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE)
    var host_obs = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE * OBS_DIM)
    ctx.enqueue_copy(host_rewards, rewards)
    ctx.enqueue_copy(host_dones, dones)
    ctx.enqueue_copy(host_obs, obs)
    ctx.enqueue_copy(host_states, states)
    ctx.synchronize()

    print("Env 0 state after 100 NOOP steps:")
    for j in range(STATE_SIZE):
        print("  [", j, "]", host_states[j])

    print("\nEnv 0 obs:")
    for j in range(OBS_DIM):
        print("  [", j, "]", host_obs[j])

    print("\nRewards/dones (first 8 envs):")
    for j in range(8):
        print(
            "  env",
            j,
            ": reward=",
            host_rewards[j],
            ", done=",
            host_dones[j],
        )

    # Run with UP action until some episodes end
    print("\n--- Running with UP action until episodes end ---")
    for j in range(BATCH_SIZE):
        host_actions[j] = 1.0  # UP
    ctx.enqueue_copy(actions, host_actions)

    var total_dones = 0
    for step in range(5000):
        PongEnv[DType.float64].step_kernel_gpu[BATCH_SIZE, STATE_SIZE, OBS_DIM](
            ctx, states, actions, rewards, dones, terminated, obs,
            rng_seed=UInt64(step + 100),
        )

        if step % 1000 == 999:
            ctx.synchronize()
            ctx.enqueue_copy(host_dones, dones)
            ctx.synchronize()
            var done_count = 0
            for j in range(BATCH_SIZE):
                if host_dones[j] > 0.5:
                    done_count += 1
            total_dones += done_count
            if done_count > 0:
                print("Step", step + 1, ": ", done_count, "envs done")

            # Selective reset
            PongEnv[DType.float64].selective_reset_kernel_gpu[
                BATCH_SIZE, STATE_SIZE
            ](ctx, states, dones, rng_seed=UInt64(step + 200))

    ctx.synchronize()
    print("Total episodes completed:", total_dones)

    print("\n=== GPU Test DONE ===")
