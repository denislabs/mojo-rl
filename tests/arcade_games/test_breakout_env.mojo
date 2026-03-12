"""Test native Breakout environment — CPU + GPU."""

from mojo_rl.envs.arcade_games.breakout import BreakoutEnv
from std.gpu.host import DeviceContext

comptime dtype = DType.float32
comptime BATCH_SIZE = 64
comptime STATE_SIZE = BreakoutEnv[DType.float64].STATE_SIZE
comptime OBS_DIM = BreakoutEnv[DType.float64].OBS_DIM


fn main() raises:
    print("=== Testing BreakoutEnv ===")
    print("STATE_SIZE:", STATE_SIZE, ", OBS_DIM:", OBS_DIM)

    # --- CPU Test ---
    print("\n--- CPU Test ---")
    var env = BreakoutEnv[DType.float64]()
    var obs = env.reset_obs_list()
    print("Reset obs (", len(obs), "dims):")
    for i in range(len(obs)):
        print("  [", i, "]", obs[i])

    # Run episodes: alternate FIRE and random LEFT/RIGHT
    var total_reward: Float64 = 0.0
    for episode in range(2):
        _ = env.reset_obs_list()
        var ep_reward: Float64 = 0.0
        var ep_steps = 0
        while True:
            # Alternate: fire, then move randomly
            var action = 1 if ep_steps < 5 else (2 + ep_steps % 2)
            var result = env.step_obs(action)
            ep_reward += Float64(result[1])
            ep_steps += 1
            if result[2] or ep_steps >= 12000:
                print(
                    "Episode",
                    episode,
                    ": steps=",
                    ep_steps,
                    ", reward=",
                    ep_reward,
                )
                total_reward += ep_reward
                break
    print("CPU Total reward:", total_reward)

    # --- GPU Test ---
    print("\n--- GPU Test ---")
    var ctx = DeviceContext()

    var states = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * STATE_SIZE)
    var actions = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var rewards = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var dones = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var terminated = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OBS_DIM)

    BreakoutEnv[DType.float64].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states
    )
    ctx.synchronize()
    print("GPU reset complete.")

    # Run steps: FIRE first, then LEFT/RIGHT
    var host_actions = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE)

    # Fire for 5 steps
    for j in range(BATCH_SIZE):
        host_actions[j] = 1.0  # FIRE
    ctx.enqueue_copy(actions, host_actions)
    for step in range(5):
        BreakoutEnv[DType.float64].step_kernel_gpu[
            BATCH_SIZE, STATE_SIZE, OBS_DIM
        ](
            ctx,
            states,
            actions,
            rewards,
            dones,
            terminated,
            obs_buf,
            rng_seed=UInt64(step),
        )

    # Then alternate left/right
    for j in range(BATCH_SIZE):
        host_actions[j] = Scalar[dtype](2 + j % 2)  # LEFT or RIGHT
    ctx.enqueue_copy(actions, host_actions)

    var total_gpu_dones = 0
    for step in range(5000):
        BreakoutEnv[DType.float64].step_kernel_gpu[
            BATCH_SIZE, STATE_SIZE, OBS_DIM
        ](
            ctx,
            states,
            actions,
            rewards,
            dones,
            terminated,
            obs_buf,
            rng_seed=UInt64(step + 10),
        )
        if step % 1000 == 999:
            ctx.synchronize()
            var host_dones = ctx.enqueue_create_host_buffer[dtype](BATCH_SIZE)
            ctx.enqueue_copy(host_dones, dones)
            ctx.synchronize()
            var done_count = 0
            for j in range(BATCH_SIZE):
                if host_dones[j] > 0.5:
                    done_count += 1
            if done_count > 0:
                print("Step", step + 1, ":", done_count, "envs done")
                total_gpu_dones += done_count
            BreakoutEnv[DType.float64].selective_reset_kernel_gpu[
                BATCH_SIZE, STATE_SIZE
            ](ctx, states, dones, rng_seed=UInt64(step + 200))

    print("Total GPU episodes:", total_gpu_dones)
    print("\n=== DONE ===")
