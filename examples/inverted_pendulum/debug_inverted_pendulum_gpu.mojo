"""Diagnostic: test InvertedPendulum GPU episode tracking end-to-end.

Runs the exact same kernel sequence as the training loop:
  step → accum_rewards → incr_steps → log_reset → selective_reset → extract_obs

Run with:
    pixi run -e apple mojo run -I . examples/inverted_pendulum/debug_inverted_pendulum_gpu.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.envs.inverted_pendulum import InvertedPendulum
from mojo_rl.physics3d.gpu.constants import (
    qpos_offset,
    metadata_offset,
    META_IDX_STEP_COUNT,
)
from mojo_rl.nn import dtype
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
)


comptime N_ENVS = 32
comptime Env = InvertedPendulum[dtype, TERMINATE_ON_UNHEALTHY=True]
comptime QPOS_OFF = qpos_offset[2, 2]()
comptime META_OFF = metadata_offset[2, 2, 3, 5]()
comptime tpb = 256
comptime env_blocks = (N_ENVS + tpb - 1) // tpb

comptime accum_k = accumulate_rewards_kernel[dtype, N_ENVS]
comptime incr_k = increment_steps_kernel[dtype, N_ENVS]
comptime log_reset_k = log_and_reset_completed_kernel[dtype, N_ENVS]


def main() raises:
    seed(42)
    print("=" * 60)
    print("InvertedPendulum Episode Tracking Diagnostic")
    print("=" * 60)
    print("N_ENVS:", N_ENVS, " STATE_SIZE:", Env.STATE_SIZE)
    print()

    with DeviceContext() as ctx:
        # === Env buffers ===
        var states_buf = ctx.enqueue_create_buffer[dtype](
            N_ENVS * Env.STATE_SIZE
        )
        var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * Env.OBS_DIM)
        var actions_buf = ctx.enqueue_create_buffer[dtype](
            N_ENVS * Env.ACTION_DIM
        )
        var rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
        var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)

        # Workspace
        var ws_size = Env.STEP_WS_SHARED + N_ENVS * Env.STEP_WS_PER_ENV
        if ws_size == 0:
            ws_size = 1
        var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
        if Env.STEP_WS_SHARED + Env.STEP_WS_PER_ENV > 0:
            Env.init_step_workspace_gpu[N_ENVS](ctx, workspace_buf)

        # === Episode tracking buffers (same as training loop) ===
        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
        var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
        var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)

        # Explicitly zero them (training loop does NOT do this)
        episode_rewards_buf.enqueue_fill(Scalar[dtype](0.0))
        episode_steps_buf.enqueue_fill(Scalar[dtype](0.0))
        gpu_reward_sum_buf.enqueue_fill(Scalar[dtype](0.0))
        gpu_episode_count_buf.enqueue_fill(Scalar[dtype](0.0))

        # Host readback
        var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
        var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)
        var host_dones = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

        # Reset all envs
        Env.reset_kernel_gpu[N_ENVS, Env.STATE_SIZE](
            ctx, states_buf, rng_seed=0
        )
        actions_buf.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        # === Run training-like loop ===
        var completed = 0
        for step in range(500):
            # 1. Step
            Env.step_kernel_gpu[
                N_ENVS, Env.STATE_SIZE, Env.OBS_DIM, Env.ACTION_DIM
            ](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                terminated_buf,
                obs_buf,
                rng_seed=UInt64(step),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )

            # 2. Accumulate rewards + increment steps
            var er_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS), MutAnyOrigin
            ](episode_rewards_buf.unsafe_ptr())
            var rw_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS), MutAnyOrigin
            ](rewards_buf.unsafe_ptr())
            var es_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS), MutAnyOrigin
            ](episode_steps_buf.unsafe_ptr())
            var dn_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS), MutAnyOrigin
            ](dones_buf.unsafe_ptr())
            var rs_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_reward_sum_buf.unsafe_ptr())
            var ec_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_episode_count_buf.unsafe_ptr())

            ctx.enqueue_function[accum_k, accum_k](
                er_t, rw_t, grid_dim=(env_blocks,), block_dim=(tpb,)
            )
            ctx.enqueue_function[incr_k, incr_k](
                es_t, grid_dim=(env_blocks,), block_dim=(tpb,)
            )

            # 3. Log and reset completed episodes
            ctx.enqueue_function[log_reset_k, log_reset_k](
                dn_t, er_t, es_t, rs_t, ec_t,
                grid_dim=(1,), block_dim=(1,),
            )

            # 4. Selective reset
            Env.selective_reset_kernel_gpu[N_ENVS, Env.STATE_SIZE](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=UInt64(step + 1000),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )

            # 5. Extract obs for next step
            Env.extract_obs_kernel_gpu[N_ENVS, Env.STATE_SIZE, Env.OBS_DIM](
                ctx, states_buf, obs_buf
            )

            # Check every 50 steps
            if step % 50 == 49 or step < 3:
                ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
                ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
                ctx.enqueue_copy(host_dones, dones_buf)
                ctx.synchronize()

                var ep_count = Float64(host_episode_count[0])
                var rw_sum = Float64(host_reward_sum[0])
                var n_done = 0
                for i in range(N_ENVS):
                    if Float64(host_dones[i]) > 0.5:
                        n_done += 1

                print(
                    "Step "
                    + String(step)
                    + ": ep_count="
                    + String(ep_count)[byte=:8]
                    + " rw_sum="
                    + String(rw_sum)[byte=:10]
                    + " n_done_now="
                    + String(n_done)
                    + " Int(ep)="
                    + String(Int(ep_count))
                )

        # Final read
        ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
        ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
        ctx.synchronize()
        var final_ep = Float64(host_episode_count[0])
        var final_rw = Float64(host_reward_sum[0])
        print()
        print("FINAL: episodes=" + String(Int(final_ep))
              + " reward_sum=" + String(final_rw)[byte=:10])
        if Int(final_ep) > 0:
            print("Avg reward: " + String(final_rw / final_ep)[byte=:10])
            print(">>> Episode tracking WORKS <<<")
        else:
            print(">>> Episode tracking BROKEN - 0 episodes after 500 steps! <<<")

    print()
    print("Done.")
