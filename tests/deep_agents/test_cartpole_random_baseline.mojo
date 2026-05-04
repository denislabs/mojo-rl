"""Sanity check: pure-random CartPole episode length.

Steps `n_envs` envs with uniform-random actions for `num_steps` and reports
the cumulative avg episode return — same accumulation pipeline as
`muzero.train_gpu` (accumulate_rewards_kernel + log_and_reset_completed_kernel).
A correctly-implemented Gymnasium-equivalent CartPole should report ~22.
If MuZero CartPole is reporting 4-6 with avg episode length effectively ~3
post-warmup, this test rules out env-side issues vs. agent-side issues.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    log_and_reset_completed_kernel,
    increment_steps_kernel,
)
from mojo_rl.deep_agents.core.kernels import (
    uniform_random_discrete_actions_kernel,
)


def main() raises:
    print("=== CartPole random-baseline avg episode length ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]
    comptime dtype = DType.float32
    comptime N_ENVS = 32
    comptime NUM_STEPS = 5000  # ~total transitions across all envs
    comptime ACT = CartPoleGPU.NUM_ACTIONS
    comptime STATE_SIZE = CartPoleGPU.STATE_SIZE
    comptime OBS_DIM = CartPoleGPU.OBS_DIM
    comptime TPB = 32
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    # Buffers
    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)
    var workspace_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * 8)

    var ep_rew_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var ep_steps_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
    var episode_count_buf = ctx.enqueue_create_buffer[dtype](1)

    var rew_sum_host = ctx.enqueue_create_host_buffer[dtype](1)
    var ep_count_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset envs
    CartPoleGPU.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf)
    CartPoleGPU.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
        ctx, states_buf, obs_buf
    )

    ep_rew_buf.enqueue_fill(Scalar[dtype](0.0))
    ep_steps_buf.enqueue_fill(Scalar[dtype](0.0))
    reward_sum_buf.enqueue_fill(Scalar[dtype](0.0))
    episode_count_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    # Loop
    var total_steps = 0
    while total_steps < NUM_STEPS:
        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        comptime run_act = uniform_random_discrete_actions_kernel[
            dtype, N_ENVS, ACT
        ]
        ctx.enqueue_function[run_act, run_act](
            act_t,
            Scalar[DType.uint32](UInt32(total_steps)),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        CartPoleGPU.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=UInt64(total_steps),
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )

        var rew_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var ep_rew_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](ep_rew_buf.unsafe_ptr())
        var ep_steps_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](ep_steps_buf.unsafe_ptr())

        comptime run_accum = accumulate_rewards_kernel[dtype, N_ENVS]
        ctx.enqueue_function[run_accum, run_accum](
            ep_rew_t,
            rew_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime run_incr = increment_steps_kernel[dtype, N_ENVS]
        ctx.enqueue_function[run_incr, run_incr](
            ep_steps_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        var rew_sum_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](reward_sum_buf.unsafe_ptr())
        var ep_count_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](episode_count_buf.unsafe_ptr())
        comptime run_log = log_and_reset_completed_kernel[dtype, N_ENVS]
        ctx.enqueue_function[run_log, run_log](
            dones_t,
            ep_rew_t,
            ep_steps_t,
            rew_sum_t,
            ep_count_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Selective reset for done envs
        CartPoleGPU.selective_reset_kernel_gpu[N_ENVS, STATE_SIZE](
            ctx,
            states_buf,
            dones_buf,
            rng_seed=UInt64(total_steps),
        )
        # Re-extract obs for next step (matches train_gpu's pattern, but
        # we DON'T re-step the env — that phantom step is what we're
        # checking might be the bug)
        CartPoleGPU.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
            ctx, states_buf, obs_buf
        )

        total_steps += N_ENVS

    ctx.enqueue_copy(rew_sum_host, reward_sum_buf)
    ctx.enqueue_copy(ep_count_host, episode_count_buf)
    ctx.synchronize()

    var total_r = Float64(rew_sum_host[0])
    var total_e = Float64(ep_count_host[0])
    var avg = total_r / total_e if total_e > 0 else Float64(0.0)

    print("Total transitions:", total_steps)
    print("Total episodes:", Int(total_e))
    print("Total reward:", Int(total_r))
    print("Avg episode return (= avg episode length, reward=1/step):", avg)
    print()
    if avg > 15.0:
        print("PASS: random CartPole gives ~22 (normal range)")
    elif avg > 8.0:
        print("WARN: lower than expected ~22 but plausibly random")
    else:
        print("FAIL: random episodes are too short — env or pipeline bug")
