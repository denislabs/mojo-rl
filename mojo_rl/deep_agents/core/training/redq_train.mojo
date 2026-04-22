"""REDQ GPU training loop.

Standalone loop tailored to REDQ's UTD+subset-min training pattern.
Not graph-capturable (subset indices are sampled on the host each step).

Per env step batch (n_envs transitions):
  1. Select actions from the stochastic actor (random during warmup)
  2. Step env, store transitions
  3. Run `UTD_RATIO * n_envs` inner REDQ critic+actor updates
      (each update: subset-min target, critic loss, periodic policy+alpha,
       soft update of all N target critics)

Designed for correctness and clarity. Throughput optimizations (graph
capture, batched forward across the critic ensemble) are left for later.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import UnsafePointer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.core import (
    TrainingMetrics,
    GPUContinuousEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_actions_kernel,
)
from mojo_rl.deep_agents.core.agents.redq_agent import REDQAgent
from mojo_rl.deep_agents.core.configs.redq_config import REDQConfig


def run_redq_train_gpu[
    E: GPUContinuousEnv,
    Config: REDQConfig,
    L: Logger = NoOpLogger,
    n_envs: Int = 1,
](
    mut agent: REDQAgent[Config, max_n_envs=n_envs],
    ctx: DeviceContext,
    num_steps: Int,
    warmup_steps: Int = 5_000,
    verbose: Bool = False,
    print_every: Int = 10_000,
    environment_name: String = "Environment",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
    rng_seed: UInt64 = 42,
) raises -> TrainingMetrics:
    """REDQ training loop. `n_envs` parallel envs collect n_envs transitions
    per loop iteration; each iteration then runs `UTD_RATIO * n_envs` inner
    REDQ updates to preserve the UTD ratio per transition.

    Parameters:
        E: GPU environment type.
        Config: REDQ configuration.
        L: Logger type.
        n_envs: number of parallel environments (compile-time).

    Args:
        agent: REDQ agent (updated in place).
        ctx: GPU device context.
        num_steps: total env transitions to collect.
        warmup_steps: transitions of uniform-random action collection before
            the agent's actor is used.
        verbose: print per-iteration progress.
        print_every: env-transition interval between progress prints.
        environment_name: label for metrics.
        logger: optional logger.
        rng_seed: initial env RNG seed.

    Returns:
        TrainingMetrics.
    """
    var metrics = TrainingMetrics(
        algorithm_name="REDQ",
        environment_name=environment_name,
    )

    var gpu_state = agent.make_gpu_state(ctx)
    agent.upload_to_gpu(gpu_state, ctx)

    # --- Env buffers ---
    var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Episode tracking
    var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
    var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)
    var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
    var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)

    # Env workspace (shared + per-env)
    var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
    if ws_size == 0:
        ws_size = 1
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
    if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
        E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)

    # --- Reset envs ---
    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](
        ctx, states_buf, rng_seed=rng_seed
    )
    E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
        ctx,
        states_buf,
        actions_buf,
        rewards_buf,
        dones_buf,
        terminated_buf,
        obs_buf,
        rng_seed=rng_seed,
        workspace_ptr=workspace_buf.unsafe_ptr(),
    )
    ctx.enqueue_memset(episode_rewards_buf, 0)
    ctx.enqueue_memset(episode_steps_buf, 0)
    ctx.enqueue_memset(gpu_reward_sum_buf, 0)
    ctx.enqueue_memset(gpu_episode_count_buf, 0)

    comptime tpb = 256
    comptime env_blocks = (n_envs + tpb - 1) // tpb
    comptime accum_k = accumulate_rewards_kernel[dtype, n_envs]
    comptime incr_steps_k = increment_steps_kernel[dtype, n_envs]
    comptime log_reset_k = log_and_reset_completed_kernel[dtype, n_envs]
    comptime act_tpb = 256
    comptime act_blocks = (n_envs * E.ACTION_DIM + act_tpb - 1) // act_tpb
    comptime warmup_kernel = uniform_random_actions_kernel[
        dtype, n_envs, E.ACTION_DIM
    ]

    var total_steps = 0
    var step_seed: UInt32 = UInt32(rng_seed)
    var next_print = print_every

    var action_scale_val = Scalar[dtype](agent.action_scale)

    while total_steps < num_steps:
        # --- 1. Action selection ---
        ctx.enqueue_copy(prev_obs_buf, obs_buf)
        if total_steps < warmup_steps:
            var act_t = LayoutTensor[
                dtype,
                Layout.row_major(n_envs, E.ACTION_DIM),
                MutAnyOrigin,
            ](actions_buf.unsafe_ptr())
            ctx.enqueue_function[warmup_kernel, warmup_kernel](
                act_t,
                action_scale_val,
                Scalar[DType.uint32](step_seed),
                grid_dim=(act_blocks,),
                block_dim=(act_tpb,),
            )
        else:
            agent.sync_explore_counter(ctx, gpu_state)
            agent.select_actions_gpu[n_envs](
                ctx, gpu_state, obs_buf, actions_buf
            )

        # --- 2. Env step ---
        step_seed += 1
        E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=UInt64(step_seed),
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )

        # --- 3. Store transitions (use terminated_buf as done mask so
        #       time-limit truncations don't bootstrap to zero). ---
        agent.gpu_store[n_envs](
            ctx,
            gpu_state,
            prev_obs_buf,
            actions_buf,
            rewards_buf,
            obs_buf,
            terminated_buf,
        )

        # --- 4. Episode tracking ---
        var er_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var rw_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var es_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())
        var dn_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var rs_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            gpu_reward_sum_buf.unsafe_ptr()
        )
        var ec_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            gpu_episode_count_buf.unsafe_ptr()
        )
        ctx.enqueue_function[accum_k, accum_k](
            er_t, rw_t, grid_dim=(env_blocks,), block_dim=(tpb,)
        )
        ctx.enqueue_function[incr_steps_k, incr_steps_k](
            es_t, grid_dim=(env_blocks,), block_dim=(tpb,)
        )
        ctx.enqueue_function[log_reset_k, log_reset_k](
            dn_t,
            er_t,
            es_t,
            rs_t,
            ec_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        # --- 5. Selective reset of done envs ---
        step_seed += 1
        E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
            ctx,
            states_buf,
            dones_buf,
            rng_seed=UInt64(step_seed),
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )
        E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
            ctx, states_buf, obs_buf
        )

        total_steps += n_envs

        # --- 6. REDQ inner updates (UTD_RATIO * n_envs iterations) ---
        if total_steps >= warmup_steps:
            comptime UTD = Config.UTD_RATIO
            for _ in range(UTD * n_envs):
                agent.do_gpu_train_step(ctx, gpu_state)

        # --- 7. Periodic progress print ---
        if verbose and total_steps >= next_print:
            ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
            ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
            ctx.synchronize()
            var rsum = Float64(host_reward_sum[0])
            var ecount = Float64(host_episode_count[0])
            var avg_ret = rsum / ecount if ecount > 0.0 else 0.0
            print(
                "[REDQ]",
                environment_name,
                "step",
                total_steps,
                "/",
                num_steps,
                "episodes:",
                Int(ecount),
                "avg_return:",
                avg_ret,
                "alpha:",
                agent.alpha,
            )
            # Reset accumulators for next interval
            ctx.enqueue_memset(gpu_reward_sum_buf, 0)
            ctx.enqueue_memset(gpu_episode_count_buf, 0)
            next_print += print_every

    # Final download of weights
    agent.download_from_gpu(gpu_state, ctx)
    ctx.synchronize()
    return metrics^
