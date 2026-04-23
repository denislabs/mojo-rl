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
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
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
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
) raises -> TrainingMetrics:
    """REDQ training loop. `n_envs` parallel envs collect n_envs transitions
    per loop iteration; each iteration then runs `UTD_RATIO * n_envs` inner
    REDQ updates to preserve the UTD ratio per transition.

    Parameters:
        E: GPU environment type.
        Config: REDQ configuration.
        L: Logger type.
        n_envs: Number of parallel environments (compile-time).

    Args:
        agent: REDQ agent (updated in place).
        ctx: GPU device context.
        num_steps: Total env transitions to collect.
        warmup_steps: Transitions of uniform-random action collection before
            the agent's actor is used.
        verbose: Print per-iteration progress.
        print_every: Env-transition interval between progress prints.
        environment_name: Label for metrics.
        logger: Optional logger.
        rng_seed: Initial env RNG seed.
        checkpoint_every: Checkpoint every N steps.
        checkpoint_path: Checkpoint path.

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
    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=rng_seed)
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
    var total_train_steps = 0
    var completed_episodes = 0
    var last_avg_reward: Float64 = 0.0
    var step_seed: UInt32 = UInt32(rng_seed)
    var next_print = print_every

    # Progress bar: ~20 updates per print interval
    var progress_interval = print_every // 20
    if progress_interval < n_envs:
        progress_interval = n_envs
    var next_progress = progress_interval

    # Checkpointing: agent's own settings take precedence if set, else use
    # the function args. Either path satisfies the same `checkpoint_every`
    # contract used by SAC / MBPO.
    var ckpt_every = checkpoint_every
    var ckpt_path = checkpoint_path
    if agent.checkpoint_every > 0 and len(agent.checkpoint_path) > 0:
        ckpt_every = agent.checkpoint_every
        ckpt_path = agent.checkpoint_path
    var next_checkpoint = ckpt_every

    var action_scale_val = Scalar[dtype](agent.action_scale)
    var alg_name = String("REDQ")
    var alpha_host = ctx.enqueue_create_host_buffer[dtype](1)

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
        var er_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            episode_rewards_buf.unsafe_ptr()
        )
        var rw_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            rewards_buf.unsafe_ptr()
        )
        var es_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            episode_steps_buf.unsafe_ptr()
        )
        var dn_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            dones_buf.unsafe_ptr()
        )
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
            var n_updates = UTD * n_envs
            for _ in range(n_updates):
                agent.do_gpu_train_step(ctx, gpu_state)
                # Per-train-step diagnostics (no-op unless agent.diag_every
                # > 0 and we're on a boundary). Train-step axis matches
                # SAC/MBPO so dashboard plots line up.
                agent.maybe_log_diagnostics[L](ctx, gpu_state, logger)
            total_train_steps += n_updates

        # --- 7a. Progress bar (no GPU sync, pure CPU counters) ---
        if verbose and total_steps >= next_progress:
            var interval_start = next_print - print_every
            print_progress_bar(
                total_steps - interval_start,
                print_every,
                total_train_steps,
                alg_name,
            )
            next_progress += progress_interval

        # --- 7b. Periodic print + logger flush ---
        if (
            verbose or (logger and logger[].is_active())
        ) and total_steps >= next_print:
            ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
            ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
            ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
            ctx.synchronize()

            var recent_count = Int(host_episode_count[0])
            var recent_sum = Float64(host_reward_sum[0])
            var cur_alpha = Float64(alpha_host[0])
            completed_episodes += recent_count
            if recent_count > 0:
                last_avg_reward = recent_sum / Float64(recent_count)
                for _ in range(recent_count):
                    metrics.log_episode(
                        completed_episodes, last_avg_reward, 0, 0.0
                    )

            # Reset GPU-side accumulators for next interval
            ctx.enqueue_memset(gpu_reward_sum_buf, 0)
            ctx.enqueue_memset(gpu_episode_count_buf, 0)

            # Logger: record scalar metrics
            if logger:
                logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
                logger[].log_scalar(
                    "episodes", Float64(completed_episodes), total_steps
                )
                logger[].log_scalar(
                    "train_steps", Float64(total_train_steps), total_steps
                )
                logger[].log_scalar("alpha", cur_alpha, total_steps)

            # Clear progress bar, then print status line
            if verbose:
                clear_progress_bar()
                var status_line = (
                    alg_name
                    + " | Step "
                    + String(total_steps)
                    + " / "
                    + String(num_steps)
                    + " | Ep: "
                    + String(completed_episodes)
                    + " | AvgR: "
                    + String(last_avg_reward)[byte=:7]
                    + " | Train: "
                    + String(total_train_steps)
                    + " | α: "
                    + String(cur_alpha)[byte=:6]
                )
                print(status_line)
            next_print += print_every

        # --- 7c. Periodic checkpoint ---
        if ckpt_every > 0 and total_steps >= next_checkpoint:
            agent.download_from_gpu(gpu_state, ctx)
            ctx.synchronize()
            agent.save_checkpoint(ckpt_path)
            if verbose:
                clear_progress_bar()
                print(
                    alg_name
                    + " | checkpoint @ step "
                    + String(total_steps)
                    + " -> "
                    + ckpt_path
                )
            next_checkpoint += ckpt_every

    # --- Final sync: download trailing episode stats ---
    ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
    ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
    ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
    ctx.synchronize()

    var final_count = Int(host_episode_count[0])
    var final_sum = Float64(host_reward_sum[0])
    completed_episodes += final_count
    if final_count > 0:
        last_avg_reward = final_sum / Float64(final_count)
        for _ in range(final_count):
            metrics.log_episode(completed_episodes, last_avg_reward, 0, 0.0)

    # Final download of weights
    agent.download_from_gpu(gpu_state, ctx)
    ctx.synchronize()

    # Final checkpoint (CPU state is now synced from GPU)
    if ckpt_every > 0 and len(ckpt_path) > 0:
        agent.save_checkpoint(ckpt_path)

    # Final logger flush
    if logger and logger[].is_active():
        logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
        logger[].log_scalar(
            "episodes", Float64(completed_episodes), total_steps
        )
        logger[].log_scalar(
            "train_steps", Float64(total_train_steps), total_steps
        )
        logger[].log_scalar("alpha", Float64(alpha_host[0]), total_steps)
        logger[].flush()

    if verbose:
        clear_progress_bar()
        print(
            alg_name
            + " | Step "
            + String(total_steps)
            + " / "
            + String(num_steps)
            + " | Ep: "
            + String(completed_episodes)
            + " | AvgR: "
            + String(last_avg_reward)[byte=:7]
            + " | Train: "
            + String(total_train_steps)
            + " [DONE]"
        )

    return metrics^
