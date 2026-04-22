"""MBPO training loop.

Differs from standard off-policy training:
1. Warmup fills the real buffer
2. Dynamics ensemble is trained periodically on real data
3. Model generates synthetic rollouts from real states
4. Multiple SAC gradient steps per environment step
5. Mixed sampling from real + synthetic buffers
"""

from std.random import random_float64
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv, GPUContinuousEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import dtype
from ..checkpoint_trait import Checkpointable
from ..agents.mbpo_agent import MBPOAgent, GPUDynamicsEnsemble
from ..replay.gpu_replay_buffer import GPUReplayBuffer
from ..configs.mbpo_config import MBPOConfig
from ..kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_actions_kernel,
)
from ..utils import print_progress_bar, clear_progress_bar
from std.sys import has_nvidia_gpu_accelerator
from mojo_rl.cuda.graph import CUDAGraph


def run_mbpo_train[
    E: BoxContinuousActionEnv,
    Config: MBPOConfig,
    L: Logger = NoOpLogger,
    TRAIN_N_ENVS: Int = 1,
](
    mut agent: MBPOAgent[Config, L, TRAIN_N_ENVS],
    mut cpu_state: MBPOAgent[Config, L, TRAIN_N_ENVS].CPUStateType,
    mut env: E,
    num_epochs: Int,
    steps_per_epoch: Int = 1000,
    max_steps_per_episode: Int = 1000,
    warmup_steps: Int = 5000,
    eval_episodes: Int = 5,
    eval_every: Int = 1,
    verbose: Bool = False,
    print_every: Int = 1,
    environment_name: String = "Environment",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """MBPO training loop.

    Structure per epoch:
    1. Collect steps_per_epoch real transitions
    2. Train dynamics ensemble every model_train_freq env steps
    3. Generate synthetic rollouts after each model training
    4. Run sac_updates_per_step SAC gradient steps per env step
    5. Update rollout length schedule

    Args:
        agent: MBPOAgent (hyperparameters + algorithm).
        cpu_state: MBPO CPU state (networks + dual buffers + ensemble).
        env: Continuous-action environment.
        num_epochs: Number of training epochs.
        steps_per_epoch: Environment steps per epoch (default: 1000).
        max_steps_per_episode: Max episode length (default: 1000).
        warmup_steps: Random exploration steps before training (default: 5000).
        eval_episodes: Episodes for evaluation (default: 5).
        eval_every: Evaluate every N epochs (default: 1).
        verbose: Print progress (default: False).
        print_every: Print every N epochs if verbose (default: 1).
        environment_name: Name for metrics labeling.
        logger: Optional metrics logger pointer.

    Returns:
        TrainingMetrics with per-evaluation rewards.
    """
    var metrics = TrainingMetrics(
        algorithm_name="MBPO",
        environment_name=environment_name,
    )

    # --- Warmup: fill real buffer with random transitions ---
    var warmup_obs = env.reset_obs_list()
    var warmup_count = 0
    var warmup_ep_steps = 0
    while warmup_count < warmup_steps:
        var action = agent.random_action[E.dtype]()
        var result = env.step_continuous_vec(action)
        var next_obs = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]
        warmup_ep_steps += 1
        # Store terminated (not done) so Q-targets bootstrap on truncation.
        var terminated = done and (warmup_ep_steps < max_steps_per_episode)
        agent.store_transition(
            cpu_state, warmup_obs, action, reward, next_obs, terminated
        )
        warmup_count += 1
        if done:
            warmup_obs = env.reset_obs_list()
            warmup_ep_steps = 0
        else:
            warmup_obs = next_obs^

    if verbose:
        print(
            "Warmup complete: "
            + String(warmup_steps)
            + " steps in real buffer"
        )

    # --- Training loop ---
    var total_env_steps = 0
    var episode_obs = env.reset_obs_list()
    var episode_reward: Float64 = 0.0
    var episode_steps = 0
    var episode_count = 0

    for epoch in range(num_epochs):
        agent.update_rollout_length(epoch)

        for step in range(steps_per_epoch):
            # Collect one real transition
            var obs_f64 = List[Float64]()
            for i in range(len(episode_obs)):
                obs_f64.append(Float64(episode_obs[i]))

            var action = agent.select_action(cpu_state, obs_f64)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]
            episode_steps += 1
            # Store terminated (not done) so Q-targets bootstrap on truncation.
            var terminated = done and (episode_steps < max_steps_per_episode)
            # Store in real buffer
            agent.store_transition(
                cpu_state, obs_f64, action, reward, next_obs, terminated
            )
            episode_reward += reward
            total_env_steps += 1

            # Train dynamics model periodically
            if total_env_steps % agent.model_train_freq == 0:
                agent.train_dynamics(cpu_state)
                agent.do_model_rollouts(cpu_state)

                if verbose:
                    print(
                        "  Model trained at step "
                        + String(total_env_steps)
                        + " | Real buffer: "
                        + String(cpu_state.real_buffer.size)
                        + " | Synth buffer: "
                        + String(cpu_state.synth_buffer.size)
                        + " | Rollout len: "
                        + String(agent.rollout_length)
                    )

            # SAC gradient steps
            if cpu_state.is_ready():
                for _ in range(agent.sac_updates_per_step):
                    _ = agent.do_cpu_train_step(cpu_state)

            # Handle episode boundaries
            if done or episode_steps >= max_steps_per_episode:
                episode_count += 1
                metrics.log_episode(
                    episode_count - 1,
                    Scalar[DType.float64](episode_reward),
                    episode_steps,
                    agent.get_explore_rate(),
                )
                if logger:
                    logger[].log_scalar(
                        "episode_reward", episode_reward, total_env_steps
                    )
                episode_obs = env.reset_obs_list()
                episode_reward = 0.0
                episode_steps = 0
            else:
                episode_obs = List[Scalar[E.dtype]]()
                for i in range(len(next_obs)):
                    episode_obs.append(Scalar[E.dtype](next_obs[i]))

        # --- Evaluation ---
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            var eval_total: Float64 = 0.0
            for _ in range(eval_episodes):
                var eval_obs_raw = env.reset_obs_list()
                var eval_obs = List[Float64]()
                for i in range(len(eval_obs_raw)):
                    eval_obs.append(Float64(eval_obs_raw[i]))
                var eval_reward: Float64 = 0.0
                for _ in range(max_steps_per_episode):
                    var eval_action = agent.select_greedy_action(
                        cpu_state, eval_obs
                    )
                    var eval_result = env.step_continuous_vec(eval_action)
                    var eval_next = List[Float64]()
                    for i in range(len(eval_result[0])):
                        eval_next.append(Float64(eval_result[0][i]))
                    eval_reward += Float64(eval_result[1])
                    if eval_result[2]:
                        break
                    eval_obs = eval_next^
                eval_total += eval_reward

            var avg_eval = eval_total / Float64(eval_episodes)

            if logger:
                logger[].log_scalar("eval_reward", avg_eval, total_env_steps)

            if verbose and (epoch + 1) % print_every == 0:
                print(
                    "Epoch "
                    + String(epoch + 1)
                    + " | Eval reward: "
                    + String(avg_eval)[byte=:8]
                    + " | Env steps: "
                    + String(total_env_steps)
                    + " | Alpha: "
                    + String(agent.alpha)[byte=:6]
                    + " | Rollout: "
                    + String(agent.rollout_length)
                )

        # Autosave checkpoint
        if (
            agent.checkpoint_every > 0
            and (epoch + 1) % agent.checkpoint_every == 0
        ):
            agent.save_checkpoint(
                agent.checkpoint_path + "_epoch_" + String(epoch + 1) + ".ckpt"
            )

    if logger:
        logger[].flush()
    return metrics^


# =============================================================================
# GPU Training Loop — MBPO with GPU env + GPU SAC + GPU dynamics
# =============================================================================


def run_mbpo_train_gpu[
    E: GPUContinuousEnv,
    Config: MBPOConfig,
    L: Logger = NoOpLogger,
    USE_CUDA_GRAPH: Bool = False,
    TRAIN_N_ENVS: Int = 1,
](
    mut agent: MBPOAgent[Config, L, TRAIN_N_ENVS],
    mut cpu_state: MBPOAgent[Config, L, TRAIN_N_ENVS].CPUStateType,
    ctx: DeviceContext,
    num_steps: Int,
    warmup_steps: Int = 5000,
    verbose: Bool = False,
    print_every: Int = 50_000,
    environment_name: String = "Environment",
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
) raises -> TrainingMetrics:
    """MBPO GPU training loop with batched GPU environments.

    Follows the standard GPU off-policy loop pattern (selective reset,
    GPU buffer store, episode tracking) but adds periodic GPU-side
    dynamics ensemble training and model rollouts.

    Flow per iteration:
    1. GPU: select actions for N_ENVS envs
    2. GPU: step all envs (selective reset for done envs)
    3. GPU: store transitions in GPU replay buffer
    4. GPU: sac_updates_per_step SAC gradient steps
    5. Periodically: GPU dynamics training + GPU model rollouts
       (data never leaves GPU)

    Args:
        agent: MBPOAgent (hyperparameters, updated in-place).
        cpu_state: MBPOCPUState (dynamics ensemble + CPU buffers).
        ctx: GPU device context.
        num_steps: Total env transitions across all parallel envs.
        warmup_steps: Transitions before training (default: 5000).
        verbose: Print progress (default: False).
        print_every: Print interval in transitions (default: 50000).
        environment_name: Name for metrics.
        logger: Optional metrics logger.

    Returns:
        TrainingMetrics with episode-level statistics.
    """
    comptime n_envs = MBPOAgent[Config, L, TRAIN_N_ENVS].GPU_N_ENVS

    var metrics = TrainingMetrics(
        algorithm_name="MBPO-GPU",
        environment_name=environment_name,
    )

    comptime GPUState = MBPOAgent[Config, L, TRAIN_N_ENVS].GPUStateType
    var gpu_state = GPUState(ctx)

    var gpu_dynamics = GPUDynamicsEnsemble[
        Config.DynamicsModel,
        Config.DynOpt,
        Config.ENSEMBLE_SIZE,
        Config.ELITE_SIZE,
        Config.obs_dim,
        Config.action_dim,
    ](ctx)
    gpu_dynamics.upload_from(cpu_state.dynamics, ctx)

    gpu_state.actor.upload_from(cpu_state.actor, ctx)
    gpu_state.critics.upload_from(cpu_state.critics, ctx)

    # Upload SAC alpha state to GPU scalars (matches SAC's upload_to_gpu).
    # Without this, gpu_scalars is zero-filled: GPU_ALPHA=0 → alpha kernel
    # sets alpha=exp(log_alpha)=exp(0)=1.0 (not the intended 0.2), and
    # GPU_ALPHA_LR=0 freezes alpha there. With saturated tanh actions,
    # log_pi → -∞ and TD target = r + γ*(min_Q - α*log_pi) → +∞,
    # blowing Q-values to 10^30.
    comptime GPUStateT = MBPOAgent[Config, L, TRAIN_N_ENVS].GPUStateType
    var scalars_host = ctx.enqueue_create_host_buffer[dtype](
        GPUStateT.GPU_SCALARS_SIZE
    )
    scalars_host[GPUStateT.GPU_ALPHA] = Scalar[dtype](agent.alpha)
    scalars_host[GPUStateT.GPU_LOG_ALPHA] = Scalar[dtype](agent.log_alpha)
    scalars_host[GPUStateT.GPU_ADAM_M] = Scalar[dtype](agent.alpha_adam_m)
    scalars_host[GPUStateT.GPU_ADAM_V] = Scalar[dtype](agent.alpha_adam_v)
    scalars_host[GPUStateT.GPU_ADAM_T] = Scalar[dtype](agent.alpha_adam_t)
    scalars_host[GPUStateT.GPU_TARGET_ENT] = Scalar[dtype](agent.target_entropy)
    scalars_host[GPUStateT.GPU_ALPHA_LR] = Scalar[dtype](agent.alpha_lr)
    ctx.enqueue_copy(gpu_state.gpu_scalars, scalars_host)

    # Pre-allocated host buffer for alpha/scalars D2H (reused by diagnostics
    # and print-boundary downloads to avoid per-call host allocations).
    # Sized to GPU_SCALARS_SIZE so the print-boundary path can read both
    # alpha (index 0) and log_alpha (index 1).
    var alpha_host = ctx.enqueue_create_host_buffer[dtype](
        GPUStateT.GPU_SCALARS_SIZE
    )

    # Synthetic replay buffer (separate from real buffer in gpu_state)
    var synth_buffer = GPUReplayBuffer[
        Config.SYNTH_CAPACITY, Config.obs_dim, Config.action_dim
    ](ctx)
    # Scratch index buffers for mixed sampling
    comptime REAL_BS = MBPOAgent[Config, L, TRAIN_N_ENVS].REAL_BS
    comptime SYNTH_BS = MBPOAgent[Config, L, TRAIN_N_ENVS].SYNTH_BS
    var s_real_idx = ctx.enqueue_create_buffer[DType.int32](REAL_BS)
    var s_synth_idx = ctx.enqueue_create_buffer[DType.int32](SYNTH_BS)

    # Allocate environment buffers
    var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Episode tracking buffers
    var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
    var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)
    var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
    var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)

    # Env workspace
    var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
    if ws_size == 0:
        ws_size = 1
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
    if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
        E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)

    # Initial reset
    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=0)
    E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, terminated_buf,
        obs_buf, rng_seed=0, workspace_ptr=workspace_buf.unsafe_ptr(),
    )

    # Initialize tracking
    ctx.enqueue_memset(episode_rewards_buf, 0)
    ctx.enqueue_memset(episode_steps_buf, 0)
    ctx.enqueue_memset(gpu_reward_sum_buf, 0)
    ctx.enqueue_memset(gpu_episode_count_buf, 0)

    # Kernel aliases
    comptime tpb = 256
    comptime env_blocks = (n_envs + tpb - 1) // tpb
    comptime accum_k = accumulate_rewards_kernel[dtype, n_envs]
    comptime incr_k = increment_steps_kernel[dtype, n_envs]
    comptime log_reset_k = log_and_reset_completed_kernel[dtype, n_envs]
    comptime act_blocks = (n_envs * E.ACTION_DIM + tpb - 1) // tpb
    comptime warmup_k = uniform_random_actions_kernel[dtype, n_envs, E.ACTION_DIM]
    var action_scale_val = Scalar[dtype](agent.action_scale)

    var total_steps = 0
    var total_train_steps = 0
    var step_seed: UInt32 = 42
    var completed_episodes = 0
    var last_avg_reward: Float64 = 0.0
    var next_print = print_every
    var next_model_train = agent.model_train_freq
    var epoch = 0

    # CUDA graph state for SAC train step capture
    var _train_graph: Optional[CUDAGraph] = None

    # Progress bar: ~20 updates per print interval
    var progress_interval = print_every // 20
    if progress_interval < n_envs:
        progress_interval = n_envs
    var next_progress = progress_interval

    while total_steps < num_steps:

        # Save prev_obs
        ctx.enqueue_copy(prev_obs_buf, obs_buf)

        # Select actions
        if total_steps < warmup_steps:
            var act_t = LayoutTensor[
                dtype, Layout.row_major(n_envs, E.ACTION_DIM), MutAnyOrigin,
            ](actions_buf.unsafe_ptr())
            ctx.enqueue_function[warmup_k, warmup_k](
                act_t, action_scale_val, Scalar[DType.uint32](step_seed),
                grid_dim=(act_blocks,), block_dim=(tpb,),
            )
        else:
            agent.select_actions_gpu[n_envs](
                ctx, gpu_state, obs_buf, actions_buf
            )

        agent.total_steps += n_envs

        # Step environment
        E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf,
            terminated_buf, obs_buf, rng_seed=UInt64(step_seed),
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )

        # Store transitions in GPU buffer
        gpu_state.gpu_store[n_envs](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, terminated_buf
        )

        # Episode tracking (GPU-side)
        var ep_rew_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            episode_rewards_buf.unsafe_ptr()
        )
        var rew_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            rewards_buf.unsafe_ptr()
        )
        var ep_steps_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            episode_steps_buf.unsafe_ptr()
        )
        var dones_t = LayoutTensor[dtype, Layout.row_major(n_envs), MutAnyOrigin](
            dones_buf.unsafe_ptr()
        )
        var rsum_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            gpu_reward_sum_buf.unsafe_ptr()
        )
        var ecount_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            gpu_episode_count_buf.unsafe_ptr()
        )

        ctx.enqueue_function[accum_k, accum_k](
            ep_rew_t, rew_t, grid_dim=(env_blocks,), block_dim=(tpb,),
        )
        ctx.enqueue_function[incr_k, incr_k](
            ep_steps_t, grid_dim=(env_blocks,), block_dim=(tpb,),
        )
        ctx.enqueue_function[log_reset_k, log_reset_k](
            dones_t, ep_rew_t, ep_steps_t, rsum_t, ecount_t,
            grid_dim=(1,), block_dim=(1,),
        )

        # Selective reset done environments
        E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
            ctx, states_buf, dones_buf, rng_seed=UInt64(step_seed + 1),
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )
        E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
            ctx, states_buf, obs_buf
        )

        # GPU SAC gradient steps
        # Soft target update must run per gradient step (matches SAC GPU loop).
        # Calling it once per env iteration makes targets update at 1/sac_updates_per_step
        # the intended rate and causes Q-value divergence with synthetic data.
        if gpu_state.gpu_buffer_is_ready():
            if synth_buffer.is_ready[SYNTH_BS]():
                # Mixed sampling: REAL_BS real + SYNTH_BS synthetic
                comptime if USE_CUDA_GRAPH and has_nvidia_gpu_accelerator():
                    if not _train_graph:
                        agent._gpu_train_kernels(
                            ctx, gpu_state, synth_buffer,
                            s_real_idx, s_synth_idx,
                        )
                        agent.soft_update_targets_gpu(ctx, gpu_state)
                        ctx.synchronize()
                        var graph = CUDAGraph(ctx)
                        graph.begin_capture()
                        agent._gpu_train_kernels(
                            ctx, gpu_state, synth_buffer,
                            s_real_idx, s_synth_idx,
                        )
                        agent.soft_update_targets_gpu(ctx, gpu_state)
                        graph.end_capture()
                        if verbose:
                            print(
                                "[CUDA Graph] Captured MBPO SAC train step with "
                                + String(graph.num_nodes())
                                + " nodes"
                            )
                        _train_graph = graph^
                    for _ in range(agent.sac_updates_per_step):
                        _train_graph.value().replay_async()
                    _train_graph.value().sync()
                    agent._gpu_train_diagnostics(
                        ctx, gpu_state, agent.sac_updates_per_step, alpha_host
                    )
                else:
                    for _ in range(agent.sac_updates_per_step):
                        agent.do_gpu_train_step(
                            ctx, gpu_state, synth_buffer,
                            s_real_idx, s_synth_idx, alpha_host,
                        )
                        agent.soft_update_targets_gpu(ctx, gpu_state)
            else:
                # No synthetic data yet: train on 100% real (like CPU path)
                for _ in range(agent.sac_updates_per_step):
                    agent.do_gpu_train_step_real_only(ctx, gpu_state)
                    agent.soft_update_targets_gpu(ctx, gpu_state)
            total_train_steps += agent.sac_updates_per_step

        # Periodic dynamics training
        total_steps += n_envs
        step_seed += 1

        if total_steps >= next_model_train and total_steps >= warmup_steps:
            # GPU dynamics training on REAL data only (paper design)
            var mean_holdout = gpu_dynamics.train_on_buffer[
                MBPOAgent[Config, L, TRAIN_N_ENVS].GPU_BUF_CAP
            ](ctx, gpu_state.buffer)
            if logger:
                logger[].log_scalar(
                    "dyn_holdout_loss", mean_holdout, total_steps
                )
            agent.update_rollout_length(epoch)
            epoch += 1

            # GPU model rollouts → store in synth_buffer
            # num_rollouts_per_step controls total rollouts per training call
            # Reference uses 100K; GPU does rollout_batch (400) per call
            var n_rollout_batches = max(
                1,
                agent.num_rollouts_per_step // gpu_dynamics.rollout_batch,
            )
            for _ in range(n_rollout_batches):
                agent.do_model_rollouts_gpu[E](
                    ctx, gpu_dynamics, gpu_state, synth_buffer
                )
            next_model_train += agent.model_train_freq

        # Progress bar (no GPU sync, pure CPU counters)
        if verbose and total_steps >= next_progress:
            var interval_start = next_print - print_every
            print_progress_bar(
                total_steps - interval_start,
                print_every,
                total_train_steps,
                "MBPO-GPU",
            )
            next_progress += progress_interval

        # Print / log at print boundaries
        if (
            verbose or (logger and logger[].is_active())
        ) and total_steps >= next_print:
            # Download GPU-side episode stats + alpha for logging (reuses
            # the hoisted alpha_host buffer allocated at loop start).
            ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
            ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
            ctx.enqueue_copy(alpha_host, gpu_state.gpu_scalars)
            ctx.synchronize()
            agent.alpha = Float64(alpha_host[GPUStateT.GPU_ALPHA])
            agent.log_alpha = Float64(alpha_host[GPUStateT.GPU_LOG_ALPHA])

            var raw_count = Float64(host_episode_count[0])
            var recent_count = Int(raw_count) if raw_count > 0.0 and raw_count == raw_count else 0
            var raw_sum = Float64(host_reward_sum[0])
            var recent_sum = raw_sum if raw_sum == raw_sum else 0.0
            if recent_count > 0 and raw_sum != raw_sum:
                print(
                    "[MBPO WARN] NaN in episode reward sum at step "
                    + String(total_steps)
                    + " (count="
                    + String(recent_count)
                    + ")"
                )
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

            # Logger: record all MBPO-relevant metrics
            if logger:
                logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
                logger[].log_scalar(
                    "episodes", Float64(completed_episodes), total_steps
                )
                logger[].log_scalar(
                    "train_steps", Float64(total_train_steps), total_steps
                )
                logger[].log_scalar("alpha", agent.alpha, total_steps)
                logger[].log_scalar(
                    "rollout_length",
                    Float64(agent.rollout_length),
                    total_steps,
                )
                logger[].log_scalar(
                    "real_buffer_size",
                    Float64(gpu_state.buffer.size),
                    total_steps,
                )
                logger[].log_scalar(
                    "synth_buffer_size",
                    Float64(synth_buffer.size),
                    total_steps,
                )
                logger[].log_scalar(
                    "model_epoch", Float64(epoch), total_steps
                )

            # Clear progress bar, then full stats line
            if verbose:
                clear_progress_bar()
                print(
                    "MBPO-GPU | Step "
                    + String(total_steps)
                    + " / "
                    + String(num_steps)
                    + " | Ep: "
                    + String(completed_episodes)
                    + " | AvgR: "
                    + String(last_avg_reward)[byte=:7]
                    + " | Alpha: "
                    + String(agent.alpha)[byte=:6]
                    + " | Train: "
                    + String(total_train_steps)
                    + " | R: "
                    + String(gpu_state.buffer.size)
                    + " S: "
                    + String(synth_buffer.size)
                )

            # Autosave checkpoint
            if agent.checkpoint_every > 0 and total_steps >= agent.checkpoint_every and total_steps % agent.checkpoint_every < print_every:
                gpu_state.actor.download_to(cpu_state.actor, ctx)
                gpu_state.critics.download_to(cpu_state.critics, ctx)
                ctx.synchronize()
                agent.save_checkpoint(
                    agent.checkpoint_path
                    + "_step_"
                    + String(total_steps)
                    + ".ckpt"
                )

            next_print += print_every

    # Final sync
    ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
    ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
    ctx.synchronize()
    var final_raw = Float64(host_episode_count[0])
    var final_count = Int(final_raw) if final_raw > 0.0 and final_raw == final_raw else 0
    if final_count > 0:
        var final_avg = Float64(host_reward_sum[0]) / Float64(final_count)
        completed_episodes += final_count
        for _ in range(final_count):
            metrics.log_episode(completed_episodes, final_avg, 0, 0.0)

    # Download trained weights
    gpu_state.actor.download_to(cpu_state.actor, ctx)
    gpu_state.critics.download_to(cpu_state.critics, ctx)
    ctx.synchronize()

    if logger:
        logger[].flush()
    return metrics^
