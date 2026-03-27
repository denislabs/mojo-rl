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
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv, GPUContinuousEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import dtype
from ..checkpoint_trait import Checkpointable
from ..agents.mbpo_agent import MBPOAgent, GPUDynamicsEnsemble
from ..configs.mbpo_config import MBPOConfig
from ..kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_actions_kernel,
)
from ..utils import print_progress_bar, clear_progress_bar


def run_mbpo_train[
    E: BoxContinuousActionEnv,
    Config: MBPOConfig,
    L: Logger = NoOpLogger,
](
    mut agent: MBPOAgent[Config, L],
    mut cpu_state: MBPOAgent[Config, L].CPUStateType,
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
    while warmup_count < warmup_steps:
        var action = agent.random_action[E.dtype]()
        var result = env.step_continuous_vec(action)
        var next_obs = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_transition(
            cpu_state, warmup_obs, action, reward, next_obs, done
        )
        warmup_count += 1
        if done:
            warmup_obs = env.reset_obs_list()
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

            # Store in real buffer
            agent.store_transition(
                cpu_state, obs_f64, action, reward, next_obs, done
            )
            episode_reward += reward
            episode_steps += 1
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
](
    mut agent: MBPOAgent[Config, L],
    mut cpu_state: MBPOAgent[Config, L].CPUStateType,
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
    comptime n_envs = MBPOAgent[Config, L].GPU_N_ENVS

    var metrics = TrainingMetrics(
        algorithm_name="MBPO-GPU",
        environment_name=environment_name,
    )

    print("[MBPO-GPU] Creating GPU SAC state...")
    comptime GPUState = MBPOAgent[Config, L].GPUStateType
    var gpu_state = GPUState(ctx)
    print("[MBPO-GPU] GPU SAC state created")

    print("[MBPO-GPU] Creating GPU dynamics ensemble...")
    var gpu_dynamics = GPUDynamicsEnsemble[
        Config.DynamicsModel,
        Config.DynOpt,
        Config.ENSEMBLE_SIZE,
        Config.ELITE_SIZE,
        Config.obs_dim,
        Config.action_dim,
    ](ctx)
    print("[MBPO-GPU] GPU dynamics ensemble created")

    print("[MBPO-GPU] Uploading CPU ensemble weights...")
    gpu_dynamics.upload_from(cpu_state.dynamics, ctx)
    print("[MBPO-GPU] Ensemble weights uploaded")

    print("[MBPO-GPU] Uploading CPU SAC weights...")
    gpu_state.actor.upload_from(cpu_state.actor, ctx)
    gpu_state.critics.upload_from(cpu_state.critics, ctx)
    print("[MBPO-GPU] SAC weights uploaded")

    print("[MBPO-GPU] Allocating environment buffers (n_envs=" + String(n_envs) + ")...")
    var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    print("[MBPO-GPU] Environment buffers allocated")

    print("[MBPO-GPU] Allocating episode tracking buffers...")
    var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
    var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)
    var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
    var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)
    print("[MBPO-GPU] Episode tracking buffers allocated")

    print("[MBPO-GPU] Allocating env workspace...")
    var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
    if ws_size == 0:
        ws_size = 1
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
    if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
        E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)
    print("[MBPO-GPU] Env workspace allocated (size=" + String(ws_size) + ")")

    print("[MBPO-GPU] Initial reset...")
    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=0)
    print("[MBPO-GPU] Initial step...")
    E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
        ctx, states_buf, actions_buf, rewards_buf, dones_buf, terminated_buf,
        obs_buf, rng_seed=0, workspace_ptr=workspace_buf.unsafe_ptr(),
    )
    print("[MBPO-GPU] Initial reset/step done")

    ctx.enqueue_memset(episode_rewards_buf, 0)
    ctx.enqueue_memset(episode_steps_buf, 0)
    ctx.enqueue_memset(gpu_reward_sum_buf, 0)
    ctx.enqueue_memset(gpu_episode_count_buf, 0)
    ctx.synchronize()
    print("[MBPO-GPU] Initialization complete, entering training loop...")

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
    var step_seed: UInt32 = 42
    var completed_episodes = 0
    var last_avg_reward: Float64 = 0.0
    var next_print = print_every
    var next_model_train = agent.model_train_freq
    var epoch = 0

    while total_steps < num_steps:
        if total_steps % (n_envs * 100) == 0 and total_steps < warmup_steps:
            ctx.synchronize()  # DEBUG: flush async errors
            print("[MBPO-GPU] Warmup step " + String(total_steps) + "/" + String(warmup_steps))
        elif total_steps == warmup_steps:
            print("[MBPO-GPU] Warmup complete, starting training...")

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
        try:
            gpu_state.gpu_store[n_envs](
                ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, terminated_buf
            )
        except e:
            print("[MBPO-GPU] CRASH at gpu_store, step=" + String(total_steps))
            raise e^

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
        try:
            E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
                ctx, states_buf, dones_buf, rng_seed=UInt64(step_seed + 1),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )
            E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
                ctx, states_buf, obs_buf
            )
        except e:
            print("[MBPO-GPU] CRASH at selective_reset/extract_obs, step=" + String(total_steps))
            raise e^

        # GPU SAC gradient steps
        if total_steps >= warmup_steps and gpu_state.gpu_buffer_is_ready():
            for _ in range(agent.sac_updates_per_step):
                agent.do_gpu_train_step(ctx, gpu_state)
            agent.soft_update_targets_gpu(ctx, gpu_state)

        # Periodic dynamics training on CPU
        total_steps += n_envs
        step_seed += 1

        if total_steps >= next_model_train and total_steps >= warmup_steps:
            # GPU dynamics training: data stays on GPU
            gpu_dynamics.train_on_buffer[MBPOAgent[Config, L].GPU_BUF_CAP](
                ctx, gpu_state.buffer,
            )
            agent.update_rollout_length(epoch)
            epoch += 1

            # GPU model rollouts: actor + dynamics forward on GPU,
            # synthetic transitions stored directly in GPU buffer
            agent.do_model_rollouts_gpu(ctx, gpu_dynamics, gpu_state)

            if verbose:
                print(
                    "  Model trained (GPU) | Step "
                    + String(total_steps)
                    + " | Buffer: "
                    + String(gpu_state.buffer.size)
                    + " | Rollout: "
                    + String(agent.rollout_length)
                )

            next_model_train += agent.model_train_freq

        # Print / log
        if (verbose or (logger and logger[].is_active())) and total_steps >= next_print:
            ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
            ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
            ctx.synchronize()

            var recent_count = Int(host_episode_count[0])
            var recent_sum = Float64(host_reward_sum[0])
            completed_episodes += recent_count

            if recent_count > 0:
                last_avg_reward = recent_sum / Float64(recent_count)
                for _ in range(recent_count):
                    metrics.log_episode(
                        completed_episodes, last_avg_reward, 0, 0.0
                    )

            ctx.enqueue_memset(gpu_reward_sum_buf, 0)
            ctx.enqueue_memset(gpu_episode_count_buf, 0)

            if logger:
                logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
                logger[].log_scalar(
                    "episodes", Float64(completed_episodes), total_steps
                )
                logger[].log_scalar(
                    "alpha", agent.alpha, total_steps
                )
                logger[].log_scalar(
                    "rollout_length",
                    Float64(agent.rollout_length),
                    total_steps,
                )
                logger[].log_scalar(
                    "gpu_buffer_size",
                    Float64(gpu_state.buffer.size),
                    total_steps,
                )

            if verbose:
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
                )

            # Autosave checkpoint
            if agent.checkpoint_every > 0 and total_steps % agent.checkpoint_every < print_every:
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
    var final_count = Int(host_episode_count[0])
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
