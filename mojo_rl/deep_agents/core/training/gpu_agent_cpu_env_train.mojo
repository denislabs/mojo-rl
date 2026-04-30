"""GPU-agent / CPU-env hybrid off-policy training loop.

Built specifically as a diagnostic: lets us run our SAC/TD3/DDPG GPU agent
against a Python-driven CPU env (e.g. Gymnasium MuJoCo Hopper-v5) so we can
attribute training failure to "the env" or "the algorithm".

Mirrors `run_offpolicy_continuous_train_gpu` but:
  - Env step happens on CPU (any TerminationAwareEnv).
  - Obs / actions / rewards / done flow across the H↔D boundary each step.
  - No CUDA graphs (env can't be captured).
  - No GPU episode-tracking (counters live on CPU since we're already there).
  - No periodic eval (skip; run a separate eval script if needed).

The trait `TerminationAwareEnv` extends BoxContinuousActionEnv with
`was_terminated()` so the bootstrap mask is correct under time-limit
truncation (we don't want to drop the bootstrap when Gym truncates at 1000
steps).
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import UnsafePointer
from std.random import random_float64

from layout import Layout, LayoutTensor

from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    TerminationAwareEnv,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import dtype
from ..checkpoint_trait import Checkpointable
from .gpu_offpolicy_train import GPUOffPolicyAgent


# =============================================================================
# Hybrid Training Loop
# =============================================================================


def run_offpolicy_continuous_train_cpu_env_gpu_agent[
    E: TerminationAwareEnv,
    A: GPUOffPolicyAgent & Checkpointable,
    L: Logger = NoOpLogger,
](
    mut agent: A,
    ctx: DeviceContext,
    mut envs: List[UnsafePointer[E, MutAnyOrigin]],
    num_steps: Int,
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
    warmup_steps: Int = 1000,
    gradient_steps: Int = 0,
    sync_every: Int = 5000,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 50_000,
    environment_name: String = "Environment",
    algorithm_name: String = "GPUAgentCPUEnv",
    reward_scale: Float64 = 1.0,
) raises -> TrainingMetrics:
    """Train a GPU off-policy agent against CPU-stepped environments.

    Args:
        agent: Off-policy agent with GPU support (mutated in place).
        ctx: GPU device context.
        envs: List of CPU envs. Length determines `n_envs` for the loop.
        num_steps: Total env transitions (across all envs) before stopping.
        logger: Optional metrics logger.
        warmup_steps: Transitions of uniform-random actions before policy
            forward starts driving (per CleanRL convention).
        gradient_steps: Gradient updates per collection iteration. 0 → n_envs.
        sync_every: GPU→CPU param sync interval, in transitions.
        checkpoint_every: Checkpoint cadence (0 disables).
        checkpoint_path: Path prefix for checkpoints.
        verbose: Print status lines at print boundaries.
        print_every: Print/log cadence in transitions.
        algorithm_name: Used in the verbose status line.
        reward_scale: Multiplier applied to env rewards before storing in the
            replay buffer (matches `train_gpu` `reward_scale`).

    Returns:
        TrainingMetrics with per-episode statistics.
    """
    # n_envs is fixed at the agent's compile-time MAX_N_ENVS so that the
    # GPU-side trait methods (`select_actions_gpu[N]`, `gpu_store[N]`) get a
    # comptime parameter. Caller must pass exactly that many env pointers.
    comptime n_envs = A.MAX_N_ENVS
    comptime OBS_DIM = A.OBS_DIM
    comptime ACTION_DIM = A.ACTION_DIM

    if len(envs) != n_envs:
        raise Error(
            "run_offpolicy_continuous_train_cpu_env_gpu_agent: expected "
            + String(n_envs)
            + " envs (= agent.MAX_N_ENVS), got "
            + String(len(envs))
        )

    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # ------------------------------------------------------------------
    # GPU state + agent buffers
    # ------------------------------------------------------------------
    var gpu_state = agent.make_gpu_state(ctx)
    agent.upload_to_gpu(gpu_state, ctx)

    # ------------------------------------------------------------------
    # Per-iteration GPU buffers (sized to runtime n_envs).
    # We allocate once; n_envs is fixed for the whole run.
    # ------------------------------------------------------------------
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * OBS_DIM)
    var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Host mirror buffers for H↔D marshalling.
    var obs_host = ctx.enqueue_create_host_buffer[dtype](n_envs * OBS_DIM)
    var actions_host = ctx.enqueue_create_host_buffer[dtype](
        n_envs * ACTION_DIM
    )
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](n_envs)
    var terminated_host = ctx.enqueue_create_host_buffer[dtype](n_envs)

    # ------------------------------------------------------------------
    # Reset all envs and seed obs_buf
    # ------------------------------------------------------------------
    var per_env_reward = List[Float64]()
    var per_env_steps = List[Int]()
    for i in range(n_envs):
        var obs0 = envs[i][].reset_obs_list()
        for j in range(OBS_DIM):
            obs_host[i * OBS_DIM + j] = Scalar[dtype](Float64(obs0[j]))
        per_env_reward.append(0.0)
        per_env_steps.append(0)
    ctx.enqueue_copy(obs_buf, obs_host)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Counters / triggers
    # ------------------------------------------------------------------
    var total_steps = 0
    var total_train_steps = 0
    var completed_episodes = 0
    var recent_reward_sum: Float64 = 0.0
    var recent_episode_count = 0
    var last_avg_reward: Float64 = 0.0

    var next_print = print_every
    var next_sync = sync_every
    var next_checkpoint = checkpoint_every if checkpoint_every > 0 else (
        num_steps + 1
    )

    var grad_steps = gradient_steps if gradient_steps > 0 else n_envs
    var action_scale = agent.get_action_scale()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while total_steps < num_steps:
        # 1. Save current obs as prev_obs for the (s, a, r, s', done) tuple.
        ctx.enqueue_copy(prev_obs_buf, obs_buf)

        # 2. Action selection.
        if total_steps < warmup_steps:
            # Uniform-random in [-action_scale, action_scale] per CleanRL.
            for i in range(n_envs):
                for j in range(ACTION_DIM):
                    var u = random_float64() * 2.0 - 1.0
                    actions_host[i * ACTION_DIM + j] = Scalar[dtype](
                        u * action_scale
                    )
            ctx.enqueue_copy(actions_buf, actions_host)
        else:
            agent.sync_explore_counter(ctx, gpu_state)
            agent.select_actions_gpu[n_envs](
                ctx, gpu_state, obs_buf, actions_buf
            )
            # Materialize actions to host so CPU envs can step.
            ctx.enqueue_copy(actions_host, actions_buf)

        # Sync so actions_host is valid before stepping CPU envs.
        ctx.synchronize()

        # Bump the agent's transition counter (used by exploration RNG).
        agent.set_total_steps(agent.get_total_steps() + n_envs)

        # 3. Step each CPU env with its action.
        for i in range(n_envs):
            var action = List[Float64]()
            for j in range(ACTION_DIM):
                action.append(Float64(actions_host[i * ACTION_DIM + j]))

            var result = envs[i][].step_continuous_vec(action)
            var raw_reward = Float64(result[1])
            var done = result[2]
            var terminated = envs[i][].was_terminated()

            # Marshal step results (next obs, scaled reward, terminated mask).
            for j in range(OBS_DIM):
                obs_host[i * OBS_DIM + j] = Scalar[dtype](
                    Float64(result[0][j])
                )
            rewards_host[i] = Scalar[dtype](raw_reward * reward_scale)
            terminated_host[i] = Scalar[dtype](
                1.0 if terminated else 0.0
            )

            per_env_reward[i] = per_env_reward[i] + raw_reward
            per_env_steps[i] = per_env_steps[i] + 1

            if done:
                metrics.log_episode(
                    completed_episodes,
                    Scalar[DType.float64](per_env_reward[i]),
                    per_env_steps[i],
                    0.0,
                )
                recent_reward_sum += per_env_reward[i]
                recent_episode_count += 1
                completed_episodes += 1
                per_env_reward[i] = 0.0
                per_env_steps[i] = 0

        # 4. Upload step results, store the transition.
        # `obs_host` here holds the post-step (possibly terminal) next obs —
        # the correct value to store; the bootstrap is masked by terminated.
        ctx.enqueue_copy(rewards_buf, rewards_host)
        ctx.enqueue_copy(terminated_buf, terminated_host)
        ctx.enqueue_copy(obs_buf, obs_host)

        gpu_state.gpu_store[n_envs](
            ctx,
            prev_obs_buf,
            actions_buf,
            rewards_buf,
            obs_buf,
            terminated_buf,
        )

        # 5. Envs whose episode ended this iteration (per_env_steps cleared
        #    to 0 above): reset and overwrite obs_host[i] so the *next*
        #    iteration's prev_obs is the reset state, not the terminal one.
        var any_reset = False
        for i in range(n_envs):
            if per_env_steps[i] == 0:
                var obs_reset = envs[i][].reset_obs_list()
                for j in range(OBS_DIM):
                    obs_host[i * OBS_DIM + j] = Scalar[dtype](
                        Float64(obs_reset[j])
                    )
                any_reset = True
        if any_reset:
            ctx.enqueue_copy(obs_buf, obs_host)

        # 6. Train if buffer is ready.
        if gpu_state.gpu_buffer_is_ready():
            for _ in range(grad_steps):
                agent.do_gpu_train_step(ctx, gpu_state)
                total_train_steps += 1

        total_steps += n_envs

        # 7. Periodic GPU→CPU sync (so checkpoints / external eval see fresh weights).
        if total_steps >= next_sync:
            agent.download_from_gpu(gpu_state, ctx)
            next_sync += sync_every

        # 8. Periodic checkpoint.
        if checkpoint_every > 0 and total_steps >= next_checkpoint:
            if total_steps < next_sync - sync_every + n_envs:
                agent.download_from_gpu(gpu_state, ctx)
            agent.save_checkpoint(checkpoint_path)
            next_checkpoint += checkpoint_every

        # 9. Periodic print + log.
        if (
            verbose or (logger and logger[].is_active())
        ) and total_steps >= next_print:
            if recent_episode_count > 0:
                last_avg_reward = recent_reward_sum / Float64(
                    recent_episode_count
                )
            recent_reward_sum = 0.0
            recent_episode_count = 0

            if logger:
                logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
                logger[].log_scalar(
                    "episodes", Float64(completed_episodes), total_steps
                )
                logger[].log_scalar(
                    "train_steps",
                    Float64(total_train_steps),
                    total_steps,
                )

            if verbose:
                print(
                    algorithm_name
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
                )
            next_print += print_every

    # ------------------------------------------------------------------
    # Final sync + checkpoint
    # ------------------------------------------------------------------
    agent.download_from_gpu(gpu_state, ctx)
    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        agent.save_checkpoint(checkpoint_path)

    if logger and logger[].is_active():
        logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
        logger[].log_scalar(
            "episodes", Float64(completed_episodes), total_steps
        )
        logger[].log_scalar(
            "train_steps", Float64(total_train_steps), total_steps
        )
        logger[].flush()

    if verbose:
        print(
            algorithm_name
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
            + " (final)"
        )

    return metrics^
