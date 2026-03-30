"""GPU-accelerated off-policy training infrastructure.

Provides two traits and shared GPU training loop functions:

  GPUOffPolicyState  — algorithm-specific GPU buffer container
                       (networks, replay buffer, scratch buffers).
                       Implements gpu_store[N_ENVS] and gpu_buffer_is_ready.

  GPUOffPolicyAgent  — unified CPU+GPU agent trait.
                       Has comptime GPUStateType: GPUOffPolicyState.
                       Implements make_gpu_state, upload_to_gpu,
                       download_from_gpu, select_actions_gpu,
                       do_gpu_train_step, soft_update_targets_gpu.

Design principles:
  - The shared loop owns: env state/obs/action/reward/done buffers + step tracking.
  - GPUOffPolicyState owns: GPU replay buffer + GPU network states + scratch buffers.
  - GPUOffPolicyAgent (the CPU struct) owns: CPU network states + hyperparameters.
  - Methods take (mut self, ctx, mut gpu_state) so agent hyperparams stay on the
    CPU struct and GPU buffers live in the state — mirroring the Env/State pattern.
  - comptime MAX_N_ENVS on the agent fixes buffer sizes at compile time.

Step counting:
  - total_steps counts TOTAL env transitions (n_envs per loop iteration),
    matching on-policy convention.  num_steps, warmup_steps, print_every,
    sync_every, checkpoint_every are all in transition units.

Usage:
    # 1. Define a GPU state container
    struct MyGPUState[...](GPUOffPolicyState):
        var actor_online: GPUNetworkState[...]
        var buffer: GPUReplayBuffer[...]
        var scratch: DeviceBuffer[dtype]
        ...
        def gpu_store[N](mut self, ctx, prev_obs, act, rew, obs, done): ...
        def gpu_buffer_is_ready(self) -> Bool: ...

    # 2. Make your agent implement GPUOffPolicyAgent
    struct MyAgent[..., max_n_envs: Int = 64](OffPolicyAgent & GPUOffPolicyAgent):
        comptime MAX_N_ENVS: Int = max_n_envs
        comptime GPUStateType = MyGPUState[...]
        ...
        def make_gpu_state(self, ctx) raises -> MyGPUState[...]: ...
        def upload_to_gpu(self, mut gpu_state: MyGPUState[...], ctx) raises: ...
        def download_from_gpu(mut self, gpu_state: MyGPUState[...], ctx) raises: ...
        def select_actions_gpu[N](mut self, ctx, mut gpu_state, obs, act) raises: ...
        def do_gpu_train_step(mut self, ctx, mut gpu_state) raises: ...
        def soft_update_targets_gpu(mut self, ctx, mut gpu_state) raises: ...

    # 3. Train
    var metrics = run_offpolicy_continuous_train_gpu[MyEnv, MyAgent](
        agent, ctx, num_steps=1_000_000, warmup_steps=25_000,
    )
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from ..checkpoint_trait import Checkpointable
from mojo_rl.core import (
    TrainingMetrics,
    GPUDiscreteEnv,
    GPUContinuousEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_actions_kernel,
    uniform_random_discrete_actions_kernel,
    _extract_obs_kernel,
)
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)
from mojo_rl.deep_agents.core.perf_timer import PerfTimer


# =============================================================================
# GPUOffPolicyState Trait
# =============================================================================


trait GPUOffPolicyState(ImplicitlyDestructible):
    """GPU-side buffer container for off-policy agents.

    Holds all GPU-resident state: network DeviceBuffers (online + target),
    GPU replay buffer, and algorithm-specific scratch buffers.

    The shared training loop calls gpu_store and gpu_buffer_is_ready directly
    on the state object, so these must be implemented by each algorithm.
    The algorithm-specific logic (train step, action selection) lives on the
    GPUOffPolicyAgent and receives the state as a parameter.
    """

    def gpu_store[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        rewards_buf: DeviceBuffer[dtype],
        obs_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Push N_ENVS transitions into the GPU replay buffer.

        Args:
            ctx: GPU device context.
            prev_obs_buf: Previous observations [N_ENVS * OBS_DIM].
            actions_buf: Actions taken [N_ENVS * ACTION_DIM].
            rewards_buf: Rewards received [N_ENVS].
            obs_buf: Next observations [N_ENVS * OBS_DIM].
            dones_buf: Done flags [N_ENVS] (1.0 = done).
        """
        ...

    def gpu_buffer_is_ready(self) -> Bool:
        """Return True if the GPU replay buffer has enough samples to train."""
        ...


# =============================================================================
# GPUOffPolicyAgent Trait
# =============================================================================


trait GPUOffPolicyAgent:
    """Off-policy agent with GPU-accelerated training.

    The agent (CPU struct) owns hyperparameters and CPU network states.
    GPU buffers (networks, replay, scratch) live in GPUStateType.
    The shared training loop creates the GPU state once via make_gpu_state,
    then passes it to every GPU method call.

    Compile-time constants:
        OBS_DIM:          Observation space dimension.
        ACTION_DIM:       Action space dimension.
        BUFFER_CAPACITY:  GPU replay buffer capacity.
        MAX_N_ENVS:       Max parallel environments (sizes GPU exploration buffers).
        GPUStateType:     Concrete type implementing GPUOffPolicyState.
    """

    comptime OBS_DIM: Int
    """Observation space dimension (must match GPUContinuousEnv.OBS_DIM)."""

    comptime ACTION_DIM: Int
    """Action space dimension (must match GPUContinuousEnv.ACTION_DIM)."""

    comptime BUFFER_CAPACITY: Int
    """GPU replay buffer capacity."""

    comptime MAX_N_ENVS: Int
    """Max parallel environments — sizes exploration buffers in GPUStateType."""

    comptime GPUStateType: GPUOffPolicyState
    """Concrete GPU state type holding all device buffers for this algorithm."""

    def get_action_scale(self) -> Float64:
        """Return action range bound [-scale, scale] for warmup random actions.
        """
        ...

    def get_total_steps(self) -> Int:
        """Return total env transitions collected so far."""
        ...

    def set_total_steps(mut self, steps: Int):
        """Set total env transitions counter (for exploration RNG seeding)."""
        ...

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for this agent (networks, replay, scratch).

        Called once at the start of GPU training. Does NOT upload CPU weights —
        call upload_to_gpu separately after make_gpu_state.

        Args:
            ctx: GPU device context.

        Returns:
            Freshly allocated GPU state container.
        """
        ...

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights and replay buffer to GPU.

        Args:
            gpu_state: GPU state to populate (mutated in-place).
            ctx: GPU device context.
        """
        ...

    def download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states.

        Args:
            gpu_state: GPU state to read from.
            ctx: GPU device context.
        """
        ...

    def select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward pass on GPU for all N_ENVS environments in parallel.

        For deterministic agents (DDPG/TD3): add exploration noise on GPU
        using persistent RNG state stored in gpu_state.
        For stochastic agents (SAC): reparameterize.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with actor network + exploration RNG buffers.
            obs_buf: Observations buffer [N_ENVS * OBS_DIM].
            actions_buf: Output actions buffer [N_ENVS * ACTION_DIM].
        """
        ...

    def do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Sample from std.gpu replay buffer and perform one full gradient update.

        Typical phases: sample → target Q → critic update → actor update.
        Uses self for hyperparams (gamma, tau, etc.) and gpu_state for buffers.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with replay buffer, networks, scratch buffers.
        """
        ...

    def soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update all target networks on GPU: θ_t ← τ*θ + (1-τ)*θ_t.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with online and target network buffers.
        """
        ...

    def decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        """Decay exploration rate based on training progress.

        Called once per collection step in GPU training loops.
        No-op for agents without epsilon-greedy exploration (SAC, TD3, DDPG).

        Args:
            total_steps: Total env transitions so far.
            num_steps: Total planned transitions (for linear schedule).
        """
        ...


# =============================================================================
# Shared GPU Training Loop — Continuous Actions
# =============================================================================


def run_offpolicy_continuous_train_gpu[
    E: GPUContinuousEnv,
    A: GPUOffPolicyAgent & Checkpointable,
    PROFILE: Int = 0,
    L: Logger = NoOpLogger,
    CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
](
    mut agent: A,
    ctx: DeviceContext,
    num_steps: Int,
    mut timer: PerfTimer[PROFILE >= 1],
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
    warmup_steps: Int = 1000,
    gradient_steps: Int = 0,
    sync_every: Int = 5000,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 50_000,
    environment_name: String = "Environment",
    algorithm_name: String = "GPUOffPolicy",
    target_total_steps: Int = 0,
) raises -> TrainingMetrics:
    """Shared GPU training loop for continuous-action off-policy agents.

    Responsibility split:
      Loop:      allocates env buffers, drives E.step_kernel_gpu / reset kernels,
                 calls gpu_state.gpu_store, checks gpu_state.gpu_buffer_is_ready.
      Agent:     implements action selection, training step, soft updates.
      GPU state: holds all device buffers (networks, replay, scratch).

    The number of parallel environments is fixed at compile time as A.MAX_N_ENVS,
    so env buffer sizes are fully known without runtime heap allocation.

    Step counting uses total env transitions (n_envs per loop iteration),
    matching on-policy convention. All step-based parameters use this unit.

    Each iteration collects n_envs transitions, then performs `gradient_steps`
    training steps to maintain the correct replay ratio. Default (0) uses
    n_envs gradient steps per iteration, matching the 1:1 replay ratio of
    single-env CleanRL implementations.

    Parameters:
        E: GPU environment type implementing GPUContinuousEnv.
        A: Agent type implementing GPUOffPolicyAgent.
        PROFILE: Whether to profile the training loop.
        L: Logger for diagnostics.
        CurriculumType: Curriculum scheduler type.

    Args:
        agent: Off-policy agent with GPU support (updated in-place).
        ctx: GPU device context.
        num_steps: Total env transitions across all parallel envs.
        timer: PerfTimer to add slots to.
        logger: Logger for diagnostics.
        warmup_steps: Transitions before training starts (default: 1000).
        gradient_steps: Training steps per env collection iteration.
            Default 0 = n_envs (1:1 replay ratio with CleanRL convention).
        sync_every: GPU→CPU parameter sync interval in transitions (default: 5000).
        checkpoint_every: Checkpoint interval in transitions (0 to disable).
        checkpoint_path: Path prefix for checkpoints.
        verbose: Print progress (default: False).
        print_every: Print interval in transitions (default: 50000).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        target_total_steps: Total steps for curriculum/annealing progress (default: 0 = disabled).

    Returns:
        TrainingMetrics with episode-level statistics.
    """
    comptime n_envs = A.MAX_N_ENVS

    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # ------------------------------------------------------------------
    # Create GPU state and upload CPU weights
    # ------------------------------------------------------------------
    var gpu_state = agent.make_gpu_state(ctx)
    agent.upload_to_gpu(gpu_state, ctx)

    # ------------------------------------------------------------------
    # Allocate environment buffers (loop-owned, comptime sizes)
    # ------------------------------------------------------------------
    var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Episode tracking: per-env accumulators + GPU-side stats
    var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
    var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)

    # Host buffers for periodic readback (only at print boundaries)
    var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
    var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)

    # Workspace buffer (shared model state for physics envs)
    var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
    if ws_size == 0:
        ws_size = 1
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)

    if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
        E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)

    # ------------------------------------------------------------------
    # Initial reset
    # ------------------------------------------------------------------
    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=0)
    E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM, E.ACTION_DIM](
        ctx,
        states_buf,
        actions_buf,
        rewards_buf,
        dones_buf,
        terminated_buf,
        obs_buf,
        rng_seed=0,
        workspace_ptr=workspace_buf.unsafe_ptr(),
    )

    # Initialize episode tracking
    ctx.enqueue_memset(episode_rewards_buf, 0)
    ctx.enqueue_memset(episode_steps_buf, 0)
    ctx.enqueue_memset(gpu_reward_sum_buf, 0)
    ctx.enqueue_memset(gpu_episode_count_buf, 0)

    # ------------------------------------------------------------------
    # Kernel wrappers (defined once outside the loop)
    # ------------------------------------------------------------------
    comptime tpb = 256
    comptime env_blocks = (n_envs + tpb - 1) // tpb

    comptime accum_rewards_wrapper = accumulate_rewards_kernel[dtype, n_envs]
    comptime incr_steps_wrapper = increment_steps_kernel[dtype, n_envs]
    comptime log_reset_wrapper = log_and_reset_completed_kernel[dtype, n_envs]

    from layout import Layout, LayoutTensor

    # Resolve gradient_steps: 0 means n_envs (1:1 replay ratio)
    var grad_steps = gradient_steps
    if grad_steps <= 0:
        grad_steps = n_envs

    var total_steps = 0
    var total_train_steps = 0
    var step_seed: UInt32 = 42
    var completed_episodes = 0
    var last_avg_reward: Float64 = 0.0

    # Threshold-based triggers (avoids modular alignment issues with n_envs)
    var next_print = print_every
    var next_sync = sync_every
    var next_checkpoint = checkpoint_every

    # Progress bar: ~20 updates per print interval
    var progress_interval = print_every // 20
    if progress_interval < n_envs:
        progress_interval = n_envs
    var next_progress = progress_interval

    # Warmup kernel wrapper (uniform random actions in [-action_scale, action_scale])
    comptime act_tpb = 256
    comptime act_blocks = (n_envs * E.ACTION_DIM + act_tpb - 1) // act_tpb
    comptime warmup_kernel = uniform_random_actions_kernel[
        dtype, n_envs, E.ACTION_DIM
    ]
    var action_scale_val = Scalar[dtype](agent.get_action_scale())

    while total_steps < num_steps:
        # Curriculum update (once per step batch, before environment step)
        comptime if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
            var progress = Float64(0.0)
            if target_total_steps > 0:
                progress = Float64(total_steps) / Float64(target_total_steps)
            if progress > 1.0:
                progress = 1.0
            var curriculum_values = CurriculumType.get_params[dtype](
                Scalar[dtype](progress)
            )
            E.update_curriculum_gpu(ctx, workspace_buf, curriculum_values)

        # ------------------------------------------------------------------
        # Save current obs as prev_obs (before environment step)
        # ------------------------------------------------------------------
        comptime if PROFILE >= 1:
            timer.sync_and_mark(ctx)
        ctx.enqueue_copy(prev_obs_buf, obs_buf)
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(0, ctx)
            timer.mark()

        # ------------------------------------------------------------------
        # Select actions: warmup uses uniform random, then agent's policy
        # ------------------------------------------------------------------
        if total_steps < warmup_steps:
            # Uniform random actions matching CleanRL's env.action_space.sample()
            var act_t = LayoutTensor[
                dtype,
                Layout.row_major(n_envs, E.ACTION_DIM),
                MutAnyOrigin,
            ](actions_buf.unsafe_ptr())
            var warmup_seed = Scalar[DType.uint32](step_seed)
            ctx.enqueue_function[warmup_kernel, warmup_kernel](
                act_t,
                action_scale_val,
                warmup_seed,
                grid_dim=(act_blocks,),
                block_dim=(act_tpb,),
            )
        else:
            agent.select_actions_gpu[n_envs](
                ctx, gpu_state, obs_buf, actions_buf
            )
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(1, ctx)
            timer.mark()

        # Update agent's total_steps so exploration RNG seed varies each call
        agent.set_total_steps(agent.get_total_steps() + n_envs)

        # ------------------------------------------------------------------
        # Step environment (obs_buf now holds next observations)
        # ------------------------------------------------------------------
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
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(2, ctx)
            timer.mark()

        # DEBUG: one-shot dones check right after step (before anything clears it)
        if total_steps == 0:
            var _dd = ctx.enqueue_create_host_buffer[dtype](n_envs)
            var _dt = ctx.enqueue_create_host_buffer[dtype](n_envs)
            var _dr = ctx.enqueue_create_host_buffer[dtype](n_envs)
            ctx.enqueue_copy(_dd, dones_buf)
            ctx.enqueue_copy(_dt, terminated_buf)
            ctx.enqueue_copy(_dr, rewards_buf)
            ctx.synchronize()
            print("  [DEBUG step 0] dones:", end="")
            for _i in range(min(n_envs, 8)):
                print(" ", Float64(_dd[_i]), end="")
            print()
            print("  [DEBUG step 0] term: ", end="")
            for _i in range(min(n_envs, 8)):
                print(" ", Float64(_dt[_i]), end="")
            print()
            print("  [DEBUG step 0] rew:  ", end="")
            for _i in range(min(n_envs, 8)):
                print(" ", Float64(_dr[_i]), end="")
            print()

        # ------------------------------------------------------------------
        # Store transitions: (prev_obs, action, reward, next_obs, terminated)
        # Use terminated_buf (not dones_buf) so TD targets bootstrap on truncation
        # ------------------------------------------------------------------
        gpu_state.gpu_store[n_envs](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, terminated_buf
        )
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(3, ctx)

        # ------------------------------------------------------------------
        # Accumulate episode rewards/steps + log completed (all on GPU)
        # ------------------------------------------------------------------
        var episode_rewards_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var episode_steps_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var reward_sum_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_reward_sum_buf.unsafe_ptr())
        var episode_count_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_episode_count_buf.unsafe_ptr())

        ctx.enqueue_function[accum_rewards_wrapper, accum_rewards_wrapper](
            episode_rewards_t,
            rewards_t,
            grid_dim=(env_blocks,),
            block_dim=(tpb,),
        )
        ctx.enqueue_function[incr_steps_wrapper, incr_steps_wrapper](
            episode_steps_t,
            grid_dim=(env_blocks,),
            block_dim=(tpb,),
        )

        # Log completed episodes to GPU-side stats and reset per-env counters
        ctx.enqueue_function[log_reset_wrapper, log_reset_wrapper](
            dones_t,
            episode_rewards_t,
            episode_steps_t,
            reward_sum_t,
            episode_count_t,
            grid_dim=(1,),
            block_dim=(1,),
        )
        comptime if PROFILE >= 1:
            timer.sync_and_mark(ctx)
            timer.accumulate(4)
            timer.mark()

        # ------------------------------------------------------------------
        # Reset done environments (reuse model from workspace)
        # ------------------------------------------------------------------
        E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
            ctx,
            states_buf,
            dones_buf,
            rng_seed=UInt64(step_seed + 1),
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )
        # Update obs_buf for reset environments — must happen after selective_reset
        # so that the next step's prev_obs copy sees the initial obs of the new
        # episode, not the terminal obs of the previous one.
        E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
            ctx, states_buf, obs_buf
        )
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(5, ctx)
            timer.mark()

        # ------------------------------------------------------------------
        # Training steps (gradient_steps per env collection iteration)
        # ------------------------------------------------------------------
        if total_steps >= warmup_steps and gpu_state.gpu_buffer_is_ready():
            for _ in range(grad_steps):
                agent.do_gpu_train_step(ctx, gpu_state)
            agent.soft_update_targets_gpu(ctx, gpu_state)
            total_train_steps += grad_steps
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(6, ctx)

        # ------------------------------------------------------------------
        # Periodic GPU→CPU sync (for evaluate() and checkpointing)
        # ------------------------------------------------------------------
        if total_steps >= next_sync:
            agent.download_from_gpu(gpu_state, ctx)
            next_sync += sync_every

        # Periodic checkpoint
        if checkpoint_every > 0 and total_steps >= next_checkpoint:
            if total_steps < next_sync - sync_every + n_envs:
                agent.download_from_gpu(gpu_state, ctx)
            agent.save_checkpoint(checkpoint_path)
            next_checkpoint += checkpoint_every

        total_steps += n_envs
        step_seed += 1

        # ------------------------------------------------------------------
        # Progress bar (no GPU sync, pure CPU counters)
        # Shows progress within the current print interval (0% → 100%)
        # ------------------------------------------------------------------
        if verbose and total_steps >= next_progress:
            var interval_start = next_print - print_every
            print_progress_bar(
                total_steps - interval_start,
                print_every,
                total_train_steps,
                algorithm_name,
            )
            next_progress += progress_interval

        # ------------------------------------------------------------------
        # Collect episode stats + print/log at print boundaries
        # ------------------------------------------------------------------
        if (
            verbose or (logger and logger[].is_active())
        ) and total_steps >= next_print:
            # Download GPU-side episode stats (only sync point for tracking)
            ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
            ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
            ctx.synchronize()

            # DEBUG: download dones_buf to check if it's ever non-zero
            var _dbg_dones = ctx.enqueue_create_host_buffer[dtype](n_envs)
            ctx.enqueue_copy(_dbg_dones, dones_buf)
            ctx.synchronize()
            var _dbg_done_sum: Float64 = 0
            for _di in range(n_envs):
                _dbg_done_sum += Float64(_dbg_dones[_di])
            print(
                "  [DEBUG] ep_count=",
                Float64(host_episode_count[0]),
                " dones_sum=",
                _dbg_done_sum,
            )

            var recent_count = Int(host_episode_count[0])
            var recent_sum = Float64(host_reward_sum[0])
            completed_episodes += recent_count

            if recent_count > 0:
                last_avg_reward = recent_sum / Float64(recent_count)
                # Log synthetic episodes for TrainingMetrics
                for _ in range(recent_count):
                    metrics.log_episode(
                        completed_episodes, last_avg_reward, 0, 0.0
                    )

            # Reset GPU-side accumulators for next interval
            ctx.enqueue_memset(gpu_reward_sum_buf, 0)
            ctx.enqueue_memset(gpu_episode_count_buf, 0)

            # Logger: record metrics
            if logger:
                logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
                logger[].log_scalar(
                    "episodes", Float64(completed_episodes), total_steps
                )
                logger[].log_scalar(
                    "train_steps", Float64(total_train_steps), total_steps
                )

            # Clear progress bar, then full stats line
            if verbose:
                clear_progress_bar()
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

    # Final sync: download episode stats + params
    ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
    ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
    ctx.synchronize()

    var final_count = Int(host_episode_count[0])
    var final_sum = Float64(host_reward_sum[0])
    completed_episodes += final_count
    if final_count > 0:
        last_avg_reward = final_sum / Float64(final_count)
        for _ in range(final_count):
            metrics.log_episode(completed_episodes, last_avg_reward, 0, 0.0)

    comptime if PROFILE >= 1:
        timer.sync_and_mark(ctx)
    agent.download_from_gpu(gpu_state, ctx)
    comptime if PROFILE >= 1:
        timer.accumulate(7)

    # Final logger flush + print
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
        clear_progress_bar()
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
            + " [DONE]"
        )

    return metrics^


# =============================================================================
# Shared GPU Training Loop — Discrete Actions
# =============================================================================


def run_offpolicy_discrete_train_gpu[
    E: GPUDiscreteEnv,
    A: GPUOffPolicyAgent & Checkpointable,
    PROFILE: Int = 0,
    L: Logger = NoOpLogger,
    CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
](
    mut agent: A,
    ctx: DeviceContext,
    num_steps: Int,
    mut timer: PerfTimer[PROFILE >= 1],
    logger: UnsafePointer[L, MutAnyOrigin] = UnsafePointer[L, MutAnyOrigin](),
    warmup_steps: Int = 1000,
    gradient_steps: Int = 0,
    sync_every: Int = 5000,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 50_000,
    environment_name: String = "Environment",
    algorithm_name: String = "GPUOffPolicy",
    target_total_steps: Int = 0,
) raises -> TrainingMetrics:
    """Shared GPU training loop for discrete-action off-policy agents (DQN etc.).

    Same pattern as run_offpolicy_continuous_train_gpu but for
    GPUDiscreteEnv + discrete actions (actions stored as Float32 indices).

    Step counting uses total env transitions (n_envs per loop iteration).

    Parameters:
        E: GPU environment type implementing GPUDiscreteEnv.
        A: Agent type implementing GPUOffPolicyAgent.
        PROFILE: Whether to profile the training loop.
        L: Logger type for diagnostics.
        CurriculumType: Curriculum scheduler type.

    Args:
        agent: Off-policy agent with GPU support (updated in-place).
        ctx: GPU device context.
        num_steps: Total env transitions across all parallel envs.
        timer: PerfTimer to add slots to.
        logger: Optional metrics logger pointer (default: null = no logging).
        warmup_steps: Transitions before training starts (default: 1000).
        gradient_steps: Training steps per env collection iteration.
            Default 0 = n_envs (1:1 replay ratio with CleanRL convention).
        sync_every: GPU→CPU parameter sync interval in transitions (default: 5000).
        checkpoint_every: Checkpoint interval in transitions (0 to disable).
        checkpoint_path: Path prefix for checkpoints.
        verbose: Print progress (default: False).
        print_every: Print interval in transitions (default: 50000).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.
        target_total_steps: Total steps for curriculum/annealing progress (default: 0 = disabled).

    Returns:
        TrainingMetrics with episode-level statistics.
    """
    comptime n_envs = A.MAX_N_ENVS

    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # Create GPU state and upload CPU weights
    var gpu_state = agent.make_gpu_state(ctx)
    agent.upload_to_gpu(gpu_state, ctx)

    # Allocate environment buffers
    var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var prev_obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    # For discrete envs, actions are Float32 indices (shape: [n_envs])
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Episode tracking: per-env accumulators + GPU-side stats
    var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
    var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)

    # Host buffers for periodic readback (only at print boundaries)
    var host_reward_sum = ctx.enqueue_create_host_buffer[dtype](1)
    var host_episode_count = ctx.enqueue_create_host_buffer[dtype](1)

    var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
    if ws_size == 0:
        ws_size = 1
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)

    if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
        E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)

    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=0)
    E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
        ctx,
        states_buf,
        actions_buf,
        rewards_buf,
        dones_buf,
        terminated_buf,
        obs_buf,
        rng_seed=0,
        workspace_ptr=workspace_buf.unsafe_ptr(),
    )

    # Initialize episode tracking
    ctx.enqueue_memset(episode_rewards_buf, 0)
    ctx.enqueue_memset(episode_steps_buf, 0)
    ctx.enqueue_memset(gpu_reward_sum_buf, 0)
    ctx.enqueue_memset(gpu_episode_count_buf, 0)

    # Kernel wrappers
    comptime tpb = 256
    comptime env_blocks = (n_envs + tpb - 1) // tpb

    comptime accum_rewards_wrapper = accumulate_rewards_kernel[dtype, n_envs]
    comptime incr_steps_wrapper = increment_steps_kernel[dtype, n_envs]
    comptime log_reset_wrapper = log_and_reset_completed_kernel[dtype, n_envs]

    from layout import Layout, LayoutTensor

    # Discrete warmup kernel wrapper (uniform random action indices)
    comptime act_tpb = 256
    comptime act_blocks = (n_envs + act_tpb - 1) // act_tpb
    comptime discrete_warmup_kernel = uniform_random_discrete_actions_kernel[
        dtype, n_envs, E.NUM_ACTIONS
    ]

    # Extract obs from state after selective reset (obs = state[:OBS_DIM])
    comptime extract_obs_after_reset = _extract_obs_kernel[
        dtype, n_envs, E.STATE_SIZE, E.OBS_DIM
    ]

    # Resolve gradient_steps: 0 means n_envs (1:1 replay ratio)
    var grad_steps = gradient_steps
    if grad_steps <= 0:
        grad_steps = n_envs

    var total_steps = 0
    var total_train_steps = 0
    var step_seed: UInt32 = 42
    var completed_episodes = 0
    var last_avg_reward: Float64 = 0.0

    # Threshold-based triggers
    var next_print = print_every
    var next_sync = sync_every
    var next_checkpoint = checkpoint_every

    # Progress bar: ~20 updates per print interval
    var progress_interval = print_every // 20
    if progress_interval < n_envs:
        progress_interval = n_envs
    var next_progress = progress_interval

    while total_steps < num_steps:
        # Curriculum update (once per step batch, before environment step)
        comptime if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
            var progress = Float64(0.0)
            if target_total_steps > 0:
                progress = Float64(total_steps) / Float64(target_total_steps)
            if progress > 1.0:
                progress = 1.0
            var curriculum_values = CurriculumType.get_params[dtype](
                Scalar[dtype](progress)
            )
            E.update_curriculum_gpu(ctx, workspace_buf, curriculum_values)

        comptime if PROFILE >= 1:
            timer.sync_and_mark(ctx)
        ctx.enqueue_copy(prev_obs_buf, obs_buf)
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(0, ctx)
            timer.mark()

        # --- Action selection: warmup uses uniform random, then agent's policy ---
        if total_steps < warmup_steps:
            var act_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](actions_buf.unsafe_ptr())
            var warmup_seed = Scalar[DType.uint32](step_seed)
            ctx.enqueue_function[
                discrete_warmup_kernel, discrete_warmup_kernel
            ](
                act_t,
                warmup_seed,
                grid_dim=(act_blocks,),
                block_dim=(act_tpb,),
            )
        else:
            agent.select_actions_gpu[n_envs](
                ctx, gpu_state, obs_buf, actions_buf
            )
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(1, ctx)
            timer.mark()

        # --- Environment step ---
        E.step_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
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
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(2, ctx)
            timer.mark()

        # --- Replay buffer store ---
        # Use terminated_buf (not dones_buf) so TD targets bootstrap on truncation
        gpu_state.gpu_store[n_envs](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, terminated_buf
        )
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(3, ctx)

        # ------------------------------------------------------------------
        # Accumulate episode rewards/steps + log completed (all on GPU)
        # ------------------------------------------------------------------
        var episode_rewards_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var episode_steps_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var reward_sum_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_reward_sum_buf.unsafe_ptr())
        var episode_count_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_episode_count_buf.unsafe_ptr())

        ctx.enqueue_function[accum_rewards_wrapper, accum_rewards_wrapper](
            episode_rewards_t,
            rewards_t,
            grid_dim=(env_blocks,),
            block_dim=(tpb,),
        )
        ctx.enqueue_function[incr_steps_wrapper, incr_steps_wrapper](
            episode_steps_t,
            grid_dim=(env_blocks,),
            block_dim=(tpb,),
        )

        ctx.enqueue_function[log_reset_wrapper, log_reset_wrapper](
            dones_t,
            episode_rewards_t,
            episode_steps_t,
            reward_sum_t,
            episode_count_t,
            grid_dim=(1,),
            block_dim=(1,),
        )
        comptime if PROFILE >= 1:
            timer.sync_and_mark(ctx)
            timer.accumulate(4)
            timer.mark()

        # Reuse model from workspace for reset
        E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
            ctx,
            states_buf,
            dones_buf,
            rng_seed=UInt64(step_seed + 1),
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )
        # Update obs_buf for reset environments — must happen after selective_reset
        # so that the next step's prev_obs copy sees the initial obs of the new
        # episode, not the terminal obs of the previous one.
        var states_t_reset = LayoutTensor[
            dtype, Layout.row_major(n_envs, E.STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var obs_t_reset = LayoutTensor[
            dtype, Layout.row_major(n_envs, E.OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        ctx.enqueue_function[extract_obs_after_reset, extract_obs_after_reset](
            states_t_reset,
            obs_t_reset,
            grid_dim=(env_blocks,),
            block_dim=(tpb,),
        )
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(5, ctx)
            timer.mark()

        # ------------------------------------------------------------------
        # Training steps (gradient_steps per env collection iteration)
        # ------------------------------------------------------------------
        if total_steps >= warmup_steps and gpu_state.gpu_buffer_is_ready():
            for _ in range(grad_steps):
                agent.do_gpu_train_step(ctx, gpu_state)
            agent.soft_update_targets_gpu(ctx, gpu_state)
            total_train_steps += grad_steps
        comptime if PROFILE >= 1:
            timer.sync_and_accumulate(6, ctx)

        if total_steps >= next_sync:
            agent.download_from_gpu(gpu_state, ctx)
            next_sync += sync_every

        # Periodic checkpoint
        if checkpoint_every > 0 and total_steps >= next_checkpoint:
            if total_steps < next_sync - sync_every + n_envs:
                agent.download_from_gpu(gpu_state, ctx)
            agent.save_checkpoint(checkpoint_path)
            next_checkpoint += checkpoint_every

        total_steps += n_envs
        step_seed += 1

        # Decay exploration (e.g. epsilon for DQN, no-op for continuous agents)
        agent.decay_explore_gpu(total_steps, num_steps)

        # Progress bar (no GPU sync, pure CPU counters)
        # Shows progress within the current print interval (0% → 100%)
        if verbose and total_steps >= next_progress:
            var interval_start = next_print - print_every
            print_progress_bar(
                total_steps - interval_start,
                print_every,
                total_train_steps,
                algorithm_name,
            )
            next_progress += progress_interval

        if (
            verbose or (logger and logger[].is_active())
        ) and total_steps >= next_print:
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

            # Logger: record metrics
            if logger:
                logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
                logger[].log_scalar(
                    "episodes", Float64(completed_episodes), total_steps
                )
                logger[].log_scalar(
                    "train_steps", Float64(total_train_steps), total_steps
                )

            if verbose:
                clear_progress_bar()
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

    # Final sync
    ctx.enqueue_copy(host_reward_sum, gpu_reward_sum_buf)
    ctx.enqueue_copy(host_episode_count, gpu_episode_count_buf)
    ctx.synchronize()

    var final_count = Int(host_episode_count[0])
    var final_sum = Float64(host_reward_sum[0])
    completed_episodes += final_count
    if final_count > 0:
        last_avg_reward = final_sum / Float64(final_count)
        for _ in range(final_count):
            metrics.log_episode(completed_episodes, last_avg_reward, 0, 0.0)

    comptime if PROFILE >= 1:
        timer.sync_and_mark(ctx)
    agent.download_from_gpu(gpu_state, ctx)
    comptime if PROFILE >= 1:
        timer.accumulate(7)

    # Final logger flush + print
    if logger:
        logger[].log_scalar("avg_reward", last_avg_reward, total_steps)
        logger[].log_scalar(
            "episodes", Float64(completed_episodes), total_steps
        )
        logger[].log_scalar(
            "train_steps", Float64(total_train_steps), total_steps
        )
        logger[].flush()

    if verbose:
        clear_progress_bar()
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
            + " [DONE]"
        )

    return metrics^
