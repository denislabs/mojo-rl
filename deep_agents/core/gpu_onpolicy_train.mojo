"""GPU-accelerated on-policy training infrastructure.

Provides two trait hierarchies and shared GPU training-loop functions:

  GPUOnPolicyState  — GPU rollout buffer container
                      (actor/critic GPUNetworkStates, rollout buffers,
                       scratch buffers). Implements gpu_store_pre_step[N_ENVS]
                       and gpu_rollout_is_full.

  GPUOnPolicyDiscreteAgent  — on-policy agent with GPU support (discrete actions).
                              Has comptime GPUStateType: GPUOnPolicyState.
                              Implements make_gpu_state, upload_to_gpu,
                              download_from_gpu, select_actions_with_meta_gpu,
                              compute_advantages_gpu, update_epochs_gpu.

  GPUOnPolicyContinuousAgent — same for continuous action spaces.

Design principles:
  - The shared loop owns: env state/obs/action/reward/done buffers + episode tracking.
  - GPUOnPolicyState owns: rollout buffers + GPU network states + scratch buffers.
  - GPUOnPolicyDiscreteAgent (CPU struct) owns: CPU network states + hyperparameters.
  - Methods take (mut self, ctx, mut gpu_state) so agent hyperparams stay on the
    CPU struct and GPU buffers live in the state.
  - comptime MAX_N_ENVS on the agent fixes buffer sizes at compile time.

Usage:
    # 1. Define a GPU state container
    struct MyGPUState[...](GPUOnPolicyState):
        var gpu_actor: GPUNetworkState[...]
        var rollout_obs: DeviceBuffer[dtype]
        var rollout_step: Int
        ...
        fn gpu_store_pre_step[N](mut self, ctx, obs, actions, log_probs, values): ...
        fn gpu_store_post_step[N](mut self, ctx, rewards, dones): ...
        fn gpu_rollout_is_full(self) -> Bool: ...

    # 2. Make your agent implement GPUOnPolicyDiscreteAgent
    struct MyAgent[..., max_n_envs: Int = 1024](OnPolicyDiscreteAgent & GPUOnPolicyDiscreteAgent):
        comptime MAX_N_ENVS: Int = max_n_envs
        comptime GPUStateType = MyGPUState[...]
        ...
        fn make_gpu_state(self, ctx) raises -> MyGPUState[...]: ...
        fn upload_to_gpu(self, mut gpu_state: MyGPUState[...], ctx) raises: ...
        fn download_from_gpu(mut self, gpu_state: MyGPUState[...], ctx) raises: ...
        fn select_actions_with_meta_gpu[N](mut self, ctx, mut gpu_state, obs, actions, log_probs, values) raises: ...
        fn compute_advantages_gpu(mut self, ctx, mut gpu_state) raises: ...
        fn update_epochs_gpu(mut self, ctx, mut gpu_state) raises: ...

    # 3. Train
    var metrics = run_onpolicy_discrete_train_gpu[MyEnv, MyAgent](
        agent, ctx, num_updates=500,
    )
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from .checkpoint_trait import Checkpointable
from core import (
    TrainingMetrics,
    GPUDiscreteEnv,
    GPUContinuousEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from nn.constants import dtype
from deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    extract_completed_episodes_kernel,
    selective_reset_tracking_kernel,
)
from deep_agents.core.utils import print_progress_bar


# =============================================================================
# GPUOnPolicyState Trait
# =============================================================================


trait GPUOnPolicyState(ImplicitlyDestructible):
    """GPU-side rollout buffer container for on-policy agents.

    Holds all GPU-resident state: actor/critic network DeviceBuffers,
    rollout buffers (obs, actions, log_probs, rewards, values, dones),
    advantage/return buffers, and algorithm-specific scratch buffers.

    The shared training loop calls gpu_store_pre_step BEFORE the env step
    (storing obs, actions, log_probs, values) and gpu_store_post_step AFTER
    (storing rewards, dones and incrementing rollout_step).
    This mirrors the _store_pre_step_kernel / _store_post_step_kernel pattern
    in ppo/kernels.mojo.

    Algorithm-specific logic (advantages, epochs) lives on the agent.
    """

    fn gpu_rollout_reset(mut self) -> None:
        """Reset the rollout write pointer to 0 for the next update cycle."""
        ...

    fn gpu_store_pre_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        log_probs_buf: DeviceBuffer[dtype],
        values_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store pre-step data (obs, actions, log_probs, values) into rollout buffers.

        Called BEFORE the environment step so that obs_buf still holds the
        current (pre-step) observations.  Writes to rollout position rollout_step
        but does NOT yet increment rollout_step.

        Args:
            ctx: GPU device context.
            obs_buf: Pre-step observations [N_ENVS * OBS_DIM].
            actions_buf: Sampled actions [N_ENVS] (discrete) or [N_ENVS * ACTION_DIM].
            log_probs_buf: Log probabilities of sampled actions [N_ENVS].
            values_buf: Critic value estimates [N_ENVS].
        """
        ...

    fn gpu_store_post_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        rewards_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store post-step data (rewards, dones) and advance rollout pointer.

        Called AFTER the environment step.  Writes to rollout position rollout_step
        then increments rollout_step.

        Args:
            ctx: GPU device context.
            rewards_buf: Rewards received [N_ENVS].
            dones_buf: Done flags [N_ENVS] (1.0 = done).
        """
        ...

    fn gpu_rollout_is_full(self) -> Bool:
        """Return True when rollout_len steps have been stored.

        Typically: rollout_step >= ROLLOUT_LEN.
        """
        ...


# =============================================================================
# GPUOnPolicyDiscreteAgent Trait
# =============================================================================


trait GPUOnPolicyDiscreteAgent:
    """Discrete on-policy agent with GPU-accelerated training.

    The agent (CPU struct) owns hyperparameters and CPU network states.
    GPU rollout/scratch buffers live in GPUStateType.
    The shared training loop creates the GPU state once via make_gpu_state,
    then passes it to every GPU method call.

    Compile-time constants:
        OBS_DIM:      Observation space dimension.
        NUM_ACTIONS:  Number of discrete actions.
        ROLLOUT_LEN:  Steps per rollout per environment.
        MAX_N_ENVS:   Max parallel environments (sizes GPU rollout buffers).
        GPUStateType: Concrete type implementing GPUOnPolicyState.
    """

    comptime OBS_DIM: Int
    """Observation space dimension (must match GPUDiscreteEnv.OBS_DIM)."""

    comptime NUM_ACTIONS: Int
    """Number of discrete actions."""

    comptime ROLLOUT_LEN: Int
    """Steps per rollout per environment (baked into buffer sizes)."""

    comptime MAX_N_ENVS: Int
    """Max parallel environments — sizes rollout and scratch buffers."""

    comptime GPUStateType: GPUOnPolicyState
    """Concrete GPU state type holding all device buffers for this algorithm."""

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for this agent (networks, rollout, scratch).

        Called once at the start of GPU training. Does NOT upload CPU weights —
        call upload_to_gpu separately after make_gpu_state.

        Args:
            ctx: GPU device context.

        Returns:
            Freshly allocated GPU state container.
        """
        ...

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU state.

        Args:
            gpu_state: GPU state to populate (mutated in-place).
            ctx: GPU device context.
        """
        ...

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states.

        Args:
            gpu_state: GPU state to read from (mut because download_to requires it).
            ctx: GPU device context.
        """
        ...

    fn select_actions_with_meta_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut log_probs_buf: DeviceBuffer[dtype],
        mut values_buf: DeviceBuffer[dtype],
        rng_seed: UInt32 = 0,
    ) raises -> None:
        """Forward actor + critic on GPU for all N_ENVS environments.

        Actor: obs → logits → sample action + log_prob.
        Critic: obs → value.
        Results written into actions_buf, log_probs_buf, values_buf.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with actor/critic network buffers.
            obs_buf: Observations [N_ENVS * OBS_DIM].
            actions_buf: Output sampled actions [N_ENVS].
            log_probs_buf: Output log probabilities [N_ENVS].
            values_buf: Output critic values [N_ENVS].
            rng_seed: Per-step RNG seed for action sampling.
        """
        ...

    fn compute_advantages_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        final_obs_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Compute GAE advantages from the collected rollout.

        Typically: copy rollout rewards/values/dones to host, run GAE
        computation, copy advantages/returns back to GPU.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with rollout buffers.
            final_obs_buf: Final observations for bootstrapping [N_ENVS * OBS_DIM].
        """
        ...

    fn update_epochs_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        update_idx: Int,
    ) raises -> None:
        """Run PPO multi-epoch minibatch updates using the stored rollout.

        Called once per rollout. Performs num_epochs passes over the rollout
        with minibatch sampling and PPO clipped surrogate loss.

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with rollout + advantage buffers.
            update_idx: Current update index (for LR annealing progress).
        """
        ...


# =============================================================================
# GPUOnPolicyContinuousAgent Trait
# =============================================================================


trait GPUOnPolicyContinuousAgent:
    """Continuous on-policy agent with GPU-accelerated training.

    Same as GPUOnPolicyDiscreteAgent but for continuous action spaces.
    Adds ACTION_DIM compile-time constant; select_actions_with_meta_gpu
    outputs [N_ENVS * ACTION_DIM] actions instead of [N_ENVS].
    """

    comptime OBS_DIM: Int
    """Observation space dimension."""

    comptime ACTION_DIM: Int
    """Continuous action space dimension."""

    comptime ROLLOUT_LEN: Int
    """Steps per rollout per environment."""

    comptime MAX_N_ENVS: Int
    """Max parallel environments."""

    comptime GPUStateType: GPUOnPolicyState
    """Concrete GPU state type."""

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for this agent."""
        ...

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU state."""
        ...

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        ...

    fn select_actions_with_meta_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut log_probs_buf: DeviceBuffer[dtype],
        mut values_buf: DeviceBuffer[dtype],
        rng_seed: UInt32 = 0,
    ) raises -> None:
        """Forward actor + critic for continuous actions.

        Actor: obs → sample (mean, std) → reparameterize → action + log_prob.
        Critic: obs → value.
        """
        ...

    fn compute_advantages_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        final_obs_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Compute GAE advantages from the collected rollout."""
        ...

    fn update_epochs_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        update_idx: Int,
    ) raises -> None:
        """Run PPO multi-epoch minibatch updates."""
        ...


# =============================================================================
# Shared GPU Training Loop — Discrete Actions
# =============================================================================


fn run_onpolicy_discrete_train_gpu[
    E: GPUDiscreteEnv,
    A: GPUOnPolicyDiscreteAgent & Checkpointable,
](
    mut agent: A,
    ctx: DeviceContext,
    num_updates: Int,
    sync_every: Int = 50,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "GPUOnPolicy",
) raises -> TrainingMetrics:
    """Shared GPU training loop for discrete-action on-policy agents (PPO).

    Responsibility split:
      Loop:      allocates env buffers, drives E.step_kernel_gpu / reset kernels,
                 calls gpu_state.gpu_store_pre_step / gpu_store_post_step, drives episode tracking.
      Agent:     implements action selection, advantage computation, epoch updates.
      GPU state: holds rollout + scratch buffers.

    The number of parallel environments is fixed at compile time as A.MAX_N_ENVS.
    Each update cycle: collect ROLLOUT_LEN steps → compute advantages → update epochs.

    Parameters:
        E: GPU environment type implementing GPUDiscreteEnv.
        A: Agent type implementing GPUOnPolicyDiscreteAgent.

    Args:
        agent: On-policy agent with GPU support (updated in-place).
        ctx: GPU device context.
        num_updates: Number of rollout + update cycles.
        sync_every: GPU→CPU parameter sync interval in updates (default: 50).
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with episode-level statistics.
    """
    comptime n_envs = A.MAX_N_ENVS
    comptime tpb = 256

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
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var log_probs_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var values_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Episode tracking buffers
    var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var completed_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var completed_mask_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Host buffers for episode tracking readback
    var completed_rewards_host = ctx.enqueue_create_host_buffer[dtype](n_envs)
    var completed_steps_host = ctx.enqueue_create_host_buffer[dtype](n_envs)
    var completed_mask_host = ctx.enqueue_create_host_buffer[dtype](n_envs)

    # Step workspace (for physics environments)
    var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
    if ws_size == 0:
        ws_size = 1
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)

    if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
        E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)
        ctx.synchronize()

    # ------------------------------------------------------------------
    # Initial reset + extract obs
    # ------------------------------------------------------------------
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
    ctx.synchronize()

    # Initialize episode tracking
    ctx.enqueue_memset(episode_rewards_buf, 0)
    ctx.enqueue_memset(episode_steps_buf, 0)

    # ------------------------------------------------------------------
    # Kernel wrappers (defined once outside the loop)
    # ------------------------------------------------------------------
    comptime env_blocks = (n_envs + tpb - 1) // tpb

    comptime accum_rewards_wrapper = accumulate_rewards_kernel[dtype, n_envs]
    comptime incr_steps_wrapper = increment_steps_kernel[dtype, n_envs]
    comptime extract_completed_wrapper = extract_completed_episodes_kernel[
        dtype, n_envs
    ]
    comptime reset_tracking_wrapper = selective_reset_tracking_kernel[
        dtype, n_envs
    ]

    from layout import Layout, LayoutTensor

    var total_steps = 0
    var step_seed: UInt32 = 42
    var completed_episodes = 0

    # Progress bar: ~20 updates per print interval
    var progress_interval = print_every // 20
    if progress_interval < 1:
        progress_interval = 1
    var next_progress = progress_interval

    for update in range(num_updates):
        # ==================================================================
        # Phase 1: Collect rollout (ROLLOUT_LEN steps across n_envs envs)
        # ==================================================================
        gpu_state.gpu_rollout_reset()
        for _t in range(A.ROLLOUT_LEN):
            # Select actions (actor + critic forward + sampling)
            agent.select_actions_with_meta_gpu[n_envs](
                ctx,
                gpu_state,
                obs_buf,
                actions_buf,
                log_probs_buf,
                values_buf,
                rng_seed=step_seed,
            )

            # Store pre-step data (obs, actions, log_probs, values) BEFORE env step
            # so that obs_buf still contains the current (pre-step) observations.
            gpu_state.gpu_store_pre_step[n_envs](
                ctx,
                obs_buf,
                actions_buf,
                log_probs_buf,
                values_buf,
            )

            # Step environment (obs_buf is updated to next_obs after this call)
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

            # Store post-step data (rewards, dones) and advance rollout pointer
            gpu_state.gpu_store_post_step[n_envs](
                ctx,
                rewards_buf,
                dones_buf,
            )

            # Accumulate episode rewards and steps
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
            var completed_rewards_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](completed_rewards_buf.unsafe_ptr())
            var completed_steps_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](completed_steps_buf.unsafe_ptr())
            var completed_mask_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](completed_mask_buf.unsafe_ptr())

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

            # Extract completed episodes
            ctx.enqueue_function[
                extract_completed_wrapper, extract_completed_wrapper
            ](
                dones_t,
                episode_rewards_t,
                episode_steps_t,
                completed_rewards_t,
                completed_steps_t,
                completed_mask_t,
                grid_dim=(env_blocks,),
                block_dim=(tpb,),
            )

            # Copy to host and log
            ctx.enqueue_copy(completed_rewards_host, completed_rewards_buf)
            ctx.enqueue_copy(completed_steps_host, completed_steps_buf)
            ctx.enqueue_copy(completed_mask_host, completed_mask_buf)
            ctx.synchronize()

            for i in range(n_envs):
                if Float64(completed_mask_host[i]) > 0.5:
                    var ep_reward = Float64(completed_rewards_host[i])
                    var ep_steps = Int(completed_steps_host[i])
                    metrics.log_episode(
                        completed_episodes, ep_reward, ep_steps, 0.0
                    )
                    completed_episodes += 1

            # Reset episode tracking for done environments
            ctx.enqueue_function[
                reset_tracking_wrapper, reset_tracking_wrapper
            ](
                dones_t,
                episode_rewards_t,
                episode_steps_t,
                grid_dim=(env_blocks,),
                block_dim=(tpb,),
            )

            # Auto-reset done environments
            E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=UInt64(step_seed + 1),
            )

            total_steps += n_envs
            step_seed += 2

        # ==================================================================
        # Phase 2: Compute GAE advantages
        # ==================================================================
        agent.compute_advantages_gpu(ctx, gpu_state, obs_buf)

        # ==================================================================
        # Phase 3: Update epochs
        # ==================================================================
        agent.update_epochs_gpu(ctx, gpu_state, update)

        # ==================================================================
        # Periodic GPU → CPU sync
        # ==================================================================
        if update % sync_every == 0:
            agent.download_from_gpu(gpu_state, ctx)

        # Periodic checkpoint (after sync)
        if checkpoint_every > 0 and update % checkpoint_every == 0:
            if update % sync_every != 0:  # avoid double download
                agent.download_from_gpu(gpu_state, ctx)
            agent.save_checkpoint(checkpoint_path)

        # Progress bar (no GPU sync, pure CPU counters)
        # Shows progress within the current print interval (0% → 100%)
        if verbose and update + 1 >= next_progress:
            var interval_pos = (update + 1) % print_every
            if interval_pos == 0:
                interval_pos = print_every
            print_progress_bar(
                interval_pos, print_every, total_steps, algorithm_name
            )
            next_progress += progress_interval

        if verbose and (update + 1) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(
                min(100, completed_episodes)
            )
            print()
            print(
                algorithm_name
                + " | Update "
                + String(update + 1)
                + " / "
                + String(num_updates)
                + " | Episodes: "
                + String(completed_episodes)
                + " | AvgR(100): "
                + String(avg_reward)[:7]
                + " | Steps: "
                + String(total_steps)
            )

    # Final sync to ensure CPU params are up to date
    ctx.synchronize()
    agent.download_from_gpu(gpu_state, ctx)

    return metrics^


# =============================================================================
# Shared GPU Training Loop — Continuous Actions
# =============================================================================


fn run_onpolicy_continuous_train_gpu[
    E: GPUContinuousEnv,
    A: GPUOnPolicyContinuousAgent & Checkpointable,
    CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
](
    mut agent: A,
    ctx: DeviceContext,
    num_updates: Int,
    target_episodes: Int = 0,
    target_total_steps: Int = 0,
    sync_every: Int = 50,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "GPUOnPolicy",
) raises -> TrainingMetrics:
    """Shared GPU training loop for continuous-action on-policy agents (PPO).

    Same structure as run_onpolicy_discrete_train_gpu but for continuous
    action spaces. Uses GPUContinuousEnv and A.ACTION_DIM for action buffers.

    Parameters:
        E: GPU environment type implementing GPUContinuousEnv.
        A: Agent type implementing GPUOnPolicyContinuousAgent.
        CurriculumType: Curriculum scheduler type (default: NoCurriculumScheduler).

    Args:
        agent: On-policy agent with GPU support (updated in-place).
        ctx: GPU device context.
        num_updates: Number of rollout + update cycles.
        target_episodes: Target number of episodes to complete (default: 0 = unlimited).
        target_total_steps: Total steps for curriculum/annealing progress (default: 0 = disabled).
        sync_every: GPU→CPU parameter sync interval in updates (default: 50).
        checkpoint_every: Save checkpoint every N updates (default: 0 = disabled).
        checkpoint_path: Path to save checkpoint (default: "").
        verbose: Print progress (default: False).
        print_every: Print every N updates if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with episode-level statistics.
    """
    comptime n_envs = A.MAX_N_ENVS
    comptime tpb = 256

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
    # Allocate environment buffers
    # ------------------------------------------------------------------
    var states_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](n_envs * E.ACTION_DIM)
    var log_probs_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var values_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var dones_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](n_envs)

    # Episode tracking
    var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var episode_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var completed_steps_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var completed_mask_buf = ctx.enqueue_create_buffer[dtype](n_envs)
    var completed_rewards_host = ctx.enqueue_create_host_buffer[dtype](n_envs)
    var completed_steps_host = ctx.enqueue_create_host_buffer[dtype](n_envs)
    var completed_mask_host = ctx.enqueue_create_host_buffer[dtype](n_envs)

    var ws_size = E.STEP_WS_SHARED + n_envs * E.STEP_WS_PER_ENV
    if ws_size == 0:
        ws_size = 1
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)

    if E.STEP_WS_SHARED + E.STEP_WS_PER_ENV > 0:
        E.init_step_workspace_gpu[n_envs](ctx, workspace_buf)
        ctx.synchronize()

    # ------------------------------------------------------------------
    # Initial reset + extract initial observations
    # ------------------------------------------------------------------
    E.reset_kernel_gpu[n_envs, E.STATE_SIZE](ctx, states_buf, rng_seed=0)
    E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
        ctx, states_buf, obs_buf
    )
    ctx.synchronize()

    ctx.enqueue_memset(episode_rewards_buf, 0)
    ctx.enqueue_memset(episode_steps_buf, 0)

    comptime env_blocks = (n_envs + tpb - 1) // tpb

    comptime accum_rewards_wrapper = accumulate_rewards_kernel[dtype, n_envs]
    comptime incr_steps_wrapper = increment_steps_kernel[dtype, n_envs]
    comptime extract_completed_wrapper = extract_completed_episodes_kernel[
        dtype, n_envs
    ]
    comptime reset_tracking_wrapper = selective_reset_tracking_kernel[
        dtype, n_envs
    ]

    from layout import Layout, LayoutTensor

    var total_steps = 0
    var step_seed: UInt32 = 42
    var completed_episodes = 0

    # Progress bar: ~20 updates per print interval
    var progress_interval = print_every // 20
    if progress_interval < 1:
        progress_interval = 1
    var next_progress = progress_interval

    for update in range(num_updates):
        if target_episodes > 0 and completed_episodes >= target_episodes:
            break

        # Curriculum update (once per rollout, before collecting steps)
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

        # Phase 1: Collect rollout
        gpu_state.gpu_rollout_reset()
        for _t in range(A.ROLLOUT_LEN):
            agent.select_actions_with_meta_gpu[n_envs](
                ctx,
                gpu_state,
                obs_buf,
                actions_buf,
                log_probs_buf,
                values_buf,
                rng_seed=step_seed,
            )

            # Store pre-step data BEFORE env step (obs_buf = current obs)
            gpu_state.gpu_store_pre_step[n_envs](
                ctx,
                obs_buf,
                actions_buf,
                log_probs_buf,
                values_buf,
            )

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

            # Store post-step data AFTER env step and advance rollout pointer
            gpu_state.gpu_store_post_step[n_envs](
                ctx,
                rewards_buf,
                dones_buf,
            )

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
            var completed_rewards_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](completed_rewards_buf.unsafe_ptr())
            var completed_steps_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](completed_steps_buf.unsafe_ptr())
            var completed_mask_t = LayoutTensor[
                dtype, Layout.row_major(n_envs), MutAnyOrigin
            ](completed_mask_buf.unsafe_ptr())

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
            ctx.enqueue_function[
                extract_completed_wrapper, extract_completed_wrapper
            ](
                dones_t,
                episode_rewards_t,
                episode_steps_t,
                completed_rewards_t,
                completed_steps_t,
                completed_mask_t,
                grid_dim=(env_blocks,),
                block_dim=(tpb,),
            )

            ctx.enqueue_copy(completed_rewards_host, completed_rewards_buf)
            ctx.enqueue_copy(completed_steps_host, completed_steps_buf)
            ctx.enqueue_copy(completed_mask_host, completed_mask_buf)
            ctx.synchronize()

            for i in range(n_envs):
                if Float64(completed_mask_host[i]) > 0.5:
                    var ep_reward = Float64(completed_rewards_host[i])
                    var ep_steps = Int(completed_steps_host[i])
                    metrics.log_episode(
                        completed_episodes, ep_reward, ep_steps, 0.0
                    )
                    completed_episodes += 1

            ctx.enqueue_function[
                reset_tracking_wrapper, reset_tracking_wrapper
            ](
                dones_t,
                episode_rewards_t,
                episode_steps_t,
                grid_dim=(env_blocks,),
                block_dim=(tpb,),
            )

            E.selective_reset_kernel_gpu[n_envs, E.STATE_SIZE](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=UInt64(step_seed + 1),
            )
            # Update obs_buf for reset environments — must happen after selective_reset
            # so that the next step's actor sees the initial obs of the new episode,
            # not the terminal obs of the previous one.
            E.extract_obs_kernel_gpu[n_envs, E.STATE_SIZE, E.OBS_DIM](
                ctx, states_buf, obs_buf
            )

            total_steps += n_envs
            # Increment seed by the full action-sampling range width + 2 (for env reset seed).
            # Each step uses seeds [step_seed, step_seed + n_envs*E.ACTION_DIM - 1].
            # With step_seed += 2, consecutive steps had ~1534 seed collisions (range overlap),
            # causing correlated action noise across timesteps and biased PPO gradients.
            # Increment must be >= n_envs * E.ACTION_DIM to ensure non-overlapping ranges.
            comptime seed_stride = n_envs * E.ACTION_DIM + 2
            step_seed += UInt32(seed_stride)

        # Phase 2: Compute advantages
        agent.compute_advantages_gpu(ctx, gpu_state, obs_buf)

        # Phase 3: Update epochs
        agent.update_epochs_gpu(ctx, gpu_state, update)

        if update % sync_every == 0:
            agent.download_from_gpu(gpu_state, ctx)

        # Periodic checkpoint (after sync)
        if checkpoint_every > 0 and update % checkpoint_every == 0:
            if update % sync_every != 0:  # avoid double download
                agent.download_from_gpu(gpu_state, ctx)
            agent.save_checkpoint(checkpoint_path)

        # Progress bar (no GPU sync, pure CPU counters)
        # Shows progress within the current print interval (0% → 100%)
        if verbose and update + 1 >= next_progress:
            var interval_pos = (update + 1) % print_every
            if interval_pos == 0:
                interval_pos = print_every
            print_progress_bar(
                interval_pos, print_every, total_steps, algorithm_name
            )
            next_progress += progress_interval

        if verbose and (update + 1) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(
                min(100, completed_episodes)
            )
            var ep_progress = String(completed_episodes) + (
                " / " + String(target_episodes) if target_episodes > 0 else ""
            )
            print()
            print(
                algorithm_name
                + " | Episodes: "
                + ep_progress
                + " | Update: "
                + String(update + 1)
                + " | AvgR(100): "
                + String(avg_reward)[:7]
                + " | Steps: "
                + String(total_steps)
            )

    ctx.synchronize()
    agent.download_from_gpu(gpu_state, ctx)

    return metrics^
