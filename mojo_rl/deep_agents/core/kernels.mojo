"""Shared GPU kernels for deep RL agents.

## Network Operations
- soft_update_kernel: Target network soft update (tau blending)
- zero_buffer_kernel: Zero out a buffer
- copy_buffer_kernel: Copy one buffer to another

## Episode Tracking
- accumulate_rewards_kernel: Add step rewards to episode totals
- increment_steps_kernel: Increment step counters
- extract_completed_episodes_kernel: Extract completed episode data
- selective_reset_tracking_kernel: Reset tracking for done envs

## Replay Buffer Operations
- store_transitions_kernel: Store transitions to GPU replay buffer
- sample_indices_kernel: Generate random sample indices
- gather_batch_kernel: Gather sampled transitions into batch
- store_transitions_kernel_nd: Multi-dimensional action store
- gather_batch_kernel_nd: Multi-dimensional action gather

## Continuous Control (shared across DDPG/TD3/SAC)
- td_target_continuous_kernel: DDPG target — r + γ * Q_t(s',a') * (1-done)
- td_target_min_twin_kernel: TD3/SAC target — r + γ * min(Q1,Q2) * (1-done) [- α*log_π]
- actor_grad_from_critic_kernel: Extract ∂Q/∂a from critic's input gradient
- concat_obs_action_kernel: Concatenate obs and actions for critic input
- scale_clip_actions_kernel: Scale tanh output and clip to action range
- ddpg_exploration_kernel: Scale + Gaussian noise (DDPG/TD3 exploration)
- td_mse_grad_kernel: MSE gradient for TD critic

## DQN Operations
- dqn_td_target_kernel: Standard DQN TD target — r + γ * max_a Q(s',a) * (1-done)
- dqn_double_td_target_kernel: Double DQN TD target — r + γ * Q_t(s', argmax_a Q_o(s',a)) * (1-done)

## Dueling DQN Operations
- dueling_combine_kernel: Combine V(s) + A(s,a) - mean(A) into Q-values
- dueling_grad_kernel: Transform dQ gradients to dueling [V, A] gradients

## TD3 Noise
- add_gaussian_noise_kernel: Clipped Gaussian noise for TD3 target smoothing

## SAC Reparameterization
- sac_sample_actions_kernel: Inference — sample from stochastic actor output
- sac_rsample_with_cache_kernel: Training forward — sample + log_prob + save eps
- sac_rsample_bwd_kernel: Training backward — grad through reparameterization trick
- min_q_dq_kernel: Masked dq gradients based on min(Q1, Q2)
- add_ci_grads_kernel: Elementwise add critic input gradients

## PPO Operations (Continuous)
- _sample_continuous_actions_kernel: Sample from unbounded Gaussian policy
- _store_continuous_pre_step_kernel: Store pre-step data for continuous actions
- _store_post_step_kernel: Store post-step rewards and dones
- ppo_continuous_gather_minibatch_kernel: Gather continuous action minibatch
- ppo_continuous_actor_grad_kernel: PPO actor gradient (unbounded Gaussian)
- ppo_critic_grad_kernel: PPO critic MSE gradient
- ppo_critic_grad_clipped_kernel: PPO critic gradient with value clipping
- normalize_advantages_kernel: Normalize advantages with pre-computed stats
- clamp_log_std_params_kernel: Clamp log_std parameters to valid range

## PPO Operations (Discrete)
- ppo_gather_minibatch_kernel: Gather discrete action minibatch
- ppo_actor_grad_with_kl_kernel: PPO discrete actor gradient with KL tracking
- _store_pre_step_kernel: Store pre-step data for discrete actions
- _store_pre_step_obs_parallel_kernel: Parallel obs store (one thread per element)
- ppo_gather_minibatch_obs_parallel_kernel: Parallel obs gather (one thread per element)

## Gradient Clipping
- gradient_norm_kernel: Partial sum of squared gradients
- gradient_reduce_and_compute_scale_kernel: Reduce partials and compute clip scale
- gradient_apply_scale_kernel: Apply precomputed scale to gradients
- gradient_reduce_apply_fused_kernel: Fused reduce + apply (2 kernels instead of 3)
"""

from std.gpu import block_dim, block_idx, thread_idx, barrier
from layout import Layout, LayoutTensor
from std.math import exp, log, tanh, sqrt, max, min, pi, cos
from std.memory import UnsafePointer
from std.random.philox import Random as PhiloxRandom


# =============================================================================
# Network Operations
# =============================================================================


@always_inline
def soft_update_kernel[
    dtype: DType,
    SIZE: Int,
](
    target: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    tau: Scalar[dtype],
):
    """Soft update: target = tau * source + (1 - tau) * target.

    Used for target network updates in DQN, DDPG, TD3, SAC.

    Args:
        target: Target network parameters (updated in-place).
        source: Source (online) network parameters.
        tau: Blending factor (typically 0.001 - 0.01).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return

    var src_val = source[i]
    var tgt_val = target[i]
    target[i] = tau * src_val + (Scalar[dtype](1.0) - tau) * tgt_val


@always_inline
def zero_buffer_kernel[
    dtype: DType,
    SIZE: Int,
](buffer: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
    """Zero out a buffer.

    Args:
        buffer: Buffer to zero (updated in-place).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    buffer[i] = Scalar[dtype](0.0)


@always_inline
def copy_buffer_kernel[
    dtype: DType,
    SIZE: Int,
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Copy src buffer to dst buffer.

    Args:
        dst: Destination buffer (updated in-place).
        src: Source buffer.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    dst[i] = src[i]


# =============================================================================
# Episode Tracking Kernels
# =============================================================================


@always_inline
def accumulate_rewards_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    episode_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    step_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
):
    """Accumulate step rewards into episode totals.

    Args:
        episode_rewards: Running episode reward totals (updated in-place).
        step_rewards: Rewards from current step.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return
    episode_rewards[i] += step_rewards[i]


@always_inline
def increment_steps_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    episode_steps: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
):
    """Increment step counters for each environment.

    Args:
        episode_steps: Step counters (updated in-place).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return
    episode_steps[i] += Scalar[dtype](1.0)


@always_inline
def extract_completed_episodes_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    episode_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    episode_steps: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    completed_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    completed_steps: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    completed_mask: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
):
    """Extract completed episode rewards and reset accumulators for done environments.

    For each done environment:
    1. Copy episode reward to completed_rewards
    2. Copy episode steps to completed_steps
    3. Set completed_mask to 1.0 (indicating valid data)
    4. Reset episode_rewards and episode_steps to 0

    Args:
        dones: Done flags for each environment.
        episode_rewards: Running episode reward totals (reset for done envs).
        episode_steps: Running step counters (reset for done envs).
        completed_rewards: Output buffer for completed episode rewards.
        completed_steps: Output buffer for completed episode step counts.
        completed_mask: Output mask (1.0 if episode completed, 0.0 otherwise).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    if dones[i] > Scalar[dtype](0.5):
        # Episode completed - extract and reset
        completed_rewards[i] = episode_rewards[i]
        completed_steps[i] = episode_steps[i]
        completed_mask[i] = Scalar[dtype](1.0)
        # Reset accumulators for next episode
        episode_rewards[i] = Scalar[dtype](0.0)
        episode_steps[i] = Scalar[dtype](0.0)
    else:
        # Episode ongoing
        completed_mask[i] = Scalar[dtype](0.0)


@always_inline
def selective_reset_tracking_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    episode_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    episode_steps: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
):
    """Reset episode tracking only for done environments.

    Args:
        dones: Done flags for each environment.
        episode_rewards: Running episode reward totals (reset for done envs).
        episode_steps: Running step counters (reset for done envs).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    if dones[i] > Scalar[dtype](0.5):
        episode_rewards[i] = Scalar[dtype](0.0)
        episode_steps[i] = Scalar[dtype](0.0)


@always_inline
def log_and_reset_completed_kernel[
    dtype: DType,
    N_ENVS: Int,
](
    dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    episode_rewards: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    episode_steps: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    reward_sum: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    episode_count: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
):
    """Single-threaded kernel: accumulate completed episode stats and reset.

    For done environments, adds the episode reward to a running sum,
    increments the episode count, and resets per-env accumulators.
    Single-threaded to avoid atomics (N_ENVS is small).

    Args:
        dones: Done flags for each environment.
        episode_rewards: Running episode reward totals (reset for done envs).
        episode_steps: Running step counters (reset for done envs).
        reward_sum: Running sum of completed episode rewards [1].
        episode_count: Running count of completed episodes [1].
    """
    if thread_idx.x != 0 or block_idx.x != 0:
        return
    var s = reward_sum[0]
    var c = episode_count[0]
    for i in range(N_ENVS):
        if dones[i] > Scalar[dtype](0.5):
            s += episode_rewards[i]
            c += Scalar[dtype](1.0)
            episode_rewards[i] = Scalar[dtype](0.0)
            episode_steps[i] = Scalar[dtype](0.0)
    reward_sum[0] = s
    episode_count[0] = c


# =============================================================================
# Replay Buffer Operations
# =============================================================================


@always_inline
def store_transitions_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    CAPACITY: Int,
](
    # Inputs: current transitions from vectorized envs
    states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    next_states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Replay buffer storage (circular buffer)
    buf_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_rewards: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_next_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_dones: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    # Write position (current head of circular buffer)
    write_idx: Scalar[DType.int32],
):
    """Store a batch of transitions into the replay buffer.

    Each thread stores one transition at position (write_idx + thread_idx) % CAPACITY.

    Args:
        states: Current states [BATCH_SIZE, OBS_DIM].
        actions: Actions taken [BATCH_SIZE].
        rewards: Rewards received [BATCH_SIZE].
        next_states: Next states [BATCH_SIZE, OBS_DIM].
        dones: Done flags [BATCH_SIZE].
        buf_states: Replay buffer states storage [CAPACITY, OBS_DIM].
        buf_actions: Replay buffer actions storage [CAPACITY].
        buf_rewards: Replay buffer rewards storage [CAPACITY].
        buf_next_states: Replay buffer next states storage [CAPACITY, OBS_DIM].
        buf_dones: Replay buffer dones storage [CAPACITY].
        write_idx: Current write position in circular buffer.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var buf_idx = (Int(write_idx) + i) % CAPACITY

    # Copy observation and next observation
    for d in range(OBS_DIM):
        buf_states[buf_idx, d] = states[i, d]
        buf_next_states[buf_idx, d] = next_states[i, d]

    buf_actions[buf_idx] = actions[i]
    buf_rewards[buf_idx] = rewards[i]
    buf_dones[buf_idx] = dones[i]


@always_inline
def store_obs_parallel_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    CAPACITY: Int,
](
    states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    next_states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    buf_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_next_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    write_idx: Scalar[DType.int32],
):
    """Parallel store for large observations. One thread per element.

    Grid: (ceil(OBS_DIM / TPB), BATCH_SIZE)
    block_idx.y selects the env, thread within block selects obs dimension.
    """
    var d = Int(block_dim.x * block_idx.x + thread_idx.x)
    var i = Int(block_idx.y)
    if d >= OBS_DIM:
        return

    var buf_idx = (Int(write_idx) + i) % CAPACITY
    buf_states[buf_idx, d] = states[i, d]
    buf_next_states[buf_idx, d] = next_states[i, d]


@always_inline
def increment_rng_counter_kernel(
    counter: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
):
    """Increment GPU-side RNG counter by 1. Launch with grid=(1,), block=(1,).

    Used inside CUDA graph capture so each replay gets a fresh seed.
    The counter lives in a DeviceBuffer and persists between replays.
    """
    if Int(thread_idx.x) == 0:
        counter[0] = counter[0] + UInt32(1)


@always_inline
def sample_indices_kernel[
    dtype: DType,
    SAMPLE_SIZE: Int,
](
    indices: LayoutTensor[
        DType.int32, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    buffer_size: Scalar[DType.int32],
    rng_counter: LayoutTensor[
        DType.uint32, Layout.row_major(1), MutAnyOrigin
    ],
):
    """Generate random indices for sampling from replay buffer.

    Each thread generates one random index in [0, buffer_size).
    Uses PhiloxRandom for GPU-safe randomness (no seed collisions).
    Reads seed from GPU-side rng_counter (CUDA graph compatible).

    Args:
        indices: Output buffer for random indices [SAMPLE_SIZE].
        buffer_size: Current size of replay buffer (samples from [0, buffer_size)).
        rng_counter: GPU-side RNG counter [1] (read, not modified).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SAMPLE_SIZE:
        return

    var rng_seed = UInt64(rng_counter[0].cast[DType.uint64]())
    # PhiloxRandom: unique seed per thread, no collisions
    var philox = PhiloxRandom(
        seed=rng_seed + UInt64(i),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var u: Scalar[dtype] = Scalar[dtype](rand_vals[0])
    var idx = Int(u * Scalar[dtype](buffer_size))
    indices[i] = Scalar[DType.int32](idx)


@always_inline
def gather_batch_kernel[
    dtype: DType,
    SAMPLE_SIZE: Int,
    OBS_DIM: Int,
    CAPACITY: Int,
](
    # Output batch (sampled transitions)
    batch_states: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, OBS_DIM), MutAnyOrigin
    ],
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    batch_next_states: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, OBS_DIM), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    # Replay buffer storage
    buf_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_rewards: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_next_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_dones: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    # Sampled indices
    indices: LayoutTensor[
        DType.int32, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
):
    """Gather sampled transitions from replay buffer into batch tensors.

    Each thread gathers one transition based on its corresponding index.
    Used for small OBS_DIM. For large OBS_DIM (pixels), use
    gather_obs_parallel_kernel + gather_scalars_kernel instead.

    Args:
        batch_states: Output batch states [SAMPLE_SIZE, OBS_DIM].
        batch_actions: Output batch actions [SAMPLE_SIZE].
        batch_rewards: Output batch rewards [SAMPLE_SIZE].
        batch_next_states: Output batch next states [SAMPLE_SIZE, OBS_DIM].
        batch_dones: Output batch dones [SAMPLE_SIZE].
        buf_states: Replay buffer states storage [CAPACITY, OBS_DIM].
        buf_actions: Replay buffer actions storage [CAPACITY].
        buf_rewards: Replay buffer rewards storage [CAPACITY].
        buf_next_states: Replay buffer next states storage [CAPACITY, OBS_DIM].
        buf_dones: Replay buffer dones storage [CAPACITY].
        indices: Sampled indices [SAMPLE_SIZE].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SAMPLE_SIZE:
        return

    var buf_idx = Int(indices[i])

    # Gather observation and next observation
    for d in range(OBS_DIM):
        batch_states[i, d] = buf_states[buf_idx, d]
        batch_next_states[i, d] = buf_next_states[buf_idx, d]

    batch_actions[i] = buf_actions[buf_idx]
    batch_rewards[i] = buf_rewards[buf_idx]
    batch_dones[i] = buf_dones[buf_idx]


@always_inline
def gather_obs_parallel_kernel[
    dtype: DType,
    SAMPLE_SIZE: Int,
    OBS_DIM: Int,
    CAPACITY: Int,
](
    batch_states: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, OBS_DIM), MutAnyOrigin
    ],
    batch_next_states: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, OBS_DIM), MutAnyOrigin
    ],
    buf_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_next_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
):
    """Parallel gather for large observations (e.g. pixel frames).

    Grid: (ceil(OBS_DIM / TPB), SAMPLE_SIZE) — each thread copies one element.
    block_idx.y selects the sample, thread within block selects obs dimension.
    Copies both obs and next_obs in one launch.
    """
    var d = Int(block_dim.x * block_idx.x + thread_idx.x)
    var sample = Int(block_idx.y)
    if d >= OBS_DIM:
        return

    var buf_idx = Int(indices[sample])
    batch_states[sample, d] = buf_states[buf_idx, d]
    batch_next_states[sample, d] = buf_next_states[buf_idx, d]


@always_inline
def gather_scalars_kernel[
    dtype: DType,
    SAMPLE_SIZE: Int,
    CAPACITY: Int,
](
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_rewards: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_dones: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
):
    """Gather scalar fields (actions, rewards, dones) for sampled transitions.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SAMPLE_SIZE:
        return

    var buf_idx = Int(indices[i])
    batch_actions[i] = buf_actions[buf_idx]
    batch_rewards[i] = buf_rewards[buf_idx]
    batch_dones[i] = buf_dones[buf_idx]


@always_inline
def gather_scalars_nd_kernel[
    dtype: DType,
    SAMPLE_SIZE: Int,
    ACTION_DIM: Int,
    CAPACITY: Int,
](
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, ACTION_DIM), MutAnyOrigin
    ],
    buf_rewards: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_dones: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
):
    """Gather scalar+action fields for multi-dimensional action transitions."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SAMPLE_SIZE:
        return

    var buf_idx = Int(indices[i])
    for a in range(ACTION_DIM):
        batch_actions[i, a] = buf_actions[buf_idx, a]
    batch_rewards[i] = buf_rewards[buf_idx]
    batch_dones[i] = buf_dones[buf_idx]


# =============================================================================
# ND (multi-dimensional action) Replay Buffer Kernels
# These variants support ACTION_DIM > 1 for continuous control (DDPG/TD3/SAC).
# =============================================================================


@always_inline
def store_transitions_kernel_nd[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    CAPACITY: Int,
](
    states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    next_states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    buf_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, ACTION_DIM), MutAnyOrigin
    ],
    buf_rewards: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_next_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_dones: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    write_idx: Scalar[DType.int32],
):
    """Store a batch of transitions with multi-dimensional actions.

    Extends store_transitions_kernel to support ACTION_DIM > 1 for
    continuous control algorithms (DDPG, TD3, SAC).

    Args:
        states: Current states [BATCH_SIZE, OBS_DIM].
        actions: Actions taken [BATCH_SIZE, ACTION_DIM].
        rewards: Rewards received [BATCH_SIZE].
        next_states: Next states [BATCH_SIZE, OBS_DIM].
        dones: Done flags [BATCH_SIZE].
        buf_states: Replay buffer states storage [CAPACITY, OBS_DIM].
        buf_actions: Replay buffer actions storage [CAPACITY, ACTION_DIM].
        buf_rewards: Replay buffer rewards storage [CAPACITY].
        buf_next_states: Replay buffer next states storage [CAPACITY, OBS_DIM].
        buf_dones: Replay buffer dones storage [CAPACITY].
        write_idx: Current write position in circular buffer.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var buf_idx = (Int(write_idx) + i) % CAPACITY

    for d in range(OBS_DIM):
        buf_states[buf_idx, d] = states[i, d]
        buf_next_states[buf_idx, d] = next_states[i, d]

    for a in range(ACTION_DIM):
        buf_actions[buf_idx, a] = actions[i, a]

    buf_rewards[buf_idx] = rewards[i]
    buf_dones[buf_idx] = dones[i]


@always_inline
def gather_batch_kernel_nd[
    dtype: DType,
    SAMPLE_SIZE: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    CAPACITY: Int,
](
    batch_states: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, OBS_DIM), MutAnyOrigin
    ],
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    batch_next_states: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE, OBS_DIM), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[
        dtype, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    buf_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, ACTION_DIM), MutAnyOrigin
    ],
    buf_rewards: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    buf_next_states: LayoutTensor[
        dtype, Layout.row_major(CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buf_dones: LayoutTensor[dtype, Layout.row_major(CAPACITY), MutAnyOrigin],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
):
    """Gather sampled transitions with multi-dimensional actions.

    Extends gather_batch_kernel to support ACTION_DIM > 1 for
    continuous control algorithms (DDPG, TD3, SAC).
    Used for small OBS_DIM. For large OBS_DIM (pixels), use
    gather_obs_parallel_kernel + gather_scalars_nd_kernel instead.

    Args:
        batch_states: Output batch states [SAMPLE_SIZE, OBS_DIM].
        batch_actions: Output batch actions [SAMPLE_SIZE, ACTION_DIM].
        batch_rewards: Output batch rewards [SAMPLE_SIZE].
        batch_next_states: Output batch next states [SAMPLE_SIZE, OBS_DIM].
        batch_dones: Output batch dones [SAMPLE_SIZE].
        buf_states: Replay buffer states storage [CAPACITY, OBS_DIM].
        buf_actions: Replay buffer actions storage [CAPACITY, ACTION_DIM].
        buf_rewards: Replay buffer rewards storage [CAPACITY].
        buf_next_states: Replay buffer next states storage [CAPACITY, OBS_DIM].
        buf_dones: Replay buffer dones storage [CAPACITY].
        indices: Sampled indices [SAMPLE_SIZE].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SAMPLE_SIZE:
        return

    var buf_idx = Int(indices[i])

    for d in range(OBS_DIM):
        batch_states[i, d] = buf_states[buf_idx, d]
        batch_next_states[i, d] = buf_next_states[buf_idx, d]

    for a in range(ACTION_DIM):
        batch_actions[i, a] = buf_actions[buf_idx, a]

    batch_rewards[i] = buf_rewards[buf_idx]
    batch_dones[i] = buf_dones[buf_idx]


# =============================================================================
# TD Target Computation (shared by DDPG/TD3/SAC)
# =============================================================================


@always_inline
def td_target_continuous_kernel[
    dtype: DType,
    BATCH: Int,
](
    td_targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    next_q_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute DDPG TD targets: y = r + γ * Q_target(s', a') * (1 - done).

    One thread per batch sample.

    Args:
        td_targets:    Output TD targets [BATCH].
        rewards:       Sampled rewards [BATCH].
        next_q_values: Q-values from target critic at next states [BATCH].
        dones:         Done flags [BATCH] (1.0 = terminal).
        gamma:         Discount factor.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return

    var one = Scalar[dtype](1.0)
    td_targets[i] = rewards[i] + gamma * next_q_values[i] * (one - dones[i])


@always_inline
def td_target_min_twin_kernel[
    dtype: DType,
    BATCH: Int,
    use_entropy: Bool,
](
    td_targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    q1_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    q2_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    gamma: Scalar[dtype],
    alpha: Scalar[dtype],
):
    """Compute TD3/SAC TD targets with min(Q1,Q2) twin critics.

    TD3: y = r + γ * min(Q1_t, Q2_t)(s', a'_smoothed) * (1-done)
    SAC: y = r + γ * (min(Q1_t, Q2_t)(s', a') - α * log_π(a'|s')) * (1-done)

    Set use_entropy=False for TD3, use_entropy=True for SAC.
    For TD3, log_probs and alpha are unused (set alpha=0 or use_entropy=False).

    Args:
        td_targets: Output TD targets [BATCH].
        rewards:    Sampled rewards [BATCH].
        q1_values:  Q-values from target critic 1 at next states [BATCH].
        q2_values:  Q-values from target critic 2 at next states [BATCH].
        dones:      Done flags [BATCH] (1.0 = terminal).
        log_probs:  Log-probabilities of next actions (SAC only) [BATCH].
        gamma:      Discount factor.
        alpha:      Entropy coefficient (SAC) or 0.0 (TD3).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return

    var one = Scalar[dtype](1.0)
    var q_min = q1_values[i] if q1_values[i] < q2_values[i] else q2_values[i]

    comptime if use_entropy:
        # SAC: entropy bonus in target
        td_targets[i] = rewards[i] + gamma * (q_min - alpha * log_probs[i]) * (
            one - dones[i]
        )
    else:
        # TD3: no entropy
        td_targets[i] = rewards[i] + gamma * q_min * (one - dones[i])


# =============================================================================
# Actor Gradient Extraction (shared by DDPG/TD3/SAC)
# =============================================================================


@always_inline
def actor_grad_from_critic_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    CRITIC_IN: Int = OBS_DIM + ACTION_DIM,
](
    d_actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    d_critic_in: LayoutTensor[
        dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
    ],
):
    """Extract actor output gradients from critic input gradients (∂Q/∂a).

    After running critic backward with gradient [-1/BATCH, ...] (gradient ascent),
    the critic's input gradient d_critic_in has shape [BATCH, CRITIC_IN].
    The action portion (columns OBS_DIM:OBS_DIM+ACTION_DIM) is the policy gradient.
    CRITIC_IN defaults to OBS_DIM + ACTION_DIM but can be passed explicitly.

    One thread per (batch, action_dim) element.

    Args:
        d_actor_out: Output actor gradient [BATCH, ACTION_DIM].
        d_critic_in: Critic input gradient [BATCH, CRITIC_IN].
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return

    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM
    d_actor_out[b, a] = d_critic_in[b, OBS_DIM + a]


# =============================================================================
# Batch Obs-Action Concatenation (DDPG/TD3/SAC critic input construction)
# =============================================================================


@always_inline
def concat_obs_action_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    CONCAT_DIM: Int = OBS_DIM + ACTION_DIM,
](
    dst: LayoutTensor[dtype, Layout.row_major(BATCH, CONCAT_DIM), MutAnyOrigin],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    act: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
):
    """Concatenate [BATCH, OBS_DIM] and [BATCH, ACTION_DIM] → [BATCH, CONCAT_DIM].

    Used by DDPG/TD3/SAC to build critic inputs (obs ‖ action) on GPU.
    One thread per output element. CONCAT_DIM defaults to OBS_DIM + ACTION_DIM
    but can be passed explicitly to avoid type unification issues.

    Args:
        dst: Output tensor [BATCH, CONCAT_DIM].
        obs: Observations [BATCH, OBS_DIM].
        act: Actions [BATCH, ACTION_DIM].
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * CONCAT_DIM
    if tid >= total:
        return
    var b = tid // CONCAT_DIM
    var c = tid % CONCAT_DIM
    if c < OBS_DIM:
        dst[b, c] = obs[b, c]
    else:
        dst[b, c] = act[b, c - OBS_DIM]


# =============================================================================
# DDPG Action Kernels (scale + optional exploration noise)
# =============================================================================


@always_inline
def scale_clip_actions_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    raw_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    action_scale: Scalar[dtype],
):
    """Scale tanh actor output by action_scale and clip to [-scale, scale].

    Used for greedy (deterministic) action selection on GPU.
    One thread per (batch, action) element.

    Args:
        actions_out: Scaled and clipped output [BATCH, ACTION_DIM].
        raw_actions: Tanh actor output in [-1, 1] [BATCH, ACTION_DIM].
        action_scale: Scaling factor (action range = [-scale, scale]).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return
    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM
    var val = raw_actions[b, a] * action_scale
    if val > action_scale:
        val = action_scale
    elif val < -action_scale:
        val = -action_scale
    actions_out[b, a] = val


@always_inline
def ddpg_exploration_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    raw_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    noise_std: Scalar[dtype],
    action_scale: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
):
    """Scale actor output and add Gaussian exploration noise (DDPG-style).

    action = clip(raw * action_scale + noise_std * action_scale * N(0,1),
                  -action_scale, +action_scale)

    Uses PhiloxRandom for GPU-safe noise generation (no Float64).
    One thread per (batch, action) element.

    Args:
        actions_out: Output noisy actions [BATCH, ACTION_DIM].
        raw_actions: Tanh actor output in [-1, 1] [BATCH, ACTION_DIM].
        noise_std:   Noise std relative to action scale (e.g. 0.1).
        action_scale: Action range bound (output clipped to [-scale, scale]).
        rng_seed:    Random seed (should vary per call).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return
    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM

    # PhiloxRandom Box-Muller for Gaussian noise
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(b) * UInt64(ACTION_DIM) + UInt64(a),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var u1 = Float32(rand_vals[0]) + Float32(1e-8)
    var u2 = Float32(rand_vals[1])
    var mag = sqrt(Float32(-2.0) * log(u1))
    var z = Scalar[dtype](mag * cos(u2 * Float32(6.283185307179586)))

    var val = raw_actions[b, a] * action_scale + noise_std * action_scale * z
    if val > action_scale:
        val = action_scale
    elif val < -action_scale:
        val = -action_scale
    actions_out[b, a] = val


# =============================================================================
# Uniform Random Actions (warmup exploration)
# =============================================================================


@always_inline
def uniform_random_actions_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    action_scale: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
):
    """Fill actions with uniform random values in [-action_scale, action_scale].

    Used during warmup to match CleanRL's env.action_space.sample() behavior.
    One thread per (batch, action) element.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return
    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM

    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(b) * UInt64(ACTION_DIM) + UInt64(a),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    # Map [0, 1] -> [-action_scale, action_scale]
    var u = Scalar[dtype](Float32(rand_vals[0]))
    actions_out[b, a] = (u * 2 - 1) * action_scale


# =============================================================================
# Discrete Warmup: Uniform Random Action Indices
# =============================================================================


@always_inline
def uniform_random_discrete_actions_kernel[
    dtype: DType,
    BATCH: Int,
    NUM_ACTIONS: Int,
](
    actions_out: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    rng_seed: Scalar[DType.uint32],
):
    """Fill actions with uniform random integer indices in [0, NUM_ACTIONS).

    Used during warmup for discrete action environments.
    One thread per environment.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH:
        return

    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(tid) * UInt64(2654435761),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var u = Float32(rand_vals[0])
    actions_out[tid] = Scalar[dtype](
        Int(u * Float32(NUM_ACTIONS)) % NUM_ACTIONS
    )


def uniform_random_legal_actions_kernel[
    dtype: DType,
    BATCH: Int,
    NUM_ACTIONS: Int,
](
    actions_out: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    legal_masks: LayoutTensor[
        dtype, Layout.row_major(BATCH * NUM_ACTIONS), MutAnyOrigin
    ],
    rng_seed: Scalar[DType.uint32],
):
    """Pick a uniform random LEGAL action using the legal mask.

    For board games where illegal moves end the game with -1 reward.
    Falls back to action 0 if no legal actions (shouldn't happen).
    One thread per environment.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH:
        return

    # Count legal actions
    var n_legal = 0
    for a in range(NUM_ACTIONS):
        if rebind[Scalar[dtype]](legal_masks[tid * NUM_ACTIONS + a]) > Scalar[
            dtype
        ](0.5):
            n_legal += 1

    if n_legal == 0:
        actions_out[tid] = Scalar[dtype](0)
        return

    # Pick random index among legal actions
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(tid) * UInt64(2654435761),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var target = Int(Float32(rand_vals[0]) * Float32(n_legal)) % n_legal

    var count = 0
    for a in range(NUM_ACTIONS):
        if rebind[Scalar[dtype]](legal_masks[tid * NUM_ACTIONS + a]) > Scalar[
            dtype
        ](0.5):
            if count == target:
                actions_out[tid] = Scalar[dtype](a)
                return
            count += 1

    actions_out[tid] = Scalar[dtype](0)


# =============================================================================
# Extract Observations from State (after selective reset)
# =============================================================================


@always_inline
def _extract_obs_kernel[
    dtype: DType,
    BATCH: Int,
    STATE_SIZE: Int,
    OBS_DIM: Int,
](
    states: LayoutTensor[
        dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
):
    """Copy state[:OBS_DIM] → obs for all environments.

    Called after selective_reset_kernel_gpu to update obs_buf so that the
    next iteration's prev_obs sees the initial obs of the new episode,
    not the terminal obs of the previous one.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    for d in range(OBS_DIM):
        obs[i, d] = states[i, d]


# =============================================================================
# TD Critic MSE Gradient (shared by DDPG/TD3/SAC)
# =============================================================================


@always_inline
def td_mse_grad_kernel[
    dtype: DType,
    BATCH: Int,
    Q_DIM: Int = 1,
](
    q_grad: LayoutTensor[dtype, Layout.row_major(BATCH, Q_DIM), MutAnyOrigin],
    q: LayoutTensor[dtype, Layout.row_major(BATCH, Q_DIM), MutAnyOrigin],
    targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
):
    """MSE loss backward for scalar TD critic: q_grad[i,0] = 2*(q-target)/BATCH.

    One thread per batch element. Q_DIM defaults to 1 but can be passed
    explicitly to avoid type unification issues with generic agents.

    Args:
        q_grad:  Output gradient [BATCH, Q_DIM] (written).
        q:       Critic output Q-values [BATCH, Q_DIM].
        targets: TD targets [BATCH].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    q_grad[i, 0] = (
        Scalar[dtype](2.0) * (q[i, 0] - targets[i]) / Scalar[dtype](BATCH)
    )


# =============================================================================
# DQN TD Target Kernels
# =============================================================================


@always_inline
def dqn_td_target_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    next_q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute TD targets for standard DQN: target = r + gamma * max_a Q(s', a) * (1 - done).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var max_q = next_q_values[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = next_q_values[b, a]
        if q > max_q:
            max_q = q

    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * max_q * done_mask


@always_inline
def dqn_double_td_target_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    online_next_q: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    target_next_q: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute TD targets for Double DQN: target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var best_action = 0
    var best_q = online_next_q[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = online_next_q[b, a]
        if q > best_q:
            best_q = q
            best_action = a

    var target_q = target_next_q[b, best_action]
    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * target_q * done_mask


# =============================================================================
# Dueling DQN Kernels
# =============================================================================


@always_inline
def dueling_combine_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
    DUELING_OUT: Int,
](
    # Outputs
    q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    # Inputs
    dueling_output: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, DUELING_OUT), MutAnyOrigin
    ],
):
    """Combine dueling output [V, A1..An] into Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A)).

    Each thread handles one batch sample.

    Parameters:
        dtype: Data type.
        BATCH_SIZE: Batch size.
        NUM_ACTIONS: Number of discrete actions.
        DUELING_OUT: Dueling model output dimension (1 + NUM_ACTIONS).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # V(s) is the first element
    var v_s = dueling_output[b, 0]

    # Compute mean advantage
    var mean_adv: dueling_output.element_type = 0.0
    for a in range(NUM_ACTIONS):
        mean_adv += dueling_output[b, 1 + a]
    mean_adv /= Scalar[dtype](NUM_ACTIONS)

    # Q(s,a) = V(s) + (A(s,a) - mean(A))
    for a in range(NUM_ACTIONS):
        var adv = dueling_output[b, 1 + a]
        q_values[b, a] = v_s + (adv - mean_adv)


@always_inline
def dueling_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
    DUELING_OUT: Int,
](
    # Outputs
    dueling_grad: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, DUELING_OUT), MutAnyOrigin
    ],
    # Inputs
    dq_grad: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
):
    """Transform dQ gradients to dueling output gradients.

    dV = sum(dQ_j)
    dA_i = dQ_i - (1/n) * sum(dQ_j)

    Each thread handles one batch sample.

    Parameters:
        dtype: Data type.
        BATCH_SIZE: Batch size.
        NUM_ACTIONS: Number of discrete actions.
        DUELING_OUT: Dueling model output dimension (1 + NUM_ACTIONS).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # Compute sum of dQ gradients
    var sum_dq: dq_grad.element_type = 0.0
    for a in range(NUM_ACTIONS):
        sum_dq += dq_grad[b, a]

    # dV = sum(dQ)
    dueling_grad[b, 0] = sum_dq

    # dA_i = dQ_i - (1/n) * sum(dQ)
    var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        dueling_grad[b, 1 + a] = dq_grad[b, a] - one_over_n * sum_dq


# =============================================================================
# TD3 Gaussian Noise Kernel
# =============================================================================


@always_inline
def add_gaussian_noise_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    noisy_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    noise_std: Scalar[dtype],
    noise_clip: Scalar[dtype],
    action_min: Scalar[dtype],
    action_max: Scalar[dtype],
    rng_counter: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
):
    """Add clipped Gaussian exploration noise to actions (TD3-style).

    Each element gets independent noise from N(0, noise_std²), clipped to
    [-noise_clip, noise_clip], then the result is clipped to [action_min, action_max].

    Uses PhiloxRandom for GPU-safe noise generation (no Float64).
    One thread per (batch, action_dim) element.
    Reads seed from GPU-side rng_counter (CUDA graph compatible).

    Args:
        noisy_actions: Output noisy actions [BATCH, ACTION_DIM].
        actions:       Clean actions from actor [BATCH, ACTION_DIM].
        noise_std:     Noise standard deviation.
        noise_clip:    Maximum absolute noise value.
        action_min:    Minimum action value (e.g. -action_scale).
        action_max:    Maximum action value (e.g. +action_scale).
        rng_counter:   GPU-side RNG counter [1] (read, not modified).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return

    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM

    var rng_seed = UInt64(rng_counter[0].cast[DType.uint64]())
    # PhiloxRandom Box-Muller for Gaussian noise
    var philox = PhiloxRandom(
        seed=rng_seed + UInt64(b) * UInt64(ACTION_DIM) + UInt64(a),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var u1 = Float32(rand_vals[0]) + Float32(1e-8)
    var u2 = Float32(rand_vals[1])
    var mag = sqrt(Float32(-2.0) * log(u1))
    var z = Scalar[dtype](mag * cos(u2 * Float32(6.283185307179586)))

    # Scale and clip noise
    var noise = z * noise_std
    if noise < -noise_clip:
        noise = -noise_clip
    if noise > noise_clip:
        noise = noise_clip

    # Apply noise and clip to action range
    var noisy = actions[b, a] + noise
    if noisy < action_min:
        noisy = action_min
    if noisy > action_max:
        noisy = action_max

    noisy_actions[b, a] = noisy


# =============================================================================
# SAC Reparameterization Kernels
# =============================================================================


@always_inline
def sac_sample_actions_kernel[
    dtype: DType where dtype.is_floating_point(),
    N: Int,
    ACTION_DIM: Int,
    ACTOR_OUT_DIM: Int = ACTION_DIM + ACTION_DIM,
](
    actions: LayoutTensor[dtype, Layout.row_major(N, ACTION_DIM), MutAnyOrigin],
    actor_out: LayoutTensor[
        dtype, Layout.row_major(N, ACTOR_OUT_DIM), MutAnyOrigin
    ],
    action_scale: Scalar[dtype],
    log_std_min: Scalar[dtype],
    log_std_max: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
):
    """SAC inference: sample actions from stochastic actor output, scaled by action_scale.

    Takes actor_out[N, 2*ACTION_DIM] where columns [0, ACTION_DIM) are mean
    and columns [ACTION_DIM, 2*ACTION_DIM) are log_std.

    Computes: a = tanh(mean + exp(clamp(log_std)) * ε) * action_scale

    Uses PhiloxRandom for GPU-safe noise generation.
    No eps_cache saved (inference only).
    One thread per environment.

    Args:
        actions:     Output scaled actions in [-action_scale, action_scale] [N, ACTION_DIM].
        actor_out:   Actor network output [N, 2*ACTION_DIM] (mean || log_std).
        action_scale: Action range bound (output clipped to [-scale, scale]).
        log_std_min: Minimum log_std clamp value.
        log_std_max: Maximum log_std clamp value.
        rng_seed:    Random seed (should vary per call).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= N:
        return

    var one = Scalar[dtype](1.0)
    var half = Scalar[dtype](0.5)
    var ls_range = log_std_max - log_std_min

    for a in range(ACTION_DIM):
        var mean_val = actor_out[b, a]
        # Affine rescale: tanh already applied by LinearTanh head
        var tanh_out = actor_out[b, ACTION_DIM + a]
        var ls = log_std_min + half * ls_range * (tanh_out + one)

        var std_val = exp(ls)

        # Sample ε ~ N(0, 1) using PhiloxRandom Box-Muller
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(b) * UInt64(ACTION_DIM) + UInt64(a),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var u1 = Scalar[dtype](rand_vals[0]) + 1e-8
        var u2 = Scalar[dtype](rand_vals[1])
        var mag = sqrt(Scalar[dtype](-2.0) * log(u1))
        var eps = Scalar[dtype](mag * cos(u2 * 6.283185307179586))

        # Reparameterize, squash, scale
        var z = mean_val + std_val * eps
        var act = tanh(z) * action_scale
        actions[b, a] = act


@always_inline
def sac_rsample_with_cache_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    ACTION_DIM: Int,
    ACTOR_OUT_DIM: Int = ACTION_DIM + ACTION_DIM,
](
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    eps_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTOR_OUT_DIM), MutAnyOrigin
    ],
    log_std_min: Scalar[dtype],
    log_std_max: Scalar[dtype],
    rng_counter: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
):
    """SAC training forward: reparameterize, compute log_prob, save eps for backward.

    Actions are in [-1, 1] (NOT scaled by action_scale) — the scale is
    factored out during actor gradient computation.

    eps_cache[b, a] saves the noise epsilon used to sample action a for batch b.
    It is needed by sac_rsample_bwd_kernel to backpropagate through log_std.

    Computes:
        ε ~ N(0, 1)
        σ = exp(clamp(log_std))
        z = mean + σ * ε
        a = tanh(z)
        log π(a|s) = Σ_j [-0.5*ε_j² - 0.5*log(2π) - ls_j - log(1 - a_j²)]

    Uses PhiloxRandom for GPU-safe noise generation.
    Reads seed from GPU-side rng_counter (CUDA graph compatible).
    One thread per batch sample.

    Args:
        actions:    Output actions in (-1, 1) [BATCH, ACTION_DIM].
        log_probs:  Output log-probabilities (summed over action dims) [BATCH].
        eps_cache:  Output saved noise ε [BATCH, ACTION_DIM] (for backward).
        actor_out:  Actor network output [BATCH, 2*ACTION_DIM] (mean || log_std).
        log_std_min: Minimum log_std clamp value.
        log_std_max: Maximum log_std clamp value.
        rng_counter: GPU-side RNG counter [1] (read, not modified).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var rng_seed = UInt64(rng_counter[0].cast[DType.uint64]())
    var half_log_2pi = Scalar[dtype](0.9189385332046727)
    var one = Scalar[dtype](1.0)
    var half = Scalar[dtype](0.5)
    var ls_range = log_std_max - log_std_min
    var lp: log_probs.element_type = 0.0

    for a in range(ACTION_DIM):
        # Affine rescale: tanh already applied by LinearTanh head
        var tanh_out = actor_out[b, ACTION_DIM + a]
        var ls = log_std_min + half * ls_range * (tanh_out + one)

        var std_val = exp(ls)

        # Sample ε ~ N(0, 1) using PhiloxRandom Box-Muller
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(b) * UInt64(ACTION_DIM) + UInt64(a),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var u1 = Scalar[dtype](rand_vals[0]) + 1e-8
        var u2 = Scalar[dtype](rand_vals[1])
        var mag = sqrt(Scalar[dtype](-2.0) * log(u1))
        var eps = Scalar[dtype](mag * cos(u2 * 6.283185307179586))

        # Save eps for backward pass
        eps_cache[b, a] = eps

        # Reparameterize: z = mean + σ * ε, a = tanh(z)
        var z = actor_out[b, a] + std_val * eps
        var act = tanh(z)
        actions[b, a] = act

        # Log-prob contribution from this dimension
        var one_minus_tanh2 = one - act * act
        if one_minus_tanh2 < Scalar[dtype](1e-6):
            one_minus_tanh2 = Scalar[dtype](1e-6)

        lp += (
            -Scalar[dtype](0.5) * eps * eps
            - half_log_2pi
            - ls
            - log(one_minus_tanh2)
        )

    log_probs[b] = lp


@always_inline
def sac_rsample_bwd_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    ACTION_DIM: Int,
    ACTOR_OUT_DIM: Int = ACTION_DIM + ACTION_DIM,
](
    actor_grad: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTOR_OUT_DIM), MutAnyOrigin
    ],
    grad_act: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    alpha_per_sample: Scalar[dtype],
    curr_act: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    eps_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTOR_OUT_DIM), MutAnyOrigin
    ],
    log_std_min: Scalar[dtype],
    log_std_max: Scalar[dtype],
):
    """SAC backward through the reparameterization trick.

    Computes the full actor gradient actor_grad[BATCH, 2*ACTION_DIM] from:
      - grad_act[b, j]: ∂(-mean(Q))/∂a_j from critic backward with dq=-1/BATCH
      - alpha_per_sample = alpha/BATCH: entropy coefficient per sample

    Derivation for each (b, j):
        a      = curr_act[b, j]             (tanh-squashed action from forward)
        ls     = clamp(actor_out[b, ACTION_DIM+j])
        σ      = exp(ls)
        ε      = eps_cache[b, j]            (noise saved during forward)

        d_z    = grad_act[b,j] * (1 - a²)  # backward through tanh
               + alpha_per_sample * 2*a    # entropy term: d(-log(1-tanh²))/da * (1-a²)

        actor_grad[b, j]            = d_z                           # grad wrt mean
        actor_grad[b, ACTION_DIM+j] = d_z * σ * ε - alpha_per_sample  # grad wrt log_std

    One thread per batch sample.

    Args:
        actor_grad:        Output gradient [BATCH, 2*ACTION_DIM] for network backward.
        grad_act:          ∂(-mean(Q))/∂a from critic backward [BATCH, ACTION_DIM].
        alpha_per_sample:  Alpha / BATCH (entropy coefficient, scalar).
        curr_act:          Tanh-squashed actions from forward pass [BATCH, ACTION_DIM].
        eps_cache:         Saved noise ε from forward pass [BATCH, ACTION_DIM].
        actor_out:         Raw actor network output [BATCH, 2*ACTION_DIM] (mean || log_std).
        log_std_min:       Lower clamp for log_std (same as forward).
        log_std_max:       Upper clamp for log_std (same as forward).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var one = Scalar[dtype](1.0)
    var two = Scalar[dtype](2.0)
    var half = Scalar[dtype](0.5)
    var ls_range = log_std_max - log_std_min

    for a in range(ACTION_DIM):
        var act_val = curr_act[b, a]
        # Affine rescale: tanh already applied by LinearTanh head
        var tanh_out = actor_out[b, ACTION_DIM + a]
        var ls = log_std_min + half * ls_range * (tanh_out + one)

        var sigma = exp(ls)
        var eps = eps_cache[b, a]

        # d_z: gradient through tanh(z) from critic + entropy contribution
        var one_minus_a2 = one - act_val * act_val
        var d_z = (
            grad_act[b, a] * one_minus_a2 + alpha_per_sample * two * act_val
        )

        # Grad wrt mean: z = mean + σ*ε, so ∂z/∂mean = 1
        actor_grad[b, a] = d_z

        # Grad wrt log_std: ∂z/∂log_std = σ*ε, plus entropy term -1
        # Chain rule for affine: d(ls)/d(tanh_out) = 0.5 * range (constant)
        # tanh derivative is handled by LinearTanh in model backward
        var d_ls = d_z * sigma * eps - alpha_per_sample
        var d_ls_d_tanh_out = half * ls_range
        actor_grad[b, ACTION_DIM + a] = d_ls * d_ls_d_tanh_out


# =============================================================================
# SAC min(Q1, Q2) Masked Gradient Kernels
# =============================================================================


@always_inline
def min_q_dq_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    Q_DIM: Int = 1,
](
    dq1: LayoutTensor[dtype, Layout.row_major(BATCH, Q_DIM), MutAnyOrigin],
    dq2: LayoutTensor[dtype, Layout.row_major(BATCH, Q_DIM), MutAnyOrigin],
    q1: LayoutTensor[dtype, Layout.row_major(BATCH, Q_DIM), MutAnyOrigin],
    q2: LayoutTensor[dtype, Layout.row_major(BATCH, Q_DIM), MutAnyOrigin],
):
    """Create masked dq gradients based on min(Q1, Q2).

    For each sample b:
        if Q1[b] <= Q2[b]: dq1[b] = -1/BATCH, dq2[b] = 0
        else:              dq1[b] = 0,         dq2[b] = -1/BATCH

    The actor loss is: L = alpha * log_pi - min(Q1, Q2)
    Minimizing L maximizes Q and entropy. The gradient of -min(Q1,Q2)
    routes through whichever critic has the lower Q value.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var neg_inv_batch = Scalar[dtype](-1.0 / Scalar[dtype](BATCH))
    var zero = Scalar[dtype](0.0)

    if q1[b, 0] <= q2[b, 0]:
        dq1[b, 0] = neg_inv_batch
        dq2[b, 0] = zero
    else:
        dq1[b, 0] = zero
        dq2[b, 0] = neg_inv_batch


@always_inline
def add_ci_grads_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    DIM: Int,
](
    dst: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    """Add src to dst elementwise: dst[b,d] += src[b,d].

    Used to combine action gradients from Q1 and Q2 backward passes.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return

    var b = idx // DIM
    var d = idx % DIM
    dst[b, d] = dst[b, d] + src[b, d]


# =============================================================================
# PPO Continuous Action Kernels
# =============================================================================


@always_inline
def _sample_continuous_actions_kernel[
    dtype: DType,
    N_ENVS: Int,
    ACTION_DIM: Int,
](
    # Actor network output (mean and log_std concatenated)
    actor_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, ACTION_DIM * 2), MutAnyOrigin
    ],
    # Outputs
    actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, ACTION_DIM), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Random seed
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Sample continuous actions from unbounded Gaussian policy on GPU (CleanRL-style).

    Actor output layout: [mean (ACTION_DIM) | log_std (ACTION_DIM)]
    Uses reparameterization trick: action = mean + exp(log_std) * noise

    NO TANH SQUASHING - actions are unbounded, clipping happens at environment boundary.
    This matches CleanRL's PPO continuous implementation and avoids train/eval mismatch.

    Log probability is simple Gaussian (no Jacobian correction):
    log_prob = sum(-0.5 * (log(2*pi) + 2*log_std + ((action-mean)/std)^2))
    """
    comptime EPS: Scalar[dtype] = 1e-6
    comptime LOG_STD_MIN: Scalar[dtype] = -5.0  # Match StochasticActor
    comptime LOG_STD_MAX: Scalar[dtype] = 2.0
    comptime LOG_2PI: Scalar[dtype] = 1.8378770664093453

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    var total_log_prob: log_probs.element_type = 0.0

    for j in range(ACTION_DIM):
        # Extract tensor elements using [0] pattern and convert to Scalar[dtype]
        var mean_raw = actor_output[i, j]
        var log_std_raw = actor_output[i, ACTION_DIM + j]
        var mean = Scalar[dtype](mean_raw[0])
        var log_std = Scalar[dtype](log_std_raw[0])

        # Clamp log_std for numerical stability (must match gradient kernel)
        if log_std < LOG_STD_MIN:
            log_std = LOG_STD_MIN
        elif log_std > LOG_STD_MAX:
            log_std = LOG_STD_MAX

        # Sample Gaussian noise using Box-Muller transform with PhiloxRandom
        # Each (i, j) pair gets unique seed and offset for independent random streams
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(i) * UInt64(ACTION_DIM) + UInt64(j),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var u1 = rand_vals[0]
        var u2 = rand_vals[1]

        # Box-Muller transform for standard normal
        # log() requires Float32
        var u1_for_log = Float32(u1) + Float32(1e-8)
        var u2_for_cos = Float32(u2)

        var mag = sqrt(Float32(-2.0) * log(u1_for_log))
        var noise = Scalar[dtype](
            mag * cos(u2_for_cos * Float32(6.283185307179586))
        )

        # Reparameterization: action = mean + std * noise (unbounded Gaussian)
        var std = exp(log_std)
        var action = mean + std * noise

        # Store unbounded action directly (no tanh squashing)
        actions[i, j] = action

        # Simple Gaussian log probability (no squashing correction)
        var action_normalized = (action - mean) / (std + EPS)

        var neg_half: Scalar[dtype] = -0.5
        var log_gaussian = neg_half * (
            LOG_2PI
            + Scalar[dtype](2.0) * log_std
            + action_normalized * action_normalized
        )

        total_log_prob = total_log_prob + log_gaussian

    log_probs[i] = total_log_prob


@always_inline
def _store_continuous_pre_step_kernel[
    dtype: DType,
    N_ENVS: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
](
    # Outputs - rollout buffer at timestep t
    r_obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    r_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, ACTION_DIM), MutAnyOrigin
    ],
    r_log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    r_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Inputs - current step data
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, ACTION_DIM), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Store pre-step data to rollout buffer for continuous actions.

    Stores unbounded actions directly (CleanRL-style, no tanh squashing).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    for d in range(OBS_DIM):
        r_obs[i, d] = obs[i, d]
    for a in range(ACTION_DIM):
        r_actions[i, a] = actions[i, a]
    r_log_probs[i] = log_probs[i]
    r_values[i] = values[i]


@always_inline
def _store_post_step_kernel[
    dtype: DType,
    N_ENVS: Int,
](
    # Outputs - rollout buffer at timestep t
    r_rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    r_dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Inputs - current step data
    rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Store post-step data (rewards, dones) to rollout buffer."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    r_rewards[i] = rewards[i]
    r_dones[i] = dones[i]


@always_inline
def ppo_continuous_gather_minibatch_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    TOTAL_SIZE: Int,
](
    # Outputs - minibatch buffers
    mb_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    mb_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    mb_advantages: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    mb_returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    mb_old_log_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    mb_old_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    # Inputs - rollout buffers and indices
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE, OBS_DIM), MutAnyOrigin
    ],
    rollout_actions: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    rollout_log_probs: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin
    ],
    rollout_values: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin
    ],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    batch_size: Int,
):
    """Gather samples from rollout buffer using shuffled indices for continuous actions.

    Uses unbounded actions directly (CleanRL-style, no tanh squashing).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= batch_size:
        return

    var src_idx = Int(indices[i])

    # Gather observation
    for d in range(OBS_DIM):
        mb_obs[i, d] = rollout_obs[src_idx, d]

    # Gather unbounded actions directly
    for a in range(ACTION_DIM):
        mb_actions[i, a] = rollout_actions[src_idx, a]

    mb_advantages[i] = advantages[src_idx]
    mb_returns[i] = returns[src_idx]
    mb_old_log_probs[i] = rollout_log_probs[src_idx]
    mb_old_values[i] = rollout_values[src_idx]


@always_inline
def ppo_continuous_actor_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
](
    # Outputs
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM * 2), MutAnyOrigin
    ],
    kl_divergences: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    entropies: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_flags: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    actor_output: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM * 2), MutAnyOrigin
    ],
    old_log_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    clip_epsilon: Scalar[dtype],
    entropy_coef: Scalar[dtype],
    batch_size: Int,
) where dtype.is_floating_point():
    """Compute gradient for PPO actor with unbounded Gaussian policy (CleanRL-style).

    For unbounded Gaussian policy (no tanh squashing):
    - log_prob = sum_j(-0.5 * (LOG_2PI + 2*log_std[j] + ((action[j]-mean[j])/std[j])^2))
    - d_log_prob/d_mean = (action - mean) / std^2
    - d_log_prob/d_log_std = ((action - mean)^2 / std^2 - 1)

    Uses unbounded actions directly (stored from collection).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    # Local constants
    var eps: Scalar[dtype] = 1e-6
    var log_2pi: Scalar[dtype] = 1.8378770664093453
    var one: Scalar[dtype] = 1.0
    var two: Scalar[dtype] = 2.0
    var half: Scalar[dtype] = 0.5
    var neg_half: Scalar[dtype] = -0.5

    # Numerical stability constants
    comptime LOG_STD_MIN: Scalar[dtype] = -5.0  # Match StochasticActor
    comptime LOG_STD_MAX: Scalar[dtype] = 2.0
    comptime LOG_PROB_DIFF_MAX: Scalar[dtype] = 20.0  # Prevent ratio explosion
    comptime GRAD_CLIP: Scalar[dtype] = 10.0  # Clip individual gradients

    var advantage = advantages[b]
    var old_log_prob = old_log_probs[b]

    # Compute new log_prob using stored actions
    var new_log_prob: Scalar[dtype] = 0.0
    var entropy_sum: Scalar[dtype] = 0.0

    # Arrays to store intermediate values for gradient computation
    var action_vals = InlineArray[Scalar[dtype], ACTION_DIM](
        fill=Scalar[dtype](0.0)
    )
    var mean_vals = InlineArray[Scalar[dtype], ACTION_DIM](
        fill=Scalar[dtype](0.0)
    )
    var std_vals = InlineArray[Scalar[dtype], ACTION_DIM](
        fill=Scalar[dtype](0.0)
    )

    for j in range(ACTION_DIM):
        # Extract tensor elements using [0] pattern and wrap in Scalar[dtype]
        var mean_val_raw = actor_output[b, j]
        var log_std_val_raw = actor_output[b, ACTION_DIM + j]
        var action_val_raw = actions[b, j]

        # Convert to Scalar[dtype] using [0] extraction
        var mean_val = Scalar[dtype](mean_val_raw[0])
        var log_std_val = Scalar[dtype](log_std_val_raw[0])
        var action_val = Scalar[dtype](action_val_raw[0])

        # Clamp log_std for numerical stability
        if log_std_val < LOG_STD_MIN:
            log_std_val = LOG_STD_MIN
        elif log_std_val > LOG_STD_MAX:
            log_std_val = LOG_STD_MAX

        var std_val = exp(log_std_val)
        # Compute normalized action for log_prob
        var action_normalized = (action_val - mean_val) / (std_val + eps)

        # Store for gradient computation
        action_vals[j] = action_val
        mean_vals[j] = mean_val
        std_vals[j] = std_val

        # Simple Gaussian log probability (no squashing correction)
        var log_gaussian = neg_half * (
            log_2pi + two * log_std_val + action_normalized * action_normalized
        )

        new_log_prob = new_log_prob + log_gaussian

        # Entropy: H = 0.5 * (LOG_2PI + 1 + 2*log_std)
        entropy_sum = entropy_sum + half * (log_2pi + one + two * log_std_val)

    # Clamp log_prob difference to prevent ratio explosion
    var log_prob_diff = new_log_prob - old_log_prob
    if log_prob_diff > LOG_PROB_DIFF_MAX:
        log_prob_diff = LOG_PROB_DIFF_MAX
    elif log_prob_diff < -LOG_PROB_DIFF_MAX:
        log_prob_diff = -LOG_PROB_DIFF_MAX

    # Probability ratio with clamped difference
    var ratio = exp(log_prob_diff)

    # KL divergence approximation: (ratio - 1) - log(ratio)
    var kl = (ratio - one) - log_prob_diff
    # Clamp KL to reasonable range
    if kl < Scalar[dtype](0.0):
        kl = Scalar[dtype](0.0)
    elif kl > Scalar[dtype](100.0):
        kl = Scalar[dtype](100.0)
    kl_divergences[b] = kl
    entropies[b] = entropy_sum

    # Clip ratio for clipped objective
    var clipped_ratio = ratio
    if clipped_ratio < one - clip_epsilon:
        clipped_ratio = one - clip_epsilon
    elif clipped_ratio > one + clip_epsilon:
        clipped_ratio = one + clip_epsilon

    # PPO clipped objective: min(ratio * A, clipped_ratio * A)
    # Gradient is 0 when we use the clipped objective (i.e., clipped_ratio * A < ratio * A)
    var unclipped_obj = ratio * advantage
    var clipped_obj = clipped_ratio * advantage
    var is_clipped = clipped_obj < unclipped_obj
    clip_flags[b] = Scalar[dtype](1.0) if is_clipped else Scalar[dtype](0.0)

    # Compute gradients for mean and log_std
    var batch_size_scalar = Scalar[dtype](BATCH_SIZE)
    for j in range(ACTION_DIM):
        if is_clipped:
            grad_output[b, j] = Scalar[dtype](0.0)
            grad_output[b, ACTION_DIM + j] = Scalar[dtype](0.0)
        else:
            var action = action_vals[j]
            var mean = mean_vals[j]
            var std = std_vals[j]

            var action_normalized = (action - mean) / (std + eps)

            # d_log_prob/d_mean = action_normalized / std
            var d_log_prob_d_mean = action_normalized / (std + eps)

            # d_log_prob/d_log_std = (action_normalized^2 - 1)
            var d_log_prob_d_log_std = (
                action_normalized * action_normalized - one
            )

            # Entropy gradient: d_entropy/d_log_std = 1
            var d_entropy_d_log_std: Scalar[dtype] = 1.0

            # PPO gradient (negative because we maximize)
            var grad_mean = (
                -advantage * ratio * d_log_prob_d_mean
            ) / batch_size_scalar

            var grad_log_std = (
                -advantage * ratio * d_log_prob_d_log_std
                - entropy_coef * d_entropy_d_log_std
            ) / batch_size_scalar

            # Clip gradients to prevent explosion
            if grad_mean > GRAD_CLIP:
                grad_mean = GRAD_CLIP
            elif grad_mean < -GRAD_CLIP:
                grad_mean = -GRAD_CLIP

            if grad_log_std > GRAD_CLIP:
                grad_log_std = GRAD_CLIP
            elif grad_log_std < -GRAD_CLIP:
                grad_log_std = -GRAD_CLIP

            grad_output[b, j] = grad_mean
            grad_output[b, ACTION_DIM + j] = grad_log_std


# =============================================================================
# PPO Critic Gradient Kernels
# =============================================================================


@always_inline
def ppo_critic_grad_kernel[
    dtype: DType, BATCH_SIZE: Int
](
    # Outputs
    grad_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin
    ],
    # Inputs
    values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    value_loss_coef: Scalar[dtype],
    batch_size: Int,
):
    """Compute gradient for PPO critic (MSE loss scaled by value_loss_coef).

    Gradient: value_loss_coef * d(0.5 * mean((value - target)^2)) / d_value
            = value_loss_coef * (value - target) / BATCH_SIZE
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var value = values[b, 0]
    var target = returns[b]

    grad_values[b, 0] = (
        value_loss_coef * (value - target) / Scalar[dtype](BATCH_SIZE)
    )


@always_inline
def ppo_critic_grad_clipped_kernel[
    dtype: DType, BATCH_SIZE: Int
](
    # Outputs
    grad_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin
    ],
    # Inputs
    values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    old_values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_epsilon: Scalar[dtype],
    value_loss_coef: Scalar[dtype],
    batch_size: Int,
):
    """Compute gradient for PPO critic with value clipping, scaled by value_loss_coef.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var value = values[b, 0]
    var target = returns[b]
    var old_value = old_values[b]

    # Clip value prediction
    var value_clipped = old_value + max(
        min(value - old_value, clip_epsilon), -clip_epsilon
    )

    # Unclipped loss
    var loss_unclipped = (value - target) * (value - target)
    # Clipped loss
    var loss_clipped = (value_clipped - target) * (value_clipped - target)

    # Use max of clipped and unclipped
    if loss_clipped > loss_unclipped:
        # Use clipped gradient
        var clip_sign = Scalar[dtype](1.0)
        if value - old_value > clip_epsilon:
            clip_sign = Scalar[dtype](0.0)  # Gradient is 0 at boundary
        elif value - old_value < -clip_epsilon:
            clip_sign = Scalar[dtype](0.0)
        grad_values[b, 0] = (
            value_loss_coef
            * clip_sign
            * (value_clipped - target)
            / Scalar[dtype](BATCH_SIZE)
        )
    else:
        grad_values[b, 0] = (
            value_loss_coef * (value - target) / Scalar[dtype](BATCH_SIZE)
        )


# =============================================================================
# PPO Advantage Normalization
# =============================================================================


@always_inline
def normalize_advantages_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    # In/Out
    advantages: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs (pre-computed on CPU and passed in)
    mean: Scalar[dtype],
    std: Scalar[dtype],
    batch_size: Int,
):
    """Normalize advantages in-place using pre-computed mean and std."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    advantages[b] = (advantages[b] - mean) / (std + Scalar[dtype](1e-8))


# =============================================================================
# PPO Gradient Clipping Kernels
# =============================================================================


@always_inline
def gradient_norm_kernel[
    dtype: DType, PARAM_SIZE: Int, NUM_BLOCKS: Int, BLOCK_SIZE: Int
](
    partial_sums: LayoutTensor[
        dtype, Layout.row_major(NUM_BLOCKS), MutAnyOrigin
    ],
    grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
):
    """Compute partial sum of squared gradients for gradient norm."""
    var block_id = Int(block_idx.x)
    var thread_id = Int(thread_idx.x)
    var idx = block_id * BLOCK_SIZE + thread_id

    var shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if idx < PARAM_SIZE:
        var g = grads[idx]
        shared[thread_id] = g * g
    else:
        shared[thread_id] = Scalar[dtype](0.0)

    barrier()

    # Reduction within block
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2

    if thread_id == 0:
        partial_sums[block_id] = shared[0]


@always_inline
def gradient_reduce_and_compute_scale_kernel[
    dtype: DType, NUM_BLOCKS: Int, BLOCK_SIZE: Int
](
    scale_out: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    partial_sums: LayoutTensor[
        dtype, Layout.row_major(NUM_BLOCKS), MutAnyOrigin
    ],
    max_grad_norm: Scalar[dtype],
):
    """Reduce partial sums and compute clipping scale entirely on GPU.

    This kernel runs with a single block. It:
    1. Loads all partial sums into shared memory
    2. Reduces them to get total squared gradient norm
    3. Computes scale = min(1.0, max_grad_norm / norm)
    4. Stores scale to global memory for the next kernel
    """
    var thread_id = Int(thread_idx.x)

    # Shared memory for reduction (size = BLOCK_SIZE)
    var shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Load partial sums (handle case where NUM_BLOCKS > BLOCK_SIZE by striding)
    var local_sum = Scalar[dtype](0.0)
    var idx = thread_id
    while idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums[idx])
        idx += BLOCK_SIZE
    shared[thread_id] = local_sum

    barrier()

    # Reduction within block
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2

    # Thread 0 computes and stores the scale
    if thread_id == 0:
        var total_sq_sum = rebind[Scalar[dtype]](shared[0])
        var norm = Scalar[dtype](sqrt(total_sq_sum))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        scale_out[0] = scale


@always_inline
def gradient_apply_scale_kernel[
    dtype: DType, PARAM_SIZE: Int
](
    grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    scale_in: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
):
    """Apply precomputed scale to all gradients.

    This kernel reads the scale computed by gradient_reduce_and_compute_scale_kernel
    and applies it to all gradients. Always runs (no conditional), but when no
    clipping is needed, scale=1.0 so it's a no-op multiply.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= PARAM_SIZE:
        return

    var scale = scale_in[0]
    grads[i] = grads[i] * scale


@always_inline
def gradient_reduce_apply_fused_kernel[
    dtype: DType, PARAM_SIZE: Int, NUM_BLOCKS: Int, BLOCK_SIZE: Int
](
    grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    partial_sums: LayoutTensor[
        dtype, Layout.row_major(NUM_BLOCKS), MutAnyOrigin
    ],
    max_grad_norm: Scalar[dtype],
):
    """Fused kernel: reduce partial sums AND apply gradient clipping.

    Each block redundantly computes the total gradient norm by reducing
    all partial_sums (small array, ~NUM_BLOCKS elements), then applies
    the computed scale to its portion of gradients.

    This eliminates the single-block bottleneck of the 3-kernel approach.
    The redundant reduction across blocks is much cheaper than kernel
    launch overhead.
    """
    var block_id = Int(block_idx.x)
    var thread_id = Int(thread_idx.x)
    var idx = block_id * BLOCK_SIZE + thread_id

    # Shared memory for reduction (each block reduces ALL partial_sums)
    var shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Step 1: Each thread loads and sums multiple partial_sums elements
    # (striding through the partial_sums array)
    var local_sum = Scalar[dtype](0.0)
    var ps_idx = thread_id
    while ps_idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums[ps_idx])
        ps_idx += BLOCK_SIZE
    shared[thread_id] = local_sum

    barrier()

    # Step 2: Block reduction to get total squared gradient norm
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2

    # Step 3: Thread 0 computes scale, broadcasts via shared memory
    # (reuse shared[1] for scale since reduction is done)
    if thread_id == 0:
        var total_sq_sum = rebind[Scalar[dtype]](shared[0])
        var norm = Scalar[dtype](sqrt(total_sq_sum))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        shared[1] = scale  # Broadcast via shared memory

    barrier()

    # Step 4: All threads read the scale and apply to their gradient
    if idx < PARAM_SIZE:
        var scale = rebind[Scalar[dtype]](shared[1])
        grads[idx] = grads[idx] * scale


# =============================================================================
# PPO Log_std Parameter Clamping
# =============================================================================


@always_inline
def clamp_log_std_params_kernel[
    dtype: DType,
    PARAM_SIZE: Int,
    LOG_STD_OFFSET: Int,
    ACTION_DIM: Int,
](params: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],):
    """Clamp log_std parameters to valid range [-5.0, 2.0].

    This kernel should be called after each optimizer step to prevent
    log_std from drifting to extreme values during training.

    Parameters:
        dtype: Data type of the parameters.
        PARAM_SIZE: Total number of parameters.
        LOG_STD_OFFSET: Offset to log_std parameters within actor params.
        ACTION_DIM: Number of action dimensions (number of log_std params).

    Args:
        params: Actor network parameters.
    """
    comptime LOG_STD_MIN: Scalar[dtype] = -5.0
    comptime LOG_STD_MAX: Scalar[dtype] = 2.0

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= ACTION_DIM:
        return

    var param_idx = LOG_STD_OFFSET + i
    var val = params[param_idx]

    if val < LOG_STD_MIN:
        params[param_idx] = LOG_STD_MIN
    elif val > LOG_STD_MAX:
        params[param_idx] = LOG_STD_MAX


# =============================================================================
# PPO Discrete Action Kernels
# =============================================================================


@always_inline
def ppo_gather_minibatch_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    TOTAL_SIZE: Int,
](
    # Outputs - minibatch buffers
    mb_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    mb_actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    mb_advantages: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    mb_returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    mb_old_log_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    mb_old_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    # Inputs - rollout buffers and indices
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE, OBS_DIM), MutAnyOrigin
    ],
    rollout_actions: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    rollout_log_probs: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin
    ],
    rollout_values: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin
    ],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    batch_size: Int,
):
    """Gather samples from rollout buffer using shuffled indices."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= batch_size:
        return

    var src_idx = Int(indices[i])

    # Gather observation
    for d in range(OBS_DIM):
        mb_obs[i, d] = rollout_obs[src_idx, d]

    mb_actions[i] = rollout_actions[src_idx]
    mb_advantages[i] = advantages[src_idx]
    mb_returns[i] = returns[src_idx]
    mb_old_log_probs[i] = rollout_log_probs[src_idx]
    mb_old_values[i] = rollout_values[src_idx]


@always_inline
def ppo_actor_grad_with_kl_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    grad_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    kl_divergences: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    entropies: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_flags: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    old_log_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_epsilon: Scalar[dtype],
    entropy_coef: Scalar[dtype],
    batch_size: Int,
):
    """Compute gradient for PPO actor with clipped surrogate objective.

    Also computes approximate KL divergence for early stopping:
    KL ≈ old_log_prob - new_log_prob (approximation)
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var action = Int(actions[b])
    var advantage = advantages[b]

    # Compute softmax probabilities
    var max_logit = logits[b, 0]
    for a in range(1, NUM_ACTIONS):
        if logits[b, a] > max_logit:
            max_logit = logits[b, a]

    var sum_exp = max_logit - max_logit  # Initialize to zero with correct type
    for a in range(NUM_ACTIONS):
        var l = logits[b, a]
        var logit_val = l - max_logit
        sum_exp = sum_exp + exp(logit_val)

    var probs = InlineArray[Scalar[dtype], NUM_ACTIONS](fill=Scalar[dtype](0.0))
    for a in range(NUM_ACTIONS):
        var l = logits[b, a]
        var logit_val = l - max_logit
        var prob_val = exp(logit_val) / sum_exp
        probs[a] = Scalar[dtype](prob_val[0])

    # Compute new log probability
    var log_eps = Float32(1e-8)
    var prob_for_log = Float32(probs[action]) + log_eps
    var new_log_prob = Scalar[dtype](log(prob_for_log))

    # Probability ratio
    var ratio = exp(new_log_prob - old_log_probs[b])

    # Compute approximate KL divergence: (ratio - 1) - log(ratio)
    var log_ratio = new_log_prob - old_log_probs[b]
    var kl = (ratio - Scalar[dtype](1.0)) - log_ratio
    if kl < Scalar[dtype](0.0):
        kl = Scalar[dtype](0.0)
    kl_divergences[b] = kl

    # Compute entropy: H = -sum(p * log(p))
    var ent: Scalar[dtype] = 0.0
    for a in range(NUM_ACTIONS):
        if probs[a] > Scalar[dtype](1e-10):
            var p_log = Float32(probs[a]) + Float32(1e-8)
            ent = ent - probs[a] * Scalar[dtype](log(p_log))
    entropies[b] = ent

    # Clip ratio for clipped objective
    var clipped_ratio = ratio
    if clipped_ratio < Scalar[dtype](1.0) - clip_epsilon:
        clipped_ratio = Scalar[dtype](1.0) - clip_epsilon
    elif clipped_ratio > Scalar[dtype](1.0) + clip_epsilon:
        clipped_ratio = Scalar[dtype](1.0) + clip_epsilon

    # PPO clipped objective: min(ratio * A, clipped_ratio * A)
    # Gradient is 0 when we use the clipped objective
    var unclipped_obj = ratio * advantage
    var clipped_obj = clipped_ratio * advantage
    var is_clipped = clipped_obj < unclipped_obj
    clip_flags[b] = Scalar[dtype](1.0) if is_clipped else Scalar[dtype](0.0)

    # Compute gradients
    for a in range(NUM_ACTIONS):
        if is_clipped:
            grad_logits[b, a] = Scalar[dtype](0.0)
        else:
            # d_log_prob / d_logits for softmax
            var d_log_prob: Scalar[dtype]
            if a == action:
                d_log_prob = Scalar[dtype](1.0) - probs[a]
            else:
                d_log_prob = -probs[a]

            # Entropy gradient: d(-p * log(p)) / d_logits
            var prob_for_log_ent = Float32(probs[a]) + Float32(1e-8)
            var log_prob_ent = Scalar[dtype](log(prob_for_log_ent))
            var d_entropy = -probs[a] * (Scalar[dtype](1.0) + log_prob_ent)

            # PPO gradient (negative because we maximize)
            grad_logits[b, a] = (
                -advantage * ratio * d_log_prob - entropy_coef * d_entropy
            ) / Scalar[dtype](BATCH_SIZE)


# =============================================================================
# PPO Rollout Storage Kernels (Discrete)
# =============================================================================


@always_inline
def _store_pre_step_kernel[
    dtype: DType,
    N_ENVS: Int,
    OBS_DIM: Int,
](
    # Outputs - rollout buffer at timestep t
    r_obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    r_actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    r_log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    r_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Inputs - current step data
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Store pre-step data (obs, action, log_prob, value) to rollout buffer."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    for d in range(OBS_DIM):
        r_obs[i, d] = obs[i, d]
    r_actions[i] = actions[i]
    r_log_probs[i] = log_probs[i]
    r_values[i] = values[i]


@always_inline
def _store_pre_step_obs_parallel_kernel[
    dtype: DType,
    N_ENVS: Int,
    OBS_DIM: Int,
](
    r_obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
):
    """Parallel obs store for pre-step. One thread per element.

    Grid: (ceil(OBS_DIM / TPB), N_ENVS)
    """
    var d = Int(block_dim.x * block_idx.x + thread_idx.x)
    var i = Int(block_idx.y)
    if d >= OBS_DIM:
        return
    r_obs[i, d] = obs[i, d]


@always_inline
def ppo_gather_minibatch_obs_parallel_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    TOTAL_SIZE: Int,
](
    mb_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE, OBS_DIM), MutAnyOrigin
    ],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    batch_size: Int,
):
    """Parallel gather obs from rollout. One thread per element.

    Grid: (ceil(OBS_DIM / TPB), batch_size)
    """
    var d = Int(block_dim.x * block_idx.x + thread_idx.x)
    var i = Int(block_idx.y)
    if d >= OBS_DIM or i >= batch_size:
        return
    var src_idx = Int(indices[i])
    mb_obs[i, d] = rollout_obs[src_idx, d]


# =============================================================================
# MBPO Dynamics Ensemble Kernels
# =============================================================================


@always_inline
def build_dynamics_target_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    OBS_DIM: Int,
    PRED_DIM: Int = 1 + OBS_DIM,
](
    target: LayoutTensor[
        dtype, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    next_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Build dynamics training target: [reward, delta_obs].

    target[b, 0] = reward[b]
    target[b, 1+i] = next_obs[b, i] - obs[b, i]

    Grid: ceil(BATCH * PRED_DIM / TPB), block: TPB.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * PRED_DIM:
        return
    var b = idx // PRED_DIM
    var d = idx % PRED_DIM
    if d == 0:
        target[b, 0] = rewards[b]
    else:
        target[b, d] = next_obs[b, d - 1] - obs[b, d - 1]


@always_inline
def gaussian_nll_grad_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    PRED_DIM: Int,
    OUT_DIM: Int = 2 * PRED_DIM,
](
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    model_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    target: LayoutTensor[
        dtype, Layout.row_major(BATCH, PRED_DIM), MutAnyOrigin
    ],
    loss_per_sample: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    min_logvar: Scalar[dtype],
    max_logvar: Scalar[dtype],
):
    """Gaussian NLL gradient w.r.t. model output [mean, logvar].

    For each (b, d):
      mean = model_output[b, d]
      logvar = clamp(model_output[b, PRED_DIM + d], min_logvar, max_logvar)
      var = exp(logvar)
      diff = target[b, d] - mean

      grad_mean = (mean - target) / var / BATCH
      grad_logvar = 0.5 * (1 - diff^2 / var) / BATCH

      loss = 0.5 * diff^2 / var + 0.5 * logvar  (accumulated into loss_per_sample)

    Grid: ceil(BATCH * PRED_DIM / TPB), block: TPB.
    One thread per (batch, pred_dim) element.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * PRED_DIM:
        return
    var b = idx // PRED_DIM
    var d = idx % PRED_DIM

    var mean_val = model_output[b, d]
    var raw_lv = model_output[b, PRED_DIM + d]
    var lv = raw_lv
    if lv < min_logvar:
        lv = min_logvar
    if lv > max_logvar:
        lv = max_logvar
    var var_val = exp(lv)
    var tgt = target[b, d]
    var diff = tgt - mean_val
    var diff_sq = diff * diff
    var inv_batch = Scalar[dtype](1.0) / Scalar[dtype](BATCH)

    # Gradients
    grad_output[b, d] = (mean_val - tgt) / var_val * inv_batch
    grad_output[b, PRED_DIM + d] = (
        Scalar[dtype](0.5)
        * (Scalar[dtype](1.0) - diff_sq / var_val)
        * inv_batch
    )

    # Accumulate loss (atomic-free: each thread writes one dimension,
    # we sum across dims on CPU after download)
    # Store per-dim contribution; caller sums across PRED_DIM
    var sample_loss = (
        Scalar[dtype](0.5) * diff_sq / var_val + Scalar[dtype](0.5) * lv
    )
    # Use atomic add simulation: just store dim contribution
    # The loss_per_sample is accumulated via a separate reduce if needed
    if d == 0:
        loss_per_sample[b] = sample_loss
    else:
        loss_per_sample[b] = loss_per_sample[b] + sample_loss


@always_inline
def clamp_rewards_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
](
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    lo: Scalar[dtype],
    hi: Scalar[dtype],
):
    """Clamp rewards to [lo, hi] to prevent NaN cascades from model rollouts."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var r = rewards[b]
    if r != r:  # NaN check
        rewards[b] = Scalar[dtype](0.0)
    elif r < lo:
        rewards[b] = lo
    elif r > hi:
        rewards[b] = hi


@always_inline
def dynamics_sample_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    OBS_DIM: Int,
    PRED_DIM: Int = 1 + OBS_DIM,
    OUT_DIM: Int = 2 * PRED_DIM,
](
    next_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin
    ],
    sampled_rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    model_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    min_logvar: Scalar[dtype],
    max_logvar: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
):
    """Sample next_obs and reward from dynamics model output.

    For each batch element:
      noise ~ N(0,1)  via PhiloxRandom
      For reward (d=0):
        std = sqrt(exp(clamp(logvar[0])))
        reward = mean[0] + std * noise
      For obs dims (d=1..OBS_DIM):
        std = sqrt(exp(clamp(logvar[d])))
        delta = mean[d] + std * noise
        next_obs[i] = obs[i] + delta   (residual prediction)

    One thread per batch sample.
    Grid: ceil(BATCH / TPB), block: TPB.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    # Generate noise for all dimensions
    for d in range(PRED_DIM):
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(b * PRED_DIM + d),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        # Box-Muller transform
        var u1 = rand_vals[0].cast[dtype]()
        var u2 = rand_vals[1].cast[dtype]()
        var eps = Scalar[dtype](1e-7)
        var z = sqrt(Scalar[dtype](-2.0) * log(u1 + eps)) * cos(
            Scalar[dtype](6.283185307) * u2
        )

        var mean_val = model_output[b, d]
        var raw_lv = model_output[b, PRED_DIM + d]
        var lv = raw_lv
        if lv < min_logvar:
            lv = min_logvar
        if lv > max_logvar:
            lv = max_logvar
        var std = sqrt(exp(lv))
        var sample = mean_val + std * z

        if d == 0:
            sampled_rewards[b] = sample
        else:
            # Residual prediction: next_obs = obs + delta
            next_obs[b, d - 1] = obs[b, d - 1] + sample
