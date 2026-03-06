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
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.math import exp, log, tanh, sqrt, max, min, pi, cos
from std.memory import UnsafePointer
from std.random.philox import Random as PhiloxRandom


# =============================================================================
# Network Operations
# =============================================================================


@always_inline
fn soft_update_kernel[
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
fn zero_buffer_kernel[
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
fn copy_buffer_kernel[
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
fn accumulate_rewards_kernel[
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
fn increment_steps_kernel[
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
fn extract_completed_episodes_kernel[
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
fn selective_reset_tracking_kernel[
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


# =============================================================================
# Replay Buffer Operations
# =============================================================================


@always_inline
fn store_transitions_kernel[
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
fn sample_indices_kernel[
    dtype: DType,
    SAMPLE_SIZE: Int,
](
    indices: LayoutTensor[
        DType.int32, Layout.row_major(SAMPLE_SIZE), MutAnyOrigin
    ],
    buffer_size: Scalar[DType.int32],
    rng_seed: Scalar[DType.uint32],
):
    """Generate random indices for sampling from replay buffer.

    Each thread generates one random index in [0, buffer_size).
    Uses PhiloxRandom for GPU-safe randomness (no seed collisions).

    Args:
        indices: Output buffer for random indices [SAMPLE_SIZE].
        buffer_size: Current size of replay buffer (samples from [0, buffer_size)).
        rng_seed: Base seed for random number generation (should vary per call).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SAMPLE_SIZE:
        return

    # PhiloxRandom: unique seed per thread, no collisions
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(i),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var u: Scalar[dtype] = Scalar[dtype](rand_vals[0])
    var idx = Int(u * Scalar[dtype](buffer_size))
    indices[i] = Scalar[DType.int32](idx)


@always_inline
fn gather_batch_kernel[
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


# =============================================================================
# ND (multi-dimensional action) Replay Buffer Kernels
# These variants support ACTION_DIM > 1 for continuous control (DDPG/TD3/SAC).
# =============================================================================


@always_inline
fn store_transitions_kernel_nd[
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
fn gather_batch_kernel_nd[
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
fn td_target_continuous_kernel[
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
fn td_target_min_twin_kernel[
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
fn actor_grad_from_critic_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
](
    d_actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    d_critic_in: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM + ACTION_DIM), MutAnyOrigin
    ],
):
    """Extract actor output gradients from critic input gradients (∂Q/∂a).

    After running critic backward with gradient [-1/BATCH, ...] (gradient ascent),
    the critic's input gradient d_critic_in has shape [BATCH, OBS_DIM + ACTION_DIM].
    The action portion (columns OBS_DIM:OBS_DIM+ACTION_DIM) is the policy gradient.

    One thread per (batch, action_dim) element.

    Args:
        d_actor_out: Output actor gradient [BATCH, ACTION_DIM].
        d_critic_in: Critic input gradient [BATCH, OBS_DIM + ACTION_DIM].
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
fn concat_obs_action_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
](
    dst: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM + ACTION_DIM), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    act: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
):
    """Concatenate [BATCH, OBS_DIM] and [BATCH, ACTION_DIM] → [BATCH, OBS+ACT].

    Used by DDPG/TD3/SAC to build critic inputs (obs ‖ action) on GPU.
    One thread per output element.

    Args:
        dst: Output tensor [BATCH, OBS_DIM + ACTION_DIM].
        obs: Observations [BATCH, OBS_DIM].
        act: Actions [BATCH, ACTION_DIM].
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * (OBS_DIM + ACTION_DIM)
    if tid >= total:
        return
    var b = tid // (OBS_DIM + ACTION_DIM)
    var c = tid % (OBS_DIM + ACTION_DIM)
    if c < OBS_DIM:
        dst[b, c] = obs[b, c]
    else:
        dst[b, c] = act[b, c - OBS_DIM]


# =============================================================================
# DDPG Action Kernels (scale + optional exploration noise)
# =============================================================================


@always_inline
fn scale_clip_actions_kernel[
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
fn ddpg_exploration_kernel[
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
# TD Critic MSE Gradient (shared by DDPG/TD3/SAC)
# =============================================================================


@always_inline
fn td_mse_grad_kernel[
    dtype: DType,
    BATCH: Int,
](
    q_grad: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    q: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
):
    """MSE loss backward for scalar TD critic: q_grad[i,0] = 2*(q-target)/BATCH.

    One thread per batch element.

    Args:
        q_grad:  Output gradient [BATCH, 1] (written).
        q:       Critic output Q-values [BATCH, 1].
        targets: TD targets [BATCH].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    q_grad[i, 0] = (
        Scalar[dtype](2.0) * (q[i, 0] - targets[i]) / Scalar[dtype](BATCH)
    )
