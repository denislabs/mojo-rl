"""GPU kernels for reinforcement learning.

This module provides reusable GPU kernels for RL algorithms:

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
"""

from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from memory import UnsafePointer

from .random import xorshift32, random_uniform


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
    Uses xorshift32 with thread-based seeding for GPU-compatible randomness.

    Args:
        indices: Output buffer for random indices [SAMPLE_SIZE].
        buffer_size: Current size of replay buffer (samples from [0, buffer_size)).
        rng_seed: Base seed for random number generation (should vary per call).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SAMPLE_SIZE:
        return

    # GPU-compatible random: unique seed per thread
    var rng = xorshift32(rng_seed + Scalar[DType.uint32](i * 2654435761))
    var rand_result = random_uniform[DType.float32](rng)
    var idx = Int(rand_result[0] * Scalar[DType.float32](buffer_size))
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
