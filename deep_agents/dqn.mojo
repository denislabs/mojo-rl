"""DQN Agent using the new trait-based deep learning architecture.

This DQN implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building Q-networks
- ReplayBuffer from deep_rl.cpu for experience replay
- Double DQN support via compile-time parameter

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN to reduce overestimation bias (optional)
- GPU support for batch training (forward/backward/update on GPU)

Usage:
    from deep_agents.dqn import DQNAgent
    from envs import CartPoleEnv

    var env = CartPoleEnv()
    var agent = DQNAgent[4, 2, 64, 10000, 32]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=200)

    # GPU Training
    var ctx = DeviceContext()
    agent.init_gpu(ctx)
    var metrics_gpu = agent.train_gpu(ctx, env, num_episodes=200)
"""

from math import exp
from random import random_float64, seed

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Kaiming
from deep_rl.training import Network
from deep_rl.cpu.replay_buffer import ReplayBuffer
from deep_rl.gpu import random_range, xorshift32, random_uniform
from core import TrainingMetrics, BoxDiscreteActionEnv, GPUDiscreteEnv


# =============================================================================
# GPU Kernels for DQN Operations
# =============================================================================


@always_inline
fn dqn_td_target_kernel[
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

    # Find max Q-value for next state
    var max_q = next_q_values[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = next_q_values[b, a]
        if q > max_q:
            max_q = q

    # Compute TD target
    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * max_q * done_mask


@always_inline
fn dqn_double_td_target_kernel[
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

    # Online network selects best action (argmax)
    var best_action = 0
    var best_q = online_next_q[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = online_next_q[b, a]
        if q > best_q:
            best_q = q
            best_action = a

    # Target network evaluates that action
    var target_q = target_next_q[b, best_action]

    # Compute TD target
    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * target_q * done_mask


@always_inline
fn dqn_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    loss_out: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    # Inputs
    q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
):
    """Compute masked gradient for DQN loss. Only backprop through taken action.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var action = Int(actions[b])
    var q_pred = q_values[b, action]
    var td_error = q_pred - targets[b]

    # Masked gradient: only for taken action
    for a in range(NUM_ACTIONS):
        if a == action:
            grad_output[b, a] = (
                Scalar[dtype](2.0) * td_error / Scalar[dtype](BATCH_SIZE)
            )
        else:
            grad_output[b, a] = Scalar[dtype](0.0)


@always_inline
fn soft_update_kernel[
    dtype: DType,
    SIZE: Int,
](
    target: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    tau: Scalar[dtype],
):
    """Soft update: target = tau * source + (1 - tau) * target."""
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
](buffer: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],):
    """Zero out a buffer."""
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
    """Copy src buffer to dst buffer."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    dst[i] = src[i]


# =============================================================================
# GPU Episode Tracking Kernels
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
    """Accumulate step rewards into episode totals."""
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
    """Increment step counters for each environment."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return
    episode_steps[i] += Scalar[dtype](1.0)


@always_inline
fn count_dones_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    done_count: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
):
    """Count number of done environments (simple reduction, uses atomics conceptually).

    Note: For small batch sizes, a single-thread reduction is sufficient.
    For larger batches, use a proper block reduction.
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid != 0:
        return

    # Single thread counts all dones (simple but works for small batches)
    var count = Scalar[dtype](0.0)
    for i in range(BATCH_SIZE):
        if dones[i] > Scalar[dtype](0.5):
            count += Scalar[dtype](1.0)
    done_count[0] = count


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
    # Output: completed episode info (packed: [reward_0, steps_0, reward_1, steps_1, ...])
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
    """Reset episode tracking only for done environments."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    if dones[i] > Scalar[dtype](0.5):
        episode_rewards[i] = Scalar[dtype](0.0)
        episode_steps[i] = Scalar[dtype](0.0)


fn argmax_greedy_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    epsilon: Scalar[dtype],
    q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    actions: LayoutTensor[
        DType.int, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
):
    """Compute argmax of Q-values for each batch element with epsilon-greedy.

    Uses GPU-compatible xorshift32 random with thread-based seeding.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # GPU-compatible random: seed based on thread index
    var rng = xorshift32(Scalar[DType.uint32](b * 2654435761 + 98765))
    var rand_result = random_uniform[dtype](rng)
    var rand_val = rand_result[0]
    rng = rand_result[1]

    if rand_val < epsilon:
        # Random action using second random draw
        var action_result = random_uniform[dtype](rng)
        actions[b] = Int(action_result[0] * Scalar[dtype](NUM_ACTIONS))
        return

    var best_q = q_values[b, 0]
    var best_action = 0
    for a in range(1, NUM_ACTIONS):
        var q = q_values[b, a]
        if q > best_q:
            best_q = q
            best_action = a

    actions[b] = best_action


# =============================================================================
# Stateless GPU Replay Buffer Kernels
# =============================================================================
#
# These kernels implement a GPU-resident circular replay buffer without storing
# DeviceBuffers in a struct. All buffers are allocated in the training loop and
# passed to kernels as parameters.
#
# Usage:
#   1. Allocate buffers in train_gpu: rb_states_buf, rb_actions_buf, etc.
#   2. Track write_idx and buffer_size as simple Int variables
#   3. Call store_transitions_kernel to write batch of transitions
#   4. Call sample_indices_kernel to generate random sample indices
#   5. Call gather_batch_kernel to collect sampled transitions
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


fn epsilon_greedy_kernel[
    dtype: DType,
    N_ENVS: Int,
    NUM_ACTIONS: Int,
](
    # Q-values from forward pass [N_ENVS, NUM_ACTIONS]
    q_values: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, NUM_ACTIONS), MutAnyOrigin
    ],
    # RNG states [N_ENVS] - updated in place
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    # Output actions [N_ENVS]
    actions: LayoutTensor[DType.int32, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Epsilon for exploration
    epsilon: Scalar[dtype],
):
    """Batched epsilon-greedy action selection on GPU.

    Each thread handles one environment's action selection.
    """
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= N_ENVS:
        return

    var rng = rebind[Scalar[DType.uint32]](rng_states[env_idx])
    var action: Int = 0

    # Generate random number for epsilon check
    var rand_result = random_range[dtype](
        rng, Scalar[dtype](0.0), Scalar[dtype](1.0)
    )
    var rand_val = rand_result[0]
    rng = rand_result[1]

    if rand_val < epsilon:
        # Random action
        var action_result = random_range[dtype](
            rng, Scalar[dtype](0.0), Scalar[dtype](NUM_ACTIONS)
        )
        action = Int(action_result[0])
        rng = action_result[1]
    else:
        # Greedy: argmax Q
        var best_q = rebind[Scalar[dtype]](q_values[env_idx, 0])
        for a in range(1, NUM_ACTIONS):
            var q = rebind[Scalar[dtype]](q_values[env_idx, a])
            if q > best_q:
                best_q = q
                action = a

    actions[env_idx] = Int32(action)
    rng_states[env_idx] = rng


fn dqn_sample_batch_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    BUFFER_CAPACITY: Int,
](
    # Random indices for sampling [BATCH_SIZE]
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    # Source buffer
    buffer_states: LayoutTensor[
        dtype, Layout.row_major(BUFFER_CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buffer_actions: LayoutTensor[
        dtype, Layout.row_major(BUFFER_CAPACITY), MutAnyOrigin
    ],
    buffer_rewards: LayoutTensor[
        dtype, Layout.row_major(BUFFER_CAPACITY), MutAnyOrigin
    ],
    buffer_next_states: LayoutTensor[
        dtype, Layout.row_major(BUFFER_CAPACITY, OBS_DIM), MutAnyOrigin
    ],
    buffer_dones: LayoutTensor[
        dtype, Layout.row_major(BUFFER_CAPACITY), MutAnyOrigin
    ],
    # Output batch
    batch_states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    batch_next_states: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
):
    """Gather a batch of transitions from GPU replay buffer using random indices.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var idx = Int(indices[b])

    # Copy state
    for i in range(OBS_DIM):
        batch_states[b, i] = buffer_states[idx, i]
        batch_next_states[b, i] = buffer_next_states[idx, i]

    batch_actions[b] = buffer_actions[idx]
    batch_rewards[b] = buffer_rewards[idx]
    batch_dones[b] = buffer_dones[idx]


struct DQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 64,
    buffer_capacity: Int = 10000,
    batch_size: Int = 32,
    double_dqn: Bool = True,
]:
    """Deep Q-Network agent using the new trait-based architecture.

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 64).
        buffer_capacity: Replay buffer capacity (default: 10000).
        batch_size: Training batch size (default: 32).
        double_dqn: Use Double DQN (default: True).
    """

    # Q-network architecture: obs -> hidden (ReLU) -> hidden (ReLU) -> num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions

    # Compute network dimensions at compile time
    # Cache sizes for each layer:
    # - Linear[obs, hidden]: caches obs_dim
    # - ReLU[hidden]: caches hidden_dim
    # - Linear[hidden, hidden]: caches hidden_dim
    # - ReLU[hidden]: caches hidden_dim
    # - Linear[hidden, actions]: caches hidden_dim
    comptime NETWORK_CACHE_SIZE: Int = (
        Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN
    )

    # Param sizes for each layer:
    # - Linear[obs, hidden]: obs * hidden + hidden (W + b)
    # - ReLU[hidden]: 0
    # - Linear[hidden, hidden]: hidden * hidden + hidden
    # - ReLU[hidden]: 0
    # - Linear[hidden, actions]: hidden * actions + actions
    comptime NETWORK_PARAM_SIZE: Int = (
        Self.OBS * Self.HIDDEN
        + Self.HIDDEN
        + Self.HIDDEN * Self.HIDDEN
        + Self.HIDDEN
        + Self.HIDDEN * Self.ACTIONS
        + Self.ACTIONS
    )

    # Online and target networks
    var online_model: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]
    var target_model: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Replay buffer (action_dim=1 since we store discrete action as scalar)
    var buffer: ReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    # Hyperparameters
    var gamma: Float64  # Discount factor
    var tau: Float64  # Soft update rate
    var lr: Float64  # Learning rate (stored for reference)

    # Exploration
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # Training state
    var train_step_count: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        lr: Float64 = 0.001,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
    ):
        """Initialize DQN agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update rate for target network (default: 0.005).
            lr: Learning rate for Adam optimizer (default: 0.001).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
        """
        # Create Q-network model
        var q_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.ACTIONS](),
        )

        # Initialize networks
        self.online_model = Network(q_model, Adam(lr=lr), Kaiming())
        self.target_model = Network(q_model, Adam(lr=lr), Kaiming())

        # Initialize target with online's weights
        self.target_model.copy_params_from(self.online_model)

        # Initialize replay buffer
        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

        # Store hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.train_step_count = 0

    fn select_action(self, obs: SIMD[DType.float64, Self.obs_dim]) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation.

        Returns:
            Selected action index.
        """
        # Epsilon-greedy exploration
        if random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.num_actions))

        # Greedy action: argmax_a Q(s, a)
        var obs_input = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_input[i] = Scalar[dtype](obs[i])

        var q_values = InlineArray[Scalar[dtype], Self.num_actions](
            uninitialized=True
        )
        self.online_model.forward[1](obs_input, q_values)

        # Find argmax
        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.num_actions):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: SIMD[DType.float64, Self.obs_dim],
        action: Int,
        reward: Float64,
        next_obs: SIMD[DType.float64, Self.obs_dim],
        done: Bool,
    ):
        """Store transition in replay buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode ended.
        """
        var obs_arr = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        var next_obs_arr = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[dtype](obs[i])
            next_obs_arr[i] = Scalar[dtype](next_obs[i])

        var action_arr = InlineArray[Scalar[dtype], 1](Scalar[dtype](action))

        self.buffer.add(
            obs_arr, action_arr, Scalar[dtype](reward), next_obs_arr, done
        )

    fn train_step(mut self) -> Float64:
        """Perform one training step.

        Returns:
            Loss value (0 if buffer not ready).
        """
        # Check if buffer has enough samples
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch from buffer
        var batch_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_actions = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_dones = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        # Temporary for actions (action_dim=1)
        var batch_actions_tmp = InlineArray[Scalar[dtype], Self.batch_size * 1](
            uninitialized=True
        )

        self.buffer.sample[Self.batch_size](
            batch_obs,
            batch_actions_tmp,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

        # Copy actions (action_dim=1 so just copy directly)
        for i in range(Self.batch_size):
            batch_actions[i] = batch_actions_tmp[i]

        # Forward pass on online network (with cache for backward)
        var q_values = InlineArray[
            Scalar[dtype], Self.batch_size * Self.num_actions
        ](uninitialized=True)
        var cache = InlineArray[
            Scalar[dtype], Self.batch_size * Self.NETWORK_CACHE_SIZE
        ](uninitialized=True)
        self.online_model.forward_with_cache[Self.batch_size](
            batch_obs, q_values, cache
        )

        # Forward pass on target network (no cache needed)
        var next_q_values = InlineArray[
            Scalar[dtype], Self.batch_size * Self.num_actions
        ](uninitialized=True)
        self.target_model.forward[Self.batch_size](
            batch_next_obs, next_q_values
        )

        # Compute TD targets
        var targets = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        @parameter
        if Self.double_dqn:
            # Double DQN: online network selects action, target evaluates
            var online_next_q = InlineArray[
                Scalar[dtype], Self.batch_size * Self.num_actions
            ](uninitialized=True)
            self.online_model.forward[Self.batch_size](
                batch_next_obs, online_next_q
            )

            for b in range(Self.batch_size):
                # Online network selects best action
                var best_action = 0
                var best_q = online_next_q[b * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = online_next_q[b * Self.num_actions + a]
                    if q > best_q:
                        best_q = q
                        best_action = a

                # Target network evaluates that action
                var next_q = next_q_values[b * Self.num_actions + best_action]

                # TD target: r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * next_q * done_mask
                )
        else:
            # Standard DQN: max_a Q_target(s', a)
            for b in range(Self.batch_size):
                var max_next_q = next_q_values[b * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = next_q_values[b * Self.num_actions + a]
                    if q > max_next_q:
                        max_next_q = q

                # TD target: r + γ * max_a Q_target(s', a) * (1 - done)
                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * max_next_q * done_mask
                )

        # Compute loss gradient: d(Q(s,a) - target)^2 / dQ = 2 * (Q(s,a) - target)
        # We only backprop through the action that was taken
        var grad_output = InlineArray[
            Scalar[dtype], Self.batch_size * Self.num_actions
        ](uninitialized=True)

        var total_loss: Float64 = 0.0

        for b in range(Self.batch_size):
            var action = Int(batch_actions[b])
            var q_pred = q_values[b * Self.num_actions + action]
            var td_error = q_pred - targets[b]

            # MSE loss
            total_loss += Float64(td_error * td_error)

            # Gradient: only for the taken action
            for a in range(Self.num_actions):
                if a == action:
                    # d/dQ (Q - target)^2 = 2 * (Q - target) / batch_size
                    grad_output[b * Self.num_actions + a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](Self.batch_size)
                    )
                else:
                    grad_output[b * Self.num_actions + a] = Scalar[dtype](0.0)

        total_loss /= Float64(Self.batch_size)

        # Backward pass
        var grad_input = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)

        self.online_model.zero_grads()
        self.online_model.backward[Self.batch_size](
            grad_output, grad_input, cache
        )

        # Update online network
        self.online_model.update()

        # Soft update target network
        self.target_model.soft_update_from(self.online_model, self.tau)

        self.train_step_count += 1

        return total_loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call at end of each episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Get current exploration rate."""
        return self.epsilon

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

    # =========================================================================
    # High-level training and evaluation methods
    # =========================================================================

    fn _list_to_simd(
        self, obs_list: List[Float64]
    ) -> SIMD[DType.float64, Self.obs_dim]:
        """Convert List[Float64] to SIMD for internal use."""
        var obs = SIMD[DType.float64, Self.obs_dim]()
        for i in range(Self.obs_dim):
            obs[i] = obs_list[i]
        return obs

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the DQN agent on a continuous-state environment.

        Args:
            env: The environment to train on (must implement BoxDiscreteActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Number of random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            verbose: Whether to print progress (default: False).
            print_every: Print progress every N episodes if verbose (default: 10).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN" if not Self.double_dqn else "Double DQN",
            environment_name=environment_name,
        )

        # =====================================================================
        # Warmup: fill replay buffer with random actions
        # =====================================================================
        var warmup_obs = self._list_to_simd(env.reset_obs_list())
        var warmup_count = 0

        while warmup_count < warmup_steps:
            # Random action
            var action = Int(random_float64() * Float64(Self.num_actions))
            var result = env.step_obs(action)

            var next_obs = self._list_to_simd(result[0])
            self.store_transition(
                warmup_obs, action, result[1], next_obs, result[2]
            )

            warmup_obs = next_obs
            warmup_count += 1

            if result[2]:  # done
                warmup_obs = self._list_to_simd(env.reset_obs_list())

        # =====================================================================
        # Training loop
        # =====================================================================
        var total_steps = 0

        for episode in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action using epsilon-greedy
                var action = self.select_action(obs)

                # Step environment
                var result = env.step_obs(action)
                var next_obs = self._list_to_simd(result[0])
                var reward = result[1]
                var done = result[2]

                # Store transition
                self.store_transition(obs, action, reward, next_obs, done)

                # Train every N steps
                if total_steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs = next_obs
                total_steps += 1
                episode_steps += 1

                if done:
                    break

            # Decay epsilon at end of episode
            self.decay_epsilon()

            # Log metrics
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.epsilon
            )

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg reward: "
                    + String(avg_reward)[:7]
                    + " | Epsilon: "
                    + String(self.epsilon)[:5]
                    + " | Steps: "
                    + String(total_steps)
                )

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment using greedy policy.

        Note: This uses the current epsilon value. For pure greedy evaluation,
        ensure epsilon is set to 0 or a very small value before calling.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 500).
            render: Whether to render the environment (default: False).

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                # Select action (uses current epsilon - typically low after training)
                var action = self.select_action(obs)

                # Step environment
                var result = env.step_obs(action)
                episode_reward += result[1]
                obs = self._list_to_simd(result[0])

                if result[2]:  # done
                    break

            total_reward += episode_reward

        if render:
            env.close()

        return total_reward / Float64(num_episodes)

    fn select_actions_gpu(
        self,
        ctx: DeviceContext,
        mut obs_buf: DeviceBuffer[dtype],
        mut q_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
    ) raises:
        """Select action using GPU forward pass (avoids CPU param sync).

        Args:
            ctx: GPU device context.
            obs_buf: Pre-allocated GPU buffer for observation [obs_dim].
            q_buf: Pre-allocated GPU buffer for Q-values [num_actions].
            actions_buf: Pre-allocated GPU buffer for actions [batch_size].
            params_buf: GPU buffer containing current params.

        """
        var obs = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.obs_dim),
            MutAnyOrigin,
        ](obs_buf.unsafe_ptr())
        var q = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](q_buf.unsafe_ptr())

        # Forward pass on GPU
        self.online_model.forward_gpu[Self.batch_size](
            ctx, obs_buf, q_buf, params_buf
        )

        # Find argmax with epsilon-greedy
        var actions = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn argmax_kernel_wrapper(
            epsilon: Scalar[dtype],
            q_vals: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ],
            acts: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            # Inline argmax with epsilon-greedy using GPU-compatible random
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= Self.batch_size:
                return

            # GPU-compatible random: seed based on thread index
            var rng = xorshift32(Scalar[DType.uint32](b * 2654435761 + 54321))
            var rand_result = random_uniform[dtype](rng)
            var rand_val = rand_result[0]
            rng = rand_result[1]

            if rand_val < epsilon:
                # Random action using second random draw
                var action_result = random_uniform[dtype](rng)
                # Truncate to int to get valid action (0 or 1)
                acts[b] = Scalar[dtype](
                    Int(action_result[0] * Scalar[dtype](Self.num_actions))
                )
                return

            var best_q = q_vals[b, 0]
            var best_action = 0
            for a in range(1, Self.num_actions):
                var qv = q_vals[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a

            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[argmax_kernel_wrapper, argmax_kernel_wrapper](
            Scalar[dtype](self.epsilon),
            q,
            actions,
            grid_dim=((Self.batch_size + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    fn evaluate_greedy[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent using pure greedy policy (epsilon=0).

        This performs evaluation without any exploration, always selecting
        the action with highest Q-value.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 500).
            render: Whether to render the environment (default: False).

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                # Greedy action: argmax_a Q(s, a) - no epsilon
                var obs_input = InlineArray[Scalar[dtype], Self.obs_dim](
                    uninitialized=True
                )
                for i in range(Self.obs_dim):
                    obs_input[i] = Scalar[dtype](obs[i])

                var q_values = InlineArray[Scalar[dtype], Self.num_actions](
                    uninitialized=True
                )
                self.online_model.forward[1](obs_input, q_values)

                # Find argmax
                var best_action = 0
                var best_q = q_values[0]
                for a in range(1, Self.num_actions):
                    if q_values[a] > best_q:
                        best_q = q_values[a]
                        best_action = a

                # Step environment
                var result = env.step_obs(best_action)
                episode_reward += result[1]
                obs = self._list_to_simd(result[0])

                if result[2]:  # done
                    break

            total_reward += episode_reward

        if render:
            env.close()

        return total_reward / Float64(num_episodes)

    # =========================================================================
    # GPU Training Methods (Optimized with pre-allocated buffers)
    # =========================================================================

    fn train_step_gpu_online(
        mut self,
        ctx: DeviceContext,
        # Network buffers (pre-allocated)
        mut online_params_buf: DeviceBuffer[dtype],
        mut online_grads_buf: DeviceBuffer[dtype],
        mut online_state_buf: DeviceBuffer[dtype],
        mut target_params_buf: DeviceBuffer[dtype],
        # Batch buffers (pre-allocated) - uses current transition batch directly
        mut obs_buf: DeviceBuffer[dtype],  # Current observations (before step)
        mut next_obs_buf: DeviceBuffer[dtype],  # Next observations (after step)
        mut q_values_buf: DeviceBuffer[dtype],
        mut next_q_values_buf: DeviceBuffer[dtype],
        mut online_next_q_buf: DeviceBuffer[dtype],  # For Double DQN
        mut cache_buf: DeviceBuffer[dtype],
        mut grad_output_buf: DeviceBuffer[dtype],
        mut grad_input_buf: DeviceBuffer[dtype],
        mut targets_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> Float64:
        """Online GPU training step using current batch directly (no replay buffer).

        This is used for vectorized GPU training where we train on the current
        batch of transitions from parallel environments, avoiding CPU-GPU transfers
        for replay buffer operations.

        All buffers are passed in to avoid allocation overhead.
        All operations run entirely on GPU.

        Note: In online mode, obs_buf contains the state BEFORE the step,
        and next_obs_buf should contain the state AFTER the step.
        The training loop is responsible for copying obs to next_obs before stepping.
        """
        # GPU Forward pass: online network with cache
        # obs_buf contains the previous observations (before step)
        self.online_model.forward_gpu_with_cache[Self.batch_size](
            ctx, obs_buf, q_values_buf, online_params_buf, cache_buf
        )

        # GPU Forward pass: target network (no cache)
        self.target_model.forward_gpu[Self.batch_size](
            ctx, next_obs_buf, next_q_values_buf, target_params_buf
        )

        # GPU TD Target computation
        comptime BATCH_BLOCKS = (Self.batch_size + TPB - 1) // TPB
        var gamma_scalar = Scalar[dtype](self.gamma)

        # Create LayoutTensor views for kernels
        var targets_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](targets_buf.unsafe_ptr())
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var next_q_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](next_q_values_buf.unsafe_ptr())

        @parameter
        if Self.double_dqn:
            # For Double DQN: forward online network on next_obs
            self.online_model.forward_gpu[Self.batch_size](
                ctx, next_obs_buf, online_next_q_buf, online_params_buf
            )

            var online_next_tensor = LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ](online_next_q_buf.unsafe_ptr())

            # Double DQN TD target kernel
            @parameter
            @always_inline
            fn double_td_kernel_wrapper(
                targets_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                online_next_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                target_next_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                rewards_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                dones_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                gamma: Scalar[dtype],
            ):
                dqn_double_td_target_kernel[
                    dtype, Self.batch_size, Self.num_actions
                ](
                    targets_t,
                    online_next_t,
                    target_next_t,
                    rewards_t,
                    dones_t,
                    gamma,
                )

            ctx.enqueue_function[
                double_td_kernel_wrapper, double_td_kernel_wrapper
            ](
                targets_tensor,
                online_next_tensor,
                next_q_tensor,
                rewards_tensor,
                dones_tensor,
                gamma_scalar,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            # Standard DQN TD target kernel
            @parameter
            @always_inline
            fn td_kernel_wrapper(
                targets_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                next_q_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                rewards_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                dones_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                gamma: Scalar[dtype],
            ):
                dqn_td_target_kernel[dtype, Self.batch_size, Self.num_actions](
                    targets_t, next_q_t, rewards_t, dones_t, gamma
                )

            ctx.enqueue_function[td_kernel_wrapper, td_kernel_wrapper](
                targets_tensor,
                next_q_tensor,
                rewards_tensor,
                dones_tensor,
                gamma_scalar,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # GPU Gradient computation
        var q_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](q_values_buf.unsafe_ptr())
        var grad_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](grad_output_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        # We don't use loss_out in the kernel for now (would need reduction)
        @parameter
        @always_inline
        fn grad_kernel_wrapper(
            grad_t: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ],
            q_t: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ],
            targets_t: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            actions_t: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            # Inline the gradient computation
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= Self.batch_size:
                return
            var action = Int(actions_t[b])
            var q_pred = q_t[b, action]
            var td_error = q_pred - targets_t[b]
            for a in range(Self.num_actions):
                if a == action:
                    grad_t[b, a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](Self.batch_size)
                    )
                else:
                    grad_t[b, a] = Scalar[dtype](0.0)

        ctx.enqueue_function[grad_kernel_wrapper, grad_kernel_wrapper](
            grad_tensor,
            q_tensor,
            targets_tensor,
            actions_tensor,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # GPU Zero gradients
        comptime PARAM_BLOCKS = (Self.NETWORK_PARAM_SIZE + TPB - 1) // TPB
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](online_grads_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn zero_kernel_wrapper(
            buf: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
        ):
            zero_buffer_kernel[dtype, Self.NETWORK_PARAM_SIZE](buf)

        ctx.enqueue_function[zero_kernel_wrapper, zero_kernel_wrapper](
            grads_tensor,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        # GPU Backward pass
        self.online_model.backward_gpu[Self.batch_size](
            ctx,
            grad_output_buf,
            grad_input_buf,
            online_params_buf,
            cache_buf,
            online_grads_buf,
        )

        # GPU Optimizer update
        self.online_model.update_gpu(
            ctx, online_params_buf, online_grads_buf, online_state_buf
        )

        # GPU Soft update target network
        var tau_scalar = Scalar[dtype](self.tau)
        var online_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](online_params_buf.unsafe_ptr())
        var target_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](target_params_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn soft_update_wrapper(
            target_t: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
            source_t: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
            tau: Scalar[dtype],
        ):
            soft_update_kernel[dtype, Self.NETWORK_PARAM_SIZE](
                target_t, source_t, tau
            )

        ctx.enqueue_function[soft_update_wrapper, soft_update_wrapper](
            target_params_tensor,
            online_params_tensor,
            tau_scalar,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        self.train_step_count += 1
        return 0.0  # Loss computation would need GPU reduction

    fn train_gpu_full[
        EnvType: GPUDiscreteEnv
    ](
        mut self,
        ctx: DeviceContext,
        mut env: EnvType,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        sync_every: Int = 5,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the DQN agent on GPU with optimized buffer management.

        All GPU buffers are pre-allocated once at the start of training.
        Action selection uses GPU forward pass (no CPU param sync needed).
        Environment interaction happens on CPU, all training operations on GPU.

        Args:
            ctx: GPU device context.
            env: The environment to train on (must implement BoxDiscreteActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Number of random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            sync_every: Sync GPU params to CPU every N episodes for backup (default: 5).
            verbose: Whether to print progress (default: False).
            print_every: Print progress every N episodes if verbose (default: 10).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN (GPU)" if not Self.double_dqn else "Double DQN (GPU)",
            environment_name=environment_name,
        )

        # =====================================================================
        # Pre-allocate ALL GPU buffers (done once!)
        # =====================================================================
        comptime PARAM_SIZE = Self.NETWORK_PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * 2  # Adam has 2 state values per param
        comptime BATCH_OBS_SIZE = Self.batch_size * Self.obs_dim
        comptime BATCH_Q_SIZE = Self.batch_size * Self.num_actions
        comptime BATCH_CACHE_SIZE = Self.batch_size * Self.NETWORK_CACHE_SIZE

        # Network parameter buffers
        var online_params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var online_grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var online_state_buf = ctx.enqueue_create_buffer[dtype](STATE_SIZE)
        var target_params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        # Batch data buffers (GPU)
        var prev_obs_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_OBS_SIZE
        )  # For online training
        var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var obs = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.obs_dim),
            MutAnyOrigin,
        ](obs_buf.unsafe_ptr())
        var next_obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var next_q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var online_next_q_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_Q_SIZE
        )  # Double DQN
        var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH_CACHE_SIZE)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var targets_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var rewards = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size),
            MutAnyOrigin,
        ](rewards_buf.unsafe_ptr())
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var dones = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size),
            MutAnyOrigin,
        ](dones_buf.unsafe_ptr())
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var actions = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size),
            MutAnyOrigin,
        ](actions_buf.unsafe_ptr())

        # Host buffers for CPU-GPU transfer
        var obs_host = ctx.enqueue_create_host_buffer[dtype](BATCH_OBS_SIZE)
        var next_obs_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH_OBS_SIZE
        )
        var rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        var dones_host = ctx.enqueue_create_host_buffer[dtype](Self.batch_size)
        var actions_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )

        # =====================================================================
        # GPU Replay Buffer Storage (stateless - just buffers, no struct)
        # =====================================================================
        comptime RB_CAPACITY = Self.buffer_capacity
        comptime RB_OBS_SIZE = RB_CAPACITY * Self.obs_dim

        var rb_states_buf = ctx.enqueue_create_buffer[dtype](RB_OBS_SIZE)
        var rb_actions_buf = ctx.enqueue_create_buffer[dtype](RB_CAPACITY)
        var rb_rewards_buf = ctx.enqueue_create_buffer[dtype](RB_CAPACITY)
        var rb_next_states_buf = ctx.enqueue_create_buffer[dtype](RB_OBS_SIZE)
        var rb_dones_buf = ctx.enqueue_create_buffer[dtype](RB_CAPACITY)

        # Sample indices buffer (for random sampling)
        var sample_indices_buf = ctx.enqueue_create_buffer[DType.int32](
            Self.batch_size
        )

        # Sampled batch buffers (for training from replay buffer)
        var sampled_obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var sampled_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var sampled_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var sampled_next_obs_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_OBS_SIZE
        )
        var sampled_dones_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )

        # Replay buffer state (managed on CPU)
        var rb_write_idx: Int = 0
        var rb_size: Int = 0

        # Episode tracking buffers (GPU)
        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var completed_steps_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var completed_mask_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )

        # Episode tracking host buffers (for reading back completed episodes)
        var completed_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        var completed_steps_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        var completed_mask_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )

        # Action selection buffers (batch=1 for single observation)
        var action_obs_buf = ctx.enqueue_create_buffer[dtype](Self.obs_dim)
        var action_q_buf = ctx.enqueue_create_buffer[dtype](Self.num_actions)
        var action_obs_host = ctx.enqueue_create_host_buffer[dtype](
            Self.obs_dim
        )
        var action_q_host = ctx.enqueue_create_host_buffer[dtype](
            Self.num_actions
        )

        # Copy CPU params to GPU
        self.online_model.copy_params_to_device(ctx, online_params_buf)
        self.online_model.copy_state_to_device(ctx, online_state_buf)
        self.target_model.copy_params_to_device(ctx, target_params_buf)

        # =====================================================================
        # Initialize episode tracking buffers to zero
        # =====================================================================
        comptime BATCH_BLOCKS = (Self.batch_size + TPB - 1) // TPB

        var episode_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var episode_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn zero_episode_rewards(
            buf: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            zero_buffer_kernel[dtype, Self.batch_size](buf)

        ctx.enqueue_function[zero_episode_rewards, zero_episode_rewards](
            episode_rewards_tensor,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[zero_episode_rewards, zero_episode_rewards](
            episode_steps_tensor,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # =====================================================================
        # Initial environment reset (all batch_size environments)
        # =====================================================================
        EnvType.reset_kernel_gpu[Self.batch_size, Self.obs_dim](ctx, obs_buf)

        if verbose:
            print("GPU buffers allocated. Starting training...")
            print(
                "Running " + String(Self.batch_size) + " parallel environments"
            )

        # =====================================================================
        # Vectorized Training loop
        # Runs batch_size environments in parallel, counting episodes as they complete
        # =====================================================================
        var total_steps = 0
        var completed_episodes = 0
        var last_print_episode = 0

        # Create tensor views for episode tracking kernels
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var completed_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](completed_rewards_buf.unsafe_ptr())
        var completed_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](completed_steps_buf.unsafe_ptr())
        var completed_mask_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](completed_mask_buf.unsafe_ptr())

        # Define kernel wrappers for episode tracking
        @parameter
        @always_inline
        fn accum_rewards_wrapper(
            ep_rewards: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            step_rewards: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            accumulate_rewards_kernel[dtype, Self.batch_size](
                ep_rewards, step_rewards
            )

        @parameter
        @always_inline
        fn incr_steps_wrapper(
            ep_steps: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            increment_steps_kernel[dtype, Self.batch_size](ep_steps)

        @parameter
        @always_inline
        fn extract_completed_wrapper(
            d: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            ep_r: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            ep_s: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            comp_r: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            comp_s: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            comp_m: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            extract_completed_episodes_kernel[dtype, Self.batch_size](
                d, ep_r, ep_s, comp_r, comp_s, comp_m
            )

        # Create tensor views for copy kernel (BATCH_OBS_SIZE already defined above)
        var prev_obs_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_OBS_SIZE), MutAnyOrigin
        ](prev_obs_buf.unsafe_ptr())
        var obs_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_OBS_SIZE), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn copy_obs_wrapper(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH_OBS_SIZE), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH_OBS_SIZE), MutAnyOrigin
            ],
        ):
            copy_buffer_kernel[dtype, BATCH_OBS_SIZE](dst, src)

        comptime OBS_BLOCKS = (BATCH_OBS_SIZE + TPB - 1) // TPB

        # =====================================================================
        # Replay buffer tensor views and kernel wrappers
        # =====================================================================
        var rb_states_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
        ](rb_states_buf.unsafe_ptr())
        var rb_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
        ](rb_actions_buf.unsafe_ptr())
        var rb_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
        ](rb_rewards_buf.unsafe_ptr())
        var rb_next_states_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
        ](rb_next_states_buf.unsafe_ptr())
        var rb_dones_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
        ](rb_dones_buf.unsafe_ptr())

        var sample_indices_tensor = LayoutTensor[
            DType.int32, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sample_indices_buf.unsafe_ptr())

        var sampled_obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size, Self.obs_dim), MutAnyOrigin
        ](sampled_obs_buf.unsafe_ptr())
        var sampled_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sampled_actions_buf.unsafe_ptr())
        var sampled_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sampled_rewards_buf.unsafe_ptr())
        var sampled_next_obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size, Self.obs_dim), MutAnyOrigin
        ](sampled_next_obs_buf.unsafe_ptr())
        var sampled_dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sampled_dones_buf.unsafe_ptr())

        # Prev obs as 2D tensor for store kernel
        var prev_obs_2d_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size, Self.obs_dim), MutAnyOrigin
        ](prev_obs_buf.unsafe_ptr())
        var obs_2d_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size, Self.obs_dim), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn store_transitions_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.obs_dim),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            next_states: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.obs_dim),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            buf_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_actions: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_rewards: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_next_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_dones: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            write_idx: Scalar[DType.int32],
        ):
            store_transitions_kernel[
                dtype, Self.batch_size, Self.obs_dim, RB_CAPACITY
            ](
                states,
                actions,
                rewards,
                next_states,
                dones,
                buf_states,
                buf_actions,
                buf_rewards,
                buf_next_states,
                buf_dones,
                write_idx,
            )

        @parameter
        @always_inline
        fn sample_indices_wrapper(
            indices: LayoutTensor[
                DType.int32, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            buffer_size: Scalar[DType.int32],
            rng_seed: Scalar[DType.uint32],
        ):
            sample_indices_kernel[dtype, Self.batch_size](
                indices, buffer_size, rng_seed
            )

        @parameter
        @always_inline
        fn gather_batch_wrapper(
            batch_states: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.obs_dim),
                MutAnyOrigin,
            ],
            batch_actions: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            batch_rewards: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            batch_next_states: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.obs_dim),
                MutAnyOrigin,
            ],
            batch_dones: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            buf_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_actions: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_rewards: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_next_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_dones: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            indices: LayoutTensor[
                DType.int32, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            gather_batch_kernel[
                dtype, Self.batch_size, Self.obs_dim, RB_CAPACITY
            ](
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
                buf_states,
                buf_actions,
                buf_rewards,
                buf_next_states,
                buf_dones,
                indices,
            )

        # =====================================================================
        # Warmup phase: fill replay buffer with random transitions
        # =====================================================================
        if verbose:
            print(
                "Warmup: collecting "
                + String(warmup_steps)
                + " random transitions..."
            )

        var saved_epsilon = self.epsilon
        self.epsilon = 1.0  # Force random actions during warmup

        var warmup_count = 0
        while warmup_count < warmup_steps:
            # Copy current obs to prev_obs
            ctx.enqueue_function[copy_obs_wrapper, copy_obs_wrapper](
                prev_obs_tensor,
                obs_flat_tensor,
                grid_dim=(OBS_BLOCKS,),
                block_dim=(TPB,),
            )

            # Select random actions (epsilon=1.0)
            self.select_actions_gpu(
                ctx,
                obs_buf,
                q_values_buf,
                actions_buf,
                online_params_buf,
            )

            # Step environments
            EnvType.step_kernel_gpu[Self.batch_size, Self.obs_dim](
                ctx, obs_buf, actions_buf, rewards_buf, dones_buf
            )

            # Store transitions to replay buffer
            ctx.enqueue_function[
                store_transitions_wrapper, store_transitions_wrapper
            ](
                prev_obs_2d_tensor,
                actions_tensor,
                rewards_tensor,
                obs_2d_tensor,
                dones_tensor,
                rb_states_tensor,
                rb_actions_tensor,
                rb_rewards_tensor,
                rb_next_states_tensor,
                rb_dones_tensor,
                Scalar[DType.int32](rb_write_idx),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # Update replay buffer state
            rb_write_idx = (rb_write_idx + Self.batch_size) % RB_CAPACITY
            rb_size = min(rb_size + Self.batch_size, RB_CAPACITY)
            warmup_count += Self.batch_size

            # Reset done environments
            var rng_seed = UInt32(warmup_count * 7919 + 42)
            EnvType.selective_reset_kernel_gpu[Self.batch_size, Self.obs_dim](
                ctx, obs_buf, dones_buf, rng_seed
            )

        self.epsilon = saved_epsilon  # Restore epsilon

        # Reset ALL environments after warmup to start fresh episodes
        # (warmup may leave envs mid-episode, which would give incorrect episode rewards)
        EnvType.reset_kernel_gpu[Self.batch_size, Self.obs_dim](ctx, obs_buf)

        if verbose:
            print("Warmup complete. Replay buffer size: " + String(rb_size))

        # =====================================================================
        # Timing counters (for debugging performance)
        # =====================================================================
        from time import perf_counter_ns

        var time_action_select: UInt = 0
        var time_env_step: UInt = 0
        var time_store: UInt = 0
        var time_train: UInt = 0
        var time_episode_track: UInt = 0
        var iteration_count = 0

        # =====================================================================
        # Main Training Loop
        # =====================================================================
        while completed_episodes < num_episodes:
            var t0 = perf_counter_ns()

            # =================================================================
            # Copy current observations to prev_obs for training
            # =================================================================
            ctx.enqueue_function[copy_obs_wrapper, copy_obs_wrapper](
                prev_obs_tensor,
                obs_flat_tensor,
                grid_dim=(OBS_BLOCKS,),
                block_dim=(TPB,),
            )

            # =================================================================
            # Select actions using GPU forward pass
            # =================================================================
            self.select_actions_gpu(
                ctx,
                obs_buf,
                q_values_buf,
                actions_buf,
                online_params_buf,
            )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            time_action_select += t1 - t0

            # =================================================================
            # Step all environments on GPU
            # After step, obs_buf contains next_obs
            # =================================================================
            EnvType.step_kernel_gpu[Self.batch_size, Self.obs_dim](
                ctx, obs_buf, actions_buf, rewards_buf, dones_buf
            )
            ctx.synchronize()
            var t2 = perf_counter_ns()
            time_env_step += t2 - t1

            # =================================================================
            # Accumulate rewards and increment steps on GPU
            # =================================================================
            ctx.enqueue_function[accum_rewards_wrapper, accum_rewards_wrapper](
                episode_rewards_tensor,
                rewards_tensor,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[incr_steps_wrapper, incr_steps_wrapper](
                episode_steps_tensor,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            total_steps += (
                Self.batch_size
            )  # Each step processes batch_size transitions

            # =================================================================
            # Store transitions to GPU replay buffer
            # prev_obs = state before step, obs = next_state after step
            # =================================================================
            ctx.enqueue_function[
                store_transitions_wrapper, store_transitions_wrapper
            ](
                prev_obs_2d_tensor,
                actions_tensor,
                rewards_tensor,
                obs_2d_tensor,  # After step, obs contains next_obs
                dones_tensor,
                rb_states_tensor,
                rb_actions_tensor,
                rb_rewards_tensor,
                rb_next_states_tensor,
                rb_dones_tensor,
                Scalar[DType.int32](rb_write_idx),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # Update replay buffer state (CPU-side tracking)
            rb_write_idx = (rb_write_idx + Self.batch_size) % RB_CAPACITY
            rb_size = min(rb_size + Self.batch_size, RB_CAPACITY)
            ctx.synchronize()
            var t3 = perf_counter_ns()
            time_store += t3 - t2

            # Initialize t4 for timing (will be updated after training)
            var t4 = t3

            # =================================================================
            # Train multiple times per iteration to increase training frequency
            # GPU processes batch_size transitions per iteration. We train
            # multiple times per iteration to get more gradient updates.
            # With train_every=1, we train batch_size times. With train_every=4,
            # we train batch_size//4 times. Cap at 8 to avoid being too slow.
            # =================================================================
            if rb_size >= Self.batch_size:
                # Number of training steps to do this iteration
                # Each training step has significant kernel launch overhead (~30ms),
                # so we balance learning speed vs wall-clock time.
                # With batch_size=32 and train_every=1, we target batch_size gradient
                # updates per iteration to match CPU training frequency.
                var num_train_steps = min(
                    Self.batch_size, max(1, Self.batch_size // train_every)
                )

                for train_idx in range(num_train_steps):
                    # Sample random indices from replay buffer
                    var rng_seed = Scalar[DType.uint32](
                        total_steps * 31337 + train_idx * 12345 + 67890
                    )
                    ctx.enqueue_function[
                        sample_indices_wrapper, sample_indices_wrapper
                    ](
                        sample_indices_tensor,
                        Scalar[DType.int32](rb_size),
                        rng_seed,
                        grid_dim=(BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Gather sampled transitions into batch tensors
                    ctx.enqueue_function[
                        gather_batch_wrapper, gather_batch_wrapper
                    ](
                        sampled_obs_tensor,
                        sampled_actions_tensor,
                        sampled_rewards_tensor,
                        sampled_next_obs_tensor,
                        sampled_dones_tensor,
                        rb_states_tensor,
                        rb_actions_tensor,
                        rb_rewards_tensor,
                        rb_next_states_tensor,
                        rb_dones_tensor,
                        sample_indices_tensor,
                        grid_dim=(BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Train on sampled batch
                    _ = self.train_step_gpu_online(
                        ctx,
                        online_params_buf,
                        online_grads_buf,
                        online_state_buf,
                        target_params_buf,
                        sampled_obs_buf,  # Sampled observations from replay buffer
                        sampled_next_obs_buf,  # Sampled next observations
                        q_values_buf,
                        next_q_values_buf,
                        online_next_q_buf,
                        cache_buf,
                        grad_output_buf,
                        grad_input_buf,
                        targets_buf,
                        sampled_rewards_buf,  # Sampled rewards
                        sampled_dones_buf,  # Sampled dones
                        sampled_actions_buf,  # Sampled actions
                    )
                ctx.synchronize()
                t4 = perf_counter_ns()
                time_train += t4 - t3

            # =================================================================
            # Extract completed episodes and reset done environments
            # =================================================================
            ctx.enqueue_function[
                extract_completed_wrapper, extract_completed_wrapper
            ](
                dones_tensor,
                episode_rewards_tensor,
                episode_steps_tensor,
                completed_rewards_tensor,
                completed_steps_tensor,
                completed_mask_tensor,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # Reset done environments (where dones > 0.5)
            # The reset_kernel will reset all envs but step_kernel already updates state
            # So we need a conditional reset - for now, copy dones to check and reset selectively
            ctx.enqueue_copy(completed_rewards_host, completed_rewards_buf)
            ctx.enqueue_copy(completed_steps_host, completed_steps_buf)
            ctx.enqueue_copy(completed_mask_host, completed_mask_buf)
            ctx.synchronize()

            # Process completed episodes on CPU and reset done envs on GPU
            var any_done = False
            for i in range(Self.batch_size):
                if completed_mask_host[i] > 0.5:
                    any_done = True
                    # Log completed episode
                    var ep_reward = Float64(completed_rewards_host[i])
                    var ep_steps = Int(completed_steps_host[i])
                    metrics.log_episode(
                        completed_episodes, ep_reward, ep_steps, self.epsilon
                    )
                    completed_episodes += 1

                    # Decay epsilon after each episode
                    self.decay_epsilon()

            # Selectively reset only done environments
            # Note: extract_completed_episodes_kernel already reset episode tracking for done envs
            if any_done:
                # Use varying seed based on total_steps for different random init each time
                var rng_seed = UInt32(
                    total_steps * 7919 + 42
                )  # Prime-based variation
                EnvType.selective_reset_kernel_gpu[
                    Self.batch_size, Self.obs_dim
                ](ctx, obs_buf, dones_buf, rng_seed)

            var t5 = perf_counter_ns()
            time_episode_track += (
                t5 - t4
            )  # Episode tracking time (from after training)
            iteration_count += 1

            # =================================================================
            # Sync GPU params to CPU periodically
            # =================================================================
            if (
                completed_episodes > 0
                and (completed_episodes % sync_every == 0)
                and completed_episodes != last_print_episode
            ):
                self.online_model.copy_params_from_device(
                    ctx, online_params_buf
                )

            # =================================================================
            # Print progress (handle batch completions that may skip milestones)
            # =================================================================
            if verbose and completed_episodes > 0:
                # Calculate next print milestone
                var next_milestone = (
                    (last_print_episode // print_every) + 1
                ) * print_every
                if completed_episodes >= next_milestone:
                    last_print_episode = completed_episodes
                    var avg_reward = metrics.mean_reward_last_n(
                        min(print_every, completed_episodes)
                    )
                    print(
                        "Episode "
                        + String(completed_episodes)
                        + " | Avg reward: "
                        + String(avg_reward)[:7]
                        + " | Epsilon: "
                        + String(self.epsilon)[:5]
                        + " | Steps: "
                        + String(total_steps)
                    )

        # Copy GPU params back to CPU for evaluation
        self.online_model.copy_params_from_device(ctx, online_params_buf)
        self.target_model.copy_params_from_device(ctx, target_params_buf)

        # Print timing summary
        if verbose:
            var total_time = (
                time_action_select
                + time_env_step
                + time_store
                + time_train
                + time_episode_track
            )
            print()
            print("Timing breakdown (ms):")
            print(
                "  Action select: "
                + String(Float64(time_action_select) / 1e6)[:8]
            )
            print(
                "  Env step:      " + String(Float64(time_env_step) / 1e6)[:8]
            )
            print("  Store trans:   " + String(Float64(time_store) / 1e6)[:8])
            print("  Training:      " + String(Float64(time_train) / 1e6)[:8])
            print(
                "  Episode track: "
                + String(Float64(time_episode_track) / 1e6)[:8]
            )
            print("  Total:         " + String(Float64(total_time) / 1e6)[:8])
            print("  Iterations:    " + String(iteration_count))
            if iteration_count > 0:
                print(
                    "  Avg per iter:  "
                    + String(
                        Float64(total_time) / Float64(iteration_count) / 1e6
                    )[:8]
                    + " ms"
                )

        return metrics^

    fn train_step_gpu(
        mut self,
        ctx: DeviceContext,
        # Network buffers (pre-allocated)
        mut online_params_buf: DeviceBuffer[dtype],
        mut online_grads_buf: DeviceBuffer[dtype],
        mut online_state_buf: DeviceBuffer[dtype],
        mut target_params_buf: DeviceBuffer[dtype],
        # Batch buffers (pre-allocated)
        mut obs_buf: DeviceBuffer[dtype],
        mut next_obs_buf: DeviceBuffer[dtype],
        mut q_values_buf: DeviceBuffer[dtype],
        mut next_q_values_buf: DeviceBuffer[dtype],
        mut online_next_q_buf: DeviceBuffer[dtype],  # For Double DQN
        mut cache_buf: DeviceBuffer[dtype],
        mut grad_output_buf: DeviceBuffer[dtype],
        mut grad_input_buf: DeviceBuffer[dtype],
        mut targets_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        # Host buffers for CPU-GPU transfer (pre-allocated)
        mut obs_host: HostBuffer[dtype],
        mut next_obs_host: HostBuffer[dtype],
        mut rewards_host: HostBuffer[dtype],
        mut dones_host: HostBuffer[dtype],
        mut actions_host: HostBuffer[dtype],
    ) raises -> Float64:
        """Optimized GPU training step with pre-allocated buffers.

        All buffers are passed in to avoid allocation overhead.
        All operations run on GPU except replay buffer sampling.
        """
        # Check if buffer has enough samples
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch from replay buffer (CPU - must be random access)
        comptime BATCH_OBS_SIZE = Self.batch_size * Self.obs_dim
        var batch_obs = InlineArray[Scalar[dtype], BATCH_OBS_SIZE](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[Scalar[dtype], BATCH_OBS_SIZE](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_dones = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_actions_tmp = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        self.buffer.sample[Self.batch_size](
            batch_obs,
            batch_actions_tmp,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

        # Copy batch to pre-allocated host buffers
        for i in range(BATCH_OBS_SIZE):
            obs_host[i] = batch_obs[i]
            next_obs_host[i] = batch_next_obs[i]
        for i in range(Self.batch_size):
            rewards_host[i] = batch_rewards[i]
            dones_host[i] = batch_dones[i]
            actions_host[i] = batch_actions_tmp[i]

        # Copy to GPU (async)
        ctx.enqueue_copy(obs_buf, obs_host)
        ctx.enqueue_copy(next_obs_buf, next_obs_host)
        ctx.enqueue_copy(rewards_buf, rewards_host)
        ctx.enqueue_copy(dones_buf, dones_host)
        ctx.enqueue_copy(actions_buf, actions_host)

        # GPU Forward pass: online network with cache
        self.online_model.forward_gpu_with_cache[Self.batch_size](
            ctx, obs_buf, q_values_buf, online_params_buf, cache_buf
        )

        # GPU Forward pass: target network (no cache)
        self.target_model.forward_gpu[Self.batch_size](
            ctx, next_obs_buf, next_q_values_buf, target_params_buf
        )

        # GPU TD Target computation
        comptime BATCH_BLOCKS = (Self.batch_size + TPB - 1) // TPB
        var gamma_scalar = Scalar[dtype](self.gamma)

        # Create LayoutTensor views for kernels
        var targets_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](targets_buf.unsafe_ptr())
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var next_q_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](next_q_values_buf.unsafe_ptr())

        @parameter
        if Self.double_dqn:
            # For Double DQN: forward online network on next_obs
            self.online_model.forward_gpu[Self.batch_size](
                ctx, next_obs_buf, online_next_q_buf, online_params_buf
            )

            var online_next_tensor = LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ](online_next_q_buf.unsafe_ptr())

            # Double DQN TD target kernel
            @parameter
            @always_inline
            fn double_td_kernel_wrapper(
                targets_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                online_next_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                target_next_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                rewards_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                dones_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                gamma: Scalar[dtype],
            ):
                dqn_double_td_target_kernel[
                    dtype, Self.batch_size, Self.num_actions
                ](
                    targets_t,
                    online_next_t,
                    target_next_t,
                    rewards_t,
                    dones_t,
                    gamma,
                )

            ctx.enqueue_function[
                double_td_kernel_wrapper, double_td_kernel_wrapper
            ](
                targets_tensor,
                online_next_tensor,
                next_q_tensor,
                rewards_tensor,
                dones_tensor,
                gamma_scalar,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            # Standard DQN TD target kernel
            @parameter
            @always_inline
            fn td_kernel_wrapper(
                targets_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                next_q_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                rewards_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                dones_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                gamma: Scalar[dtype],
            ):
                dqn_td_target_kernel[dtype, Self.batch_size, Self.num_actions](
                    targets_t, next_q_t, rewards_t, dones_t, gamma
                )

            ctx.enqueue_function[td_kernel_wrapper, td_kernel_wrapper](
                targets_tensor,
                next_q_tensor,
                rewards_tensor,
                dones_tensor,
                gamma_scalar,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # GPU Gradient computation
        var q_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](q_values_buf.unsafe_ptr())
        var grad_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](grad_output_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        # We don't use loss_out in the kernel for now (would need reduction)
        @parameter
        @always_inline
        fn grad_kernel_wrapper(
            grad_t: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ],
            q_t: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ],
            targets_t: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            actions_t: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            # Inline the gradient computation
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= Self.batch_size:
                return
            var action = Int(actions_t[b])
            var q_pred = q_t[b, action]
            var td_error = q_pred - targets_t[b]
            for a in range(Self.num_actions):
                if a == action:
                    grad_t[b, a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](Self.batch_size)
                    )
                else:
                    grad_t[b, a] = Scalar[dtype](0.0)

        ctx.enqueue_function[grad_kernel_wrapper, grad_kernel_wrapper](
            grad_tensor,
            q_tensor,
            targets_tensor,
            actions_tensor,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # GPU Zero gradients
        comptime PARAM_BLOCKS = (Self.NETWORK_PARAM_SIZE + TPB - 1) // TPB
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](online_grads_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn zero_kernel_wrapper(
            buf: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
        ):
            zero_buffer_kernel[dtype, Self.NETWORK_PARAM_SIZE](buf)

        ctx.enqueue_function[zero_kernel_wrapper, zero_kernel_wrapper](
            grads_tensor,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        # GPU Backward pass
        self.online_model.backward_gpu[Self.batch_size](
            ctx,
            grad_output_buf,
            grad_input_buf,
            online_params_buf,
            cache_buf,
            online_grads_buf,
        )

        # GPU Optimizer update
        self.online_model.update_gpu(
            ctx, online_params_buf, online_grads_buf, online_state_buf
        )

        # GPU Soft update target network
        var tau_scalar = Scalar[dtype](self.tau)
        var online_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](online_params_buf.unsafe_ptr())
        var target_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](target_params_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn soft_update_wrapper(
            target_t: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
            source_t: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
            tau: Scalar[dtype],
        ):
            soft_update_kernel[dtype, Self.NETWORK_PARAM_SIZE](
                target_t, source_t, tau
            )

        ctx.enqueue_function[soft_update_wrapper, soft_update_wrapper](
            target_params_tensor,
            online_params_tensor,
            tau_scalar,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        self.train_step_count += 1
        return 0.0  # Loss computation would need GPU reduction

    fn select_action_gpu(
        self,
        obs: SIMD[DType.float64, Self.obs_dim],
        ctx: DeviceContext,
        mut obs_buf: DeviceBuffer[dtype],
        mut q_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        mut obs_host: HostBuffer[dtype],
        mut q_host: HostBuffer[dtype],
    ) raises -> Int:
        """Select action using GPU forward pass (avoids CPU param sync).

        Args:
            obs: Current observation.
            ctx: GPU device context.
            obs_buf: Pre-allocated GPU buffer for observation [obs_dim].
            q_buf: Pre-allocated GPU buffer for Q-values [num_actions].
            params_buf: GPU buffer containing current params.
            obs_host: Pre-allocated host buffer for observation [obs_dim].
            q_host: Pre-allocated host buffer for Q-values [num_actions].

        Returns:
            Selected action index.
        """
        # Epsilon-greedy exploration
        if random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.num_actions))

        # Copy obs to host buffer
        for i in range(Self.obs_dim):
            obs_host[i] = Scalar[dtype](obs[i])

        # Copy to GPU
        ctx.enqueue_copy(obs_buf, obs_host)

        # Forward pass on GPU (batch=1)
        self.online_model.forward_gpu[1](ctx, obs_buf, q_buf, params_buf)

        # Copy Q-values back
        ctx.enqueue_copy(q_host, q_buf)
        ctx.synchronize()

        # Find argmax
        var best_action = 0
        var best_q = q_host[0]
        for a in range(1, Self.num_actions):
            if q_host[a] > best_q:
                best_q = q_host[a]
                best_action = a

        return best_action

    fn train_gpu[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        ctx: DeviceContext,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        sync_every: Int = 5,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the DQN agent on GPU with optimized buffer management.

        All GPU buffers are pre-allocated once at the start of training.
        Action selection uses GPU forward pass (no CPU param sync needed).
        Environment interaction happens on CPU, all training operations on GPU.

        Args:
            ctx: GPU device context.
            env: The environment to train on (must implement BoxDiscreteActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Number of random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            sync_every: Sync GPU params to CPU every N episodes for backup (default: 5).
            verbose: Whether to print progress (default: False).
            print_every: Print progress every N episodes if verbose (default: 10).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN (GPU)" if not Self.double_dqn else "Double DQN (GPU)",
            environment_name=environment_name,
        )

        # =====================================================================
        # Pre-allocate ALL GPU buffers (done once!)
        # =====================================================================
        comptime PARAM_SIZE = Self.NETWORK_PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * 2  # Adam has 2 state values per param
        comptime BATCH_OBS_SIZE = Self.batch_size * Self.obs_dim
        comptime BATCH_Q_SIZE = Self.batch_size * Self.num_actions
        comptime BATCH_CACHE_SIZE = Self.batch_size * Self.NETWORK_CACHE_SIZE

        # Network parameter buffers
        var online_params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var online_grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var online_state_buf = ctx.enqueue_create_buffer[dtype](STATE_SIZE)
        var target_params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        # Batch data buffers (GPU)
        var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var next_obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var next_q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var online_next_q_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_Q_SIZE
        )  # Double DQN
        var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH_CACHE_SIZE)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var targets_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)

        # Host buffers for CPU-GPU transfer
        var obs_host = ctx.enqueue_create_host_buffer[dtype](BATCH_OBS_SIZE)
        var next_obs_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH_OBS_SIZE
        )
        var rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        var dones_host = ctx.enqueue_create_host_buffer[dtype](Self.batch_size)
        var actions_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )

        # Action selection buffers (batch=1 for single observation)
        var action_obs_buf = ctx.enqueue_create_buffer[dtype](Self.obs_dim)
        var action_q_buf = ctx.enqueue_create_buffer[dtype](Self.num_actions)
        var action_obs_host = ctx.enqueue_create_host_buffer[dtype](
            Self.obs_dim
        )
        var action_q_host = ctx.enqueue_create_host_buffer[dtype](
            Self.num_actions
        )

        # Copy CPU params to GPU
        self.online_model.copy_params_to_device(ctx, online_params_buf)
        self.online_model.copy_state_to_device(ctx, online_state_buf)
        self.target_model.copy_params_to_device(ctx, target_params_buf)

        if verbose:
            print("GPU buffers allocated. Starting training...")

        # =====================================================================
        # Warmup: fill replay buffer with random actions (CPU)
        # =====================================================================
        var warmup_obs = self._list_to_simd(env.reset_obs_list())
        var warmup_count = 0

        while warmup_count < warmup_steps:
            var action = Int(random_float64() * Float64(Self.num_actions))
            var result = env.step_obs(action)
            var next_obs = self._list_to_simd(result[0])
            self.store_transition(
                warmup_obs, action, result[1], next_obs, result[2]
            )
            warmup_obs = next_obs
            warmup_count += 1
            if result[2]:
                warmup_obs = self._list_to_simd(env.reset_obs_list())

        # =====================================================================
        # Training loop
        # =====================================================================
        var total_steps = 0

        for episode in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action using GPU forward pass (no CPU param sync needed)
                var action = self.select_action_gpu(
                    obs,
                    ctx,
                    action_obs_buf,
                    action_q_buf,
                    online_params_buf,
                    action_obs_host,
                    action_q_host,
                )

                # Step environment (CPU)
                var result = env.step_obs(action)
                var next_obs = self._list_to_simd(result[0])
                var reward = result[1]
                var done = result[2]

                # Store transition (CPU)
                self.store_transition(obs, action, reward, next_obs, done)

                # Train every N steps (GPU - all buffers pre-allocated!)
                if total_steps % train_every == 0:
                    _ = self.train_step_gpu(
                        ctx,
                        online_params_buf,
                        online_grads_buf,
                        online_state_buf,
                        target_params_buf,
                        obs_buf,
                        next_obs_buf,
                        q_values_buf,
                        next_q_values_buf,
                        online_next_q_buf,
                        cache_buf,
                        grad_output_buf,
                        grad_input_buf,
                        targets_buf,
                        rewards_buf,
                        dones_buf,
                        actions_buf,
                        obs_host,
                        next_obs_host,
                        rewards_host,
                        dones_host,
                        actions_host,
                    )

                episode_reward += reward
                obs = next_obs
                total_steps += 1
                episode_steps += 1

                if done:
                    break

            # Decay epsilon
            self.decay_epsilon()

            # Sync GPU params to CPU periodically (for backup, not needed for action selection)
            if (episode + 1) % sync_every == 0:
                self.online_model.copy_params_from_device(
                    ctx, online_params_buf
                )

            # Log metrics
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.epsilon
            )

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg reward: "
                    + String(avg_reward)[:7]
                    + " | Epsilon: "
                    + String(self.epsilon)[:5]
                    + " | Steps: "
                    + String(total_steps)
                )

        # Copy GPU params back to CPU for evaluation
        self.online_model.copy_params_from_device(ctx, online_params_buf)
        self.target_model.copy_params_from_device(ctx, target_params_buf)

        return metrics^
