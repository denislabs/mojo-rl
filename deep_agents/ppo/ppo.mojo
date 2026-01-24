"""Deep PPO (Proximal Policy Optimization) Agent using the new trait-based architecture.

This PPO implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building actor and critic networks
- Clipped surrogate objective for stable policy updates
- GAE (Generalized Advantage Estimation) for variance reduction

Key features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Clipped policy ratio for stable updates
- Multiple epochs of optimization per rollout
- Entropy bonus for exploration
- Advantage normalization

Architecture:
- Actor: obs -> hidden (ReLU) -> hidden (ReLU) -> num_actions (Softmax)
- Critic: obs -> hidden (ReLU) -> hidden (ReLU) -> 1 (value)

Usage:
    from deep_agents.ppo import DeepPPOAgent
    from envs import CartPoleNative

    var env = CartPoleNative()
    var agent = DeepPPOAgent[4, 2, 128]()

    var metrics = agent.train(env, num_episodes=1000)

Features:
 - Per-minibatch advantage normalization (norm_adv_per_minibatch)
 - Value clipping for stable critic updates (clip_value)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from math import exp, log, sqrt
from random import random_float64, seed
from time import perf_counter_ns
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, LinearReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Xavier
from deep_rl.training import Network
from deep_rl.checkpoint import (
    split_lines,
    find_section_start,
    save_checkpoint_file,
    read_checkpoint_file,
)
from deep_rl.gpu import (
    random_range,
    xorshift32,
    random_uniform,
    soft_update_kernel,
    zero_buffer_kernel,
    copy_buffer_kernel,
    accumulate_rewards_kernel,
    increment_steps_kernel,
    extract_completed_episodes_kernel,
    selective_reset_tracking_kernel,
)
from core import TrainingMetrics, BoxDiscreteActionEnv, GPUDiscreteEnv
from render import RendererBase
from memory import UnsafePointer
from core.utils.gae import compute_gae_inline
from core.utils.softmax import (
    softmax_inline,
    sample_from_probs_inline,
    argmax_probs_inline,
)
from core.utils.normalization import normalize_inline
from core.utils.shuffle import shuffle_indices_inline


# =============================================================================
# GPU Kernels for PPO Operations
# =============================================================================


@always_inline
fn ppo_store_rollout_kernel[
    dtype: DType,
    N_ENVS: Int,
    OBS_DIM: Int,
](
    # Outputs - rollout buffer storage
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin
    ],
    rollout_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    rollout_rewards: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    rollout_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    rollout_log_probs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    rollout_dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Inputs - current step data
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Store transition data for one timestep (n_envs transitions).

    This kernel stores data at timestep t. The rollout buffer tensors
    passed in should be views at offset t * n_envs.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    # Store observation
    for d in range(OBS_DIM):
        rollout_obs[i, d] = obs[i, d]

    rollout_actions[i] = actions[i]
    rollout_rewards[i] = rewards[i]
    rollout_values[i] = values[i]
    rollout_log_probs[i] = log_probs[i]
    rollout_dones[i] = dones[i]


@always_inline
fn ppo_gather_minibatch_kernel[
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
fn ppo_actor_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    grad_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
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

    Gradient is zero if ratio is clipped, otherwise:
    grad = -advantage * ratio * d_log_prob - entropy_coef * d_entropy
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

    # Check if clipped
    var is_clipped = (ratio < Scalar[dtype](1.0) - clip_epsilon) or (
        ratio > Scalar[dtype](1.0) + clip_epsilon
    )

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


@always_inline
fn ppo_actor_grad_with_kl_kernel[
    dtype: DType,
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

    # Compute approximate KL divergence: old_log_prob - new_log_prob
    var kl = old_log_probs[b] - new_log_prob
    kl_divergences[b] = kl

    # Probability ratio
    var ratio = exp(new_log_prob - old_log_probs[b])

    # Check if clipped
    var is_clipped = (ratio < Scalar[dtype](1.0) - clip_epsilon) or (
        ratio > Scalar[dtype](1.0) + clip_epsilon
    )

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


@always_inline
fn gradient_norm_kernel[
    dtype: DType,
    SIZE: Int,
    NUM_BLOCKS: Int,
    BLOCK_SIZE: Int,
](
    # Output
    partial_sums: LayoutTensor[
        dtype, Layout.row_major(NUM_BLOCKS), MutAnyOrigin
    ],
    # Input
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Compute partial squared sums for gradient norm calculation.

    Each block computes a partial sum of squared gradients.
    Final norm = sqrt(sum of all partial_sums).
    """
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var i = bid * BLOCK_SIZE + tid

    # Shared memory for block reduction
    shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Load and square
    if i < SIZE:
        var g = grads[i]
        shared[tid] = g * g
    else:
        shared[tid] = Scalar[dtype](0.0)

    barrier()

    # Block reduction
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = shared[tid] + shared[tid + stride]
        barrier()
        stride //= 2

    # Write result
    if tid == 0:
        partial_sums[bid] = shared[0]


@always_inline
fn gradient_clip_kernel[
    dtype: DType,
    SIZE: Int,
](
    # In/Out
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    # Input
    scale: Scalar[dtype],
):
    """Scale gradients by a factor (for gradient clipping).

    Called when grad_norm > max_grad_norm with scale = max_grad_norm / grad_norm.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return

    grads[i] = grads[i] * scale


@always_inline
fn gradient_reduce_and_compute_scale_kernel[
    dtype: DType,
    NUM_BLOCKS: Int,
    BLOCK_SIZE: Int,
](
    # Output
    scale_out: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    # Input
    partial_sums: LayoutTensor[dtype, Layout.row_major(NUM_BLOCKS), MutAnyOrigin],
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

    # Shared memory for reduction
    var shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
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
fn gradient_apply_scale_kernel[
    dtype: DType,
    PARAM_SIZE: Int,
](
    # In/Out
    grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    # Input
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
fn ppo_critic_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
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
    """Compute gradient for critic value loss: MSE(value, return)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # Gradient of MSE loss: 2 * (value - return) / N
    grad_values[b, 0] = (
        Scalar[dtype](2.0)
        * value_loss_coef
        * (values[b, 0] - returns[b])
        / Scalar[dtype](BATCH_SIZE)
    )


@always_inline
fn ppo_critic_grad_clipped_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    # Outputs
    grad_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin
    ],
    # Inputs
    values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin],
    old_values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    value_loss_coef: Scalar[dtype],
    clip_epsilon: Scalar[dtype],
    batch_size: Int,
):
    """Compute gradient for critic with value clipping.

    Value clipping prevents the value function from changing too drastically:
    V_clipped = V_old + clip(V - V_old, -ε, +ε)
    L_value = max((V - returns)², (V_clipped - returns)²)

    We take gradient of the max, which means we use the gradient of whichever
    loss is larger (more pessimistic update).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var v_new_simd = values[b, 0]
    var v_old_simd = old_values[b]
    var ret_simd = returns[b]

    # Extract scalar values
    var v_new = Scalar[dtype](v_new_simd[0])
    var v_old = Scalar[dtype](v_old_simd[0])
    var ret = Scalar[dtype](ret_simd[0])

    # Clipped value
    var v_diff = v_new - v_old
    var v_clipped: Scalar[dtype]
    if v_diff > clip_epsilon:
        v_clipped = v_old + clip_epsilon
    elif v_diff < -clip_epsilon:
        v_clipped = v_old - clip_epsilon
    else:
        v_clipped = v_new

    # Unclipped and clipped losses
    var loss_unclipped = (v_new - ret) * (v_new - ret)
    var loss_clipped = (v_clipped - ret) * (v_clipped - ret)

    # Use gradient of the larger loss (pessimistic)
    var grad: Scalar[dtype]
    if loss_unclipped > loss_clipped:
        # Gradient of unclipped loss
        grad = Scalar[dtype](2.0) * (v_new - ret)
    else:
        # Gradient of clipped loss - but v_clipped might be clamped
        if v_diff > clip_epsilon or v_diff < -clip_epsilon:
            # v_clipped doesn't depend on v_new, so gradient is 0
            grad = Scalar[dtype](0.0)
        else:
            # v_clipped = v_new, same as unclipped
            grad = Scalar[dtype](2.0) * (v_new - ret)

    grad_values[b, 0] = value_loss_coef * grad / Scalar[dtype](BATCH_SIZE)


@always_inline
fn normalize_advantages_kernel[
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
# GPU Kernels: Store transition data
# =============================================================================


@always_inline
fn _store_pre_step_kernel[
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
fn _store_post_step_kernel[
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
fn _extract_obs_from_state_kernel[
    dtype: DType,
    N_ENVS: Int,
    STATE_SIZE: Int,
    OBS_DIM: Int,
](
    # Output - observation buffer for neural network input
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    # Input - full state buffer from environment
    states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, STATE_SIZE), MutAnyOrigin
    ],
):
    """Extract observations from full state buffer.

    For environments where STATE_SIZE > OBS_DIM, observations are stored
    at the beginning of each environment's state (offset 0).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    # Copy first OBS_DIM elements from state to obs
    for d in range(OBS_DIM):
        obs[i, d] = states[i, d]


# =============================================================================
# Deep PPO Agent
# =============================================================================


struct DeepPPOAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 64,
    rollout_len: Int = 128,
    n_envs: Int = 1024,
    gpu_minibatch_size: Int = 256,
]:
    """Deep Proximal Policy Optimization Agent using new trait-based architecture.

    Uses clipped surrogate objective for stable policy updates:
    L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
    where r(θ) = π_θ(a|s) / π_θ_old(a|s)

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 64).
        rollout_len: Steps per rollout per environment (default: 128 for GPU).
        n_envs: Number of parallel environments for GPU training (default: 1024).
        gpu_minibatch_size: Minibatch size for GPU training (default: 256).

    Note on GPU training:
        - n_envs: Parallel environments on GPU (affects data collection rate)
        - rollout_len: Steps before training (total transitions = n_envs × rollout_len)
        - gpu_minibatch_size: Samples per gradient update
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime ROLLOUT = Self.rollout_len

    # Cache sizes
    comptime ACTOR_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN
    comptime CRITIC_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Network parameter sizes (for GPU buffer allocation)
    # Actor: Linear[obs, hidden] + ReLU + Linear[hidden, hidden] + ReLU + Linear[hidden, actions]
    comptime ACTOR_PARAM_SIZE: Int = (
        Self.OBS * Self.HIDDEN
        + Self.HIDDEN  # Linear 1
        + Self.HIDDEN * Self.HIDDEN
        + Self.HIDDEN  # Linear 2
        + Self.HIDDEN * Self.ACTIONS
        + Self.ACTIONS  # Linear 3
    )
    # Critic: Linear[obs, hidden] + ReLU + Linear[hidden, hidden] + ReLU + Linear[hidden, 1]
    comptime CRITIC_PARAM_SIZE: Int = (
        Self.OBS * Self.HIDDEN
        + Self.HIDDEN  # Linear 1
        + Self.HIDDEN * Self.HIDDEN
        + Self.HIDDEN  # Linear 2
        + Self.HIDDEN * 1
        + 1  # Linear 3
    )

    # GPU-specific sizes
    comptime TOTAL_ROLLOUT_SIZE: Int = Self.n_envs * Self.rollout_len
    comptime GPU_MINIBATCH = Self.gpu_minibatch_size

    # Actor network: obs -> hidden (ReLU) -> hidden (ReLU) -> action logits
    var actor: Network[
        type_of(
            seq(
                LinearReLU[Self.OBS, Self.HIDDEN](),
                LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Xavier,
    ]

    # Critic network: obs -> hidden (ReLU) -> hidden (ReLU) -> value
    var critic: Network[
        type_of(
            seq(
                LinearReLU[Self.OBS, Self.HIDDEN](),
                LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            )
        ),
        Adam,
        Xavier,
    ]

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var actor_lr: Float64
    var critic_lr: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var minibatch_size: Int
    var normalize_advantages: Bool

    # Advanced hyperparameters (environment-agnostic improvements)
    var target_kl: Float64  # KL threshold for early epoch stopping
    var max_grad_norm: Float64  # Gradient clipping threshold
    var anneal_lr: Bool  # Whether to linearly anneal learning rate
    var anneal_entropy: Bool  # Whether to anneal entropy coefficient
    var target_total_steps: Int  # Target steps for annealing (0 = auto-calculate)
    var clip_value: Bool  # Whether to clip value function updates
    var norm_adv_per_minibatch: Bool  # Normalize advantages per minibatch

    # Rollout buffers (heap-allocated to avoid stack overflow for large ROLLOUT/OBS)
    var buffer_obs: List[Scalar[dtype]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Scalar[dtype]]
    var buffer_values: List[Scalar[dtype]]
    var buffer_log_probs: List[Scalar[dtype]]
    var buffer_dones: List[Bool]
    var buffer_idx: Int

    # Training state
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int  # Save checkpoint every N episodes (0 to disable)
    var checkpoint_path: String  # Path for auto-checkpointing

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        minibatch_size: Int = 64,
        normalize_advantages: Bool = True,
        # Advanced hyperparameters
        target_kl: Float64 = 0.015,
        max_grad_norm: Float64 = 0.5,
        anneal_lr: Bool = True,
        anneal_entropy: Bool = False,
        target_total_steps: Int = 0,
        clip_value: Bool = True,
        norm_adv_per_minibatch: Bool = True,
        # Checkpoint settings
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep PPO agent.

        Args:
            gamma: Discount factor (default: 0.99).
            gae_lambda: GAE lambda parameter (default: 0.95).
            clip_epsilon: PPO clipping parameter (default: 0.2).
            actor_lr: Actor learning rate (default: 0.0003).
            critic_lr: Critic learning rate (default: 0.001).
            entropy_coef: Entropy bonus coefficient (default: 0.01).
            value_loss_coef: Value loss coefficient (default: 0.5).
            num_epochs: Number of optimization epochs per update (default: 4).
            minibatch_size: Size of minibatches (default: 64).
            normalize_advantages: Whether to normalize advantages (default: True).
            target_kl: KL divergence threshold for early epoch stopping (default: 0.015).
            max_grad_norm: Maximum gradient norm for clipping (default: 0.5).
            anneal_lr: Whether to linearly anneal learning rate (default: True).
            anneal_entropy: Whether to anneal entropy coefficient (default: False).
            target_total_steps: Target total steps for annealing, 0=auto (default: 0).
            clip_value: Whether to clip value function updates (default: True).
            norm_adv_per_minibatch: Normalize advantages per minibatch (default: True).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints.
        """
        # Build actor and critic models
        var actor_model = seq(
            LinearReLU[Self.OBS, Self.HIDDEN](),
            LinearReLU[Self.HIDDEN, Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.ACTIONS](),
        )

        var critic_model = seq(
            LinearReLU[Self.OBS, Self.HIDDEN](),
            LinearReLU[Self.HIDDEN, Self.HIDDEN](),
            Linear[Self.HIDDEN, 1](),
        )

        # Initialize networks
        self.actor = Network(actor_model, Adam(lr=actor_lr), Xavier())
        self.critic = Network(critic_model, Adam(lr=critic_lr), Xavier())

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.normalize_advantages = normalize_advantages

        # Store advanced hyperparameters
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.anneal_entropy = anneal_entropy
        self.target_total_steps = target_total_steps
        self.clip_value = clip_value
        self.norm_adv_per_minibatch = norm_adv_per_minibatch

        # Initialize buffers (heap-allocated Lists)
        self.buffer_obs = List[Scalar[dtype]](capacity=Self.ROLLOUT * Self.OBS)
        for _ in range(Self.ROLLOUT * Self.OBS):
            self.buffer_obs.append(Scalar[dtype](0))

        self.buffer_actions = List[Int](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_actions.append(0)

        self.buffer_rewards = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_rewards.append(Scalar[dtype](0))

        self.buffer_values = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_values.append(Scalar[dtype](0))

        self.buffer_log_probs = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_log_probs.append(Scalar[dtype](0))

        self.buffer_dones = List[Bool](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_dones.append(False)

        self.buffer_idx = 0

        # Training state
        self.train_step_count = 0

        # Auto-checkpoint settings
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        training: Bool = True,
    ) -> Tuple[Int, Scalar[dtype], Scalar[dtype]]:
        """Select action from policy and compute log probability and value.

        Args:
            obs: Current observation.
            training: If True, sample action; else use greedy.

        Returns:
            Tuple of (action, log_prob, value).
        """
        # Forward actor to get logits
        var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        self.actor.forward[1](obs, logits)

        # Compute softmax probabilities
        var probs = softmax_inline[dtype, Self.ACTIONS](logits)

        # Forward critic to get value
        var value_out = InlineArray[Scalar[dtype], 1](uninitialized=True)
        self.critic.forward[1](obs, value_out)
        var value = value_out[0]

        # Sample or select greedy action
        var action: Int
        if training:
            action = sample_from_probs_inline[dtype, Self.ACTIONS](probs)
        else:
            action = argmax_probs_inline[dtype, Self.ACTIONS](probs)

        # Compute log probability
        var log_prob = log(probs[action] + Scalar[dtype](1e-8))

        return (action, log_prob, value)

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        action: Int,
        reward: Float64,
        log_prob: Scalar[dtype],
        value: Scalar[dtype],
        done: Bool,
    ):
        """Store transition in rollout buffer."""
        # Store observation
        for i in range(Self.OBS):
            self.buffer_obs[self.buffer_idx * Self.OBS + i] = obs[i]

        self.buffer_actions[self.buffer_idx] = action
        self.buffer_rewards[self.buffer_idx] = Scalar[dtype](reward)
        self.buffer_log_probs[self.buffer_idx] = log_prob
        self.buffer_values[self.buffer_idx] = value
        self.buffer_dones[self.buffer_idx] = done

        self.buffer_idx += 1

    fn update(
        mut self,
        next_obs: InlineArray[Scalar[dtype], Self.OBS],
    ) -> Float64:
        """Update actor and critic using PPO with clipped objective.

        Args:
            next_obs: Next observation for bootstrapping.

        Returns:
            Total loss value.
        """
        if self.buffer_idx == 0:
            return 0.0

        var buffer_len = self.buffer_idx

        # Get bootstrap value
        var next_value_out = InlineArray[Scalar[dtype], 1](uninitialized=True)
        self.critic.forward[1](next_obs, next_value_out)
        var next_value = next_value_out[0]

        # Compute GAE advantages and returns (inline for List compatibility)
        var advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        var returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(buffer_len):
            advantages.append(Scalar[dtype](0))
            returns.append(Scalar[dtype](0))

        # GAE computation
        var gae = Scalar[dtype](0.0)
        var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
        for t in range(buffer_len - 1, -1, -1):
            var next_val: Scalar[dtype]
            if t == buffer_len - 1:
                next_val = next_value
            else:
                next_val = self.buffer_values[t + 1]

            # Reset GAE at episode boundary
            if self.buffer_dones[t]:
                next_val = Scalar[dtype](0.0)
                gae = Scalar[dtype](0.0)

            # TD residual: δ = r + γV(s') - V(s)
            var delta = (
                self.buffer_rewards[t]
                + Scalar[dtype](self.gamma) * next_val
                - self.buffer_values[t]
            )

            # GAE accumulation: A = δ + γλA'
            gae = delta + gae_decay * gae

            advantages[t] = gae
            returns[t] = gae + self.buffer_values[t]

        # Normalize advantages
        if self.normalize_advantages and buffer_len > 1:
            var mean = Scalar[dtype](0.0)
            for i in range(buffer_len):
                mean += advantages[i]
            mean /= Scalar[dtype](buffer_len)

            var var_sum = Scalar[dtype](0.0)
            for i in range(buffer_len):
                var diff = advantages[i] - mean
                var_sum += diff * diff

            var std = sqrt(
                var_sum / Scalar[dtype](buffer_len) + Scalar[dtype](1e-8)
            )
            for i in range(buffer_len):
                advantages[i] = (advantages[i] - mean) / std

        # =====================================================================
        # Multiple epochs of optimization
        # =====================================================================

        var total_loss = Scalar[dtype](0.0)
        var indices = List[Int](capacity=buffer_len)
        for i in range(buffer_len):
            indices.append(i)

        for epoch in range(self.num_epochs):
            # Shuffle indices for minibatch sampling using Fisher-Yates
            for i in range(buffer_len - 1, 0, -1):
                var j = Int(random_float64() * Float64(i + 1))
                var temp = indices[i]
                indices[i] = indices[j]
                indices[j] = temp

            var batch_start = 0
            while batch_start < buffer_len:
                var batch_end = batch_start + self.minibatch_size
                if batch_end > buffer_len:
                    batch_end = buffer_len

                var mb_size = batch_end - batch_start

                # Per-minibatch advantage normalization
                var mb_advantages = List[Scalar[dtype]](capacity=mb_size)
                for b in range(batch_start, batch_end):
                    var t = indices[b]
                    mb_advantages.append(advantages[t])

                if self.norm_adv_per_minibatch and mb_size > 1:
                    var mb_mean = Scalar[dtype](0.0)
                    for i in range(mb_size):
                        mb_mean += mb_advantages[i]
                    mb_mean /= Scalar[dtype](mb_size)

                    var mb_var_sum = Scalar[dtype](0.0)
                    for i in range(mb_size):
                        var diff = mb_advantages[i] - mb_mean
                        mb_var_sum += diff * diff

                    var mb_std = sqrt(
                        mb_var_sum / Scalar[dtype](mb_size)
                        + Scalar[dtype](1e-8)
                    )
                    for i in range(mb_size):
                        mb_advantages[i] = (mb_advantages[i] - mb_mean) / mb_std

                # Process minibatch
                for b in range(batch_start, batch_end):
                    var t = indices[b]
                    var mb_idx = b - batch_start

                    # Get observation for this timestep
                    var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
                    for i in range(Self.OBS):
                        obs[i] = self.buffer_obs[t * Self.OBS + i]

                    var action = self.buffer_actions[t]
                    var old_log_prob = self.buffer_log_probs[t]
                    var old_value = self.buffer_values[t]
                    var advantage = mb_advantages[mb_idx]
                    var return_t = returns[t]

                    # ==========================================================
                    # Actor forward and update
                    # ==========================================================
                    var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    self.actor.forward[1](obs, logits)

                    var probs = softmax_inline[dtype, Self.ACTIONS](logits)
                    var new_log_prob = log(probs[action] + Scalar[dtype](1e-8))

                    # Probability ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
                    var ratio = exp(new_log_prob - old_log_prob)

                    # Clipped surrogate objective
                    var surr1 = ratio * advantage
                    var clipped_ratio: Scalar[dtype]
                    if advantage >= Scalar[dtype](0.0):
                        clipped_ratio = min(
                            ratio, Scalar[dtype](1.0 + self.clip_epsilon)
                        )
                    else:
                        clipped_ratio = max(
                            ratio, Scalar[dtype](1.0 - self.clip_epsilon)
                        )
                    var surr2 = clipped_ratio * advantage

                    # Policy loss: -min(surr1, surr2)
                    var policy_loss: Scalar[dtype]
                    if surr1 < surr2:
                        policy_loss = -surr1
                    else:
                        policy_loss = -surr2

                    # Entropy bonus
                    var entropy = Scalar[dtype](0.0)
                    for a in range(Self.ACTIONS):
                        if probs[a] > Scalar[dtype](1e-8):
                            entropy -= probs[a] * log(probs[a])

                    # Check if ratio is clipped
                    var is_clipped = (
                        ratio < Scalar[dtype](1.0 - self.clip_epsilon)
                    ) or (ratio > Scalar[dtype](1.0 + self.clip_epsilon))

                    # Actor gradient (only if not clipped)
                    var d_logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        fill=0
                    )
                    if not is_clipped:
                        for a in range(Self.ACTIONS):
                            var d_log_prob: Scalar[dtype]
                            if a == action:
                                d_log_prob = Scalar[dtype](1.0) - probs[a]
                            else:
                                d_log_prob = -probs[a]

                            # Entropy gradient
                            var d_entropy = -probs[a] * (
                                Scalar[dtype](1.0)
                                + log(probs[a] + Scalar[dtype](1e-8))
                            )

                            d_logits[a] = (
                                -advantage * ratio * d_log_prob
                                - Scalar[dtype](self.entropy_coef) * d_entropy
                            )

                    # Backward through actor (use heap-allocated cache for large HIDDEN)
                    var actor_cache = List[Scalar[dtype]](
                        capacity=Self.ACTOR_CACHE
                    )
                    for _ in range(Self.ACTOR_CACHE):
                        actor_cache.append(Scalar[dtype](0))
                    self.actor.forward_with_cache_heap[1](
                        obs, logits, actor_cache
                    )

                    var actor_grad_input = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    self.actor.zero_grads()
                    self.actor.backward_heap[1](
                        d_logits, actor_grad_input, actor_cache
                    )
                    self.actor.update()

                    # ==========================================================
                    # Critic forward and update
                    # ==========================================================
                    var value_out = InlineArray[Scalar[dtype], 1](
                        uninitialized=True
                    )
                    var critic_cache = List[Scalar[dtype]](
                        capacity=Self.CRITIC_CACHE
                    )
                    for _ in range(Self.CRITIC_CACHE):
                        critic_cache.append(Scalar[dtype](0))
                    self.critic.forward_with_cache_heap[1](
                        obs, value_out, critic_cache
                    )

                    var value = value_out[0]

                    # Value loss: (return - value)^2
                    var value_loss = (return_t - value) * (return_t - value)

                    # Critic gradient (with optional value clipping)
                    var d_value = InlineArray[Scalar[dtype], 1](fill=0)
                    if self.clip_value:
                        # Clipped value function
                        var v_diff = value - old_value
                        var v_clipped: Scalar[dtype]
                        if v_diff > Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value + Scalar[dtype](
                                self.clip_epsilon
                            )
                        elif v_diff < -Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value - Scalar[dtype](
                                self.clip_epsilon
                            )
                        else:
                            v_clipped = value

                        # Unclipped and clipped losses
                        var loss_unclipped = (value - return_t) * (
                            value - return_t
                        )
                        var loss_clipped = (v_clipped - return_t) * (
                            v_clipped - return_t
                        )

                        # Use gradient of the larger loss (pessimistic)
                        if loss_unclipped > loss_clipped:
                            # Gradient of unclipped loss
                            d_value[0] = (
                                Scalar[dtype](2.0)
                                * Scalar[dtype](self.value_loss_coef)
                                * (value - return_t)
                            )
                        else:
                            # Gradient of clipped loss
                            if v_diff > Scalar[dtype](
                                self.clip_epsilon
                            ) or v_diff < -Scalar[dtype](self.clip_epsilon):
                                # v_clipped doesn't depend on value, gradient is 0
                                d_value[0] = Scalar[dtype](0.0)
                            else:
                                # v_clipped = value
                                d_value[0] = (
                                    Scalar[dtype](2.0)
                                    * Scalar[dtype](self.value_loss_coef)
                                    * (value - return_t)
                                )
                    else:
                        # Regular gradient
                        d_value[0] = (
                            Scalar[dtype](2.0)
                            * Scalar[dtype](self.value_loss_coef)
                            * (value - return_t)
                        )

                    # Backward through critic
                    var critic_grad_input = InlineArray[
                        Scalar[dtype], Self.OBS
                    ](fill=0)
                    self.critic.zero_grads()
                    self.critic.backward_heap[1](
                        d_value, critic_grad_input, critic_cache
                    )
                    self.critic.update()

                    total_loss += (
                        policy_loss
                        + Scalar[dtype](self.value_loss_coef) * value_loss
                        - Scalar[dtype](self.entropy_coef) * entropy
                    )

                batch_start = batch_end

        # Clear buffer
        self.buffer_idx = 0
        self.train_step_count += 1

        return Float64(total_loss / Scalar[dtype](self.num_epochs * buffer_len))

    fn _list_to_inline[
        dtype: DType
    ](self, obs_list: List[Scalar[dtype]]) -> InlineArray[
        Scalar[dtype], Self.OBS
    ]:
        """Convert List[Float64] to InlineArray."""
        var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
        for i in range(Self.OBS):
            if i < len(obs_list):
                obs[i] = Scalar[dtype](obs_list[i])
        return obs

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the PPO agent on a discrete action environment.

        Args:
            env: The environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep PPO",
            environment_name=environment_name,
        )

        var total_steps = 0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action
                var action_result = self.select_action(obs, training=True)
                var action = action_result[0]
                var log_prob = action_result[1]
                var value = action_result[2]

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                var next_obs = self._list_to_inline(next_obs_list)

                # Store transition
                self.store_transition(
                    obs, action, reward, log_prob, value, done
                )

                episode_reward += reward
                obs = next_obs
                total_steps += 1
                episode_steps += 1

                # Update at rollout boundary or episode end
                if self.buffer_idx >= Self.ROLLOUT or done:
                    _ = self.update(obs)

                if done:
                    break

            # Log metrics
            metrics.log_episode(episode, episode_reward, episode_steps, 0.0)

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode",
                    episode + 1,
                    "| Avg reward:",
                    String(avg_reward)[:7],
                    "| Steps:",
                    total_steps,
                )

            # Auto-checkpoint
            if self.checkpoint_every > 0 and len(self.checkpoint_path) > 0:
                if (episode + 1) % self.checkpoint_every == 0:
                    try:
                        self.save_checkpoint(self.checkpoint_path)
                        if verbose:
                            print("Checkpoint saved at episode", episode + 1)
                    except:
                        print(
                            "Warning: Failed to save checkpoint at episode",
                            episode + 1,
                        )

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 1000,
        verbose: Bool = False,
        renderer: UnsafePointer[RendererBase, MutAnyOrigin] = UnsafePointer[
            RendererBase, MutAnyOrigin
        ](),
    ) -> Float64:
        """Evaluate the agent using greedy policy.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            renderer: Optional pointer to renderer for visualization.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps):
                # Greedy action
                var action_result = self.select_action(obs, training=False)
                var action = action_result[0]

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                if renderer:
                    env.render(renderer[])

                episode_reward += reward
                obs = self._list_to_inline(next_obs_list)
                episode_steps += 1

                if done:
                    break

            total_reward += episode_reward

            if verbose:
                print(
                    "Eval Episode",
                    episode + 1,
                    "| Reward:",
                    String(episode_reward)[:10],
                    "| Steps:",
                    episode_steps,
                )

        return total_reward / Float64(num_episodes)

    # =========================================================================
    # GPU Training
    # =========================================================================

    fn train_gpu[
        EnvType: GPUDiscreteEnv
    ](
        mut self,
        ctx: DeviceContext,
        num_episodes: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train PPO on GPU with parallel environments.

        Args:
            ctx: GPU device context.
            num_episodes: Target number of episodes to complete.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep PPO (GPU)",
            environment_name="GPU Environment",
        )

        # =====================================================================
        # Compile-time constants for buffer sizes
        # =====================================================================
        comptime ACTOR_PARAMS = Self.ACTOR_PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CRITIC_PARAM_SIZE
        comptime ACTOR_STATE = ACTOR_PARAMS * 2  # Adam: 2 states per param
        comptime CRITIC_STATE = CRITIC_PARAMS * 2

        comptime ENV_OBS_SIZE = Self.n_envs * Self.OBS
        # Full environment state size (may be larger than OBS for complex physics)
        comptime ENV_STATE_SIZE = Self.n_envs * EnvType.STATE_SIZE
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime ROLLOUT_OBS_SIZE = ROLLOUT_TOTAL * Self.OBS

        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_OBS_SIZE = MINIBATCH * Self.OBS
        comptime MINIBATCH_LOGITS_SIZE = MINIBATCH * Self.ACTIONS
        comptime MINIBATCH_CACHE_ACTOR = MINIBATCH * Self.ACTOR_CACHE
        comptime MINIBATCH_CACHE_CRITIC = MINIBATCH * Self.CRITIC_CACHE

        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ROLLOUT_BLOCKS = (ROLLOUT_TOTAL + TPB - 1) // TPB

        # Workspace sizes for forward passes (5-layer network = 4*HIDDEN intermediates)
        # Formula: for seq(L,R,L,R,L), workspace = 4 * HIDDEN per sample
        comptime WORKSPACE_PER_SAMPLE = 4 * Self.HIDDEN
        comptime ENV_WORKSPACE_SIZE = Self.n_envs * WORKSPACE_PER_SAMPLE
        comptime MINIBATCH_WORKSPACE_SIZE = MINIBATCH * WORKSPACE_PER_SAMPLE

        # =====================================================================
        # Network parameter buffers
        # =====================================================================
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_grads_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_state_buf = ctx.enqueue_create_buffer[dtype](ACTOR_STATE)

        var critic_params_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_grads_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_state_buf = ctx.enqueue_create_buffer[dtype](CRITIC_STATE)

        # Pre-allocated workspace buffers (prevents GPU memory leak!)
        var actor_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var critic_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var actor_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )
        var critic_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )

        # =====================================================================
        # Environment buffers (n_envs parallel environments)
        # =====================================================================
        # Full state buffer for environment (includes physics state, metadata, etc.)
        var states_buf = ctx.enqueue_create_buffer[dtype](ENV_STATE_SIZE)
        # Observation buffer for neural network input (extracted from states)
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var values_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * Self.ACTIONS
        )

        # Episode tracking buffers
        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs
        )
        var completed_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_mask_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Host buffers for episode tracking
        var completed_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )
        var completed_steps_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )
        var completed_mask_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )

        # =====================================================================
        # Rollout buffers (store transitions for one rollout)
        # =====================================================================
        var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_OBS_SIZE)
        var rollout_actions_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Advantages and returns (computed after rollout)
        var advantages_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var returns_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Host buffers for GAE computation
        var rollout_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var advantages_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var returns_host = ctx.enqueue_create_host_buffer[dtype](ROLLOUT_TOTAL)
        var bootstrap_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )

        # =====================================================================
        # Minibatch buffers (for training)
        # =====================================================================
        var mb_obs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH_OBS_SIZE)
        var mb_actions_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_advantages_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_returns_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_log_probs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_values_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH
        )  # For value clipping
        var mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](MINIBATCH)
        var mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            MINIBATCH
        )

        # Training workspace
        var actor_logits_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_LOGITS_SIZE
        )
        var actor_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_ACTOR
        )
        var actor_grad_output_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_LOGITS_SIZE
        )
        var actor_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        var critic_values_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_CRITIC
        )
        var critic_grad_output_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        # =====================================================================
        # KL divergence and gradient clipping buffers
        # =====================================================================
        var kl_divergences_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )

        # Per-minibatch advantage normalization buffers
        var mb_advantages_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )

        # Gradient norm computation buffers (fused kernel keeps everything on GPU)
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB
        var actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            ACTOR_GRAD_BLOCKS
        )
        var critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            CRITIC_GRAD_BLOCKS
        )
        # Scale buffers for fused gradient clipping (single scalar each)
        var actor_scale_buf = ctx.enqueue_create_buffer[dtype](1)
        var critic_scale_buf = ctx.enqueue_create_buffer[dtype](1)

        # =====================================================================
        # Initialize network parameters on GPU
        # =====================================================================
        self.actor.copy_params_to_device(ctx, actor_params_buf)
        self.actor.copy_state_to_device(ctx, actor_state_buf)
        self.critic.copy_params_to_device(ctx, critic_params_buf)
        self.critic.copy_state_to_device(ctx, critic_state_buf)

        # =====================================================================
        # Create LayoutTensor views
        # =====================================================================
        # Full state tensor for environment (physics, metadata, etc.)
        var states_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, EnvType.STATE_SIZE),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        # Observation tensor for neural network input
        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        var episode_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var episode_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())
        var completed_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_rewards_buf.unsafe_ptr())
        var completed_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_steps_buf.unsafe_ptr())
        var completed_mask_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_mask_buf.unsafe_ptr())

        var mb_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.OBS),
            MutAnyOrigin,
        ](mb_obs_buf.unsafe_ptr())
        var mb_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_actions_buf.unsafe_ptr())
        var mb_advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_advantages_buf.unsafe_ptr())
        var mb_returns_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_returns_buf.unsafe_ptr())
        var mb_old_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_log_probs_buf.unsafe_ptr())
        var mb_old_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_values_buf.unsafe_ptr())
        var rollout_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(ROLLOUT_TOTAL, Self.OBS),
            MutAnyOrigin,
        ](rollout_obs_buf.unsafe_ptr())
        var rollout_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_actions_buf.unsafe_ptr())
        var advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](advantages_buf.unsafe_ptr())
        var returns_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](returns_buf.unsafe_ptr())
        var rollout_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_log_probs_buf.unsafe_ptr())
        var rollout_values_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_values_buf.unsafe_ptr())
        var mb_indices_tensor = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_indices_buf.unsafe_ptr())

        var actor_logits_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTIONS),
            MutAnyOrigin,
        ](actor_logits_buf.unsafe_ptr())
        var actor_grad_output_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTIONS),
            MutAnyOrigin,
        ](actor_grad_output_buf.unsafe_ptr())
        var actor_old_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_log_probs_buf.unsafe_ptr())
        var actor_advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_advantages_buf.unsafe_ptr())
        var actor_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_actions_buf.unsafe_ptr())
        var critic_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_values_buf.unsafe_ptr())
        var critic_grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_grad_output_buf.unsafe_ptr())
        var critic_returns_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_returns_buf.unsafe_ptr())

        var logits_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.ACTIONS),
            MutAnyOrigin,
        ](logits_buf.unsafe_ptr())

        var log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())

        var kl_divergences_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](kl_divergences_buf.unsafe_ptr())

        var actor_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_PARAMS), MutAnyOrigin
        ](actor_grads_buf.unsafe_ptr())
        var critic_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_PARAMS), MutAnyOrigin
        ](critic_grads_buf.unsafe_ptr())
        var actor_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_GRAD_BLOCKS), MutAnyOrigin
        ](actor_grad_partial_sums_buf.unsafe_ptr())
        var critic_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_GRAD_BLOCKS), MutAnyOrigin
        ](critic_grad_partial_sums_buf.unsafe_ptr())
        var actor_scale_tensor = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](actor_scale_buf.unsafe_ptr())
        var critic_scale_tensor = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](critic_scale_buf.unsafe_ptr())

        # Define extract_obs_wrapper early (needed for initial reset)
        comptime extract_obs_wrapper = _extract_obs_from_state_kernel[
            dtype, Self.n_envs, EnvType.STATE_SIZE, Self.OBS
        ]

        # Initialize episode tracking to zero
        ctx.enqueue_memset(episode_rewards_buf, 0)
        ctx.enqueue_memset(episode_steps_buf, 0)

        # =====================================================================
        # Initialize all environments on GPU
        # =====================================================================
        # Reset environments (writes to full state buffer)
        EnvType.reset_kernel_gpu[Self.n_envs, EnvType.STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Extract observations from state buffer for neural network input
        ctx.enqueue_function[extract_obs_wrapper, extract_obs_wrapper](
            obs_tensor,
            states_tensor,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.synchronize()

        # =====================================================================
        # Training state
        # =====================================================================
        var completed_episodes = 0
        var total_steps = 0
        var rollout_count = 0

        # Annealing: compute target total steps
        # If not set, estimate based on num_episodes * average episode length
        # Use ROLLOUT_TOTAL as rough estimate of steps per rollout batch
        var annealing_target_steps = self.target_total_steps
        if annealing_target_steps == 0:
            # Estimate: num_episodes * 200 steps average (conservative for most envs)
            annealing_target_steps = num_episodes * 200

        # Store initial learning rates for annealing
        var initial_actor_lr = self.actor_lr
        var initial_critic_lr = self.critic_lr
        var initial_entropy_coef = self.entropy_coef

        # Kernel wrappers
        comptime accum_rewards_wrapper = accumulate_rewards_kernel[
            dtype, Self.n_envs
        ]
        comptime incr_steps_wrapper = increment_steps_kernel[dtype, Self.n_envs]
        comptime extract_completed_wrapper = extract_completed_episodes_kernel[
            dtype, Self.n_envs
        ]

        comptime reset_tracking_wrapper = selective_reset_tracking_kernel[
            dtype, Self.n_envs
        ]

        # Define wrappers OUTSIDE the loop to avoid recompilation
        comptime store_post_step_wrapper = _store_post_step_kernel[
            dtype, Self.n_envs
        ]

        comptime store_pre_step_wrapper = _store_pre_step_kernel[
            dtype, Self.n_envs, Self.OBS
        ]

        # Phase 3 kernel wrappers - defined ONCE outside the training loop
        comptime gather_wrapper = ppo_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_wrapper = ppo_actor_grad_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime actor_grad_with_kl_wrapper = ppo_actor_grad_with_kl_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_wrapper = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_advantages_wrapper = normalize_advantages_kernel[
            dtype, MINIBATCH
        ]

        # Gradient clipping kernel wrappers (fused version eliminates host copy)
        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        # Fused two-kernel approach: reduce+scale then apply
        comptime actor_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime actor_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime critic_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, CRITIC_PARAMS
        ]

        comptime sample_actions_wrapper = _sample_actions_kernel[
            dtype, Self.n_envs, Self.ACTIONS
        ]

        # =====================================================================
        # Timing accumulators (for final summary)
        # =====================================================================
        var total_phase1_ns: UInt = 0
        var total_phase2_ns: UInt = 0
        var total_phase3_ns: UInt = 0
        var total_shuffle_ns: UInt = 0
        var total_indices_ns: UInt = 0
        var total_gather_ns: UInt = 0
        var total_actor_train_ns: UInt = 0
        var total_critic_train_ns: UInt = 0

        # =====================================================================
        # Main Training Loop
        # =====================================================================

        while completed_episodes < num_episodes:
            # Track episodes at start of rollout for aggregated logging
            var rollout_start_episodes = completed_episodes
            rollout_count += 1
            var rollout_start = perf_counter_ns()

            # =================================================================
            # Phase 1: Collect rollout (rollout_len steps across n_envs envs)
            # =================================================================
            var phase1_start = perf_counter_ns()

            for t in range(Self.rollout_len):
                # Select actions for all environments
                var rng_seed = UInt32(total_steps * 2654435761 + t * 7919)
                # Forward actor to get logits (using pre-allocated workspace)
                self.actor.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    logits_buf,
                    obs_buf,
                    actor_params_buf,
                    actor_env_workspace_buf,
                )

                # Forward critic to get values (using pre-allocated workspace)
                self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    values_buf,
                    obs_buf,
                    critic_params_buf,
                    critic_env_workspace_buf,
                )
                # No sync needed - GPU ops execute in order within stream

                # Sample actions and compute log probs on GPU
                ctx.enqueue_function[
                    sample_actions_wrapper, sample_actions_wrapper
                ](
                    logits_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    Scalar[DType.uint32](rng_seed),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                # No sync needed - GPU ops execute in order within stream

                # Store pre-step observation to rollout buffer using kernel
                var t_offset = t * Self.n_envs

                # Create views at the correct offset for this timestep
                var rollout_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.OBS),
                    MutAnyOrigin,
                ](rollout_obs_buf.unsafe_ptr() + t_offset * Self.OBS)
                var rollout_actions_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_actions_buf.unsafe_ptr() + t_offset)
                var rollout_log_probs_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_log_probs_buf.unsafe_ptr() + t_offset)
                var rollout_values_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_values_buf.unsafe_ptr() + t_offset)

                var values_tensor = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](values_buf.unsafe_ptr())
                var log_probs_tensor = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](log_probs_buf.unsafe_ptr())

                ctx.enqueue_function[
                    store_pre_step_wrapper, store_pre_step_wrapper
                ](
                    rollout_obs_t,
                    rollout_actions_t,
                    rollout_log_probs_t,
                    rollout_values_t,
                    obs_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    values_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Step all environments + extract observations (fused kernel)
                # Use a different multiplier to get independent seed from action sampling
                var env_step_seed = UInt64(total_steps * 1103515245 + t * 12345)
                EnvType.step_kernel_gpu[
                    Self.n_envs, EnvType.STATE_SIZE, Self.OBS
                ](
                    ctx,
                    states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    obs_buf,
                    env_step_seed,
                )
                # No sync needed - obs extraction is fused into step kernel

                # Store rewards and dones
                var rollout_rewards_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_rewards_buf.unsafe_ptr() + t_offset)
                var rollout_dones_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_dones_buf.unsafe_ptr() + t_offset)

                ctx.enqueue_function[
                    store_post_step_wrapper, store_post_step_wrapper
                ](
                    rollout_rewards_t,
                    rollout_dones_t,
                    rewards_tensor,
                    dones_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Accumulate episode rewards and steps
                ctx.enqueue_function[
                    accum_rewards_wrapper, accum_rewards_wrapper
                ](
                    episode_rewards_tensor,
                    rewards_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[incr_steps_wrapper, incr_steps_wrapper](
                    episode_steps_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                total_steps += Self.n_envs

                # Extract completed episodes
                ctx.enqueue_function[
                    extract_completed_wrapper, extract_completed_wrapper
                ](
                    dones_tensor,
                    episode_rewards_tensor,
                    episode_steps_tensor,
                    completed_rewards_tensor,
                    completed_steps_tensor,
                    completed_mask_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Copy to CPU and process
                ctx.enqueue_copy(completed_rewards_host, completed_rewards_buf)
                ctx.enqueue_copy(completed_steps_host, completed_steps_buf)
                ctx.enqueue_copy(completed_mask_host, completed_mask_buf)
                ctx.synchronize()

                # Log completed episodes
                for i in range(Self.n_envs):
                    if Float64(completed_mask_host[i]) > 0.5:
                        var ep_reward = Float64(completed_rewards_host[i])
                        var ep_steps = Int(completed_steps_host[i])
                        metrics.log_episode(
                            completed_episodes, ep_reward, ep_steps, 0.0
                        )
                        completed_episodes += 1

                        # Auto-checkpoint (GPU)
                        if (
                            self.checkpoint_every > 0
                            and self.checkpoint_path != ""
                            and completed_episodes % self.checkpoint_every == 0
                        ):
                            # Copy params and state from GPU to CPU
                            self.actor.copy_params_from_device(
                                ctx, actor_params_buf
                            )
                            self.actor.copy_state_from_device(
                                ctx, actor_state_buf
                            )
                            self.critic.copy_params_from_device(
                                ctx, critic_params_buf
                            )
                            self.critic.copy_state_from_device(
                                ctx, critic_state_buf
                            )
                            ctx.synchronize()

                            # Save checkpoint
                            self.save_checkpoint(self.checkpoint_path)
                            if verbose:
                                print(
                                    "  [Checkpoint saved at episode",
                                    completed_episodes,
                                    "]",
                                )

                # Reset episode tracking for done environments
                ctx.enqueue_function[
                    reset_tracking_wrapper, reset_tracking_wrapper
                ](
                    dones_tensor,
                    episode_rewards_tensor,
                    episode_steps_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Auto-reset done environments
                EnvType.selective_reset_kernel_gpu[
                    Self.n_envs, EnvType.STATE_SIZE
                ](
                    ctx,
                    states_buf,
                    dones_buf,
                    UInt64(total_steps * 1013904223 + t * 2654435761),
                )
                # No sync needed - GPU ops execute in order within stream

                # Extract observations from state buffer after selective reset
                ctx.enqueue_function[extract_obs_wrapper, extract_obs_wrapper](
                    obs_tensor,
                    states_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                # No sync needed - next iteration's GPU ops will wait automatically

            # Early exit if we've reached target episodes
            if completed_episodes >= num_episodes:
                break

            var phase1_end = perf_counter_ns()

            # =================================================================
            # Phase 2: Compute GAE advantages on CPU
            # =================================================================
            var phase2_start = perf_counter_ns()

            # Get bootstrap values from final observations (using pre-allocated workspace)
            self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                ctx,
                values_buf,
                obs_buf,
                critic_params_buf,
                critic_env_workspace_buf,
            )

            ctx.enqueue_copy(bootstrap_values_host, values_buf)

            # Copy rollout data to CPU
            ctx.enqueue_copy(rollout_rewards_host, rollout_rewards_buf)
            ctx.enqueue_copy(rollout_values_host, rollout_values_buf)
            ctx.enqueue_copy(rollout_dones_host, rollout_dones_buf)
            ctx.synchronize()

            # Compute GAE for each environment
            for env_idx in range(Self.n_envs):
                var gae = Scalar[dtype](0.0)
                var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
                var bootstrap_val = Scalar[dtype](
                    bootstrap_values_host[env_idx]
                )

                # Iterate backwards through timesteps for this environment
                for t in range(Self.rollout_len - 1, -1, -1):
                    var idx = t * Self.n_envs + env_idx
                    var reward = rollout_rewards_host[idx]
                    var value = rollout_values_host[idx]
                    var done = rollout_dones_host[idx]

                    # Get next value
                    var next_val: Scalar[dtype]
                    if t == Self.rollout_len - 1:
                        next_val = bootstrap_val
                    else:
                        var next_idx = (t + 1) * Self.n_envs + env_idx
                        next_val = rollout_values_host[next_idx]

                    # Reset GAE at episode boundary
                    if done > Scalar[dtype](0.5):
                        next_val = Scalar[dtype](0.0)
                        gae = Scalar[dtype](0.0)

                    # TD residual
                    var delta = (
                        reward + Scalar[dtype](self.gamma) * next_val - value
                    )

                    # GAE accumulation
                    gae = delta + gae_decay * gae

                    advantages_host[idx] = gae
                    returns_host[idx] = gae + value

            # Normalize advantages
            if self.normalize_advantages:
                var mean = Scalar[dtype](0.0)
                var var_sum = Scalar[dtype](0.0)
                for i in range(ROLLOUT_TOTAL):
                    mean += advantages_host[i]
                mean /= Scalar[dtype](ROLLOUT_TOTAL)
                for i in range(ROLLOUT_TOTAL):
                    var diff = advantages_host[i] - mean
                    var_sum += diff * diff

                var variance = var_sum / Scalar[dtype](ROLLOUT_TOTAL)
                var std = sqrt(variance + Scalar[dtype](1e-8))
                for i in range(ROLLOUT_TOTAL):
                    advantages_host[i] = (advantages_host[i] - mean) / (
                        std + Scalar[dtype](1e-8)
                    )

            # Copy advantages and returns to GPU
            ctx.enqueue_copy(advantages_buf, advantages_host)
            ctx.enqueue_copy(returns_buf, returns_host)
            ctx.synchronize()

            var phase2_end = perf_counter_ns()

            # =================================================================
            # Phase 3: Train actor and critic with minibatches
            # =================================================================
            var phase3_start = perf_counter_ns()

            # Compute annealing progress (0.0 to 1.0)
            var progress = Float64(total_steps) / Float64(
                annealing_target_steps
            )
            if progress > 1.0:
                progress = 1.0

            # Apply learning rate annealing
            var current_actor_lr = initial_actor_lr
            var current_critic_lr = initial_critic_lr
            var current_entropy_coef = initial_entropy_coef
            if self.anneal_lr:
                var lr_multiplier = 1.0 - progress
                current_actor_lr = initial_actor_lr * lr_multiplier
                current_critic_lr = initial_critic_lr * lr_multiplier
                # Update optimizer learning rates
                self.actor.optimizer.lr = current_actor_lr
                self.critic.optimizer.lr = current_critic_lr

            # Apply entropy coefficient annealing
            if self.anneal_entropy:
                current_entropy_coef = initial_entropy_coef * (1.0 - progress)

            # Sub-timers for phase 3
            var shuffle_time_ns: UInt = 0
            var indices_copy_time_ns: UInt = 0
            var gather_time_ns: UInt = 0
            var actor_train_time_ns: UInt = 0
            var critic_train_time_ns: UInt = 0
            var sync_time_ns: UInt = 0

            # KL early stopping flag
            var kl_early_stop = False

            for epoch in range(self.num_epochs):
                # Check if we should early stop due to KL
                if kl_early_stop:
                    break
                # Generate shuffled indices on CPU
                var shuffle_start = perf_counter_ns()
                var indices_list = List[Int]()
                for i in range(ROLLOUT_TOTAL):
                    indices_list.append(i)

                # Fisher-Yates shuffle
                for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                    var j = Int(random_float64() * Float64(i + 1))
                    var temp = indices_list[i]
                    indices_list[i] = indices_list[j]
                    indices_list[j] = temp
                shuffle_time_ns += perf_counter_ns() - shuffle_start

                # Process minibatches
                var num_minibatches = ROLLOUT_TOTAL // MINIBATCH
                for mb_idx in range(num_minibatches):
                    var start_idx = mb_idx * MINIBATCH

                    # Copy minibatch indices to host buffer
                    var indices_copy_start = perf_counter_ns()
                    for i in range(MINIBATCH):
                        mb_indices_host[i] = Int32(indices_list[start_idx + i])

                    # Copy indices to GPU
                    ctx.enqueue_copy(mb_indices_buf, mb_indices_host)
                    indices_copy_time_ns += (
                        perf_counter_ns() - indices_copy_start
                    )

                    # Gather minibatch data (inlined)
                    var gather_start = perf_counter_ns()

                    ctx.enqueue_function[gather_wrapper, gather_wrapper](
                        mb_obs_tensor,
                        mb_actions_tensor,
                        mb_advantages_tensor,
                        mb_returns_tensor,
                        mb_old_log_probs_tensor,
                        mb_old_values_tensor,
                        rollout_obs_tensor,
                        rollout_actions_tensor,
                        advantages_tensor,
                        returns_tensor,
                        rollout_log_probs_tensor,
                        rollout_values_tensor,
                        mb_indices_tensor,
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()
                    gather_time_ns += perf_counter_ns() - gather_start

                    # Per-minibatch advantage normalization
                    if self.norm_adv_per_minibatch:
                        # Copy advantages to CPU, compute mean/std, normalize on GPU
                        ctx.enqueue_copy(mb_advantages_host, mb_advantages_buf)
                        ctx.synchronize()

                        # Compute mean and std on CPU
                        var adv_mean = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            adv_mean += mb_advantages_host[i]
                        adv_mean /= Scalar[dtype](MINIBATCH)

                        var adv_var_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            var diff = mb_advantages_host[i] - adv_mean
                            adv_var_sum += diff * diff

                        var adv_std = sqrt(
                            adv_var_sum / Scalar[dtype](MINIBATCH)
                            + Scalar[dtype](1e-8)
                        )

                        # Normalize advantages on GPU
                        ctx.enqueue_function[
                            normalize_advantages_wrapper,
                            normalize_advantages_wrapper,
                        ](
                            mb_advantages_tensor,
                            adv_mean,
                            adv_std,
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                    # Train actor (inlined)
                    var actor_start = perf_counter_ns()

                    # Zero actor gradients
                    ctx.enqueue_memset(actor_grads_buf, 0)

                    # Forward pass with cache (using pre-allocated workspace)
                    self.actor.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_logits_buf,
                        mb_obs_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_minibatch_workspace_buf,
                    )
                    ctx.synchronize()

                    # Compute PPO gradient with KL divergence tracking
                    ctx.enqueue_function[
                        actor_grad_with_kl_wrapper, actor_grad_with_kl_wrapper
                    ](
                        actor_grad_output_tensor,
                        kl_divergences_tensor,
                        actor_logits_tensor,
                        actor_old_log_probs_tensor,
                        actor_advantages_tensor,
                        actor_actions_tensor,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](current_entropy_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                    # Check KL divergence for early stopping
                    if self.target_kl > 0.0:
                        ctx.enqueue_copy(
                            kl_divergences_host, kl_divergences_buf
                        )
                        ctx.synchronize()

                        # Compute mean KL
                        var kl_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            kl_sum += kl_divergences_host[i]
                        var mean_kl = Float64(kl_sum) / Float64(MINIBATCH)

                        if mean_kl > self.target_kl:
                            kl_early_stop = True
                            if verbose:
                                print(
                                    "    KL early stop at epoch",
                                    epoch,
                                    "minibatch",
                                    mb_idx,
                                    "| KL:",
                                    String(mean_kl)[:7],
                                )
                            break  # Break from minibatch loop

                    # Backward pass (using pre-allocated workspace)
                    self.actor.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_grad_input_buf,
                        actor_grad_output_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_grads_buf,
                        actor_minibatch_workspace_buf,
                    )

                    # Gradient clipping for actor (fused: no host copy)
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            actor_grad_norm_wrapper, actor_grad_norm_wrapper
                        ](
                            actor_grad_partial_sums_tensor,
                            actor_grads_tensor,
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums and compute scale (single block)
                        ctx.enqueue_function[
                            actor_reduce_scale_wrapper,
                            actor_reduce_scale_wrapper,
                        ](
                            actor_scale_tensor,
                            actor_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(1,),
                            block_dim=(TPB,),
                        )
                        # Step 3: Apply scale to all gradients
                        ctx.enqueue_function[
                            actor_apply_scale_wrapper,
                            actor_apply_scale_wrapper,
                        ](
                            actor_grads_tensor,
                            actor_scale_tensor,
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                    # Update actor parameters
                    self.actor.optimizer.step_gpu[Self.ACTOR_PARAM_SIZE](
                        ctx, actor_params_buf, actor_grads_buf, actor_state_buf
                    )
                    ctx.synchronize()
                    actor_train_time_ns += perf_counter_ns() - actor_start

                    # Train critic (inlined)
                    var critic_start = perf_counter_ns()

                    # Zero critic gradients
                    ctx.enqueue_memset(critic_grads_buf, 0)

                    # Forward pass with cache (using pre-allocated workspace)
                    self.critic.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_values_buf,
                        mb_obs_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_minibatch_workspace_buf,
                    )
                    ctx.synchronize()

                    # Compute value loss gradient
                    if self.clip_value:
                        # Use clipped critic gradient
                        ctx.enqueue_function[
                            critic_grad_clipped_wrapper,
                            critic_grad_clipped_wrapper,
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            mb_old_values_tensor,
                            critic_returns_tensor,
                            Scalar[dtype](self.value_loss_coef),
                            Scalar[dtype](self.clip_epsilon),
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    else:
                        # Use regular critic gradient
                        ctx.enqueue_function[
                            critic_grad_wrapper, critic_grad_wrapper
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            critic_returns_tensor,
                            Scalar[dtype](self.value_loss_coef),
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    ctx.synchronize()

                    # Backward pass (using pre-allocated workspace)
                    self.critic.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_grad_input_buf,
                        critic_grad_output_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_grads_buf,
                        critic_minibatch_workspace_buf,
                    )

                    # Gradient clipping for critic (fused: no host copy)
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            critic_grad_norm_wrapper, critic_grad_norm_wrapper
                        ](
                            critic_grad_partial_sums_tensor,
                            critic_grads_tensor,
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums and compute scale (single block)
                        ctx.enqueue_function[
                            critic_reduce_scale_wrapper,
                            critic_reduce_scale_wrapper,
                        ](
                            critic_scale_tensor,
                            critic_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(1,),
                            block_dim=(TPB,),
                        )
                        # Step 3: Apply scale to all gradients
                        ctx.enqueue_function[
                            critic_apply_scale_wrapper,
                            critic_apply_scale_wrapper,
                        ](
                            critic_grads_tensor,
                            critic_scale_tensor,
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                    # Update critic parameters
                    self.critic.optimizer.step_gpu[Self.CRITIC_PARAM_SIZE](
                        ctx,
                        critic_params_buf,
                        critic_grads_buf,
                        critic_state_buf,
                    )
                    ctx.synchronize()
                    critic_train_time_ns += perf_counter_ns() - critic_start

            var phase3_end = perf_counter_ns()

            # Accumulate timing for final summary
            total_phase1_ns += phase1_end - phase1_start
            total_phase2_ns += phase2_end - phase2_start
            total_phase3_ns += phase3_end - phase3_start
            total_shuffle_ns += shuffle_time_ns
            total_indices_ns += indices_copy_time_ns
            total_gather_ns += gather_time_ns
            total_actor_train_ns += actor_train_time_ns
            total_critic_train_ns += critic_train_time_ns

            # Print rollout summary with episode range (aggregated)
            if verbose and rollout_count % print_every == 0:
                var rollout_end_episodes = completed_episodes
                var episodes_this_rollout = (
                    rollout_end_episodes - rollout_start_episodes
                )
                var avg_reward = metrics.mean_reward_last_n(
                    min(100, completed_episodes)
                )
                print(
                    "Rollout",
                    rollout_count,
                    "| Episodes",
                    rollout_start_episodes + 1,
                    "-",
                    rollout_end_episodes,
                    "(+" + String(episodes_this_rollout) + ")",
                    "| Avg(100):",
                    String(avg_reward)[:7],
                    "| Steps:",
                    total_steps,
                )

        # =====================================================================
        # Copy final parameters back to CPU
        # =====================================================================
        self.actor.copy_params_from_device(ctx, actor_params_buf)
        self.actor.copy_state_from_device(ctx, actor_state_buf)
        self.critic.copy_params_from_device(ctx, critic_params_buf)
        self.critic.copy_state_from_device(ctx, critic_state_buf)
        ctx.synchronize()

        # =====================================================================
        # Print final timing summary
        # =====================================================================
        if verbose:
            var total_time_ns = (
                total_phase1_ns + total_phase2_ns + total_phase3_ns
            )
            var p1_pct = Float64(total_phase1_ns) / Float64(total_time_ns) * 100
            var p2_pct = Float64(total_phase2_ns) / Float64(total_time_ns) * 100
            var p3_pct = Float64(total_phase3_ns) / Float64(total_time_ns) * 100
            print()
            print("-" * 60)
            print(
                "Performance Summary (" + String(rollout_count) + " rollouts)"
            )
            print("-" * 60)
            print(
                "  Phase 1 (collect):  ",
                String(Float64(total_phase1_ns) / 1e9)[:6],
                "s (",
                String(p1_pct)[:4],
                "%)",
            )
            print(
                "  Phase 2 (GAE):      ",
                String(Float64(total_phase2_ns) / 1e9)[:6],
                "s (",
                String(p2_pct)[:4],
                "%)",
            )
            print(
                "  Phase 3 (train):    ",
                String(Float64(total_phase3_ns) / 1e9)[:6],
                "s (",
                String(p3_pct)[:4],
                "%)",
            )
            print()
            print("  Phase 3 breakdown:")
            print(
                "    Shuffle:      ",
                String(Float64(total_shuffle_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Indices copy: ",
                String(Float64(total_indices_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Gather:       ",
                String(Float64(total_gather_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Actor train:  ",
                String(Float64(total_actor_train_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Critic train: ",
                String(Float64(total_critic_train_ns) / 1e6)[:8],
                "ms",
            )
            print("-" * 60)

        return metrics^

    # =========================================================================
    # GPU Training with CPU Environments (Hybrid)
    # =========================================================================

    fn train_gpu_cpu_env[
        EnvType: BoxDiscreteActionEnv & Copyable & Movable,
    ](
        mut self,
        ctx: DeviceContext,
        mut envs: List[EnvType],
        num_episodes: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train PPO on GPU with CPU environments.

        This hybrid approach uses CPU environments for accurate physics simulation
        (e.g., Box2D for LunarLander) while leveraging GPU for neural network
        computations (forward/backward passes, parameter updates).

        The training loop:
        1. Forward pass on GPU to get actions
        2. Step environments on CPU (accurate physics)
        3. Store transitions and compute GAE on CPU
        4. Train actor/critic on GPU

        Args:
            ctx: GPU device context.
            envs: List of n_envs CPU environments (must match Self.n_envs).
            num_episodes: Target number of episodes to complete.
            verbose: Whether to print progress.
            print_every: Print progress every N rollouts.

        Returns:
            TrainingMetrics with episode rewards and statistics.

        Note:
            The envs list must have exactly n_envs environments.
            Each environment should be independently initialized.
        """
        # Validate environment count
        if len(envs) != Self.n_envs:
            print(
                "Error: Expected",
                Self.n_envs,
                "environments, got",
                len(envs),
            )
            return TrainingMetrics(
                algorithm_name="Deep PPO (GPU+CPU)",
                environment_name="Error",
            )

        var metrics = TrainingMetrics(
            algorithm_name="Deep PPO (GPU+CPU)",
            environment_name="CPU Environment",
        )

        # =====================================================================
        # Compile-time constants for buffer sizes
        # =====================================================================
        comptime ACTOR_PARAMS = Self.ACTOR_PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CRITIC_PARAM_SIZE
        comptime ACTOR_STATE = ACTOR_PARAMS * 2  # Adam: 2 states per param
        comptime CRITIC_STATE = CRITIC_PARAMS * 2

        comptime ENV_OBS_SIZE = Self.n_envs * Self.OBS
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime ROLLOUT_OBS_SIZE = ROLLOUT_TOTAL * Self.OBS

        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_OBS_SIZE = MINIBATCH * Self.OBS
        comptime MINIBATCH_LOGITS_SIZE = MINIBATCH * Self.ACTIONS
        comptime MINIBATCH_CACHE_ACTOR = MINIBATCH * Self.ACTOR_CACHE
        comptime MINIBATCH_CACHE_CRITIC = MINIBATCH * Self.CRITIC_CACHE

        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ROLLOUT_BLOCKS = (ROLLOUT_TOTAL + TPB - 1) // TPB

        # Workspace sizes for forward passes
        comptime WORKSPACE_PER_SAMPLE = 4 * Self.HIDDEN
        comptime ENV_WORKSPACE_SIZE = Self.n_envs * WORKSPACE_PER_SAMPLE
        comptime MINIBATCH_WORKSPACE_SIZE = MINIBATCH * WORKSPACE_PER_SAMPLE

        # =====================================================================
        # Network parameter buffers (GPU)
        # =====================================================================
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_grads_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_state_buf = ctx.enqueue_create_buffer[dtype](ACTOR_STATE)

        var critic_params_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_grads_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_state_buf = ctx.enqueue_create_buffer[dtype](CRITIC_STATE)

        # Pre-allocated workspace buffers
        var actor_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var critic_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var actor_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )
        var critic_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )

        # =====================================================================
        # Environment buffers (GPU device + CPU host for transfers)
        # =====================================================================
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var obs_host = ctx.enqueue_create_host_buffer[dtype](ENV_OBS_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var values_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * Self.ACTIONS
        )

        # =====================================================================
        # Rollout buffers
        # =====================================================================
        var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_OBS_SIZE)
        var rollout_actions_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Advantages and returns
        var advantages_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var returns_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Host buffers for GAE computation
        var rollout_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var advantages_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var returns_host = ctx.enqueue_create_host_buffer[dtype](ROLLOUT_TOTAL)
        var bootstrap_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )

        # =====================================================================
        # Minibatch buffers (for training)
        # =====================================================================
        var mb_obs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH_OBS_SIZE)
        var mb_actions_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_advantages_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_returns_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_log_probs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_values_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](MINIBATCH)
        var mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            MINIBATCH
        )

        # Training workspace
        var actor_logits_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_LOGITS_SIZE
        )
        var actor_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_ACTOR
        )
        var actor_grad_output_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_LOGITS_SIZE
        )
        var actor_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        var critic_values_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_CRITIC
        )
        var critic_grad_output_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        # =====================================================================
        # KL divergence and gradient clipping buffers
        # =====================================================================
        var kl_divergences_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )

        var mb_advantages_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )

        # Gradient norm computation buffers (fused kernel keeps everything on GPU)
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB
        var actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            ACTOR_GRAD_BLOCKS
        )
        var critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            CRITIC_GRAD_BLOCKS
        )
        # Scale buffers for fused gradient clipping (single scalar each)
        var actor_scale_buf = ctx.enqueue_create_buffer[dtype](1)
        var critic_scale_buf = ctx.enqueue_create_buffer[dtype](1)

        # =====================================================================
        # Initialize network parameters on GPU
        # =====================================================================
        self.actor.copy_params_to_device(ctx, actor_params_buf)
        self.actor.copy_state_to_device(ctx, actor_state_buf)
        self.critic.copy_params_to_device(ctx, critic_params_buf)
        self.critic.copy_state_to_device(ctx, critic_state_buf)

        # =====================================================================
        # Create LayoutTensor views
        # =====================================================================
        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        var mb_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.OBS),
            MutAnyOrigin,
        ](mb_obs_buf.unsafe_ptr())
        var mb_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_actions_buf.unsafe_ptr())
        var mb_advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_advantages_buf.unsafe_ptr())
        var mb_returns_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_returns_buf.unsafe_ptr())
        var mb_old_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_log_probs_buf.unsafe_ptr())
        var mb_old_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_values_buf.unsafe_ptr())
        var rollout_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(ROLLOUT_TOTAL, Self.OBS),
            MutAnyOrigin,
        ](rollout_obs_buf.unsafe_ptr())
        var rollout_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_actions_buf.unsafe_ptr())
        var advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](advantages_buf.unsafe_ptr())
        var returns_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](returns_buf.unsafe_ptr())
        var rollout_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_log_probs_buf.unsafe_ptr())
        var rollout_values_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_values_buf.unsafe_ptr())
        var mb_indices_tensor = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_indices_buf.unsafe_ptr())

        var actor_logits_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTIONS),
            MutAnyOrigin,
        ](actor_logits_buf.unsafe_ptr())
        var actor_grad_output_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTIONS),
            MutAnyOrigin,
        ](actor_grad_output_buf.unsafe_ptr())
        var actor_old_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_log_probs_buf.unsafe_ptr())
        var actor_advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_advantages_buf.unsafe_ptr())
        var actor_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_actions_buf.unsafe_ptr())
        var critic_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_values_buf.unsafe_ptr())
        var critic_grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_grad_output_buf.unsafe_ptr())
        var critic_returns_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_returns_buf.unsafe_ptr())

        var logits_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.ACTIONS),
            MutAnyOrigin,
        ](logits_buf.unsafe_ptr())

        var log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())

        var kl_divergences_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](kl_divergences_buf.unsafe_ptr())

        var actor_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_PARAMS), MutAnyOrigin
        ](actor_grads_buf.unsafe_ptr())
        var critic_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_PARAMS), MutAnyOrigin
        ](critic_grads_buf.unsafe_ptr())
        var actor_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_GRAD_BLOCKS), MutAnyOrigin
        ](actor_grad_partial_sums_buf.unsafe_ptr())
        var critic_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_GRAD_BLOCKS), MutAnyOrigin
        ](critic_grad_partial_sums_buf.unsafe_ptr())
        var actor_scale_tensor = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](actor_scale_buf.unsafe_ptr())
        var critic_scale_tensor = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](critic_scale_buf.unsafe_ptr())

        # =====================================================================
        # Initialize all CPU environments and copy observations to GPU
        # =====================================================================
        for i in range(Self.n_envs):
            var obs_list = envs[i].reset_obs_list()
            for d in range(Self.OBS):
                obs_host[i * Self.OBS + d] = Scalar[dtype](obs_list[d])

        ctx.enqueue_copy(obs_buf, obs_host)
        ctx.synchronize()

        # =====================================================================
        # Episode tracking (CPU-side)
        # =====================================================================
        var episode_rewards = List[Scalar[dtype]]()
        var episode_steps = List[Int]()
        for _ in range(Self.n_envs):
            episode_rewards.append(0.0)
            episode_steps.append(0)

        # =====================================================================
        # Training state
        # =====================================================================
        var completed_episodes = 0
        var total_steps = 0
        var rollout_count = 0

        # Annealing target
        var annealing_target_steps = self.target_total_steps
        if annealing_target_steps == 0:
            annealing_target_steps = num_episodes * 200

        var initial_actor_lr = self.actor_lr
        var initial_critic_lr = self.critic_lr
        var initial_entropy_coef = self.entropy_coef

        # Phase 3 kernel wrappers
        comptime gather_wrapper = ppo_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_with_kl_wrapper = ppo_actor_grad_with_kl_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_wrapper = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_advantages_wrapper = normalize_advantages_kernel[
            dtype, MINIBATCH
        ]

        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        # Fused two-kernel approach: reduce+scale then apply
        comptime actor_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime actor_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime critic_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, CRITIC_PARAMS
        ]

        comptime sample_actions_wrapper = _sample_actions_kernel[
            dtype, Self.n_envs, Self.ACTIONS
        ]

        # Timing accumulators
        var total_phase1_ns: UInt = 0
        var total_phase2_ns: UInt = 0
        var total_phase3_ns: UInt = 0

        # =====================================================================
        # Main Training Loop
        # =====================================================================

        while completed_episodes < num_episodes:
            var rollout_start_episodes = completed_episodes
            rollout_count += 1

            # =================================================================
            # Phase 1: Collect rollout (CPU environments + GPU forward passes)
            # =================================================================
            var phase1_start = perf_counter_ns()

            for t in range(Self.rollout_len):
                var rng_seed = UInt32(total_steps * 2654435761 + t * 7919)

                # Forward actor on GPU to get logits
                self.actor.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    logits_buf,
                    obs_buf,
                    actor_params_buf,
                    actor_env_workspace_buf,
                )

                # Forward critic on GPU to get values
                self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    values_buf,
                    obs_buf,
                    critic_params_buf,
                    critic_env_workspace_buf,
                )
                ctx.synchronize()

                # Sample actions on GPU
                ctx.enqueue_function[
                    sample_actions_wrapper, sample_actions_wrapper
                ](
                    logits_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    Scalar[DType.uint32](rng_seed),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # Copy actions to CPU for environment stepping
                ctx.enqueue_copy(actions_host, actions_buf)
                ctx.synchronize()

                # Store pre-step data to rollout buffer (observations on GPU)
                var t_offset = t * Self.n_envs

                var rollout_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.OBS),
                    MutAnyOrigin,
                ](rollout_obs_buf.unsafe_ptr() + t_offset * Self.OBS)
                var rollout_actions_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_actions_buf.unsafe_ptr() + t_offset)
                var rollout_log_probs_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_log_probs_buf.unsafe_ptr() + t_offset)
                var rollout_values_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_values_buf.unsafe_ptr() + t_offset)

                var values_tensor_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](values_buf.unsafe_ptr())
                var log_probs_tensor_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](log_probs_buf.unsafe_ptr())

                comptime store_pre_step_wrapper = _store_pre_step_kernel[
                    dtype, Self.n_envs, Self.OBS
                ]

                ctx.enqueue_function[
                    store_pre_step_wrapper, store_pre_step_wrapper
                ](
                    rollout_obs_t,
                    rollout_actions_t,
                    rollout_log_probs_t,
                    rollout_values_t,
                    obs_tensor,
                    actions_tensor,
                    log_probs_tensor_t,
                    values_tensor_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # =============================================================
                # Step CPU environments (the key difference from train_gpu!)
                # =============================================================
                for i in range(Self.n_envs):
                    var action = Int(actions_host[i])
                    var result = envs[i].step_obs(action)
                    var next_obs = result[0].copy()
                    var reward = result[1]
                    var done = result[2]

                    # Store reward and done in host buffer for later copy
                    rollout_rewards_host[t_offset + i] = Scalar[dtype](reward)
                    rollout_dones_host[t_offset + i] = Scalar[dtype](
                        1.0 if done else 0.0
                    )

                    # Update episode tracking
                    episode_rewards[i] += Scalar[dtype](reward)
                    episode_steps[i] += 1

                    # Handle episode completion
                    if done:
                        metrics.log_episode(
                            completed_episodes,
                            episode_rewards[i],
                            episode_steps[i],
                            0.0,
                        )
                        completed_episodes += 1

                        # Auto-checkpoint
                        if (
                            self.checkpoint_every > 0
                            and self.checkpoint_path != ""
                            and completed_episodes % self.checkpoint_every == 0
                        ):
                            self.actor.copy_params_from_device(
                                ctx, actor_params_buf
                            )
                            self.actor.copy_state_from_device(
                                ctx, actor_state_buf
                            )
                            self.critic.copy_params_from_device(
                                ctx, critic_params_buf
                            )
                            self.critic.copy_state_from_device(
                                ctx, critic_state_buf
                            )
                            ctx.synchronize()
                            self.save_checkpoint(self.checkpoint_path)
                            if verbose:
                                print(
                                    "  [Checkpoint saved at episode",
                                    completed_episodes,
                                    "]",
                                )

                        # Reset episode tracking
                        episode_rewards[i] = 0.0
                        episode_steps[i] = 0

                        # Reset environment
                        var reset_obs = envs[i].reset_obs_list()
                        for d in range(Self.OBS):
                            obs_host[i * Self.OBS + d] = Scalar[dtype](
                                reset_obs[d]
                            )
                    else:
                        # Copy next observation
                        for d in range(Self.OBS):
                            obs_host[i * Self.OBS + d] = Scalar[dtype](
                                next_obs[d]
                            )

                total_steps += Self.n_envs

                # Copy updated observations to GPU for next iteration
                ctx.enqueue_copy(obs_buf, obs_host)
                ctx.synchronize()

            # Early exit if we've reached target episodes
            if completed_episodes >= num_episodes:
                break

            var phase1_end = perf_counter_ns()

            # =================================================================
            # Phase 2: Compute GAE advantages on CPU
            # =================================================================
            var phase2_start = perf_counter_ns()

            # Get bootstrap values from final observations
            self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                ctx,
                values_buf,
                obs_buf,
                critic_params_buf,
                critic_env_workspace_buf,
            )

            ctx.enqueue_copy(bootstrap_values_host, values_buf)

            # Copy rollout values to CPU
            ctx.enqueue_copy(rollout_values_host, rollout_values_buf)
            ctx.synchronize()

            # Compute GAE for each environment
            for env_idx in range(Self.n_envs):
                var gae = Scalar[dtype](0.0)
                var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
                var bootstrap_val = Scalar[dtype](
                    bootstrap_values_host[env_idx]
                )

                for t in range(Self.rollout_len - 1, -1, -1):
                    var idx = t * Self.n_envs + env_idx
                    var reward = rollout_rewards_host[idx]
                    var value = rollout_values_host[idx]
                    var done = rollout_dones_host[idx]

                    var next_val: Scalar[dtype]
                    if t == Self.rollout_len - 1:
                        next_val = bootstrap_val
                    else:
                        var next_idx = (t + 1) * Self.n_envs + env_idx
                        next_val = rollout_values_host[next_idx]

                    if done > Scalar[dtype](0.5):
                        next_val = Scalar[dtype](0.0)
                        gae = Scalar[dtype](0.0)

                    var delta = (
                        reward + Scalar[dtype](self.gamma) * next_val - value
                    )
                    gae = delta + gae_decay * gae

                    advantages_host[idx] = gae
                    returns_host[idx] = gae + value

            # Normalize advantages
            if self.normalize_advantages:
                var mean = Scalar[dtype](0.0)
                var var_sum = Scalar[dtype](0.0)
                for i in range(ROLLOUT_TOTAL):
                    mean += advantages_host[i]
                mean /= Scalar[dtype](ROLLOUT_TOTAL)
                for i in range(ROLLOUT_TOTAL):
                    var diff = advantages_host[i] - mean
                    var_sum += diff * diff

                var variance = var_sum / Scalar[dtype](ROLLOUT_TOTAL)
                var std = sqrt(variance + Scalar[dtype](1e-8))
                for i in range(ROLLOUT_TOTAL):
                    advantages_host[i] = (advantages_host[i] - mean) / (
                        std + Scalar[dtype](1e-8)
                    )

            # Copy advantages, returns, rewards, dones to GPU
            ctx.enqueue_copy(advantages_buf, advantages_host)
            ctx.enqueue_copy(returns_buf, returns_host)
            ctx.enqueue_copy(rollout_rewards_buf, rollout_rewards_host)
            ctx.enqueue_copy(rollout_dones_buf, rollout_dones_host)
            ctx.synchronize()

            var phase2_end = perf_counter_ns()

            # =================================================================
            # Phase 3: Train actor and critic with minibatches (GPU)
            # =================================================================
            var phase3_start = perf_counter_ns()

            var progress = Float64(total_steps) / Float64(
                annealing_target_steps
            )
            if progress > 1.0:
                progress = 1.0

            var current_actor_lr = initial_actor_lr
            var current_critic_lr = initial_critic_lr
            var current_entropy_coef = initial_entropy_coef
            if self.anneal_lr:
                var lr_multiplier = 1.0 - progress
                current_actor_lr = initial_actor_lr * lr_multiplier
                current_critic_lr = initial_critic_lr * lr_multiplier
                self.actor.optimizer.lr = current_actor_lr
                self.critic.optimizer.lr = current_critic_lr

            if self.anneal_entropy:
                current_entropy_coef = initial_entropy_coef * (1.0 - progress)

            var kl_early_stop = False

            for epoch in range(self.num_epochs):
                if kl_early_stop:
                    break

                # Generate shuffled indices
                var indices_list = List[Int]()
                for i in range(ROLLOUT_TOTAL):
                    indices_list.append(i)

                for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                    var j = Int(random_float64() * Float64(i + 1))
                    var temp = indices_list[i]
                    indices_list[i] = indices_list[j]
                    indices_list[j] = temp

                var num_minibatches = ROLLOUT_TOTAL // MINIBATCH
                for mb_idx in range(num_minibatches):
                    var start_idx = mb_idx * MINIBATCH

                    for i in range(MINIBATCH):
                        mb_indices_host[i] = Int32(indices_list[start_idx + i])

                    ctx.enqueue_copy(mb_indices_buf, mb_indices_host)

                    ctx.enqueue_function[gather_wrapper, gather_wrapper](
                        mb_obs_tensor,
                        mb_actions_tensor,
                        mb_advantages_tensor,
                        mb_returns_tensor,
                        mb_old_log_probs_tensor,
                        mb_old_values_tensor,
                        rollout_obs_tensor,
                        rollout_actions_tensor,
                        advantages_tensor,
                        returns_tensor,
                        rollout_log_probs_tensor,
                        rollout_values_tensor,
                        mb_indices_tensor,
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                    # Per-minibatch advantage normalization
                    if self.norm_adv_per_minibatch:
                        ctx.enqueue_copy(mb_advantages_host, mb_advantages_buf)
                        ctx.synchronize()

                        var adv_mean = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            adv_mean += mb_advantages_host[i]
                        adv_mean /= Scalar[dtype](MINIBATCH)

                        var adv_var_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            var diff = mb_advantages_host[i] - adv_mean
                            adv_var_sum += diff * diff

                        var adv_std = sqrt(
                            adv_var_sum / Scalar[dtype](MINIBATCH)
                            + Scalar[dtype](1e-8)
                        )

                        ctx.enqueue_function[
                            normalize_advantages_wrapper,
                            normalize_advantages_wrapper,
                        ](
                            mb_advantages_tensor,
                            adv_mean,
                            adv_std,
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                    # Train actor
                    ctx.enqueue_memset(actor_grads_buf, 0)

                    self.actor.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_logits_buf,
                        mb_obs_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_minibatch_workspace_buf,
                    )
                    ctx.synchronize()

                    ctx.enqueue_function[
                        actor_grad_with_kl_wrapper, actor_grad_with_kl_wrapper
                    ](
                        actor_grad_output_tensor,
                        kl_divergences_tensor,
                        actor_logits_tensor,
                        actor_old_log_probs_tensor,
                        actor_advantages_tensor,
                        actor_actions_tensor,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](current_entropy_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                    # KL divergence early stopping
                    if self.target_kl > 0.0:
                        ctx.enqueue_copy(
                            kl_divergences_host, kl_divergences_buf
                        )
                        ctx.synchronize()

                        var kl_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            kl_sum += kl_divergences_host[i]
                        var mean_kl = Float64(kl_sum) / Float64(MINIBATCH)

                        if mean_kl > self.target_kl:
                            kl_early_stop = True
                            if verbose:
                                print(
                                    "    KL early stop at epoch",
                                    epoch,
                                    "minibatch",
                                    mb_idx,
                                    "| KL:",
                                    String(mean_kl)[:7],
                                )
                            break

                    self.actor.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_grad_input_buf,
                        actor_grad_output_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_grads_buf,
                        actor_minibatch_workspace_buf,
                    )

                    # Gradient clipping for actor (fused: no host copy)
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            actor_grad_norm_wrapper, actor_grad_norm_wrapper
                        ](
                            actor_grad_partial_sums_tensor,
                            actor_grads_tensor,
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums and compute scale (single block)
                        ctx.enqueue_function[
                            actor_reduce_scale_wrapper,
                            actor_reduce_scale_wrapper,
                        ](
                            actor_scale_tensor,
                            actor_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(1,),
                            block_dim=(TPB,),
                        )
                        # Step 3: Apply scale to all gradients
                        ctx.enqueue_function[
                            actor_apply_scale_wrapper,
                            actor_apply_scale_wrapper,
                        ](
                            actor_grads_tensor,
                            actor_scale_tensor,
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                    self.actor.optimizer.step_gpu[Self.ACTOR_PARAM_SIZE](
                        ctx, actor_params_buf, actor_grads_buf, actor_state_buf
                    )
                    ctx.synchronize()

                    # Train critic
                    ctx.enqueue_memset(critic_grads_buf, 0)

                    self.critic.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_values_buf,
                        mb_obs_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_minibatch_workspace_buf,
                    )
                    ctx.synchronize()

                    if self.clip_value:
                        ctx.enqueue_function[
                            critic_grad_clipped_wrapper,
                            critic_grad_clipped_wrapper,
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            mb_old_values_tensor,
                            critic_returns_tensor,
                            Scalar[dtype](self.value_loss_coef),
                            Scalar[dtype](self.clip_epsilon),
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    else:
                        ctx.enqueue_function[
                            critic_grad_wrapper, critic_grad_wrapper
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            critic_returns_tensor,
                            Scalar[dtype](self.value_loss_coef),
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    ctx.synchronize()

                    self.critic.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_grad_input_buf,
                        critic_grad_output_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_grads_buf,
                        critic_minibatch_workspace_buf,
                    )

                    # Gradient clipping for critic (fused: no host copy)
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            critic_grad_norm_wrapper, critic_grad_norm_wrapper
                        ](
                            critic_grad_partial_sums_tensor,
                            critic_grads_tensor,
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums and compute scale (single block)
                        ctx.enqueue_function[
                            critic_reduce_scale_wrapper,
                            critic_reduce_scale_wrapper,
                        ](
                            critic_scale_tensor,
                            critic_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(1,),
                            block_dim=(TPB,),
                        )
                        # Step 3: Apply scale to all gradients
                        ctx.enqueue_function[
                            critic_apply_scale_wrapper,
                            critic_apply_scale_wrapper,
                        ](
                            critic_grads_tensor,
                            critic_scale_tensor,
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                    self.critic.optimizer.step_gpu[Self.CRITIC_PARAM_SIZE](
                        ctx,
                        critic_params_buf,
                        critic_grads_buf,
                        critic_state_buf,
                    )
                    ctx.synchronize()

            var phase3_end = perf_counter_ns()

            total_phase1_ns += phase1_end - phase1_start
            total_phase2_ns += phase2_end - phase2_start
            total_phase3_ns += phase3_end - phase3_start

            # Print rollout summary
            if verbose and rollout_count % print_every == 0:
                var rollout_end_episodes = completed_episodes
                var episodes_this_rollout = (
                    rollout_end_episodes - rollout_start_episodes
                )
                var avg_reward = metrics.mean_reward_last_n(
                    min(100, completed_episodes)
                )
                print(
                    "Rollout",
                    rollout_count,
                    "| Episodes",
                    rollout_start_episodes + 1,
                    "-",
                    rollout_end_episodes,
                    "(+" + String(episodes_this_rollout) + ")",
                    "| Avg(100):",
                    String(avg_reward)[:7],
                    "| Steps:",
                    total_steps,
                )

        # =====================================================================
        # Copy final parameters back to CPU
        # =====================================================================
        self.actor.copy_params_from_device(ctx, actor_params_buf)
        self.actor.copy_state_from_device(ctx, actor_state_buf)
        self.critic.copy_params_from_device(ctx, critic_params_buf)
        self.critic.copy_state_from_device(ctx, critic_state_buf)
        ctx.synchronize()

        # =====================================================================
        # Print final timing summary
        # =====================================================================
        if verbose:
            var total_time_ns = (
                total_phase1_ns + total_phase2_ns + total_phase3_ns
            )
            var p1_pct = Float64(total_phase1_ns) / Float64(total_time_ns) * 100
            var p2_pct = Float64(total_phase2_ns) / Float64(total_time_ns) * 100
            var p3_pct = Float64(total_phase3_ns) / Float64(total_time_ns) * 100
            print()
            print("-" * 60)
            print(
                "Performance Summary (GPU+CPU) ("
                + String(rollout_count)
                + " rollouts)"
            )
            print("-" * 60)
            print(
                "  Phase 1 (collect+CPU step):  ",
                String(Float64(total_phase1_ns) / 1e9)[:6],
                "s (",
                String(p1_pct)[:4],
                "%)",
            )
            print(
                "  Phase 2 (GAE):               ",
                String(Float64(total_phase2_ns) / 1e9)[:6],
                "s (",
                String(p2_pct)[:4],
                "%)",
            )
            print(
                "  Phase 3 (train):             ",
                String(Float64(total_phase3_ns) / 1e9)[:6],
                "s (",
                String(p3_pct)[:4],
                "%)",
            )
            print("-" * 60)

        return metrics^

    # =========================================================================
    # Checkpoint Save/Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save agent state to a checkpoint file.

        Saves actor and critic networks and hyperparameters.

        Args:
            filepath: Path to save the checkpoint file.
        """
        var actor_param_size = self.actor.PARAM_SIZE
        var critic_param_size = self.critic.PARAM_SIZE
        var actor_state_size = actor_param_size * Adam.STATE_PER_PARAM
        var critic_state_size = critic_param_size * Adam.STATE_PER_PARAM

        var content = String("# mojo-rl checkpoint v1\n")
        content += "# type: ppo_agent\n"
        content += "# actor_param_size: " + String(actor_param_size) + "\n"
        content += "# critic_param_size: " + String(critic_param_size) + "\n"

        # Actor params
        content += "actor_params:\n"
        for i in range(actor_param_size):
            content += String(Float64(self.actor.params[i])) + "\n"

        content += "actor_optimizer_state:\n"
        for i in range(actor_state_size):
            content += String(Float64(self.actor.optimizer_state[i])) + "\n"

        # Critic params
        content += "critic_params:\n"
        for i in range(critic_param_size):
            content += String(Float64(self.critic.params[i])) + "\n"

        content += "critic_optimizer_state:\n"
        for i in range(critic_state_size):
            content += String(Float64(self.critic.optimizer_state[i])) + "\n"

        # Metadata
        content += "metadata:\n"
        content += "gamma=" + String(self.gamma) + "\n"
        content += "gae_lambda=" + String(self.gae_lambda) + "\n"
        content += "clip_epsilon=" + String(self.clip_epsilon) + "\n"
        content += "actor_lr=" + String(self.actor_lr) + "\n"
        content += "critic_lr=" + String(self.critic_lr) + "\n"
        content += "entropy_coef=" + String(self.entropy_coef) + "\n"
        content += "value_loss_coef=" + String(self.value_loss_coef) + "\n"
        content += "num_epochs=" + String(self.num_epochs) + "\n"
        content += "minibatch_size=" + String(self.minibatch_size) + "\n"
        content += (
            "normalize_advantages=" + String(self.normalize_advantages) + "\n"
        )
        content += "target_kl=" + String(self.target_kl) + "\n"
        content += "max_grad_norm=" + String(self.max_grad_norm) + "\n"
        content += "train_step_count=" + String(self.train_step_count) + "\n"

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var actor_param_size = self.actor.PARAM_SIZE
        var critic_param_size = self.critic.PARAM_SIZE
        var actor_state_size = actor_param_size * Adam.STATE_PER_PARAM
        var critic_state_size = critic_param_size * Adam.STATE_PER_PARAM

        var content = read_checkpoint_file(filepath)
        var lines = split_lines(content)

        # Load actor params
        var actor_params_start = find_section_start(lines, "actor_params:")
        for i in range(actor_param_size):
            self.actor.params[i] = Scalar[dtype](
                atof(lines[actor_params_start + i])
            )

        var actor_state_start = find_section_start(
            lines, "actor_optimizer_state:"
        )
        for i in range(actor_state_size):
            self.actor.optimizer_state[i] = Scalar[dtype](
                atof(lines[actor_state_start + i])
            )

        # Load critic params
        var critic_params_start = find_section_start(lines, "critic_params:")
        for i in range(critic_param_size):
            self.critic.params[i] = Scalar[dtype](
                atof(lines[critic_params_start + i])
            )

        var critic_state_start = find_section_start(
            lines, "critic_optimizer_state:"
        )
        for i in range(critic_state_size):
            self.critic.optimizer_state[i] = Scalar[dtype](
                atof(lines[critic_state_start + i])
            )

        # Load metadata
        var metadata_start = find_section_start(lines, "metadata:")
        for i in range(metadata_start, len(lines)):
            var line = lines[i]
            if line.startswith("gamma="):
                self.gamma = atof(String(line[6:]))
            elif line.startswith("gae_lambda="):
                self.gae_lambda = atof(String(line[11:]))
            elif line.startswith("clip_epsilon="):
                self.clip_epsilon = atof(String(line[13:]))
            elif line.startswith("actor_lr="):
                self.actor_lr = atof(String(line[9:]))
            elif line.startswith("critic_lr="):
                self.critic_lr = atof(String(line[10:]))
            elif line.startswith("entropy_coef="):
                self.entropy_coef = atof(String(line[13:]))
            elif line.startswith("value_loss_coef="):
                self.value_loss_coef = atof(String(line[16:]))
            elif line.startswith("num_epochs="):
                self.num_epochs = Int(atol(String(line[11:])))
            elif line.startswith("minibatch_size="):
                self.minibatch_size = Int(atol(String(line[15:])))
            elif line.startswith("normalize_advantages="):
                self.normalize_advantages = String(line[21:]) == "True"
            elif line.startswith("target_kl="):
                self.target_kl = atof(String(line[10:]))
            elif line.startswith("max_grad_norm="):
                self.max_grad_norm = atof(String(line[14:]))
            elif line.startswith("train_step_count="):
                self.train_step_count = Int(atol(String(line[17:])))


# =============================================================================
# GPU Kernel: Sample actions from categorical distribution
# =============================================================================


@always_inline
fn _sample_actions_kernel[
    dtype: DType,
    N_ENVS: Int,
    NUM_ACTIONS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, NUM_ACTIONS), MutAnyOrigin
    ],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    seed: Scalar[DType.uint32],
):
    """Sample actions from categorical distribution and compute log probs."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    # Per-thread RNG
    var rng_state = UInt32(seed) ^ (UInt32(i) * 2654435761)
    rng_state = xorshift32(rng_state)

    # Compute softmax probabilities
    var max_logit = logits[i, 0]
    for a in range(1, NUM_ACTIONS):
        var l = logits[i, a]
        if l > max_logit:
            max_logit = l

    var sum_exp = (
        logits[i, 0] - logits[i, 0]
    )  # Initialize to zero with correct type
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        sum_exp = sum_exp + exp(logit_val)

    # Sample action
    var rand_result = random_uniform[dtype](rng_state)
    var rand_val = rand_result[0]
    rng_state = rand_result[1]

    var cumsum_val = Scalar[dtype](0.0)
    var selected_action = 0
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        var prob = exp(logit_val) / sum_exp
        var prob_scalar = Scalar[dtype](prob[0])
        cumsum_val = cumsum_val + prob_scalar
        if rand_val < cumsum_val:
            selected_action = a
            break

    actions[i] = selected_action

    # Compute log probability
    var logit_sel = logits[i, selected_action] - max_logit
    var selected_prob_simd = exp(logit_sel) / sum_exp
    var selected_prob = Float32(selected_prob_simd[0])
    var eps = Float32(1e-8)
    var log_prob_val = log(selected_prob + eps)
    log_probs[i] = Scalar[dtype](log_prob_val)
