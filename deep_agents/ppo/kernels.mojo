from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from math import exp, log, sqrt, cos
from random.philox import Random as PhiloxRandom

# =============================================================================
# GPU Kernels for Continuous PPO Operations
# =============================================================================


@always_inline
fn _sample_continuous_actions_kernel[
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
):
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
        var philox = PhiloxRandom(seed=Int(rng_seed) + i * ACTION_DIM + j, offset=0)
        var rand_vals = philox.step_uniform()
        var u1 = rand_vals[0]
        var u2 = rand_vals[1]

        # Box-Muller transform for standard normal
        # log() requires Float32
        var u1_for_log = Float32(u1) + Float32(1e-8)
        var u2_for_cos = Float32(u2)

        var mag = sqrt(-2.0 * log(u1_for_log))
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
            LOG_2PI + Scalar[dtype](2.0) * log_std + action_normalized * action_normalized
        )

        total_log_prob = total_log_prob + log_gaussian

    log_probs[i] = total_log_prob


@always_inline
fn _store_continuous_pre_step_kernel[
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
fn ppo_continuous_gather_minibatch_kernel[
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
fn ppo_continuous_actor_grad_kernel[
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
):
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
    var action_vals = InlineArray[Scalar[dtype], ACTION_DIM](fill=Scalar[dtype](0.0))
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

    # Compute gradients for mean and log_std
    var batch_size_scalar: Scalar[dtype] = BATCH_SIZE
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
            var d_log_prob_d_log_std = action_normalized * action_normalized - one

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


@always_inline
fn ppo_critic_grad_kernel[
    dtype: DType, BATCH_SIZE: Int
](
    # Outputs
    grad_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin
    ],
    # Inputs
    values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    batch_size: Int,
):
    """Compute gradient for PPO critic (MSE loss)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var value = values[b, 0]
    var target = returns[b]

    # MSE gradient: d(0.5 * (value - target)^2) / d_value = (value - target)
    grad_values[b, 0] = (value - target) / Scalar[dtype](BATCH_SIZE)


@always_inline
fn ppo_critic_grad_clipped_kernel[
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
    batch_size: Int,
):
    """Compute gradient for PPO critic with value clipping."""
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
            clip_sign * (value_clipped - target) / Scalar[dtype](BATCH_SIZE)
        )
    else:
        grad_values[b, 0] = (value - target) / Scalar[dtype](BATCH_SIZE)


@always_inline
fn normalize_advantages_kernel[
    dtype: DType, BATCH_SIZE: Int
](
    advantages: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    mean: Scalar[dtype],
    std: Scalar[dtype],
    batch_size: Int,
):
    """Normalize advantages in-place."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= batch_size:
        return

    advantages[i] = (advantages[i] - mean) / (std + Scalar[dtype](1e-8))


@always_inline
fn normalize_advantages_fused_kernel[
    dtype: DType, BATCH_SIZE: Int, BLOCK_SIZE: Int
](advantages: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],):
    """Compute mean/std and normalize advantages entirely on GPU.

    This kernel uses shared memory reduction to compute mean and std,
    then normalizes in-place. Must be launched with a single block.
    BLOCK_SIZE should be >= BATCH_SIZE and a power of 2.
    """
    var thread_id = Int(thread_idx.x)

    # Shared memory for reductions
    var shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_var = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Step 1: Load values and compute sum for mean
    var local_val = Scalar[dtype](0.0)
    var local_sum = Scalar[dtype](0.0)

    # Handle case where BATCH_SIZE > BLOCK_SIZE by striding
    var idx = thread_id
    while idx < BATCH_SIZE:
        var val = rebind[Scalar[dtype]](advantages[idx])
        local_sum += val
        if idx == thread_id:
            local_val = val  # Save first value for this thread
        idx += BLOCK_SIZE
    shared_sum[thread_id] = local_sum

    barrier()

    # Parallel reduction for sum
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared_sum[thread_id] = (
                shared_sum[thread_id] + shared_sum[thread_id + stride]
            )
        barrier()
        stride = stride // 2

    # Compute mean (thread 0 broadcasts via shared memory)
    var mean = rebind[Scalar[dtype]](shared_sum[0]) / Scalar[dtype](BATCH_SIZE)

    barrier()

    # Step 2: Compute variance sum
    var local_var_sum = Scalar[dtype](0.0)
    idx = thread_id
    while idx < BATCH_SIZE:
        var val = rebind[Scalar[dtype]](advantages[idx])
        var diff = val - mean
        local_var_sum += diff * diff
        idx += BLOCK_SIZE
    shared_var[thread_id] = local_var_sum

    barrier()

    # Parallel reduction for variance
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared_var[thread_id] = (
                shared_var[thread_id] + shared_var[thread_id + stride]
            )
        barrier()
        stride = stride // 2

    # Compute std
    var variance = rebind[Scalar[dtype]](shared_var[0]) / Scalar[dtype](
        BATCH_SIZE
    )
    var std = Scalar[dtype](sqrt(variance + 1e-8))

    barrier()

    # Step 3: Normalize in-place
    idx = thread_id
    while idx < BATCH_SIZE:
        var val = rebind[Scalar[dtype]](advantages[idx])
        advantages[idx] = (val - mean) / (std + Scalar[dtype](1e-8))
        idx += BLOCK_SIZE


@always_inline
fn gradient_norm_kernel[
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
        address_space = AddressSpace.SHARED,
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
fn gradient_clip_kernel[
    dtype: DType, PARAM_SIZE: Int
](
    grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    scale: Scalar[dtype],
):
    """Scale gradients for gradient clipping."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= PARAM_SIZE:
        return
    grads[i] = grads[i] * scale


@always_inline
fn gradient_reduce_and_compute_scale_kernel[
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


# =============================================================================
# Fully fused gradient clipping (reduces 3 kernels to 2)
# =============================================================================


@always_inline
fn gradient_reduce_apply_fused_kernel[
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
        address_space = AddressSpace.SHARED,
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


@always_inline
fn _extract_obs_from_state_continuous_kernel[
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
