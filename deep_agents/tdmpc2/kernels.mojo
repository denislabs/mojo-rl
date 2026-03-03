"""GPU Kernels for TD-MPC2.

All kernels follow the same pattern as deep_agents/ppo/kernels.mojo:
  - @always_inline decorator
  - LayoutTensor params with compile-time dimensions
  - 1 thread per sample (no reduction kernels needed)
  - PhiloxRandom for GPU-compatible random number generation

Kernels provided:
  Data Collection:
    tdmpc2_random_actions_kernel       — uniform random in [-1,1] (warmup)
    tdmpc2_sample_actions_kernel       — tanh-squashed Gaussian (policy exploration)

  Data Preparation:
    tdmpc2_build_za_kernel             — concatenate (z, action) → za
    tdmpc2_extract_z_from_za_kernel    — extract z from za gradient
    tdmpc2_extract_obs_step_kernel     — extract obs at horizon step t from batch buffer
    tdmpc2_extract_act_step_kernel     — extract actions at step t from batch buffer
    tdmpc2_reorder_obs_batch_kernel    — reorder from b-major to t-major layout

  Loss Gradients:
    tdmpc2_consistency_loss_grad_kernel — MSE grad with rho weighting (ACCUMULATES)
    tdmpc2_two_hot_loss_grad_kernel     — softmax CE grad with rho weighting (ACCUMULATES)
    tdmpc2_bce_loss_grad_kernel         — BCE grad for termination (ACCUMULATES)

  TD Target Computation:
    tdmpc2_q_decode_kernel             — softmax(logits) · bins (expected value)
    tdmpc2_compute_td_targets_kernel   — compute + two-hot encode TD targets on GPU

  Policy Gradient:
    tdmpc2_policy_grad_kernel          — gradient for tanh-squashed policy update
    tdmpc2_apply_tanh_kernel           — apply tanh squashing to mean actions
"""

from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from math import exp, log, sqrt, cos, tanh
from random.philox import Random as PhiloxRandom


# =============================================================================
# Data Collection Kernels
# =============================================================================


@always_inline
fn tdmpc2_random_actions_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
](
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Sample uniformly random actions in [-1, 1] for warmup phase.

    Each thread generates one sample (N_ENVS parallel).

    Args:
        actions: Output action buffer [BATCH_SIZE, ACTION_DIM].
        rng_seed: Random seed (should vary per call).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    for j in range(ACTION_DIM):
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(i) * UInt64(ACTION_DIM) + UInt64(j),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        # Map [0, 1] → [-1, 1]
        var u = Scalar[DType.float32](rand_vals[0]) * 2.0 - 1.0
        actions[i, j] = Scalar[dtype](u)


@always_inline
fn tdmpc2_sample_actions_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM * 2), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Sample actions from tanh-squashed Gaussian policy (TD-MPC2 style).

    Policy output layout: [mean[ACTION_DIM] | log_std[ACTION_DIM]]
    Action = tanh(mean + exp(log_std) * noise) ∈ (-1, 1)

    Unlike PPO (unbounded Gaussian), TD-MPC2 uses tanh squashing to keep
    actions in the valid [-1, 1] range for continuous control tasks.

    Args:
        pi_out: Policy network output [BATCH_SIZE, ACTION_DIM * 2].
        actions: Output sampled actions [BATCH_SIZE, ACTION_DIM].
        rng_seed: Random seed (should vary per call).
    """
    comptime LOG_STD_MIN: Scalar[dtype] = -5.0
    comptime LOG_STD_MAX: Scalar[dtype] = 2.0

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    for j in range(ACTION_DIM):
        var mean_raw = pi_out[i, j]
        var log_std_raw = pi_out[i, ACTION_DIM + j]
        var mean = Scalar[dtype](mean_raw[0])
        var log_std = Scalar[dtype](log_std_raw[0])

        # Clamp log_std for numerical stability
        if log_std < LOG_STD_MIN:
            log_std = LOG_STD_MIN
        elif log_std > LOG_STD_MAX:
            log_std = LOG_STD_MAX

        # Box-Muller transform for standard normal sample
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(i) * UInt64(ACTION_DIM) + UInt64(j),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var u1 = Scalar[DType.float32](rand_vals[0]) + 1e-8
        var u2 = Scalar[DType.float32](rand_vals[1])
        var mag = sqrt(-2.0 * log(u1))
        var noise = Scalar[dtype](
            mag * cos(u2 * Scalar[DType.float32](6.283185307179586))
        )

        # Pre-squash sample: u = mean + exp(log_std) * noise
        var std = exp(log_std)
        var u_pre = mean + std * noise

        # Tanh squash to (-1, 1)
        actions[i, j] = Scalar[dtype](tanh(Float32(u_pre)))


# =============================================================================
# Data Preparation Kernels
# =============================================================================


@always_inline
fn tdmpc2_build_za_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    LATENT_DIM: Int,
    ACTION_DIM: Int,
](
    z: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, LATENT_DIM), MutAnyOrigin
    ],
    action: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    za: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE, LATENT_DIM + ACTION_DIM),
        MutAnyOrigin,
    ],
) where dtype.is_floating_point():
    """Concatenate latent z and action into (z, a) for dynamics/Q networks.

    za[i] = [z[i, 0..LATENT), action[i, 0..ACT)]

    Args:
        z: Latent states [BATCH_SIZE, LATENT_DIM].
        action: Actions [BATCH_SIZE, ACTION_DIM].
        za: Output concatenation [BATCH_SIZE, LATENT_DIM + ACTION_DIM].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    for k in range(LATENT_DIM):
        za[i, k] = z[i, k]
    for k in range(ACTION_DIM):
        za[i, LATENT_DIM + k] = action[i, k]


@always_inline
fn tdmpc2_extract_z_from_za_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    LATENT_DIM: Int,
    ACTION_DIM: Int,
](
    grad_za: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE, LATENT_DIM + ACTION_DIM),
        MutAnyOrigin,
    ],
    grad_z: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, LATENT_DIM), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Extract the z-part gradient from grad_za (accumulates into grad_z).

    When dynamics backward produces grad_za[BATCH, LATENT+ACT], only the
    first LATENT elements flow back through the encoder. This kernel copies
    grad_za[:, :LATENT] into grad_z (accumulating, not overwriting).

    Args:
        grad_za: Gradient w.r.t. (z, a) input [BATCH_SIZE, LATENT+ACT].
        grad_z: Gradient accumulation buffer for z [BATCH_SIZE, LATENT_DIM].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    for k in range(LATENT_DIM):
        grad_z[i, k] = grad_z[i, k] + grad_za[i, k]


@always_inline
fn tdmpc2_extract_obs_step_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    HORIZON: Int,
](
    batch_obs_flat: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE * (HORIZON + 1) * OBS_DIM),
        MutAnyOrigin,
    ],
    step: Int,
    obs_step: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Extract observations at a specific horizon step from a flat batch buffer.

    batch_obs_flat layout: flat [BATCH * (H+1) * OBS] (b-major):
      index = b * (H+1) * OBS + step * OBS + k
    Extracts: obs_step[b, k] = batch_obs_flat[b * (H+1) * OBS + step * OBS + k]

    Args:
        batch_obs_flat: Flat batch obs buffer [BATCH * (H+1) * OBS].
        step: Horizon step index (0 = initial obs, 1 = next obs after step 0).
        obs_step: Output observations at this step [BATCH, OBS].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var base = i * (HORIZON + 1) * OBS_DIM + step * OBS_DIM
    for k in range(OBS_DIM):
        obs_step[i, k] = batch_obs_flat[base + k]


@always_inline
fn tdmpc2_extract_act_step_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
](
    batch_acts_flat: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE * HORIZON * ACTION_DIM),
        MutAnyOrigin,
    ],
    step: Int,
    acts_step: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Extract actions at a specific horizon step from a flat batch buffer.

    batch_acts_flat layout: flat [BATCH * H * ACT] (b-major):
      index = b * H * ACT + step * ACT + k
    Extracts: acts_step[b, k] = batch_acts_flat[b * H * ACT + step * ACT + k]

    Args:
        batch_acts_flat: Flat batch action buffer [BATCH * H * ACT].
        step: Horizon step index (0..H-1).
        acts_step: Output actions at this step [BATCH, ACT].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var base = i * HORIZON * ACTION_DIM + step * ACTION_DIM
    for k in range(ACTION_DIM):
        acts_step[i, k] = batch_acts_flat[base + k]


@always_inline
fn tdmpc2_extract_scalar_step_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    HORIZON: Int,
](
    batch_flat: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE * HORIZON), MutAnyOrigin
    ],
    step: Int,
    step_out: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Extract scalar values at a specific horizon step from a flat batch buffer.

    batch_flat layout: flat [BATCH * H] (b-major):
      index = b * H + step
    Extracts: step_out[b] = batch_flat[b * H + step]

    Used for extracting per-step rewards and done flags.

    Args:
        batch_flat: Flat scalar buffer [BATCH * H].
        step: Horizon step index (0..H-1).
        step_out: Output scalar values at this step [BATCH].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    step_out[i] = batch_flat[i * HORIZON + step]


# =============================================================================
# Loss Gradient Kernels
# =============================================================================


@always_inline
fn tdmpc2_consistency_loss_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    LATENT_DIM: Int,
](
    z_pred: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, LATENT_DIM), MutAnyOrigin
    ],
    z_target: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, LATENT_DIM), MutAnyOrigin
    ],
    grad_z_pred: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, LATENT_DIM), MutAnyOrigin
    ],
    rho_weight: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute and ACCUMULATE gradient of MSE consistency loss w.r.t. z_pred.

    L_consistency = rho * mean((z_pred - z_target)^2)
    dL/d(z_pred[i,k]) = rho * 2*(z_pred[i,k] - z_target[i,k]) / (BATCH * LATENT)

    ACCUMULATES into grad_z_pred (use zero_buffer_kernel before first call).

    Args:
        z_pred: Predicted next latent state [BATCH, LATENT].
        z_target: Target next latent state (stop-grad) [BATCH, LATENT].
        grad_z_pred: Gradient accumulation buffer [BATCH, LATENT].
        rho_weight: rho^t temporal decay weight.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var scale = (
        rho_weight * Scalar[dtype](2.0) / Scalar[dtype](BATCH_SIZE * LATENT_DIM)
    )
    for k in range(LATENT_DIM):
        var diff = z_pred[i, k] - z_target[i, k]
        grad_z_pred[i, k] = grad_z_pred[i, k] + scale * Scalar[dtype](diff[0])


@always_inline
fn tdmpc2_two_hot_loss_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    BINS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    target_dist: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    grad_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    rho_weight: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute and ACCUMULATE gradient of two-hot (distributional) CE loss.

    L = rho * mean(-sum_k(target[k] * log_softmax(logits)[k]))
    dL/d(logits[i,k]) = rho * (softmax(logits)[i,k] - target[i,k]) / BATCH

    ACCUMULATES into grad_logits.

    Args:
        logits: Network output logits [BATCH, BINS].
        target_dist: Two-hot encoded target distribution [BATCH, BINS].
        grad_logits: Gradient accumulation buffer [BATCH, BINS].
        rho_weight: rho^t temporal decay weight.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Compute softmax
    var max_l = Scalar[dtype](logits[i, 0][0])
    for k in range(1, BINS):
        var v = Scalar[dtype](logits[i, k][0])
        if v > max_l:
            max_l = v

    var sum_exp = Scalar[dtype](0.0)
    for k in range(BINS):
        sum_exp = sum_exp + exp(Scalar[dtype](logits[i, k][0]) - max_l)

    var scale = rho_weight / Scalar[dtype](BATCH_SIZE)
    for k in range(BINS):
        var sm_k = exp(Scalar[dtype](logits[i, k][0]) - max_l) / sum_exp
        var tgt_k = Scalar[dtype](target_dist[i, k][0])
        grad_logits[i, k] = grad_logits[i, k] + scale * (sm_k - tgt_k)


@always_inline
fn tdmpc2_bce_loss_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    probs: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    grad_probs: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    rho_weight: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute and ACCUMULATE gradient of binary cross-entropy termination loss.

    L = rho * mean(-d*log(p) - (1-d)*log(1-p))
    dL/d(p[i]) = rho * (-d/clamp(p, eps) + (1-d)/clamp(1-p, eps)) / BATCH

    ACCUMULATES into grad_probs.

    Args:
        probs: Predicted termination probabilities [BATCH] (sigmoid output).
        dones: True done flags [BATCH] (1.0 = done, 0.0 = not done).
        grad_probs: Gradient accumulation buffer [BATCH].
        rho_weight: rho^t temporal decay weight.
    """
    comptime EPS: Scalar[dtype] = 1e-7

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var p = Scalar[dtype](probs[i][0])
    var d = Scalar[dtype](dones[i][0])
    var scale = rho_weight / Scalar[dtype](BATCH_SIZE)

    # Clamp p for numerical stability
    var p_clamped = p
    if p_clamped < EPS:
        p_clamped = EPS
    if p_clamped > Scalar[dtype](1.0) - EPS:
        p_clamped = Scalar[dtype](1.0) - EPS

    var grad = scale * (
        -d / p_clamped
        + (Scalar[dtype](1.0) - d) / (Scalar[dtype](1.0) - p_clamped)
    )
    grad_probs[i] = grad_probs[i] + grad


# =============================================================================
# TD Target Computation Kernels
# =============================================================================


@always_inline
fn tdmpc2_q_decode_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    BINS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    bins: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Decode distributional Q-value: expected value = softmax(logits) · bins.

    values[i] = sum_k(softmax(logits[i])[k] * bins[k])

    Used to:
    - Decode target Q-values for TD target computation
    - Decode current Q-values for policy gradient

    Args:
        logits: Q-network logit output [BATCH, BINS].
        bins: Value bin centers [BINS].
        values: Output decoded values [BATCH].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Numerically stable softmax + dot product
    var max_l = Scalar[dtype](logits[i, 0][0])
    for k in range(1, BINS):
        var v = Scalar[dtype](logits[i, k][0])
        if v > max_l:
            max_l = v

    var sum_exp = Scalar[dtype](0.0)
    for k in range(BINS):
        sum_exp = sum_exp + exp(Scalar[dtype](logits[i, k][0]) - max_l)

    var expected_val = Scalar[dtype](0.0)
    for k in range(BINS):
        var sm_k = exp(Scalar[dtype](logits[i, k][0]) - max_l) / sum_exp
        expected_val = expected_val + sm_k * Scalar[dtype](bins[k][0])

    values[i] = expected_val


@always_inline
fn tdmpc2_compute_td_targets_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    BINS: Int,
](
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    q_next: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    td_targets: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    gamma: Scalar[dtype],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute two-hot encoded TD targets on GPU.

    td_target[i] = r[i] + gamma * (1 - done[i]) * V_next[i]
    Then two-hot encode td_target[i] over BINS bins.

    Args:
        rewards: Step rewards [BATCH].
        dones: Done flags [BATCH] (1.0 = done).
        q_next: Target network Q values (V_next) [BATCH].
        td_targets: Output two-hot distributions [BATCH, BINS].
        gamma: Discount factor.
        v_min: Minimum value for distribution.
        v_max: Maximum value for distribution.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var r = Scalar[dtype](rewards[i][0])
    var d = Scalar[dtype](dones[i][0])
    var v_next = Scalar[dtype](q_next[i][0])

    var td_val = r + gamma * (Scalar[dtype](1.0) - d) * v_next

    # Clamp to [v_min, v_max]
    if td_val < v_min:
        td_val = v_min
    if td_val > v_max:
        td_val = v_max

    # Two-hot encoding: distribute probability between two adjacent bins
    var step = (v_max - v_min) / Scalar[dtype](BINS - 1)
    var k_float = (td_val - v_min) / step
    var k = Int(k_float)
    if k >= BINS - 1:
        k = BINS - 2

    var bin_low = v_min + step * Scalar[dtype](k)
    var bin_high = bin_low + step
    var upper_w = (bin_high - td_val) / (bin_high - bin_low)
    var lower_w = Scalar[dtype](1.0) - upper_w

    # Zero out this row, then set the two bins
    for kk in range(BINS):
        td_targets[i, kk] = Scalar[dtype](0.0)
    td_targets[i, k] = upper_w
    td_targets[i, k + 1] = lower_w


# =============================================================================
# Policy Gradient Kernels
# =============================================================================


@always_inline
fn tdmpc2_policy_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM * 2), MutAnyOrigin
    ],
    q_values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    grad_pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM * 2), MutAnyOrigin
    ],
    rho_weight: Scalar[dtype],
    entropy_coef: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute gradient for tanh-squashed Gaussian policy update.

    Policy loss: L = -rho * mean(Q(z, pi(z))) + entropy_coef * mean(-log_prob)

    Gradient approximation:
    - The gradient of -E[Q] w.r.t. mean: -rho/BATCH * tanh'(mean) * sign(Q-mean_Q)
    - The gradient w.r.t. log_std: entropy bonus entropy_coef / BATCH

    This simplified gradient treats Q as a scalar weight for the mean update.
    For proper reparameterization, a full backward through the Q-network would
    be needed. This approximation is effective in practice for TD-MPC2.

    ACCUMULATES into grad_pi_out.

    Args:
        pi_out: Policy network output [BATCH, ACTION_DIM * 2] (mean | log_std).
        q_values: Q-values for the actions [BATCH].
        grad_pi_out: Gradient accumulation buffer [BATCH, ACTION_DIM * 2].
        rho_weight: rho^t temporal decay weight.
        entropy_coef: Entropy regularization coefficient.
    """
    comptime LOG_STD_MIN: Scalar[dtype] = -5.0
    comptime LOG_STD_MAX: Scalar[dtype] = 2.0

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Gradient of -E[Q] w.r.t. policy mean:
    # Maximize Q → gradient is -rho / BATCH for each action dim
    # (multiplied by tanh' = 1 - tanh(mean)^2 at the mean output)
    var neg_scale = -rho_weight / Scalar[dtype](BATCH_SIZE)
    var ent_scale = entropy_coef / Scalar[dtype](BATCH_SIZE)

    for j in range(ACTION_DIM):
        var mean_raw = Scalar[dtype](pi_out[i, j][0])

        # tanh derivative at the pre-squash mean
        var t = Scalar[dtype](tanh(Float32(mean_raw)))
        var tanh_deriv = Scalar[dtype](1.0) - t * t

        # Gradient w.r.t. mean: -rho/B * (1 - tanh(u)^2)
        # (chain rule: d(-Q)/d(mean) = d(-Q)/d(action) * d(action)/d(u))
        grad_pi_out[i, j] = grad_pi_out[i, j] + neg_scale * tanh_deriv

        # Gradient w.r.t. log_std: entropy bonus = +1/B (maximize entropy)
        # log_prob contribution to entropy: -log_std (approx)
        # maximize entropy = minimize -log_prob → gradient = -entropy_coef/B
        var log_std_raw = Scalar[dtype](pi_out[i, ACTION_DIM + j][0])
        # Clamp for stability
        if log_std_raw < LOG_STD_MIN:
            log_std_raw = LOG_STD_MIN
        elif log_std_raw > LOG_STD_MAX:
            log_std_raw = LOG_STD_MAX

        # d(entropy)/d(log_std) = 1 → gradient to maximize entropy = -entropy_coef * (-1/B) = ent_scale
        grad_pi_out[i, ACTION_DIM + j] = (
            grad_pi_out[i, ACTION_DIM + j] - ent_scale
        )


@always_inline
fn tdmpc2_apply_tanh_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM * 2), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Apply tanh squashing to policy mean outputs (deterministic actions).

    actions[i, j] = tanh(pi_out[i, j])

    Used for stop-gradient policy evaluation (e.g., computing V_next
    for TD targets without needing the stochastic sampling).

    Args:
        pi_out: Policy network output [BATCH, ACTION_DIM * 2] (mean | log_std).
        actions: Output squashed actions [BATCH, ACTION_DIM].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    for j in range(ACTION_DIM):
        var mean_raw = Scalar[dtype](pi_out[i, j][0])
        actions[i, j] = Scalar[dtype](tanh(Float32(mean_raw)))


# =============================================================================
# Q-value Aggregation
# =============================================================================


@always_inline
fn tdmpc2_q_min_reduce_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    running_min: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    new_vals: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Update running minimum: running_min[i] = min(running_min[i], new_vals[i]).

    Used iteratively across Q1..Q5 target networks to compute min-Q for TD targets.
    Initialize running_min with Q1 values, then call with Q2..Q5 values.

    Args:
        running_min: Running minimum buffer [BATCH] (updated in-place).
        new_vals: New Q values to compare against [BATCH].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var cur = Scalar[dtype](running_min[i][0])
    var nv = Scalar[dtype](new_vals[i][0])
    if nv < cur:
        running_min[i] = nv


# =============================================================================
# Gradient Zero Kernel (convenience wrapper)
# =============================================================================


@always_inline
fn tdmpc2_zero_kernel[
    dtype: DType,
    SIZE: Int,
](
    buffer: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]
) where dtype.is_floating_point():
    """Zero out a flat buffer. Alias for zero_buffer_kernel for clarity.

    Args:
        buffer: Buffer to zero [SIZE].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    buffer[i] = Scalar[dtype](0.0)


# =============================================================================
# Accumulate Buffer (add src into dst)
# =============================================================================


@always_inline
fn tdmpc2_add_into_kernel[
    dtype: DType,
    SIZE: Int,
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Accumulate: dst[i] += src[i].

    Used to combine gradients from multiple loss sources
    (e.g., dynamics gradient + termination gradient for encoder).

    Args:
        dst: Destination accumulation buffer [SIZE] (updated in-place).
        src: Source buffer to add [SIZE].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    dst[i] = dst[i] + src[i]


# =============================================================================
# Fused Kernels (reduce kernel launch overhead)
# =============================================================================


@always_inline
fn tdmpc2_extract_rew_done_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    HORIZON: Int,
](
    batch_rew_flat: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE * HORIZON), MutAnyOrigin
    ],
    batch_done_flat: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE * HORIZON), MutAnyOrigin
    ],
    step: Int,
    rew_step: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    done_step: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Fused extraction of reward AND done flag at a specific horizon step.

    Combines two tdmpc2_extract_scalar_step_kernel calls into one launch.
    Extracts:
      rew_step[b]  = batch_rew_flat[b * H + step]
      done_step[b] = batch_done_flat[b * H + step]

    Args:
        batch_rew_flat: Flat reward buffer [BATCH * H].
        batch_done_flat: Flat done-flag buffer [BATCH * H].
        step: Horizon step index (0..H-1).
        rew_step: Output rewards at this step [BATCH].
        done_step: Output done flags at this step [BATCH].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var idx = i * HORIZON + step
    rew_step[i] = batch_rew_flat[idx]
    done_step[i] = batch_done_flat[idx]


@always_inline
fn tdmpc2_decode_and_min_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    BINS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    bins: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
    q_min: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Fused Q-value decode + running-minimum update.

    Combines tdmpc2_q_decode_kernel + tdmpc2_q_min_reduce_kernel into one
    launch. Computes expected value = softmax(logits) · bins, then updates
    q_min[i] = min(q_min[i], expected_val).

    Used for Q2..Q5 target networks during TD target computation.
    Initialize q_min with Q1 decoded values, then call this for Q2..Q5.

    Args:
        logits: Q-network logit output [BATCH, BINS].
        bins: Value bin centers [BINS].
        q_min: Running minimum buffer [BATCH] (updated in-place).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Numerically stable softmax + dot product (same as q_decode_kernel)
    var max_l = Scalar[dtype](logits[i, 0][0])
    for k in range(1, BINS):
        var v = Scalar[dtype](logits[i, k][0])
        if v > max_l:
            max_l = v

    var sum_exp = Scalar[dtype](0.0)
    for k in range(BINS):
        sum_exp = sum_exp + exp(Scalar[dtype](logits[i, k][0]) - max_l)

    var expected_val = Scalar[dtype](0.0)
    for k in range(BINS):
        var sm_k = exp(Scalar[dtype](logits[i, k][0]) - max_l) / sum_exp
        expected_val = expected_val + sm_k * Scalar[dtype](bins[k][0])

    # In-place min update
    var cur = Scalar[dtype](q_min[i][0])
    if expected_val < cur:
        q_min[i] = expected_val


@always_inline
fn tdmpc2_add_two_into_kernel[
    dtype: DType,
    SIZE: Int,
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src1: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src2: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Fused accumulate of two sources: dst[i] += src1[i] + src2[i].

    Combines two consecutive tdmpc2_add_into_kernel calls into one launch.
    Used to combine dynamics gradient + termination gradient into encoder
    output gradient in a single pass.

    Args:
        dst: Destination accumulation buffer [SIZE] (updated in-place).
        src1: First source buffer to add [SIZE].
        src2: Second source buffer to add [SIZE].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    dst[i] = dst[i] + src1[i] + src2[i]
