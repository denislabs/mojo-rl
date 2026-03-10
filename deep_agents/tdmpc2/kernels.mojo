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
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.math import exp, log, sqrt, cos, tanh
from std.random.philox import Random as PhiloxRandom
from std.math import abs as math_abs


# =============================================================================
# Symlog / Symexp (TD-MPC2 value normalization)
# =============================================================================


@always_inline
fn _symlog[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """Symmetric logarithm: sign(x) * ln(1 + |x|).

    Compresses large values into a bounded range while preserving sign.
    Used to encode TD targets and rewards into distributional bin space.
    """
    if x >= 0:
        return log(Scalar[dtype](1.0) + x)
    else:
        return -log(Scalar[dtype](1.0) - x)


@always_inline
fn _symexp[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """Inverse of symlog: sign(x) * (exp(|x|) - 1).

    Converts from symlog space back to actual value space.
    Used to decode Q-values from distributional representation.
    """
    if x >= 0:
        return exp(x) - Scalar[dtype](1.0)
    else:
        return -(exp(-x) - Scalar[dtype](1.0))


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
    POL_OUT: Int = ACTION_DIM * 2,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
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
        var mag = sqrt(Float32(-2.0) * log(u1))
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
        rho_weight: Rho^t temporal decay weight.
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
        rho_weight: Rho^t temporal decay weight.
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
        rho_weight: Rho^t temporal decay weight.
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

    # Apply symexp: bins are in symlog space, convert to actual value space
    values[i] = _symexp[dtype](expected_val)


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
    # q_next is in actual value space (already symexp'd by q_decode_kernel)
    var v_next = Scalar[dtype](q_next[i][0])

    # Compute TD target in actual value space
    var td_val = r + gamma * (Scalar[dtype](1.0) - d) * v_next

    # Apply symlog: compress to distributional bin space
    td_val = _symlog[dtype](td_val)

    # Clamp to [v_min, v_max] (now in symlog space)
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


@always_inline
fn tdmpc2_compute_reward_targets_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    BINS: Int,
](
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    rew_targets: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute two-hot encoded IMMEDIATE reward targets on GPU.

    Unlike TD targets, this encodes just r_t (no bootstrapping).
    Used for the reward prediction head.

    Args:
        rewards: Step rewards [BATCH].
        rew_targets: Output two-hot distributions [BATCH, BINS].
        v_min: Minimum value for distribution.
        v_max: Maximum value for distribution.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var r = Scalar[dtype](rewards[i][0])

    # Apply symlog: compress reward to distributional bin space
    r = _symlog[dtype](r)

    # Clamp to [v_min, v_max] (now in symlog space)
    if r < v_min:
        r = v_min
    if r > v_max:
        r = v_max

    # Two-hot encoding: distribute probability between two adjacent bins
    var step = (v_max - v_min) / Scalar[dtype](BINS - 1)
    var k_float = (r - v_min) / step
    var k = Int(k_float)
    if k >= BINS - 1:
        k = BINS - 2

    var bin_low = v_min + step * Scalar[dtype](k)
    var bin_high = bin_low + step
    var upper_w = (bin_high - r) / (bin_high - bin_low)
    var lower_w = Scalar[dtype](1.0) - upper_w

    # Zero out this row, then set the two bins
    for kk in range(BINS):
        rew_targets[i, kk] = Scalar[dtype](0.0)
    rew_targets[i, k] = upper_w
    rew_targets[i, k + 1] = lower_w


# =============================================================================
# Policy Gradient Kernels
# =============================================================================


@always_inline
fn tdmpc2_q_decode_backward_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    BINS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    bins: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
    grad_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, BINS), MutAnyOrigin
    ],
    rho_weight: Scalar[dtype],
) where dtype.is_floating_point():
    """Backward through distributional Q decode for policy gradient.

    Q = sum(softmax(logits) * bins)
    d(-Q)/d(logits_k) = -softmax_k * (bin_k - Q) * rho / BATCH

    WRITES (not accumulates) to grad_logits.

    Args:
        logits: Q-network output logits [BATCH, BINS].
        bins: Bin centers [BINS].
        grad_logits: Output gradient [BATCH, BINS].
        rho_weight: Rho^t temporal decay weight.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Softmax
    var max_l = Scalar[dtype](logits[i, 0][0])
    for k in range(1, BINS):
        var v = Scalar[dtype](logits[i, k][0])
        if v > max_l:
            max_l = v

    var sum_exp = Scalar[dtype](0.0)
    for k in range(BINS):
        sum_exp = sum_exp + exp(Scalar[dtype](logits[i, k][0]) - max_l)

    # Compute Q = E[bins] under softmax
    var q_val = Scalar[dtype](0.0)
    for k in range(BINS):
        var prob_k = exp(Scalar[dtype](logits[i, k][0]) - max_l) / sum_exp
        q_val = q_val + prob_k * Scalar[dtype](bins[k][0])

    # With symexp decode: Q_actual = symexp(Q_symlog)
    # d(-Q_actual)/d(logits_k) = -symexp'(Q_symlog) * softmax_k * (bin_k - Q_symlog)
    # where symexp'(x) = exp(|x|) for all x
    var symexp_deriv = exp(math_abs(q_val))
    var scale = -symexp_deriv * rho_weight / Scalar[dtype](BATCH_SIZE)
    for k in range(BINS):
        var prob_k = exp(Scalar[dtype](logits[i, k][0]) - max_l) / sum_exp
        grad_logits[i, k] = scale * prob_k * (Scalar[dtype](bins[k][0]) - q_val)


@always_inline
fn tdmpc2_action_tanh_chain_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    ZA_DIM: Int = LATENT_DIM + ACTION_DIM,
    POL_OUT: Int = ACTION_DIM * 2,
](
    grad_za: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ZA_DIM), MutAnyOrigin
    ],
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
    ],
    grad_pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
    ],
    entropy_coef: Scalar[dtype],
) where dtype.is_floating_point():
    """Chain dQ/d(action) through tanh to get dQ/d(policy_mean).

    Given grad_za from Q backward (dL/d(za)):
    - Extract dL/d(action) = grad_za[:, LATENT_DIM:]
    - Chain through tanh: dL/d(mean_j) = dL/d(action_j) * tanh'(mean_j)
    - Add entropy gradient for log_std

    ACCUMULATES into grad_pi_out.

    Args:
        grad_za: Gradient from Q backward [BATCH, ZA_DIM].
        pi_out: Policy output [BATCH, POL_OUT] (mean | log_std).
        grad_pi_out: Output gradient for policy backward [BATCH, POL_OUT].
        entropy_coef: Entropy regularization coefficient.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    var ent_scale = entropy_coef / Scalar[dtype](BATCH_SIZE)

    for j in range(ACTION_DIM):
        # dL/d(action_j) from Q backward
        var grad_action = Scalar[dtype](grad_za[i, LATENT_DIM + j][0])

        # tanh derivative: d(tanh(u))/du = 1 - tanh(u)^2
        var mean_raw = Scalar[dtype](pi_out[i, j][0])
        var t = Scalar[dtype](tanh(Float32(mean_raw)))
        var tanh_deriv = Scalar[dtype](1.0) - t * t

        # dL/d(mean_j) = dL/d(action_j) * tanh'(mean_j)
        grad_pi_out[i, j] = grad_pi_out[i, j] + grad_action * tanh_deriv

        # Entropy gradient for log_std: maximize entropy → -entropy_coef/B
        grad_pi_out[i, ACTION_DIM + j] = (
            grad_pi_out[i, ACTION_DIM + j] - ent_scale
        )


# Legacy kernel kept for compatibility but replaced by proper DPG chain above
@always_inline
fn tdmpc2_policy_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
    POL_OUT: Int = ACTION_DIM * 2,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
    ],
    q_values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    grad_pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
    ],
    rho_weight: Scalar[dtype],
    entropy_coef: Scalar[dtype],
) where dtype.is_floating_point():
    """DEPRECATED: Use tdmpc2_q_decode_backward_kernel + Q backward +
    tdmpc2_action_tanh_chain_kernel for proper deterministic policy gradient."""
    pass


@always_inline
fn tdmpc2_apply_tanh_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
    POL_OUT: Int = ACTION_DIM * 2,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
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

    # Apply symexp: convert from symlog space to actual value space
    var actual_val = _symexp[dtype](expected_val)

    # In-place min update (in actual value space)
    var cur = Scalar[dtype](q_min[i][0])
    if actual_val < cur:
        q_min[i] = actual_val


@always_inline
fn tdmpc2_extract_all_build_za_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    HORIZON: Int,
](
    batch_acts_flat: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE * HORIZON * ACTION_DIM),
        MutAnyOrigin,
    ],
    batch_obs_flat: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE * (HORIZON + 1) * OBS_DIM),
        MutAnyOrigin,
    ],
    batch_rew_flat: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE * HORIZON), MutAnyOrigin
    ],
    batch_done_flat: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE * HORIZON), MutAnyOrigin
    ],
    step: Int,
    acts_step: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    obs_next_step: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    rew_step: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    done_step: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    z: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, LATENT_DIM), MutAnyOrigin
    ],
    za: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE, LATENT_DIM + ACTION_DIM),
        MutAnyOrigin,
    ],
) where dtype.is_floating_point():
    """Fused: extract act/obs/rew/done at step t AND build za = [z, act].

    Combines 4 sequential kernels into one launch:
      1. extract_act_step (step t)
      2. extract_obs_step (step t+1)
      3. extract_rew_done (step t)
      4. build_za (z, act → za)

    Safe because each thread i processes only its own row.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Extract actions at step t
    var act_base = i * HORIZON * ACTION_DIM + step * ACTION_DIM
    for k in range(ACTION_DIM):
        acts_step[i, k] = batch_acts_flat[act_base + k]

    # Extract obs at step t+1
    var obs_base = i * (HORIZON + 1) * OBS_DIM + (step + 1) * OBS_DIM
    for k in range(OBS_DIM):
        obs_next_step[i, k] = batch_obs_flat[obs_base + k]

    # Extract rew and done at step t
    var rd_idx = i * HORIZON + step
    rew_step[i] = batch_rew_flat[rd_idx]
    done_step[i] = batch_done_flat[rd_idx]

    # Build za = [z, act]
    for k in range(LATENT_DIM):
        za[i, k] = z[i, k]
    for k in range(ACTION_DIM):
        za[i, LATENT_DIM + k] = acts_step[i, k]


@always_inline
fn tdmpc2_extract_obs_rew_done_kernel[
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
    batch_rew_flat: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE * HORIZON), MutAnyOrigin
    ],
    batch_done_flat: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE * HORIZON), MutAnyOrigin
    ],
    step: Int,
    obs_next_step: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    rew_step: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    done_step: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """Fused: extract obs at step t+1 AND rew/done at step t.

    Combines extract_obs_step + extract_rew_done into one launch.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Extract obs at step t+1
    var obs_base = i * (HORIZON + 1) * OBS_DIM + (step + 1) * OBS_DIM
    for k in range(OBS_DIM):
        obs_next_step[i, k] = batch_obs_flat[obs_base + k]

    # Extract rew and done at step t
    var rd_idx = i * HORIZON + step
    rew_step[i] = batch_rew_flat[rd_idx]
    done_step[i] = batch_done_flat[rd_idx]


@always_inline
fn tdmpc2_apply_tanh_build_za_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    POL_OUT: Int = ACTION_DIM * 2,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, POL_OUT), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    z: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, LATENT_DIM), MutAnyOrigin
    ],
    za: LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE, LATENT_DIM + ACTION_DIM),
        MutAnyOrigin,
    ],
) where dtype.is_floating_point():
    """Fused: apply tanh to policy mean AND build za = [z, tanh(mean)].

    Combines apply_tanh_kernel + build_za_kernel into one launch.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_SIZE:
        return

    # Apply tanh to mean → actions
    for j in range(ACTION_DIM):
        var mean_raw = Scalar[dtype](pi_out[i, j][0])
        actions[i, j] = Scalar[dtype](tanh(Float32(mean_raw)))

    # Build za = [z, actions]
    for k in range(LATENT_DIM):
        za[i, k] = z[i, k]
    for k in range(ACTION_DIM):
        za[i, LATENT_DIM + k] = actions[i, k]


@always_inline
fn tdmpc2_soft_update_5q_kernel[
    dtype: DType,
    SIZE: Int,
](
    target1: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source1: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    target2: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source2: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    target3: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source3: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    target4: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source4: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    target5: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source5: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    tau: Scalar[dtype],
):
    """Fused soft update of 5 Q-target networks in one kernel launch.

    target_k = tau * source_k + (1 - tau) * target_k, for k=1..5.
    Saves 4 kernel launches per training step.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return

    var one_minus_tau = Scalar[dtype](1.0) - tau
    target1[i] = tau * source1[i] + one_minus_tau * target1[i]
    target2[i] = tau * source2[i] + one_minus_tau * target2[i]
    target3[i] = tau * source3[i] + one_minus_tau * target3[i]
    target4[i] = tau * source4[i] + one_minus_tau * target4[i]
    target5[i] = tau * source5[i] + one_minus_tau * target5[i]


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


# =============================================================================
# Fused 5Q Gradient Clipping + Adam (15 kernel launches → 3)
# =============================================================================


@always_inline
fn tdmpc2_gradient_norm_5q_kernel[
    dtype: DType, PARAM_SIZE: Int, NUM_BLOCKS: Int, BLOCK_SIZE: Int
](
    partial_sums_5q: LayoutTensor[
        dtype, Layout.row_major(NUM_BLOCKS * 5), MutAnyOrigin
    ],
    grads1: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads2: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads3: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads4: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads5: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
):
    """Compute partial sums of squared gradients for 5 Q networks in one launch.

    Replaces 5 sequential gradient_norm_kernel calls. Each block computes its
    partial sum for all 5 networks, writing to partial_sums_5q at offsets
    k * NUM_BLOCKS + block_id (k=0..4).

    Uses shared memory sequentially for each network to avoid 5x shared mem.
    """
    var block_id = Int(block_idx.x)
    var thread_id = Int(thread_idx.x)
    var idx = block_id * BLOCK_SIZE + thread_id

    var shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Network 1
    if idx < PARAM_SIZE:
        var g = grads1[idx]
        shared[thread_id] = g * g
    else:
        shared[thread_id] = Scalar[dtype](0.0)
    barrier()
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        partial_sums_5q[0 * NUM_BLOCKS + block_id] = shared[0]

    # Network 2
    if idx < PARAM_SIZE:
        var g = grads2[idx]
        shared[thread_id] = g * g
    else:
        shared[thread_id] = Scalar[dtype](0.0)
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        partial_sums_5q[1 * NUM_BLOCKS + block_id] = shared[0]

    # Network 3
    if idx < PARAM_SIZE:
        var g = grads3[idx]
        shared[thread_id] = g * g
    else:
        shared[thread_id] = Scalar[dtype](0.0)
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        partial_sums_5q[2 * NUM_BLOCKS + block_id] = shared[0]

    # Network 4
    if idx < PARAM_SIZE:
        var g = grads4[idx]
        shared[thread_id] = g * g
    else:
        shared[thread_id] = Scalar[dtype](0.0)
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        partial_sums_5q[3 * NUM_BLOCKS + block_id] = shared[0]

    # Network 5
    if idx < PARAM_SIZE:
        var g = grads5[idx]
        shared[thread_id] = g * g
    else:
        shared[thread_id] = Scalar[dtype](0.0)
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        partial_sums_5q[4 * NUM_BLOCKS + block_id] = shared[0]


@always_inline
fn tdmpc2_gradient_reduce_apply_5q_kernel[
    dtype: DType, PARAM_SIZE: Int, NUM_BLOCKS: Int, BLOCK_SIZE: Int
](
    grads1: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads2: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads3: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads4: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads5: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    partial_sums_5q: LayoutTensor[
        dtype, Layout.row_major(NUM_BLOCKS * 5), MutAnyOrigin
    ],
    max_grad_norm: Scalar[dtype],
):
    """Fused reduce partial sums + clip gradients for 5 Q networks in one launch.

    Replaces 5 sequential gradient_reduce_apply_fused_kernel calls.
    Each block redundantly reduces all 5 partial sum arrays, computes
    5 clip scales, and applies them to its portion of gradients.

    Scales are stored at shared[BLOCK_SIZE-5 .. BLOCK_SIZE-1] after each
    reduction to avoid a second shared allocation (BLOCK_SIZE >= 256).
    """
    var block_id = Int(block_idx.x)
    var thread_id = Int(thread_idx.x)
    var idx = block_id * BLOCK_SIZE + thread_id

    var shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # ── Network 1 ──
    var local_sum = Scalar[dtype](0.0)
    var ps_idx = thread_id
    while ps_idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums_5q[0 * NUM_BLOCKS + ps_idx])
        ps_idx += BLOCK_SIZE
    shared[thread_id] = local_sum
    barrier()
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        var norm = Scalar[dtype](sqrt(rebind[Scalar[dtype]](shared[0])))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        shared[BLOCK_SIZE - 5] = scale
    barrier()

    # ── Network 2 ──
    local_sum = Scalar[dtype](0.0)
    ps_idx = thread_id
    while ps_idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums_5q[1 * NUM_BLOCKS + ps_idx])
        ps_idx += BLOCK_SIZE
    shared[thread_id] = local_sum
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        var norm = Scalar[dtype](sqrt(rebind[Scalar[dtype]](shared[0])))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        shared[BLOCK_SIZE - 4] = scale
    barrier()

    # ── Network 3 ──
    local_sum = Scalar[dtype](0.0)
    ps_idx = thread_id
    while ps_idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums_5q[2 * NUM_BLOCKS + ps_idx])
        ps_idx += BLOCK_SIZE
    shared[thread_id] = local_sum
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        var norm = Scalar[dtype](sqrt(rebind[Scalar[dtype]](shared[0])))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        shared[BLOCK_SIZE - 3] = scale
    barrier()

    # ── Network 4 ──
    local_sum = Scalar[dtype](0.0)
    ps_idx = thread_id
    while ps_idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums_5q[3 * NUM_BLOCKS + ps_idx])
        ps_idx += BLOCK_SIZE
    shared[thread_id] = local_sum
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        var norm = Scalar[dtype](sqrt(rebind[Scalar[dtype]](shared[0])))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        shared[BLOCK_SIZE - 2] = scale
    barrier()

    # ── Network 5 ──
    local_sum = Scalar[dtype](0.0)
    ps_idx = thread_id
    while ps_idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums_5q[4 * NUM_BLOCKS + ps_idx])
        ps_idx += BLOCK_SIZE
    shared[thread_id] = local_sum
    barrier()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_id < stride:
            shared[thread_id] = shared[thread_id] + shared[thread_id + stride]
        barrier()
        stride = stride // 2
    if thread_id == 0:
        var norm = Scalar[dtype](sqrt(rebind[Scalar[dtype]](shared[0])))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        shared[BLOCK_SIZE - 1] = scale
    barrier()

    # Apply all 5 scales to gradients
    if idx < PARAM_SIZE:
        grads1[idx] = grads1[idx] * rebind[Scalar[dtype]](shared[BLOCK_SIZE - 5])
        grads2[idx] = grads2[idx] * rebind[Scalar[dtype]](shared[BLOCK_SIZE - 4])
        grads3[idx] = grads3[idx] * rebind[Scalar[dtype]](shared[BLOCK_SIZE - 3])
        grads4[idx] = grads4[idx] * rebind[Scalar[dtype]](shared[BLOCK_SIZE - 2])
        grads5[idx] = grads5[idx] * rebind[Scalar[dtype]](shared[BLOCK_SIZE - 1])


@always_inline
fn tdmpc2_adam_step_5q_kernel[
    dtype: DType, PARAM_SIZE: Int
](
    params1: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads1: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    state1: LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
    ],
    params2: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads2: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    state2: LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
    ],
    params3: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads3: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    state3: LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
    ],
    params4: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads4: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    state4: LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
    ],
    params5: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    grads5: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    state5: LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
    ],
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    bias_correction1: Scalar[dtype],
    bias_correction2: Scalar[dtype],
):
    """Fused Adam update for 5 Q networks in one kernel launch.

    Replaces 5 sequential Adam.step_gpu calls. Each thread processes
    the same parameter index across all 5 networks.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= PARAM_SIZE:
        return

    var one = Scalar[dtype](1.0)
    var one_minus_b1 = one - beta1
    var one_minus_b2 = one - beta2

    # Q1
    var g1 = rebind[Scalar[dtype]](grads1[idx])
    var m1 = beta1 * rebind[Scalar[dtype]](state1[idx, 0]) + one_minus_b1 * g1
    var v1 = beta2 * rebind[Scalar[dtype]](state1[idx, 1]) + one_minus_b2 * g1 * g1
    state1[idx, 0] = m1
    state1[idx, 1] = v1
    params1[idx] = rebind[Scalar[dtype]](params1[idx]) - lr * (m1 / bias_correction1) / (sqrt(v1 / bias_correction2) + eps)

    # Q2
    var g2 = rebind[Scalar[dtype]](grads2[idx])
    var m2 = beta1 * rebind[Scalar[dtype]](state2[idx, 0]) + one_minus_b1 * g2
    var v2 = beta2 * rebind[Scalar[dtype]](state2[idx, 1]) + one_minus_b2 * g2 * g2
    state2[idx, 0] = m2
    state2[idx, 1] = v2
    params2[idx] = rebind[Scalar[dtype]](params2[idx]) - lr * (m2 / bias_correction1) / (sqrt(v2 / bias_correction2) + eps)

    # Q3
    var g3 = rebind[Scalar[dtype]](grads3[idx])
    var m3 = beta1 * rebind[Scalar[dtype]](state3[idx, 0]) + one_minus_b1 * g3
    var v3 = beta2 * rebind[Scalar[dtype]](state3[idx, 1]) + one_minus_b2 * g3 * g3
    state3[idx, 0] = m3
    state3[idx, 1] = v3
    params3[idx] = rebind[Scalar[dtype]](params3[idx]) - lr * (m3 / bias_correction1) / (sqrt(v3 / bias_correction2) + eps)

    # Q4
    var g4 = rebind[Scalar[dtype]](grads4[idx])
    var m4 = beta1 * rebind[Scalar[dtype]](state4[idx, 0]) + one_minus_b1 * g4
    var v4 = beta2 * rebind[Scalar[dtype]](state4[idx, 1]) + one_minus_b2 * g4 * g4
    state4[idx, 0] = m4
    state4[idx, 1] = v4
    params4[idx] = rebind[Scalar[dtype]](params4[idx]) - lr * (m4 / bias_correction1) / (sqrt(v4 / bias_correction2) + eps)

    # Q5
    var g5 = rebind[Scalar[dtype]](grads5[idx])
    var m5 = beta1 * rebind[Scalar[dtype]](state5[idx, 0]) + one_minus_b1 * g5
    var v5 = beta2 * rebind[Scalar[dtype]](state5[idx, 1]) + one_minus_b2 * g5 * g5
    state5[idx, 0] = m5
    state5[idx, 1] = v5
    params5[idx] = rebind[Scalar[dtype]](params5[idx]) - lr * (m5 / bias_correction1) / (sqrt(v5 / bias_correction2) + eps)


# =============================================================================
# MPPI GPU Kernels
# =============================================================================


@always_inline
fn mppi_broadcast_z0_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
    LATENT_DIM: Int,
](
    z0: LayoutTensor[dtype, Layout.row_major(1, LATENT_DIM), MutAnyOrigin],
    z_all: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, LATENT_DIM), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Broadcast a single z0 [1, LATENT] into [TOTAL_SAMPLES, LATENT].

    One thread per sample, copies LATENT values.

    Args:
        z0: Source latent state [1, LATENT_DIM].
        z_all: Output replicated states [TOTAL_SAMPLES, LATENT_DIM].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return

    for k in range(LATENT_DIM):
        z_all[i, k] = z0[0, k]


@always_inline
fn mppi_sample_actions_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
    NUM_PI_TRAJS: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
    POL_OUT: Int = ACTION_DIM * 2,
](
    pi_out: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, POL_OUT), MutAnyOrigin
    ],
    mean: LayoutTensor[
        dtype, Layout.row_major(HORIZON * ACTION_DIM), MutAnyOrigin
    ],
    std: LayoutTensor[
        dtype, Layout.row_major(HORIZON * ACTION_DIM), MutAnyOrigin
    ],
    act_step: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, ACTION_DIM), MutAnyOrigin
    ],
    all_actions: LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES * HORIZON * ACTION_DIM),
        MutAnyOrigin,
    ],
    step: Int,
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Sample actions for all MPPI candidates at horizon step t.

    For samples [0, NUM_PI_TRAJS): sample from policy output with noise (0.1 std).
    For samples [NUM_PI_TRAJS, TOTAL_SAMPLES): sample from Gaussian(mean[t], std[t]).

    Args:
        pi_out: Policy network output [TOTAL_SAMPLES, POL_OUT] (mean | log_std).
        mean: MPPI distribution mean [H * ACT].
        std: MPPI distribution std [H * ACT].
        act_step: Output actions for current step [TOTAL_SAMPLES, ACT].
        all_actions: Full action storage [TOTAL_SAMPLES * H * ACT] (written at step t).
        step: Current horizon step index (0..H-1).
        rng_seed: Random seed.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return

    for j in range(ACTION_DIM):
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(i) * UInt64(ACTION_DIM) + UInt64(j),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var u1 = Scalar[DType.float32](rand_vals[0]) + 1e-8
        var u2 = Scalar[DType.float32](rand_vals[1])
        var mag = sqrt(Float32(-2.0) * log(u1))
        var noise = Scalar[dtype](
            mag * cos(u2 * Scalar[DType.float32](6.283185307179586))
        )

        var act: Scalar[dtype]
        if i < NUM_PI_TRAJS:
            # Policy trajectory: use policy mean + small noise
            var pi_mean = Scalar[dtype](pi_out[i, j][0])
            act = pi_mean + noise * Scalar[dtype](0.1)
        else:
            # MPPI trajectory: sample from Gaussian(mean[t,j], std[t,j])
            var mu = Scalar[dtype](mean[step * ACTION_DIM + j][0])
            var sigma = Scalar[dtype](std[step * ACTION_DIM + j][0])
            act = mu + sigma * noise

        # Clamp to [-1, 1]
        if act < Scalar[dtype](-1.0):
            act = Scalar[dtype](-1.0)
        if act > Scalar[dtype](1.0):
            act = Scalar[dtype](1.0)

        act_step[i, j] = act
        # Store into all_actions[i * H * ACT + step * ACT + j]
        all_actions[i * HORIZON * ACTION_DIM + step * ACTION_DIM + j] = act


@always_inline
fn mppi_accumulate_reward_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
    BINS: Int,
](
    rew_logits: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, BINS), MutAnyOrigin
    ],
    bins: LayoutTensor[dtype, Layout.row_major(BINS), MutAnyOrigin],
    returns: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
    ],
    discount: Scalar[dtype],
) where dtype.is_floating_point():
    """Decode reward logits and accumulate discounted reward into returns.

    returns[i] += discount * symexp(softmax(logits[i]) · bins)

    Args:
        rew_logits: Reward network logits [TOTAL_SAMPLES, BINS].
        bins: Bin centers [BINS] (in symlog space).
        returns: Running return accumulator [TOTAL_SAMPLES] (updated in-place).
        discount: Discount factor gamma^t for this step.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return

    # Softmax + dot product with bins
    var max_l = Scalar[dtype](rew_logits[i, 0][0])
    for k in range(1, BINS):
        var v = Scalar[dtype](rew_logits[i, k][0])
        if v > max_l:
            max_l = v

    var sum_exp = Scalar[dtype](0.0)
    for k in range(BINS):
        sum_exp = sum_exp + exp(Scalar[dtype](rew_logits[i, k][0]) - max_l)

    var val_symlog = Scalar[dtype](0.0)
    for k in range(BINS):
        var prob = exp(Scalar[dtype](rew_logits[i, k][0]) - max_l) / sum_exp
        val_symlog = val_symlog + prob * Scalar[dtype](bins[k][0])

    # symexp: convert from symlog to actual value space
    var reward_val = _symexp[dtype](val_symlog)
    returns[i] = returns[i] + discount * reward_val


@always_inline
fn mppi_add_terminal_value_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
](
    q_min: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
    ],
    returns: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
    ],
    discount: Scalar[dtype],
) where dtype.is_floating_point():
    """Add discounted terminal Q-value to returns.

    returns[i] += discount * q_min[i]

    Args:
        q_min: Min-Q terminal values [TOTAL_SAMPLES].
        returns: Running return accumulator [TOTAL_SAMPLES] (updated in-place).
        discount: Discount factor gamma^H.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return

    returns[i] = returns[i] + discount * q_min[i]


@always_inline
fn mppi_copy_z_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
    LATENT_DIM: Int,
](
    dst: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, LATENT_DIM), MutAnyOrigin
    ],
    src: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, LATENT_DIM), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Copy latent states: dst[i] = src[i].

    Args:
        dst: Destination latent buffer [TOTAL_SAMPLES, LATENT].
        src: Source latent buffer [TOTAL_SAMPLES, LATENT].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return

    for k in range(LATENT_DIM):
        dst[i, k] = src[i, k]
