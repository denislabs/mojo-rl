"""DreamerV3 GPU Kernels.

Custom GPU kernels for DreamerV3-specific operations that aren't covered
by the standard forward_gpu/backward_gpu network methods:
  - GRU gating (DreamerV3's non-standard 3-gate update)
  - Categorical sampling with Gumbel-max and uniform mixture
  - KL divergence between categorical distributions
  - Feature concatenation (deter || stoch)
  - Symlog/symexp elementwise transforms
  - Action normalization
  - Lambda returns (backward scan)
  - Return normalization (percentile EMA)
  - Two-hot cross-entropy gradient
  - MSE gradient (decoder loss)
  - BCE gradient (continue loss)
  - Tanh-normal sampling + log probability
  - REINFORCE gradient computation
"""

from std.math import exp, log, abs, sqrt
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.memory import AddressSpace
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from std.random.philox import Random as PhiloxRandom
from mojo_rl.nn.constants import dtype

comptime TPB = 256

# Helper alias for scalar type
comptime S = Scalar[dtype]

# Helper: read element from 2D LayoutTensor as Scalar[dtype]
@always_inline
fn _rd2[
    R: Int, C: Int
](t: LayoutTensor[dtype, Layout.row_major(R, C), MutAnyOrigin], r: Int, c: Int) -> Scalar[dtype]:
    return rebind[Scalar[dtype]](t[r, c])


# =============================================================================
# Symlog / Symexp
# =============================================================================


@always_inline
fn symlog_kernel[
    SIZE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Elementwise symlog: sign(x) * log(1 + |x|)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    var x = rebind[S](input[i])
    if x >= S(0.0):
        output[i] = log(S(1.0) + x)
    else:
        output[i] = -log(S(1.0) - x)


@always_inline
fn symexp_kernel[
    SIZE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Elementwise symexp: sign(x) * (exp(|x|) - 1)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    var x = rebind[S](input[i])
    # Clamp to prevent exp overflow (float32 max ~88)
    if x > S(20.0):
        x = S(20.0)
    if x < S(-20.0):
        x = S(-20.0)
    if x >= S(0.0):
        output[i] = exp(x) - S(1.0)
    else:
        output[i] = -(exp(-x) - S(1.0))


# =============================================================================
# GRU Gating
# =============================================================================


@always_inline
fn gru_gate_kernel[
    BATCH: Int,
    DETER: Int,
](
    new_deter: LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ],
    prev_deter: LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ],
    gate_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * DETER), MutAnyOrigin
    ],
):
    """Apply DreamerV3 GRU gating.

    gate_out contains [reset, candidate, update] each of size DETER.
    reset = sigmoid(gate[0:D])
    cand = tanh(reset * gate[D:2D])
    update = sigmoid(gate[2D:3D] - 1)  # bias toward keeping old state
    new_deter = update * cand + (1 - update) * prev_deter
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * DETER:
        return
    var b = tid // DETER
    var i = tid % DETER

    var one = S(1.0)

    # Reset gate (clamp for numerical stability)
    var reset_logit = _rd2[BATCH, 3 * DETER](gate_out, b, i)
    if reset_logit > S(15.0):
        reset_logit = S(15.0)
    if reset_logit < S(-15.0):
        reset_logit = S(-15.0)
    var reset_val = one / (one + exp(-reset_logit))

    # Candidate (clamp input to prevent exp overflow → NaN in tanh)
    var cand_logit = _rd2[BATCH, 3 * DETER](gate_out, b, DETER + i)
    var rc = reset_val * cand_logit
    if rc > S(15.0):
        rc = S(15.0)
    if rc < S(-15.0):
        rc = S(-15.0)
    var exp_rc = exp(rc)
    var exp_neg_rc = exp(-rc)
    var cand_val = (exp_rc - exp_neg_rc) / (exp_rc + exp_neg_rc)

    # Update gate (biased toward keeping old state, clamped)
    var update_logit = _rd2[BATCH, 3 * DETER](gate_out, b, 2 * DETER + i)
    var update_in = update_logit - one
    if update_in > S(15.0):
        update_in = S(15.0)
    if update_in < S(-15.0):
        update_in = S(-15.0)
    var update_val = one / (one + exp(-update_in))

    # New deterministic state
    var pd = _rd2[BATCH, DETER](prev_deter, b, i)
    new_deter[b, i] = update_val * cand_val + (one - update_val) * pd


# =============================================================================
# Concatenation
# =============================================================================


@always_inline
fn concat_feat_kernel[
    BATCH: Int,
    DETER: Int,
    STOCH: Int,
](
    feat: LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER + STOCH), MutAnyOrigin
    ],
    deter: LayoutTensor[dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin],
    stoch: LayoutTensor[dtype, Layout.row_major(BATCH, STOCH), MutAnyOrigin],
):
    """Concatenate deter and stoch into feat = [deter || stoch]."""
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime FEAT = DETER + STOCH
    if tid >= BATCH * FEAT:
        return
    var b = tid // FEAT
    var j = tid % FEAT
    if j < DETER:
        feat[b, j] = _rd2[BATCH, DETER](deter, b, j)
    else:
        feat[b, j] = _rd2[BATCH, STOCH](stoch, b, j - DETER)


@always_inline
fn concat_gru_input_kernel[
    BATCH: Int,
    DETER: Int,
    HIDDEN: Int,
](
    concat_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER + 3 * HIDDEN), MutAnyOrigin
    ],
    deter: LayoutTensor[dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin],
    proj_d: LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    proj_s: LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    proj_a: LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
):
    """Concatenate [deter, proj_d, proj_s, proj_a] for GRU input."""
    comptime TOTAL = DETER + 3 * HIDDEN
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * TOTAL:
        return
    var b = tid // TOTAL
    var j = tid % TOTAL
    if j < DETER:
        concat_out[b, j] = _rd2[BATCH, DETER](deter, b, j)
    elif j < DETER + HIDDEN:
        concat_out[b, j] = _rd2[BATCH, HIDDEN](proj_d, b, j - DETER)
    elif j < DETER + 2 * HIDDEN:
        concat_out[b, j] = _rd2[BATCH, HIDDEN](proj_s, b, j - DETER - HIDDEN)
    else:
        concat_out[b, j] = _rd2[BATCH, HIDDEN](proj_a, b, j - DETER - 2 * HIDDEN)


@always_inline
fn concat_deter_embed_kernel[
    BATCH: Int,
    DETER: Int,
    STOCH: Int,
](
    concat_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER + STOCH), MutAnyOrigin
    ],
    deter: LayoutTensor[dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin],
    embed: LayoutTensor[dtype, Layout.row_major(BATCH, STOCH), MutAnyOrigin],
):
    """Concatenate [deter, embed] for posterior input."""
    comptime TOTAL = DETER + STOCH
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * TOTAL:
        return
    var b = tid // TOTAL
    var j = tid % TOTAL
    if j < DETER:
        concat_out[b, j] = _rd2[BATCH, DETER](deter, b, j)
    else:
        concat_out[b, j] = _rd2[BATCH, STOCH](embed, b, j - DETER)


# =============================================================================
# Action Normalization
# =============================================================================


@always_inline
fn action_normalize_kernel[
    BATCH: Int,
    ACTION_DIM: Int,
](
    output: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    input: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
):
    """Normalize action: a /= max(1, |a|) elementwise."""
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return
    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM
    var val = _rd2[BATCH, ACTION_DIM](input, b, a)
    var abs_val = abs(val)
    if abs_val > S(1.0):
        output[b, a] = val / abs_val
    else:
        output[b, a] = val


# =============================================================================
# Categorical Sampling with Gumbel-Max + Unimix
# =============================================================================


@always_inline
fn categorical_sample_kernel[
    BATCH: Int,
    STOCH_DIM: Int,
    CLASSES: Int,
    UNIMIX: Float64 = 0.01,
](
    output: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    rng_seed: Scalar[DType.uint32],
    training: Scalar[DType.bool],
):
    """Sample from categorical distributions with uniform mixture.

    One thread per (batch, stoch_dim) pair.
    """
    comptime SC = STOCH_DIM * CLASSES
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * STOCH_DIM:
        return
    var b = tid // STOCH_DIM
    var s = tid % STOCH_DIM
    var base = s * CLASSES

    var unimix_val = S(UNIMIX)
    var one_minus_unimix = S(1.0 - UNIMIX)
    var uniform_prob = S(1.0 / Float64(CLASSES))
    var one = S(1.0)
    var eps = S(1e-8)

    # Softmax over CLASSES
    var max_val = _rd2[BATCH, SC](logits, b, base)
    for c in range(1, CLASSES):
        var v = _rd2[BATCH, SC](logits, b, base + c)
        if v > max_val:
            max_val = v

    var sum_exp = S(0.0)
    for c in range(CLASSES):
        var e = exp(_rd2[BATCH, SC](logits, b, base + c) - max_val)
        probs[b, base + c] = e
        sum_exp += e

    # Normalize + unimix
    for c in range(CLASSES):
        var softmax_p = _rd2[BATCH, SC](probs, b, base + c) / sum_exp
        probs[b, base + c] = one_minus_unimix * softmax_p + unimix_val * uniform_prob

    # Sample
    var best_idx = 0
    if training:
        # Gumbel-max trick
        var best_score = S(-1e10)
        for c in range(CLASSES):
            var p = _rd2[BATCH, SC](probs, b, base + c)
            var philox = PhiloxRandom(
                seed=UInt64(rng_seed) + UInt64(tid) * UInt64(CLASSES) + UInt64(c),
                offset=0,
            )
            var rand_vals = philox.step_uniform()
            var u = S(Float32(rand_vals[0]))
            if u < S(0.0001):
                u = S(0.0001)
            if u > S(0.9999):
                u = S(0.9999)
            var gumbel = -log(-log(u))
            var score = log(p + eps) + gumbel
            if score > best_score:
                best_score = score
                best_idx = Int(c)
    else:
        # Argmax
        var best_p = _rd2[BATCH, SC](probs, b, base)
        for c in range(1, CLASSES):
            var p = _rd2[BATCH, SC](probs, b, base + c)
            if p > best_p:
                best_p = p
                best_idx = Int(c)

    # Write one-hot
    for c in range(CLASSES):
        output[b, base + c] = S(0.0)
    output[b, base + best_idx] = one


# =============================================================================
# KL Divergence
# =============================================================================


@always_inline
fn kl_divergence_kernel[
    BATCH: Int,
    STOCH_DIM: Int,
    CLASSES: Int,
](
    kl_out: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    post_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    prior_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
):
    """KL(posterior || prior) per batch element."""
    comptime SC = STOCH_DIM * CLASSES
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var eps = S(1e-8)
    var total = S(0.0)
    for s in range(STOCH_DIM):
        for c in range(CLASSES):
            var idx = s * CLASSES + c
            var p = _rd2[BATCH, SC](post_probs, b, idx)
            var q = _rd2[BATCH, SC](prior_probs, b, idx)
            if p > eps:
                total += p * (log(p + eps) - log(q + eps))
    kl_out[b] = total


@always_inline
fn kl_categorical_gradient_kernel[
    BATCH: Int,
    STOCH_DIM: Int,
    CLASSES: Int,
](
    grad_post_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    grad_prior_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    post_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    prior_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_DIM * CLASSES), MutAnyOrigin
    ],
    kl_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    free_nats: Scalar[dtype],
    dyn_scale: Scalar[dtype],
    rep_scale: Scalar[dtype],
    inv_batch: Scalar[dtype],
):
    """Gradient of KL divergence w.r.t. posterior and prior logits.

    Uses dual KL balancing (DreamerV3):
    - dyn_scale (0.5): gradient to prior (dynamics loss, stop-grad posterior)
    - rep_scale (0.1): gradient to posterior (representation loss, stop-grad prior)
    Free nats gating: zero gradient if KL < free_nats.

    Gradient of KL w.r.t. softmax probs:
      dKL/d(post_p_i) = log(post_p_i / prior_p_i) + 1 - sum_j(post_p_j) [=0 for probs]
                       ≈ log(post_p_i) - log(prior_p_i) (since probs sum to 1)
      For prior: dKL/d(prior_p_i) = -post_p_i / prior_p_i

    Softmax Jacobian: d(softmax_i)/d(logit_j) = p_i * (delta_ij - p_j)
    Combined: grad_logit_j = sum_i [dKL/dp_i * p_i * (delta_ij - p_j)]
    For posterior: grad_logit_j = p_j * (log(p_j/q_j) + 1) - p_j * sum_i[p_i * (log(p_i/q_i) + 1)]
                                = p_j * [(log(p_j/q_j) + 1) - sum_i p_i*(log(p_i/q_i) + 1)]
                                = p_j * [log(p_j/q_j) - KL]  (since sum_i p_i*log(p_i/q_i) = KL)
    For prior: grad_logit_j = q_j * (p_j/q_j) - q_j * sum_i[p_i]
                             = p_j - q_j  (since sum_i p_i = 1)
    So: prior grad = (post_p - prior_p), posterior grad = post_p * (log(post_p/prior_p) - KL)
    """
    comptime SC = STOCH_DIM * CLASSES
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * STOCH_DIM:
        return
    var b = tid // STOCH_DIM
    var s = tid % STOCH_DIM
    var base = s * CLASSES
    var eps = S(1e-8)

    var kl_val = rebind[S](kl_values[b])

    if kl_val <= free_nats:
        # Below free nats threshold — zero gradient
        for c in range(CLASSES):
            grad_post_logits[b, base + c] = S(0.0)
            grad_prior_logits[b, base + c] = S(0.0)
        return

    # Compute per-category KL contribution for this stoch dim
    # to get the mean KL across categories for centering
    var kl_s = S(0.0)
    for c in range(CLASSES):
        var p = _rd2[BATCH, SC](post_probs, b, base + c)
        var q = _rd2[BATCH, SC](prior_probs, b, base + c)
        if p > eps:
            kl_s += p * (log(p + eps) - log(q + eps))

    for c in range(CLASSES):
        var p = _rd2[BATCH, SC](post_probs, b, base + c)
        var q = _rd2[BATCH, SC](prior_probs, b, base + c)

        # Prior gradient (dynamics loss): push prior toward posterior
        # d(KL)/d(prior_logit_j) = -(p_j - q_j) = q_j - p_j
        # We want to minimize KL, so gradient is (q_j - p_j) but since
        # we're computing loss gradient: prior_grad = (p - q) (for descent)
        var prior_g = dyn_scale * (p - q) * inv_batch

        # Posterior gradient (representation loss): push posterior toward prior
        # d(KL)/d(post_logit_j) = p_j * (log(p_j/q_j) - KL_s)
        var log_ratio = log(p + eps) - log(q + eps)
        var post_g = rep_scale * p * (log_ratio - kl_s) * inv_batch

        grad_prior_logits[b, base + c] = prior_g
        grad_post_logits[b, base + c] = post_g


# =============================================================================
# Lambda Returns (Backward Scan)
# =============================================================================


@always_inline
fn lambda_returns_kernel[
    HORIZON: Int,
    BATCH: Int,
](
    returns: LayoutTensor[
        dtype, Layout.row_major(HORIZON, BATCH), MutAnyOrigin
    ],
    rewards: LayoutTensor[
        dtype, Layout.row_major(HORIZON, BATCH), MutAnyOrigin
    ],
    values: LayoutTensor[
        dtype, Layout.row_major(HORIZON, BATCH), MutAnyOrigin
    ],
    continues: LayoutTensor[
        dtype, Layout.row_major(HORIZON, BATCH), MutAnyOrigin
    ],
    gamma: Scalar[dtype],
    lambda_: Scalar[dtype],
):
    """Compute lambda returns via backward scan.

    One thread per batch element; sequential across horizon.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var one = S(1.0)

    # Bootstrap from last value
    returns[HORIZON - 1, b] = _rd2[HORIZON, BATCH](values, HORIZON - 1, b)

    # Backward scan
    for h_off in range(HORIZON - 1):
        var h = HORIZON - 2 - h_off
        var r = _rd2[HORIZON, BATCH](rewards, h, b)
        var c = _rd2[HORIZON, BATCH](continues, h, b)
        var v_next = _rd2[HORIZON, BATCH](values, h + 1, b)
        var ret_next = _rd2[HORIZON, BATCH](returns, h + 1, b)
        returns[h, b] = r + gamma * c * (
            (one - lambda_) * v_next + lambda_ * ret_next
        )


# =============================================================================
# Return Normalization
# =============================================================================


@always_inline
fn normalize_returns_elementwise_kernel[
    SIZE: Int,
](
    returns: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    lo: Scalar[dtype],
    inv_scale: Scalar[dtype],
):
    """Normalize returns: returns = (returns - lo) / scale."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    returns[i] = (rebind[S](returns[i]) - lo) * inv_scale


# =============================================================================
# Two-Hot Cross-Entropy Gradient
# =============================================================================


@always_inline
fn two_hot_ce_grad_kernel[
    BATCH: Int,
    NUM_BINS: Int,
](
    grad_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, NUM_BINS), MutAnyOrigin
    ],
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, NUM_BINS), MutAnyOrigin
    ],
    targets: LayoutTensor[
        dtype, Layout.row_major(BATCH, NUM_BINS), MutAnyOrigin
    ],
    inv_batch: Scalar[dtype],
):
    """Gradient of two-hot cross-entropy: softmax(logits) - target, scaled."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var max_val = _rd2[BATCH, NUM_BINS](logits, b, 0)
    for k in range(1, NUM_BINS):
        var v = _rd2[BATCH, NUM_BINS](logits, b, k)
        if v > max_val:
            max_val = v

    var sum_exp = S(0.0)
    for k in range(NUM_BINS):
        sum_exp += exp(_rd2[BATCH, NUM_BINS](logits, b, k) - max_val)

    for k in range(NUM_BINS):
        var softmax_k = exp(_rd2[BATCH, NUM_BINS](logits, b, k) - max_val) / sum_exp
        grad_out[b, k] = (softmax_k - _rd2[BATCH, NUM_BINS](targets, b, k)) * inv_batch


# =============================================================================
# MSE Gradient (Decoder Loss)
# =============================================================================


@always_inline
fn mse_grad_kernel[
    SIZE: Int,
](
    grad: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    pred: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    target: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    scale: Scalar[dtype],
):
    """MSE gradient: 2 * (pred - target) * scale, elementwise."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    grad[i] = S(2.0) * (rebind[S](pred[i]) - rebind[S](target[i])) * scale


# =============================================================================
# BCE Gradient (Continue Loss)
# =============================================================================


@always_inline
fn bce_grad_kernel[
    BATCH: Int,
](
    grad: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    pred: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    target: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    inv_batch: Scalar[dtype],
):
    """BCE gradient for continue prediction."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var eps = S(1e-6)
    var p = _rd2[BATCH, 1](pred, b, 0)
    if p < eps:
        p = eps
    if p > S(1.0) - eps:
        p = S(1.0) - eps
    grad[b, 0] = (p - _rd2[BATCH, 1](target, b, 0)) * inv_batch


# =============================================================================
# Tanh-Normal Sampling
# =============================================================================


@always_inline
fn tanh_normal_sample_kernel[
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * ACTION_DIM), MutAnyOrigin
    ],
    rng_seed: Scalar[DType.uint32],
):
    """Sample from tanh-normal distribution and compute log probability.

    actor_out = [mean, log_std] each of size ACTION_DIM.
    One thread per batch element.
    """
    comptime AD2 = 2 * ACTION_DIM
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var one = S(1.0)
    var eps = S(1e-6)
    var pi_const = S(0.9189385332046727)  # 0.5 * log(2*pi)
    var total_lp = S(0.0)

    for a in range(ACTION_DIM):
        var mean_val = _rd2[BATCH, AD2](actor_out, b, a)
        var log_std_val = _rd2[BATCH, AD2](actor_out, b, ACTION_DIM + a)

        if log_std_val < S(-5.0):
            log_std_val = S(-5.0)
        if log_std_val > S(2.0):
            log_std_val = S(2.0)

        var std_val = exp(log_std_val)
        if std_val < eps:
            std_val = eps

        # Box-Muller from Philox
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(b) * UInt64(ACTION_DIM) + UInt64(a),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var u1 = S(Float32(rand_vals[0]))
        var u2 = S(Float32(rand_vals[1]))
        if u1 < S(0.0001):
            u1 = S(0.0001)
        if u1 > S(0.9999):
            u1 = S(0.9999)
        var neg2log = S(-2.0) * log(u1)
        var z: S
        if neg2log > S(0.0):
            var sq = exp(S(0.5) * log(neg2log))
            z = sq * (S(2.0) * u2 - one)
        else:
            z = S(0.0)

        var pre_tanh = mean_val + std_val * z

        # tanh (clamp to prevent exp overflow)
        if pre_tanh > S(15.0):
            pre_tanh = S(15.0)
        if pre_tanh < S(-15.0):
            pre_tanh = S(-15.0)
        var ep = exp(pre_tanh)
        var en = exp(-pre_tanh)
        var action_val = (ep - en) / (ep + en)

        if action_val > one:
            action_val = one
        if action_val < -one:
            action_val = -one

        actions[b, a] = action_val

        # Log probability
        var z_norm = (pre_tanh - mean_val) / std_val
        var log_normal = S(-0.5) * z_norm * z_norm - log_std_val - pi_const
        var log_det = log(one - action_val * action_val + eps)
        total_lp += log_normal - log_det

    log_probs[b] = total_lp


# =============================================================================
# REINFORCE Actor Gradient
# =============================================================================


@always_inline
fn reinforce_grad_kernel[
    BATCH: Int,
    ACTION_DIM: Int,
](
    grad_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * ACTION_DIM), MutAnyOrigin
    ],
    actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * ACTION_DIM), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    inv_batch: Scalar[dtype],
    entropy_coef: Scalar[dtype],
):
    """Compute REINFORCE gradient w.r.t. actor output (mean, log_std).

    One thread per batch element.
    """
    comptime AD2 = 2 * ACTION_DIM
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var one = S(1.0)
    var eps = S(1e-6)
    var advantage = rebind[S](advantages[b])

    for a in range(ACTION_DIM):
        var mean_val = _rd2[BATCH, AD2](actor_out, b, a)
        var log_std_val = _rd2[BATCH, AD2](actor_out, b, ACTION_DIM + a)
        if log_std_val < S(-5.0):
            log_std_val = S(-5.0)
        if log_std_val > S(2.0):
            log_std_val = S(2.0)

        var std_val = exp(log_std_val)
        if std_val < eps:
            std_val = eps

        var action_val = _rd2[BATCH, ACTION_DIM](actions, b, a)
        if action_val > one - eps:
            action_val = one - eps
        if action_val < -one + eps:
            action_val = -one + eps

        # atanh
        var pre_tanh = S(0.5) * log((one + action_val) / (one - action_val))
        var z = (pre_tanh - mean_val) / std_val

        var grad_mean = z / std_val
        var grad_log_std = z * z - one

        var weight = (-advantage + entropy_coef) * inv_batch
        grad_out[b, a] = weight * grad_mean
        grad_out[b, ACTION_DIM + a] = weight * grad_log_std


# =============================================================================
# Decode Distributional Value
# =============================================================================


@always_inline
fn decode_value_kernel[
    BATCH: Int,
    NUM_BINS: Int,
](
    values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, NUM_BINS), MutAnyOrigin
    ],
    bins: LayoutTensor[dtype, Layout.row_major(NUM_BINS), MutAnyOrigin],
    apply_symexp: Scalar[DType.bool],
):
    """Decode distributional value: symexp(sum(softmax(logits) * bins))."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var one = S(1.0)

    var max_val = _rd2[BATCH, NUM_BINS](logits, b, 0)
    for k in range(1, NUM_BINS):
        var v = _rd2[BATCH, NUM_BINS](logits, b, k)
        if v > max_val:
            max_val = v

    var sum_exp = S(0.0)
    for k in range(NUM_BINS):
        sum_exp += exp(_rd2[BATCH, NUM_BINS](logits, b, k) - max_val)

    var value_symlog = S(0.0)
    for k in range(NUM_BINS):
        var prob = exp(_rd2[BATCH, NUM_BINS](logits, b, k) - max_val) / sum_exp
        value_symlog += prob * rebind[S](bins[k])

    if apply_symexp:
        # Clamp to prevent exp overflow (float32 max ~88)
        if value_symlog > S(20.0):
            value_symlog = S(20.0)
        if value_symlog < S(-20.0):
            value_symlog = S(-20.0)
        if value_symlog >= S(0.0):
            values[b] = exp(value_symlog) - one
        else:
            values[b] = -(exp(-value_symlog) - one)
    else:
        values[b] = value_symlog


# =============================================================================
# Two-Hot Encoding
# =============================================================================


@always_inline
fn two_hot_encode_kernel[
    BATCH: Int,
    NUM_BINS: Int,
](
    targets: LayoutTensor[
        dtype, Layout.row_major(BATCH, NUM_BINS), MutAnyOrigin
    ],
    values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    bins: LayoutTensor[dtype, Layout.row_major(NUM_BINS), MutAnyOrigin],
):
    """Two-hot encode scalar values into distribution targets."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var one = S(1.0)
    var v_min = rebind[S](bins[0])
    var v_max = rebind[S](bins[NUM_BINS - 1])
    var x = rebind[S](values[b])

    if x < v_min:
        x = v_min
    if x > v_max:
        x = v_max

    # Zero out
    for k in range(NUM_BINS):
        targets[b, k] = S(0.0)

    if NUM_BINS == 1:
        targets[b, 0] = one
        return

    var step = (v_max - v_min) / S(Float64(NUM_BINS - 1))
    var k_float = (x - v_min) / step
    var k = Int(k_float)
    if k >= NUM_BINS - 1:
        k = NUM_BINS - 2

    var bin_low = rebind[S](bins[k])
    var bin_high = rebind[S](bins[k + 1])
    var width = bin_high - bin_low
    var eps = S(1e-8)

    if width < eps:
        targets[b, k] = one
        return

    var upper_weight = (bin_high - x) / width
    targets[b, k] = upper_weight
    targets[b, k + 1] = one - upper_weight


# =============================================================================
# Sigmoid Kernel (for continue head output)
# =============================================================================


@always_inline
fn sigmoid_kernel[
    SIZE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Elementwise sigmoid."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    var x = rebind[S](input[i])
    # Clamp to prevent exp overflow
    if x > S(20.0):
        x = S(20.0)
    if x < S(-20.0):
        x = S(-20.0)
    output[i] = S(1.0) / (S(1.0) + exp(-x))


# =============================================================================
# Copy Buffer Kernel
# =============================================================================


@always_inline
fn copy_kernel[
    SIZE: Int,
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Copy src to dst elementwise."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    dst[i] = src[i]


# =============================================================================
# Zero Buffer Kernel
# =============================================================================


@always_inline
fn zero_kernel[
    SIZE: Int,
](
    buf: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Zero a buffer."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    buf[i] = S(0.0)


# =============================================================================
# Advantage Computation
# =============================================================================


@always_inline
fn advantage_kernel[
    SIZE: Int,
](
    adv: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Elementwise advantage = returns - values."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    adv[i] = rebind[S](returns[i]) - rebind[S](values[i])


# =============================================================================
# Gradient clipping kernels
# =============================================================================


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
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if idx < PARAM_SIZE:
        var g = grads[idx]
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
        partial_sums[block_id] = shared[0]


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
    all partial_sums, then applies the computed scale to its portion of grads.
    """
    var block_id = Int(block_idx.x)
    var thread_id = Int(thread_idx.x)
    var idx = block_id * BLOCK_SIZE + thread_id

    var shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var local_sum = Scalar[dtype](0.0)
    var ps_idx = thread_id
    while ps_idx < NUM_BLOCKS:
        local_sum += rebind[Scalar[dtype]](partial_sums[ps_idx])
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
        var total_sq_sum = rebind[Scalar[dtype]](shared[0])
        var norm = Scalar[dtype](sqrt(total_sq_sum))
        var scale = Scalar[dtype](1.0)
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + Scalar[dtype](1e-8))
        shared[1] = scale

    barrier()

    if idx < PARAM_SIZE:
        grads[idx] = grads[idx] * rebind[Scalar[dtype]](shared[1])
