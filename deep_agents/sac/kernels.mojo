"""SAC-specific GPU kernels for reparameterization trick.

## SAC Reparameterization
- sac_reparameterize_kernel:     a = tanh(μ + exp(logσ)*ε), log_prob (reference, no cache)
- sac_sample_actions_kernel:     inference — sample from actor_out[N, 2*ACTION_DIM]
- sac_rsample_with_cache_kernel: training forward — sample + log_prob + save eps for backward
- sac_rsample_bwd_kernel:        training backward — grad through reparameterization trick
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.math import exp, log, tanh, sqrt, cos
from std.random.philox import Random as PhiloxRandom


# =============================================================================
# SAC Reparameterization
# =============================================================================


@always_inline
fn sac_reparameterize_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_std_min: Scalar[dtype],
    log_std_max: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
):
    """SAC reparameterization trick with tanh squashing and Jacobian log-prob correction.

    Computes:
        ε ~ N(0, 1)
        z = μ + exp(log_σ) * ε         (reparameterization)
        a = tanh(z)                     (squashed action in (-1, 1))
        log π(a|s) = Σ_j [ -0.5*ε_j² - 0.5*log(2π) - log_σ_j
                            - log(1 - tanh²(z_j)) ]  (change-of-variables)

    Uses PhiloxRandom for GPU-safe noise generation .
    One thread per batch sample.

    Args:
        actions:    Output squashed actions in (-1, 1) [BATCH, ACTION_DIM].
        log_probs:  Output log-probabilities (summed over action dims) [BATCH].
        mean:       Actor output mean [BATCH, ACTION_DIM].
        log_std:    Actor output log_std (before clamping) [BATCH, ACTION_DIM].
        log_std_min: Minimum log_std clamp (e.g. -20).
        log_std_max: Maximum log_std clamp (e.g. 2).
        rng_seed:   Random seed (should vary per call).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var half_log_2pi = Scalar[dtype](0.9189385332046727)  # 0.5 * log(2π)
    var one = Scalar[dtype](1.0)
    var half = Scalar[dtype](0.5)
    var ls_range = log_std_max - log_std_min

    var lp: log_probs.element_type = 0.0

    for a in range(ACTION_DIM):
        # Affine rescale: tanh already applied by LinearTanh head
        var tanh_out = log_std[b, a]
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
        var mag = sqrt(-2.0 * log(u1))
        var eps = Scalar[dtype](mag * cos(u2 * 6.283185307179586))

        # Reparameterize: z = μ + σ * ε
        var z = mean[b, a] + std_val * eps

        # Squash: a = tanh(z)
        var act = tanh(z)
        actions[b, a] = act

        # Log-prob: -0.5*(ε²) - 0.5*log(2π) - log_σ - log(1 - tanh²(z))
        var one_minus_tanh2 = one - act * act
        # Clamp to avoid log(0)
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
fn sac_sample_actions_kernel[
    dtype: DType where dtype.is_floating_point(),
    N: Int,
    ACTION_DIM: Int,
](
    actions: LayoutTensor[dtype, Layout.row_major(N, ACTION_DIM), MutAnyOrigin],
    actor_out: LayoutTensor[
        dtype, Layout.row_major(N, ACTION_DIM + ACTION_DIM), MutAnyOrigin
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
fn sac_rsample_with_cache_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    eps_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM + ACTION_DIM), MutAnyOrigin
    ],
    log_std_min: Scalar[dtype],
    log_std_max: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
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
    One thread per batch sample.

    Args:
        actions:    Output actions in (-1, 1) [BATCH, ACTION_DIM].
        log_probs:  Output log-probabilities (summed over action dims) [BATCH].
        eps_cache:  Output saved noise ε [BATCH, ACTION_DIM] (for backward).
        actor_out:  Actor network output [BATCH, 2*ACTION_DIM] (mean || log_std).
        log_std_min: Minimum log_std clamp value.
        log_std_max: Maximum log_std clamp value.
        rng_seed:   Random seed (should vary per call).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

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
fn sac_rsample_bwd_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    ACTION_DIM: Int,
](
    actor_grad: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM + ACTION_DIM), MutAnyOrigin
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
        dtype, Layout.row_major(BATCH, ACTION_DIM + ACTION_DIM), MutAnyOrigin
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
# min(Q1, Q2) masked gradient kernels
# =============================================================================


@always_inline
fn min_q_dq_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
](
    dq1: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    dq2: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    q1: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    q2: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
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

    var neg_inv_batch = Scalar[dtype](-1.0 / Float64(BATCH))
    var zero = Scalar[dtype](0.0)

    if q1[b, 0] <= q2[b, 0]:
        dq1[b, 0] = neg_inv_batch
        dq2[b, 0] = zero
    else:
        dq1[b, 0] = zero
        dq2[b, 0] = neg_inv_batch


@always_inline
fn add_ci_grads_kernel[
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
