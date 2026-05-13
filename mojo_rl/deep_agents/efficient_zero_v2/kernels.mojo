"""EfficientZero V2 GPU kernels.

Companion file to `efficient_zero_v2.mojo`'s `train_step_gpu`. The forward
+ backward + optimizer pieces use `Network.forward_gpu/backward_gpu` and
`Optimizer.step_gpu`, which already exist. The bits glued in between —
turning per-sample CE / cosine losses into per-output upstream gradients
and small scalar accumulators — are EZ-V2-specific and live here.

Naming: every kernel is prefixed with `ezv2_` so inter-file searches
stay unambiguous (the MuZero `kernels.mojo` already owns the unprefixed
CE / scaling / sampling kernels).

Convention: all kernels operate on **per-time-slice** buffers (size
`[BATCH * X]`). The host caller re-views the larger time-major scratch
tensors (e.g. `[(K+1) * BATCH * PRED_OUT]`) via `LayoutTensor` ptr-offset
arithmetic before each launch. Mirrors the MuZero kernel pattern and
avoids comptime-baking the unroll position into every kernel
specialization.
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.math import cos, exp, log, sqrt, tanh
from std.random.philox import Random as PhiloxRandom


comptime TPB: Int = 256


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample obs gather: src_obs_window[b, k, :] → out_obs[b, :]
# ═══════════════════════════════════════════════════════════════════════════
#
# `batch_obs` lives in per-sample-time-major layout matching CPU
# `train_step`: `batch_obs[(b * (K+1) + k) * OBS + d]`. We pass the
# whole [BATCH * (K+1) * OBS] tensor and let the kernel read the right
# slot via `K_STEP` runtime arg — `K_STEP` changes per call so we don't
# want it bound at comptime.


def ezv2_copy_obs_at_step_kernel[
    BATCH: Int,
    K_PLUS_1: Int,
    OBS: Int,
    dtype: DType,
](
    batch_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1 * OBS), MutAnyOrigin
    ],
    out_obs: LayoutTensor[dtype, Layout.row_major(BATCH * OBS), MutAnyOrigin],
    k_step: Int,
) where dtype.is_floating_point():
    """Gather `batch_obs[b, k_step, :]` into `out_obs[b, :]`."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var src_off = (b * K_PLUS_1 + k_step) * OBS
    var dst_off = b * OBS
    for d in range(OBS):
        out_obs[dst_off + d] = batch_obs[src_off + d]


# ═══════════════════════════════════════════════════════════════════════════
# Build dyn input: hidden[k] ‖ batch_actions[k]
# ═══════════════════════════════════════════════════════════════════════════
#
# Both buffers come in pre-sliced per-step (host views into the larger
# time-major scratches). `batch_actions` is per-sample-time-major
# (`batch_actions[(b * K + k) * ACT + a]`) so the host slices the right
# `[BATCH * K * ACT]` window directly.


def ezv2_build_dyn_input_kernel[
    BATCH: Int,
    LATENT: Int,
    ACT: Int,
    K: Int,
    dtype: DType,
](
    hidden_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH * K * ACT), MutAnyOrigin
    ],
    dyn_input: LayoutTensor[
        dtype, Layout.row_major(BATCH * (LATENT + ACT)), MutAnyOrigin
    ],
    k_step: Int,
) where dtype.is_floating_point():
    """Build `dyn_input[b, :] = hidden_step[b, :] ‖ batch_actions[b, k_step, :]`.

    `hidden_step` is a single time-slice already (host views the slot for
    `hidden[k_step]`); the action lookup needs the per-sample-time-major
    offset `(b * K + k_step) * ACT`.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var dyn_in_dim = LATENT + ACT
    var dst_off = b * dyn_in_dim
    var act_src_off = (b * K + k_step) * ACT
    for d in range(LATENT):
        dyn_input[dst_off + d] = hidden_step[b * LATENT + d]
    for a in range(ACT):
        dyn_input[dst_off + LATENT + a] = batch_actions[act_src_off + a]


# ═══════════════════════════════════════════════════════════════════════════
# dyn_out[k][b, 0:LATENT] + hidden[k, b, :] → hidden[k+1, b, :]   (residual)
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_extract_hidden_after_dyn_kernel[
    BATCH: Int,
    LATENT: Int,
    BINS: Int,
    dtype: DType,
](
    dyn_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * (LATENT + BINS)), MutAnyOrigin
    ],
    hidden_k: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
    next_hidden: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Compute `next_hidden[b, :] = dyn_out_step[b, :LATENT] + hidden_k[b, :]`.

    The residual `hidden[k+1] = dyn(hidden[k], action[k]) + hidden[k]` matches
    `ez_dmc_state.py:270` (`state = hidden + x`). At init with `delta ≈ 0`,
    dyn ≈ identity, so the K-step latent rollout preserves state info and
    the SimSiam consistency loss cannot trivially collapse both branches to
    a constant. Without this, the trivial all-same-direction fixed point
    is the easiest minimum of the consistency loss.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var dyn_out_dim = LATENT + BINS
    var src_off = b * dyn_out_dim
    var dst_off = b * LATENT
    for d in range(LATENT):
        next_hidden[dst_off + d] = (
            dyn_out_step[src_off + d] + hidden_k[dst_off + d]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample CE loss + grad — single time-slice
# ═══════════════════════════════════════════════════════════════════════════
#
# These mirror muzero's CE-grad kernels but additionally emit the
# per-sample CE loss into `per_sample_loss[BATCH]`. The host then reduces
# that buffer into the `L_*` scalar accumulator and (for value at k=0)
# saves a copy for the priority refresh.
#
# Operates on a single time-slice (the host views `pred_out[k_step]`,
# `grad_pred_out[k_step]`, etc. via ptr-offset arithmetic before each
# call). PRED_OUT-flavoured kernels write the policy/value sub-slice of
# the *same* shared output buffer so the dyn-output and pred-output grad
# tensors stay dense + ready for `backward_gpu` directly.


def ezv2_policy_loss_grad_kernel[
    BATCH: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType,
](
    pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    policy_target_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
    ],
    grad_pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    scale: Scalar[dtype],
) where dtype.is_floating_point():
    """CE(softmax(policy logits) || target) + grad on one time-slice.

    Writes scaled grad into the first ACT elements of
    `grad_pred_out_step[b]` and the per-sample CE (UNSCALED) into
    `per_sample_loss[b]`.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var logits_off = b * PRED_OUT
    var target_off = b * ACT

    var max_l = rebind[Scalar[dtype]](pred_out_step[logits_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_out_step[logits_off + a])
        if v > max_l:
            max_l = v

    var sum_e = Scalar[dtype](0.0)
    for a in range(ACT):
        sum_e = sum_e + exp(
            rebind[Scalar[dtype]](pred_out_step[logits_off + a]) - max_l
        )
    var log_z = log(sum_e) + max_l
    var inv_sum = Scalar[dtype](1.0) / sum_e

    var loss = Scalar[dtype](0.0)
    for a in range(ACT):
        var l_a = rebind[Scalar[dtype]](pred_out_step[logits_off + a])
        var t_a = rebind[Scalar[dtype]](policy_target_step[target_off + a])
        var p_a = exp(l_a - max_l) * inv_sum
        grad_pred_out_step[logits_off + a] = (p_a - t_a) * scale
        loss = loss + t_a * (log_z - l_a)

    per_sample_loss[b] = loss


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample squashed-Gaussian NLL + entropy bonus + grad — single time-slice
# ═══════════════════════════════════════════════════════════════════════════
#
# Continuous-action counterpart of `ezv2_policy_loss_grad_kernel`. Used by
# `ContinuousActionSpace.policy_loss_grad_gpu`. The policy section of
# `pred_out_step[b, 0:2*ACT_DIM]` carries (μ_raw, σ_raw); we forward through
# the squashed-Gaussian parameterization, evaluate the negative log-prob of
# the search-selected target action `a*` (paper Eq. 8 — simple-best-action
# loss), subtract an entropy bonus (paper Eq. 9), and write grads back into
# the same slice. The trailing BINS value-logit slots are left untouched
# (the value-loss kernel owns them).
#
# Forward:
#     μ_d         = MAX_ACTION · tanh(μ_raw_d / MAX_ACTION)
#     σ_d         = softplus(σ_raw_d) + MIN_STD
#     c_d         = clamp(a*_d / MAX_ACTION, ±0.999)
#     u*_d        = atanh(c_d) = 0.5 · log((1+c_d) / (1-c_d))
#     η_d         = (u*_d − μ_d) / σ_d
#     nlp_d       = 0.5·η_d² + log σ_d + 0.5·log(2π) + log(1 − c_d²)
#                       (last term is the tanh-squash log-det Jacobian)
#     H_d         ≈ -mean_j[log p_squashed(y_j)]   (MC, K=1024, reference parity)
#                  = mean_j[0.5·z_j² + log(1 − tanh²(u_j/MAX))] + log σ_d + 0.5·log(2π)
#                  where u_j = μ_d + σ_d · z_j, z_j ~ N(0,1) via Philox(seed, b, d).
#     loss[b]     = Σ_d (nlp_d − ent_scale · H_d)
#
# Backward (a* is constant ⇒ NLL tanh-correction has zero grad on μ, σ;
#  MC entropy gets reparameterized grad through u_j):
#     ∂loss/∂μ_d        = (μ_d − u*_d) / σ_d²
#                            + ent_scale · (2/MAX) · mean_j[tanh(u_j/MAX)]
#     ∂loss/∂σ_d        = (1 − η_d²) / σ_d
#                            - ent_scale · (1/σ_d - (2/MAX) · mean_j[z_j·tanh(u_j/MAX)])
#     ∂μ_d/∂μ_raw_d     = 1 − tanh²(μ_raw_d / MAX_ACTION)
#     ∂σ_d/∂σ_raw_d     = sigmoid(σ_raw_d)
#
# Numerical notes:
#     - softplus uses the max(x,0) + log(1+exp(-|x|)) form for stability.
#     - σ is bounded below by MIN_STD ⇒ inv_σ is finite even pre-training.
#     - c_d clipping bounds u*_d ∈ [-3.8, 3.8] for c=±0.999.


def ezv2_policy_loss_grad_continuous_kernel[
    BATCH: Int,
    ACT_DIM: Int,
    PRED_OUT: Int,
    dtype: DType,
](
    pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    policy_target_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * ACT_DIM), MutAnyOrigin
    ],
    grad_pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    scale: Scalar[dtype],
    ent_scale: Scalar[dtype],
    max_action: Scalar[dtype],
    min_std: Scalar[dtype],
    seed: UInt64,
) where dtype.is_floating_point():
    """Squashed-Gaussian NLL + entropy bonus + grad on one time-slice.

    Pred-out layout: `pred_out_step[b, 0:2*ACT_DIM]` = (μ_raw ‖ σ_raw),
    `pred_out_step[b, 2*ACT_DIM:PRED_OUT]` = value bins (untouched here).

    Writes grad into the first 2·ACT_DIM elements of
    `grad_pred_out_step[b]` (grads on policy logits — μ_raw then σ_raw).
    The trailing BINS slots remain untouched. Per-sample loss (UNSCALED,
    including the constant tanh-squash correction) goes into
    `per_sample_loss[b]`. `scale` is the loss-weight folded into the grad
    only, matching the discrete kernel convention.

    Entropy `H[π]` is estimated by Monte Carlo with `MC_K=1024` reparameterized
    samples per (b, d), matching the reference `loss.py:95-101`
    (`entropy = -mean(log_prob(rsample(1024)))`). The estimator
    implicitly includes the tanh log-det correction via the squashed
    `log_prob`. `seed` selects the Philox stream; different train-step
    callers should pass distinct seeds to decorrelate gradient noise.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var po_off = b * PRED_OUT
    var pt_off = b * ACT_DIM

    var inv_max = Scalar[dtype](1.0) / max_action
    # log(2π) ≈ 1.8378770664093453, log(2π·e) = log(2π) + 1
    var LOG_2PI = Scalar[dtype](1.8378770664093453)
    var HALF_LOG_2PI = Scalar[dtype](0.5) * LOG_2PI
    var LOG_2 = Scalar[dtype](0.6931471805599453)
    var TWO_PI = Scalar[dtype](6.283185307179586)

    # MC entropy with tanh correction (reference parity).
    comptime MC_K: Int = 1024
    var inv_K = Scalar[dtype](1.0) / Scalar[dtype](MC_K)

    var loss = Scalar[dtype](0.0)

    for d in range(ACT_DIM):
        var mu_raw = rebind[Scalar[dtype]](pred_out_step[po_off + d])
        var sg_raw = rebind[Scalar[dtype]](
            pred_out_step[po_off + ACT_DIM + d]
        )
        var a_star = rebind[Scalar[dtype]](policy_target_step[pt_off + d])

        # μ = max_action · tanh(μ_raw / max_action)
        var th = tanh(mu_raw * inv_max)
        var mu = max_action * th

        # σ = softplus(σ_raw) + min_std, numerically stable softplus.
        var sg_raw_neg_abs = -sg_raw if sg_raw > Scalar[dtype](0.0) else sg_raw
        var sp_pos = sg_raw if sg_raw > Scalar[dtype](0.0) else Scalar[dtype](
            0.0
        )
        var sp = sp_pos + log(Scalar[dtype](1.0) + exp(sg_raw_neg_abs))
        var sg = sp + min_std

        # u* = atanh(clamp(a*/max_action, ±0.999))
        var c = a_star * inv_max
        var c_lo = Scalar[dtype](-0.999)
        var c_hi = Scalar[dtype](0.999)
        if c > c_hi:
            c = c_hi
        if c < c_lo:
            c = c_lo
        var u_star = Scalar[dtype](0.5) * log(
            (Scalar[dtype](1.0) + c) / (Scalar[dtype](1.0) - c)
        )

        var diff = u_star - mu
        var inv_sg = Scalar[dtype](1.0) / sg
        var eta = diff * inv_sg

        var log_sg = log(sg)
        var corr_d = log(Scalar[dtype](1.0) - c * c)

        var nlp_d = (
            Scalar[dtype](0.5) * eta * eta
            + log_sg
            + HALF_LOG_2PI
            + corr_d
        )

        # ── MC entropy with tanh correction (reparameterized) ──────────
        # Per-sample stream: distinct (b, d) ⇒ distinct Philox key.
        var philox = PhiloxRandom(
            seed=(seed * UInt64(0x9E3779B97F4A7C15))
            + UInt64(b * 1664525 + 1013904223)
            + UInt64(d * 2654435761),
            offset=0,
        )

        var H_acc = Scalar[dtype](0.0)
        var mean_tanh_u = Scalar[dtype](0.0)
        var mean_z_tanh_u = Scalar[dtype](0.0)
        var eps_lo = Scalar[dtype](1e-9)
        for _ in range(MC_K):
            # Box-Muller z ~ N(0, 1) from two Philox uniforms.
            var u1 = philox.step_uniform()
            var u2 = philox.step_uniform()
            var u1f = Scalar[dtype](u1[0])
            if u1f < eps_lo:
                u1f = eps_lo
            var r = sqrt(Scalar[dtype](-2.0) * log(u1f))
            var z = r * cos(TWO_PI * Scalar[dtype](u2[0]))

            # u_k = μ + σ·z_k; x_k = u_k / max_action.
            var u_k = mu + sg * z
            var x_k = u_k * inv_max
            var th_uk = tanh(x_k)

            # log(1 - tanh²(x)) = 2·(log 2 - |x| - softplus(-2|x|))
            # stable form: |x| eliminates the sign branch on -2x.
            var ax = x_k if x_k >= Scalar[dtype](0.0) else -x_k
            var sp_term = log(Scalar[dtype](1.0) + exp(Scalar[dtype](-2.0) * ax))
            var log_1_m_tanh2 = Scalar[dtype](2.0) * (LOG_2 - ax - sp_term)

            # Per-sample squashed NLL (= -log_prob_squashed at u_k):
            #   0.5·z² + log σ + 0.5·log(2π) + log(1 - tanh²(x_k))
            # (the +log(1 - tanh²) term comes from −log|dy/du|).
            H_acc = H_acc + Scalar[dtype](0.5) * z * z + log_1_m_tanh2
            mean_tanh_u = mean_tanh_u + th_uk
            mean_z_tanh_u = mean_z_tanh_u + z * th_uk

        # H_d ≈ (1/K)·Σ(-log_prob_squashed(y_k)) plus the σ-only constants.
        var H_d = H_acc * inv_K + log_sg + HALF_LOG_2PI
        mean_tanh_u = mean_tanh_u * inv_K
        mean_z_tanh_u = mean_z_tanh_u * inv_K

        loss = loss + nlp_d - ent_scale * H_d

        # ── Gradients on (μ, σ) ────────────────────────────────────────
        # NLL contribution (a* constant ⇒ tanh-correction has zero grad):
        var dnlp_dmu = (mu - u_star) * inv_sg * inv_sg
        var dnlp_dsg = (Scalar[dtype](1.0) - eta * eta) * inv_sg
        # Entropy correction terms (reparameterized) — see kernel docstring.
        var two_inv_max = Scalar[dtype](2.0) * inv_max
        var dH_dmu = -two_inv_max * mean_tanh_u
        var dH_dsg = inv_sg - two_inv_max * mean_z_tanh_u
        var d_loss_d_mu = dnlp_dmu - ent_scale * dH_dmu
        var d_loss_d_sg = dnlp_dsg - ent_scale * dH_dsg

        # Chain through (μ_raw, σ_raw):
        var dmu_dmuraw = Scalar[dtype](1.0) - th * th
        # sigmoid(σ_raw) — d softplus / d σ_raw
        var sig_sg_raw: Scalar[dtype]
        if sg_raw > Scalar[dtype](0.0):
            var e_neg = exp(-sg_raw)
            sig_sg_raw = Scalar[dtype](1.0) / (Scalar[dtype](1.0) + e_neg)
        else:
            var e_pos = exp(sg_raw)
            sig_sg_raw = e_pos / (Scalar[dtype](1.0) + e_pos)

        grad_pred_out_step[po_off + d] = scale * d_loss_d_mu * dmu_dmuraw
        grad_pred_out_step[po_off + ACT_DIM + d] = (
            scale * d_loss_d_sg * sig_sg_raw
        )

    per_sample_loss[b] = loss


# ═══════════════════════════════════════════════════════════════════════════
# Full-π squashed-Gaussian NLL + entropy + grad — single time-slice
# ═══════════════════════════════════════════════════════════════════════════
#
# Paper Eq. 6 (EfficientZero V2) loss for continuous actions with action_dim
# == 1. Replaces the simple-best NLL of `ezv2_policy_loss_grad_continuous_
# kernel` with a *weighted* NLL over K MCTS root-sampled candidate actions:
#
#     loss[b] = Σ_k π_target[b, k] · NLL(a_target[b, k] | μ, σ)  -  ent · H
#
# where NLL is the squashed-Normal negative-log-prob (same form as the simple-
# best kernel; see that kernel's docstring for derivation), H is the analytic
# Gaussian-on-u entropy bonus, and π_target is the improved-policy distribution
# returned by sampled-Gumbel MCTS.
#
# Backward — per dim, accumulated across K candidates + MC entropy term:
#     ∂loss/∂μ_d = (μ_d · Σ_k π_k - Σ_k π_k · u*_d^(k)) / σ_d²
#                     + ent_scale · (2/MAX) · mean_j[tanh(u_j/MAX)]
#     ∂loss/∂σ_d = (Σ_k π_k - Σ_k π_k · η_d^(k)²) / σ_d
#                     - ent_scale · (1/σ_d - (2/MAX) · mean_j[z_j · tanh(u_j/MAX)])
# where u_j = μ + σ·z_j are the reparameterized MC samples used for H[π].
# Chain to (μ_raw, σ_raw) is identical to the simple-best kernel.
#
# Reference: `ez/utils/loss.py:continuous_loss` (action_dim==1 branch) in
# the upstream EZ-V2 implementation.


def ezv2_policy_loss_grad_continuous_fullpi_kernel[
    BATCH: Int,
    ACT_DIM: Int,
    K_ROOT: Int,
    PRED_OUT: Int,
    dtype: DType,
](
    pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    target_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_ROOT * ACT_DIM), MutAnyOrigin
    ],
    target_policy: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_ROOT), MutAnyOrigin
    ],
    grad_pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    scale: Scalar[dtype],
    ent_scale: Scalar[dtype],
    max_action: Scalar[dtype],
    min_std: Scalar[dtype],
    seed: UInt64,
) where dtype.is_floating_point():
    """Full-π squashed-Gaussian weighted-NLL + entropy bonus + grad.

    Pred-out layout matches the simple-best kernel: `pred_out_step[b,
    0:2*ACT_DIM]` = (μ_raw ‖ σ_raw), trailing slots are value bins
    (untouched here). Writes grad into the first 2·ACT_DIM elements
    of `grad_pred_out_step[b]`. `target_policy` row should sum to 1.0
    over the K candidates (the kernel doesn't normalize it).

    Entropy `H[π]` matches `ezv2_policy_loss_grad_continuous_kernel` —
    MC estimator with `MC_K=1024` reparameterized samples + tanh
    log-det correction (reference `loss.py:95-101`). `seed` selects the
    Philox stream.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var po_off = b * PRED_OUT
    var ta_off = b * K_ROOT * ACT_DIM
    var tp_off = b * K_ROOT

    var inv_max = Scalar[dtype](1.0) / max_action
    var LOG_2PI = Scalar[dtype](1.8378770664093453)
    var HALF_LOG_2PI = Scalar[dtype](0.5) * LOG_2PI
    var LOG_2 = Scalar[dtype](0.6931471805599453)
    var TWO_PI = Scalar[dtype](6.283185307179586)

    comptime MC_K: Int = 1024
    var inv_K = Scalar[dtype](1.0) / Scalar[dtype](MC_K)

    var loss = Scalar[dtype](0.0)

    for d in range(ACT_DIM):
        var mu_raw = rebind[Scalar[dtype]](pred_out_step[po_off + d])
        var sg_raw = rebind[Scalar[dtype]](
            pred_out_step[po_off + ACT_DIM + d]
        )

        var th = tanh(mu_raw * inv_max)
        var mu = max_action * th

        var sg_raw_neg_abs = -sg_raw if sg_raw > Scalar[dtype](
            0.0
        ) else sg_raw
        var sp_pos = sg_raw if sg_raw > Scalar[dtype](
            0.0
        ) else Scalar[dtype](0.0)
        var sp = sp_pos + log(Scalar[dtype](1.0) + exp(sg_raw_neg_abs))
        var sg = sp + min_std

        var inv_sg = Scalar[dtype](1.0) / sg
        var log_sg = log(sg)

        # Accumulate weighted NLL across K candidates.
        var sum_pi = Scalar[dtype](0.0)
        var sum_pi_u = Scalar[dtype](0.0)
        var sum_pi_eta_sq = Scalar[dtype](0.0)
        var weighted_corr = Scalar[dtype](0.0)

        for k in range(K_ROOT):
            var pi_k = rebind[Scalar[dtype]](target_policy[tp_off + k])
            var a_k = rebind[Scalar[dtype]](
                target_actions[ta_off + k * ACT_DIM + d]
            )
            var c = a_k * inv_max
            var c_lo = Scalar[dtype](-0.999)
            var c_hi = Scalar[dtype](0.999)
            if c > c_hi:
                c = c_hi
            if c < c_lo:
                c = c_lo
            var u_star = Scalar[dtype](0.5) * log(
                (Scalar[dtype](1.0) + c) / (Scalar[dtype](1.0) - c)
            )
            var diff = u_star - mu
            var eta = diff * inv_sg
            var corr = log(Scalar[dtype](1.0) - c * c)
            sum_pi = sum_pi + pi_k
            sum_pi_u = sum_pi_u + pi_k * u_star
            sum_pi_eta_sq = sum_pi_eta_sq + pi_k * eta * eta
            weighted_corr = weighted_corr + pi_k * corr

        # Per-dim weighted-NLL: Σ_k π_k · nlp_d^(k)
        #   = 0.5 · sum_pi_eta_sq + sum_pi · (log_sg + 0.5·log 2π) + weighted_corr
        var weighted_nlp_d = (
            Scalar[dtype](0.5) * sum_pi_eta_sq
            + sum_pi * (log_sg + HALF_LOG_2PI)
            + weighted_corr
        )

        # ── MC entropy with tanh correction (reparameterized) ──────────
        var philox = PhiloxRandom(
            seed=(seed * UInt64(0x9E3779B97F4A7C15))
            + UInt64(b * 1664525 + 1013904223)
            + UInt64(d * 2654435761),
            offset=0,
        )

        var H_acc = Scalar[dtype](0.0)
        var mean_tanh_u = Scalar[dtype](0.0)
        var mean_z_tanh_u = Scalar[dtype](0.0)
        var eps_lo = Scalar[dtype](1e-9)
        for _ in range(MC_K):
            var u1 = philox.step_uniform()
            var u2 = philox.step_uniform()
            var u1f = Scalar[dtype](u1[0])
            if u1f < eps_lo:
                u1f = eps_lo
            var r = sqrt(Scalar[dtype](-2.0) * log(u1f))
            var z = r * cos(TWO_PI * Scalar[dtype](u2[0]))

            var u_k = mu + sg * z
            var x_k = u_k * inv_max
            var th_uk = tanh(x_k)

            var ax = x_k if x_k >= Scalar[dtype](0.0) else -x_k
            var sp_term = log(Scalar[dtype](1.0) + exp(Scalar[dtype](-2.0) * ax))
            var log_1_m_tanh2 = Scalar[dtype](2.0) * (LOG_2 - ax - sp_term)

            H_acc = H_acc + Scalar[dtype](0.5) * z * z + log_1_m_tanh2
            mean_tanh_u = mean_tanh_u + th_uk
            mean_z_tanh_u = mean_z_tanh_u + z * th_uk

        var H_d = H_acc * inv_K + log_sg + HALF_LOG_2PI
        mean_tanh_u = mean_tanh_u * inv_K
        mean_z_tanh_u = mean_z_tanh_u * inv_K

        loss = loss + weighted_nlp_d - ent_scale * H_d

        # ── Gradients on (μ, σ) (NLL + MC entropy correction) ──────────
        var dnlp_dmu = (mu * sum_pi - sum_pi_u) * inv_sg * inv_sg
        var dnlp_dsg = (sum_pi - sum_pi_eta_sq) * inv_sg
        var two_inv_max = Scalar[dtype](2.0) * inv_max
        var dH_dmu = -two_inv_max * mean_tanh_u
        var dH_dsg = inv_sg - two_inv_max * mean_z_tanh_u
        var d_loss_d_mu = dnlp_dmu - ent_scale * dH_dmu
        var d_loss_d_sg = dnlp_dsg - ent_scale * dH_dsg

        var dmu_dmuraw = Scalar[dtype](1.0) - th * th
        var sig_sg_raw: Scalar[dtype]
        if sg_raw > Scalar[dtype](0.0):
            var e_neg = exp(-sg_raw)
            sig_sg_raw = Scalar[dtype](1.0) / (Scalar[dtype](1.0) + e_neg)
        else:
            var e_pos = exp(sg_raw)
            sig_sg_raw = e_pos / (Scalar[dtype](1.0) + e_pos)

        grad_pred_out_step[po_off + d] = scale * d_loss_d_mu * dmu_dmuraw
        grad_pred_out_step[po_off + ACT_DIM + d] = (
            scale * d_loss_d_sg * sig_sg_raw
        )

    per_sample_loss[b] = loss


def ezv2_value_loss_grad_kernel[
    BATCH: Int,
    BINS: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType,
](
    pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    value_target_dist: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    grad_pred_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    scale: Scalar[dtype],
) where dtype.is_floating_point():
    """CE(softmax(value logits) || two_hot(target)) + grad on one time-slice.

    Writes scaled grad into elements [ACT, ACT+BINS) of
    `grad_pred_out_step[b]` and the per-sample CE (UNSCALED) into
    `per_sample_loss[b]`. Reuses the (mostly stable) value-CE pattern
    from muzero's `ce_value_grad_kernel` plus the CE-loss accumulator.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var logits_off = b * PRED_OUT + ACT
    var tgt_off = b * BINS

    var max_l = rebind[Scalar[dtype]](pred_out_step[logits_off])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](pred_out_step[logits_off + i])
        if v > max_l:
            max_l = v

    var sum_e = Scalar[dtype](0.0)
    for i in range(BINS):
        sum_e = sum_e + exp(
            rebind[Scalar[dtype]](pred_out_step[logits_off + i]) - max_l
        )
    var log_z = log(sum_e) + max_l
    var inv_sum = Scalar[dtype](1.0) / sum_e

    var loss = Scalar[dtype](0.0)
    for i in range(BINS):
        var l_i = rebind[Scalar[dtype]](pred_out_step[logits_off + i])
        var t_i = rebind[Scalar[dtype]](value_target_dist[tgt_off + i])
        var p_i = exp(l_i - max_l) * inv_sum
        grad_pred_out_step[logits_off + i] = (p_i - t_i) * scale
        loss = loss + t_i * (log_z - l_i)

    per_sample_loss[b] = loss


def ezv2_reward_loss_grad_kernel[
    BATCH: Int,
    BINS: Int,
    LATENT: Int,
    DYN_OUT: Int,
    dtype: DType,
](
    dyn_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
    ],
    reward_target_dist: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    grad_dyn_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    scale: Scalar[dtype],
) where dtype.is_floating_point():
    """CE(softmax(reward logits) || two_hot(target)) + grad on one time-slice.

    Reward logits live at `dyn_out_step[b, LATENT:LATENT+BINS]`.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var logits_off = b * DYN_OUT + LATENT
    var tgt_off = b * BINS

    var max_l = rebind[Scalar[dtype]](dyn_out_step[logits_off])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](dyn_out_step[logits_off + i])
        if v > max_l:
            max_l = v

    var sum_e = Scalar[dtype](0.0)
    for i in range(BINS):
        sum_e = sum_e + exp(
            rebind[Scalar[dtype]](dyn_out_step[logits_off + i]) - max_l
        )
    var log_z = log(sum_e) + max_l
    var inv_sum = Scalar[dtype](1.0) / sum_e

    var loss = Scalar[dtype](0.0)
    for i in range(BINS):
        var l_i = rebind[Scalar[dtype]](dyn_out_step[logits_off + i])
        var t_i = rebind[Scalar[dtype]](reward_target_dist[tgt_off + i])
        var p_i = exp(l_i - max_l) * inv_sum
        grad_dyn_out_step[logits_off + i] = (p_i - t_i) * scale
        loss = loss + t_i * (log_z - l_i)

    per_sample_loss[b] = loss


# ═══════════════════════════════════════════════════════════════════════════
# Cosine consistency loss + grad on one time-slice
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_cosine_loss_grad_kernel[
    BATCH: Int,
    PROJ: Int,
    dtype: DType,
](
    pred_dyn_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
    ],
    proj_obs_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
    ],
    grad_pred_dyn_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    scale: Scalar[dtype],
) where dtype.is_floating_point():
    """Compute -cos(pred, target) and d/d pred on one time-slice.

    `proj_obs_step` is the stop-grad target; we only emit grad w.r.t.
    `pred_dyn_step`. The loss is `-cos`, matching CPU
    `consistency.cosine_consistency_loss` (and `train_step`'s L_G block).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var off = b * PROJ

    var dot = Scalar[dtype](0.0)
    var na2 = Scalar[dtype](0.0)
    var nb2 = Scalar[dtype](0.0)
    for i in range(PROJ):
        var pv = rebind[Scalar[dtype]](pred_dyn_step[off + i])
        var tv = rebind[Scalar[dtype]](proj_obs_step[off + i])
        dot = dot + pv * tv
        na2 = na2 + pv * pv
        nb2 = nb2 + tv * tv

    var na = sqrt(na2 + Scalar[dtype](1e-12))
    var nb = sqrt(nb2 + Scalar[dtype](1e-12))
    var c = dot / (na * nb)
    per_sample_loss[b] = -c

    var inv_na2 = Scalar[dtype](1.0) / (na * na)
    var inv_na_nb = Scalar[dtype](1.0) / (na * nb)
    for i in range(PROJ):
        var pv = rebind[Scalar[dtype]](pred_dyn_step[off + i])
        var tv = rebind[Scalar[dtype]](proj_obs_step[off + i])
        grad_pred_dyn_step[off + i] = (
            (c * pv * inv_na2 - tv * inv_na_nb) * scale
        )


# ═══════════════════════════════════════════════════════════════════════════
# Single-thread reductions — `[BATCH] → 1 scalar` accumulator
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_reduce_add_kernel[
    BATCH: Int,
    dtype: DType,
](
    in_buf: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    accum: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
) where dtype.is_floating_point():
    """Single-thread sum reduction: `accum[0] += Σ in_buf[i]`.

    Loss buffers are tiny (BATCH ≤ 64). One thread is fine + we avoid
    needing block-shared-memory reduction code; the host pulls the
    running accumulator at the end.
    """
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t != 0:
        return
    var s = Scalar[dtype](0.0)
    for i in range(BATCH):
        s = s + rebind[Scalar[dtype]](in_buf[i])
    accum[0] = rebind[Scalar[dtype]](accum[0]) + s


# ═══════════════════════════════════════════════════════════════════════════
# Element-wise add: dst[i] += src[i]
# ═══════════════════════════════════════════════════════════════════════════
#
# Used as the grad-hidden accumulator: after each
# {pred, projector, dyn}.backward call, the LATENT-shaped step grad
# is added into the right slot of the time-major `grad_hidden` scratch.


def ezv2_add_kernel[
    SIZE: Int,
    dtype: DType,
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
) where dtype.is_floating_point():
    """`dst[i] += src[i]` elementwise."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return
    dst[i] = (
        rebind[Scalar[dtype]](dst[i]) + rebind[Scalar[dtype]](src[i])
    )


# ═══════════════════════════════════════════════════════════════════════════
# Build per-step dyn grad_out: [grad_hidden_next ‖ grad_dyn_out_reward]
# ═══════════════════════════════════════════════════════════════════════════
#
# At dyn-backward time the step's full grad_out has two contributions:
#   • LATENT slice — gradient w.r.t. hidden[k+1], accumulated by
#     pred-backward at k+1 + projector-backward at k+1 + later
#     dyn-backwards.
#   • BINS slice — already written by `ezv2_reward_loss_grad_kernel`.
# Both are pre-sliced into single time-slot views; this kernel
# concatenates them into one [BATCH * (LATENT + BINS)] buffer the
# `Network.backward_gpu` call can consume directly.


def ezv2_assemble_grad_dyn_step_kernel[
    BATCH: Int,
    LATENT: Int,
    BINS: Int,
    dtype: DType,
](
    grad_hidden_next: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
    grad_dyn_out_reward: LayoutTensor[
        dtype, Layout.row_major(BATCH * (LATENT + BINS)), MutAnyOrigin
    ],
    grad_dyn_out_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * (LATENT + BINS)), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Build `grad_dyn_out_step[b] = [grad_hidden_next[b] || grad_dyn_out_reward[b, LATENT:]]`.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var dyn_out_dim = LATENT + BINS
    var dst_off = b * dyn_out_dim
    var rew_src_off = b * dyn_out_dim + LATENT
    for d in range(LATENT):
        grad_dyn_out_step[dst_off + d] = grad_hidden_next[b * LATENT + d]
    for i in range(BINS):
        grad_dyn_out_step[dst_off + LATENT + i] = grad_dyn_out_reward[
            rew_src_off + i
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Slice + accumulate: grad_dyn_in[b, :LATENT] → grad_hidden_step[b, :]
# ═══════════════════════════════════════════════════════════════════════════
#
# After dyn-backward writes grad_dyn_in (BATCH × DYN_IN), the LATENT-slice
# is the gradient w.r.t. hidden[k]; the action-slice is the gradient
# w.r.t. the (constant) one-hot action and gets discarded.


def ezv2_accumulate_dyn_grad_in_kernel[
    BATCH: Int,
    LATENT: Int,
    ACT: Int,
    dtype: DType,
](
    grad_dyn_in: LayoutTensor[
        dtype, Layout.row_major(BATCH * (LATENT + ACT)), MutAnyOrigin
    ],
    grad_hidden_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Accumulate `grad_dyn_in[b, :LATENT]` into `grad_hidden_step[b, :]`."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var dyn_in_dim = LATENT + ACT
    var src_off = b * dyn_in_dim
    var dst_off = b * LATENT
    for d in range(LATENT):
        grad_hidden_step[dst_off + d] = (
            rebind[Scalar[dtype]](grad_hidden_step[dst_off + d])
            + rebind[Scalar[dtype]](grad_dyn_in[src_off + d])
        )


# ═══════════════════════════════════════════════════════════════════════════
# Reward target gather: batch_rewards[b, k_step] → reward_target_scalar[b]
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_gather_reward_at_step_kernel[
    BATCH: Int,
    K: Int,
    dtype: DType,
](
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ],
    reward_target_scalar: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    k_step: Int,
) where dtype.is_floating_point():
    """Gather `batch_rewards[b, k_step]` into `reward_target_scalar[b]`.

    `batch_rewards` is per-sample-time-major
    (`batch_rewards[b * K + k_step]`).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    reward_target_scalar[b] = batch_rewards[b * K + k_step]


# ═══════════════════════════════════════════════════════════════════════════
# Value scalar gather: value_target_scalar_full[b, k_step] → out[b]
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_gather_value_target_kernel[
    BATCH: Int,
    K_PLUS_1: Int,
    dtype: DType,
](
    value_target_full: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1), MutAnyOrigin
    ],
    value_target_scalar: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    k_step: Int,
) where dtype.is_floating_point():
    """Gather `value_target_full[b, k_step]` into `value_target_scalar[b]`.

    `value_target_full` is per-sample-time-major
    (`value_target_full[b * (K+1) + k_step]`); the full mixed-value-target
    array is computed on host during sampling and uploaded once per
    train step.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    value_target_scalar[b] = value_target_full[b * K_PLUS_1 + k_step]


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample policy gather: batch_mcts_pol[b, k_step, :] → policy_target_step[b, :]
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_gather_policy_target_kernel[
    BATCH: Int,
    K_PLUS_1: Int,
    ACT: Int,
    dtype: DType,
](
    batch_mcts_pol: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1 * ACT), MutAnyOrigin
    ],
    policy_target_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
    ],
    k_step: Int,
) where dtype.is_floating_point():
    """Gather `batch_mcts_pol[b, k_step, :]` into `policy_target_step[b, :]`.

    `batch_mcts_pol` is per-sample-time-major
    (`batch_mcts_pol[(b * (K+1) + k_step) * ACT + a]`).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var src_off = (b * K_PLUS_1 + k_step) * ACT
    var dst_off = b * ACT
    for a in range(ACT):
        policy_target_step[dst_off + a] = batch_mcts_pol[src_off + a]


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample gather of full-π targets (paper Eq. 6, ACT_DIM==1):
# batch_mcts_samp_act[b, k_step, :, :] → target_actions_step[b, :, :]
# batch_mcts_imp_pi[b, k_step, :]      → target_policy_step[b, :]
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_gather_fullpi_targets_kernel[
    BATCH: Int,
    K_PLUS_1: Int,
    K_ROOT: Int,
    ACT_DIM: Int,
    dtype: DType,
](
    batch_mcts_samp_act: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * K_PLUS_1 * K_ROOT * ACT_DIM),
        MutAnyOrigin,
    ],
    batch_mcts_imp_pi: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1 * K_ROOT), MutAnyOrigin
    ],
    target_actions_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_ROOT * ACT_DIM), MutAnyOrigin
    ],
    target_policy_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_ROOT), MutAnyOrigin
    ],
    k_step: Int,
) where dtype.is_floating_point():
    """Gather per-k full-π target slices from time-major batch buffers.

    Source layouts match the upload: `batch_mcts_samp_act[(b · (K+1) +
    k_step) · K_ROOT · ACT_DIM + i · ACT_DIM + d]` and
    `batch_mcts_imp_pi[(b · (K+1) + k_step) · K_ROOT + i]`.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var samp_src_off = (b * K_PLUS_1 + k_step) * K_ROOT * ACT_DIM
    var samp_dst_off = b * K_ROOT * ACT_DIM
    for j in range(K_ROOT * ACT_DIM):
        target_actions_step[samp_dst_off + j] = batch_mcts_samp_act[
            samp_src_off + j
        ]
    var pi_src_off = (b * K_PLUS_1 + k_step) * K_ROOT
    var pi_dst_off = b * K_ROOT
    for i in range(K_ROOT):
        target_policy_step[pi_dst_off + i] = batch_mcts_imp_pi[
            pi_src_off + i
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Priority update: priorities_out[b] = per_sample_v_loss[b] + 1e-3
# ═══════════════════════════════════════════════════════════════════════════


def ezv2_priority_from_v_loss_kernel[
    BATCH: Int,
    dtype: DType,
](
    per_sample_v_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    priorities_out: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Write `priorities_out[b] = per_sample_v_loss[b] + 1e-3`.

    Mirrors CPU `train_step` — `|TD error|` is approximated by the
    per-sample value-CE; the 1e-3 floor matches `compute_loss_components`
    and prevents priorities collapsing to zero on perfectly-fit windows.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    priorities_out[b] = rebind[Scalar[dtype]](per_sample_v_loss[b]) + Scalar[
        dtype
    ](1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# Reward-prefix LSTM head — GPU helpers
# ═══════════════════════════════════════════════════════════════════════════
#
# These are the EZ-V2-specific kernels that don't already exist in
# muzero/kernels.mojo and aren't covered by the kernels above. The
# LSTM cell + MLP head themselves use `LSTMCell.step_forward_gpu` /
# `Network.forward_gpu_with_cache` directly.


def ezv2_copy_lstm_input_kernel[
    BATCH: Int,
    LSTM_HIDDEN: Int,
    dtype: DType,
](
    lstm_h_states: LayoutTensor[
        dtype, Layout.row_major(BATCH * LSTM_HIDDEN), MutAnyOrigin
    ],
    lstm_c_states: LayoutTensor[
        dtype, Layout.row_major(BATCH * LSTM_HIDDEN), MutAnyOrigin
    ],
    h_input: LayoutTensor[
        dtype, Layout.row_major(BATCH * LSTM_HIDDEN), MutAnyOrigin
    ],
    c_input: LayoutTensor[
        dtype, Layout.row_major(BATCH * LSTM_HIDDEN), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Copy a single-step slice of `lstm_h_states` / `lstm_c_states` into
    the per-step input scratch. The host pre-views the right time-slot
    via ptr-offset arithmetic before the call so the kernel sees only
    `[BATCH * LSTM_HIDDEN]`-shaped tensors."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH * LSTM_HIDDEN:
        return
    h_input[i] = lstm_h_states[i]
    c_input[i] = lstm_c_states[i]


def ezv2_reward_prefix_loss_grad_kernel[
    BATCH: Int,
    BINS: Int,
    dtype: DType,
](
    rew_pref_logits_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    rew_target_dist: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    grad_rew_pref_logits_step: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    per_sample_loss: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    scale: Scalar[dtype],
) where dtype.is_floating_point():
    """CE(softmax(reward-prefix logits) || two_hot(cumulative reward target))
    + grad on one time-slice.

    Differs from `ezv2_value_loss_grad_kernel` / `ezv2_reward_loss_grad_kernel`
    in that the logits live in their own dense `[BATCH * BINS]` buffer
    (one per unroll position) — there's no enclosing `[..., PRED_OUT]`
    or `[..., DYN_OUT]` wrapper, so logits start at offset `b * BINS`."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var logits_off = b * BINS
    var tgt_off = b * BINS

    var max_l = rebind[Scalar[dtype]](rew_pref_logits_step[logits_off])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](rew_pref_logits_step[logits_off + i])
        if v > max_l:
            max_l = v

    var sum_e = Scalar[dtype](0.0)
    for i in range(BINS):
        sum_e = sum_e + exp(
            rebind[Scalar[dtype]](rew_pref_logits_step[logits_off + i])
            - max_l
        )
    var log_z = log(sum_e) + max_l
    var inv_sum = Scalar[dtype](1.0) / sum_e

    var loss = Scalar[dtype](0.0)
    for i in range(BINS):
        var l_i = rebind[Scalar[dtype]](
            rew_pref_logits_step[logits_off + i]
        )
        var t_i = rebind[Scalar[dtype]](rew_target_dist[tgt_off + i])
        var p_i = exp(l_i - max_l) * inv_sum
        grad_rew_pref_logits_step[logits_off + i] = (
            (p_i - t_i) * scale
        )
        loss = loss + t_i * (log_z - l_i)

    per_sample_loss[b] = loss


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3b: Decode boot-V from target prediction-net output
# ═══════════════════════════════════════════════════════════════════════════
#
# Per (sample, k_step): softmax over the BINS value logits in the target
# pred-net output, expectation in scalar-transformed space, then
# `inverse_scalar_transform` → real value. Writes to
# `boot_v[b * (K+1) + k_step]`. Replaces the host loop at
# `continuous_agent.mojo:1062-1071` (and the discrete sibling).
#
# `VALUE_OFF` is the value-bin offset inside the per-sample pred-net
# output — `ACT` for discrete, `2 * ACT_DIM` for continuous (the
# `POLICY_OUT_DIM` width before the BINS slice). Threaded at comptime.


def ezv2_decode_boot_v_kernel[
    BATCH: Int,
    K_PLUS_1: Int,
    PRED_OUT: Int,
    BINS: Int,
    VALUE_OFF: Int,
    dtype: DType,
](
    pred_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    boot_v: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1), MutAnyOrigin
    ],
    k_step: Int,
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    eps: Scalar[dtype],
) where dtype.is_floating_point():
    """Decode target-net pred output → boot-V scalar (one thread per sample).

    `pred_out[b, VALUE_OFF:VALUE_OFF+BINS]` are the value-bin logits.
    Numerically-stable softmax, expectation over `[v_min, v_max]` support,
    then `h^{-1}` (inverse MuZero scalar transform, `eps`=0.001).

    The result lands in `boot_v[b, k_step]` (per-sample-time-major) so
    Phase 3c's `compute_value_target_kernel` can consume it directly.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var off = b * PRED_OUT + VALUE_OFF

    # Numerically stable softmax + expectation in transformed space.
    var max_val = rebind[Scalar[dtype]](pred_out[off])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](pred_out[off + i])
        if v > max_val:
            max_val = v

    var sum_exp = Scalar[dtype](0.0)
    for i in range(BINS):
        sum_exp += exp(rebind[Scalar[dtype]](pred_out[off + i]) - max_val)

    var step = (
        (v_max - v_min) / Scalar[dtype](BINS - 1)
        if BINS > 1 else Scalar[dtype](0.0)
    )
    var v_raw = Scalar[dtype](0.0)
    for i in range(BINS):
        var prob = exp(
            rebind[Scalar[dtype]](pred_out[off + i]) - max_val
        ) / sum_exp
        var bin_val = v_min + Scalar[dtype](i) * step
        v_raw += prob * bin_val

    # Inverse MuZero scalar transform: closed-form quadratic inverse of
    # h(x) = sign(x)*(sqrt(|x|+1)-1) + eps*x.
    var sign = (
        Scalar[dtype](1.0) if v_raw >= Scalar[dtype](0.0)
        else Scalar[dtype](-1.0)
    )
    var abs_y = v_raw if v_raw >= Scalar[dtype](0.0) else -v_raw
    var inner = sqrt(
        Scalar[dtype](1.0)
        + Scalar[dtype](4.0) * eps * (abs_y + Scalar[dtype](1.0) + eps)
    )
    var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps)
    var v_real = sign * (f * f - Scalar[dtype](1.0))

    boot_v[b * K_PLUS_1 + k_step] = v_real


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3c: Compute V-target on GPU
# ═══════════════════════════════════════════════════════════════════════════
#
# Replaces the host loop at `continuous_agent.mojo:1102-1132` (and the
# discrete sibling). One thread per (b, k) computes the per-mode value
# target:
#   SEARCH: v = sve
#   SARSA : v = Σ_j γ^j r_{k+j}  +  γ^n_eff · boot_v[k + n_eff]
#   MIXED : age < T_FRESH ? TD : SVE   (see `MixedValueTarget.compute`)
#
# Comptime-specialised on `VALUE_TARGET_MODE` so each variant compiles
# down to its minimal body — SEARCH skips the boot_v reads entirely.
# `VALUE_TARGET_MODE` constants must match `configs.mojo`:
# SEARCH=0, SARSA=1, MIXED=2.


def ezv2_compute_value_target_kernel[
    BATCH: Int,
    K_PLUS_1: Int,
    K_UNROLL: Int,
    N_TD: Int,
    VALUE_TARGET_MODE: Int,
    T_FRESH: Int,
    dtype: DType,
](
    batch_mcts_val: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_UNROLL), MutAnyOrigin
    ],
    batch_age: LayoutTensor[
        DType.int32, Layout.row_major(BATCH * K_PLUS_1), MutAnyOrigin
    ],
    boot_v: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1), MutAnyOrigin
    ],
    value_target_full: LayoutTensor[
        dtype, Layout.row_major(BATCH * K_PLUS_1), MutAnyOrigin
    ],
    gamma: Scalar[dtype],
    mixed_use_age_blend: Scalar[DType.int32],
) where dtype.is_floating_point():
    """Per-(b, k) V-target. One thread per element. Total threads =
    `BATCH * K_PLUS_1`. `K_PLUS_1` should equal `K_UNROLL + 1`.

    `mixed_use_age_blend` only matters under VALUE_TARGET_MIXED
    (0 = false, 1 = true; `Bool` isn't `DevicePassable` so we encode):
      • 0 ⇒ pure TD (mirrors discrete agent's pre-`T_STALE` gate from
        `base.py:420`: pure TD until `train_step_count >= T_STALE`).
      • 1 ⇒ per-sample age-based blend (`age < T_FRESH ? td : sve`).
    Continuous agent passes 1 unconditionally (it doesn't apply the
    train-step gate today; matches its existing host-side path).
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * K_PLUS_1:
        return
    var b = idx // K_PLUS_1
    var k = idx % K_PLUS_1

    var sve = rebind[Scalar[dtype]](batch_mcts_val[idx])

    comptime if VALUE_TARGET_MODE == 0:  # SEARCH
        value_target_full[idx] = sve
        return

    # SARSA or MIXED: build n-step TD return.
    var k_remaining = K_UNROLL - k
    var n_eff = N_TD if N_TD < k_remaining else k_remaining

    var td = Scalar[dtype](0.0)
    var disc = Scalar[dtype](1.0)
    for j in range(n_eff):
        td += disc * rebind[Scalar[dtype]](batch_rewards[b * K_UNROLL + k + j])
        disc *= gamma
    td += disc * rebind[Scalar[dtype]](boot_v[b * K_PLUS_1 + k + n_eff])

    comptime if VALUE_TARGET_MODE == 1:  # SARSA
        value_target_full[idx] = td
    else:  # MIXED — `MixedValueTarget.compute` semantics
        if Int(mixed_use_age_blend) == 0:
            value_target_full[idx] = td
            return
        var age = Int(rebind[Scalar[DType.int32]](batch_age[idx]))
        if age < T_FRESH:
            value_target_full[idx] = td
        else:
            value_target_full[idx] = sve
