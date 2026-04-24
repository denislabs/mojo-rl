"""REDQ-specific GPU kernels.

Extracted from `deep_agents/core/kernels.mojo` so REDQ owns its own
ensemble-target kernel. All other kernels used by REDQ (td_mse_grad,
sac_rsample_*, gradient clipping, etc.) still live in `core/kernels.mojo`.
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.random.philox import Random as PhiloxRandom


# =============================================================================
# REDQ Ensemble Target Kernel
# =============================================================================


@always_inline
def redq_ensemble_target_kernel[
    dtype: DType where dtype.is_floating_point(),
    BATCH: Int,
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,  # 0=min subset, 1=ave all, 2=rem (random ensemble mixture)
](
    td_targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    # Target Q values stacked contiguously: q_next[n, b] = Q_target^n(s', a')
    q_next: LayoutTensor[
        dtype, Layout.row_major(N_ENSEMBLE, BATCH), MutAnyOrigin
    ],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    # Subset indices (for MODE=0): which N_MIN of N_ENSEMBLE to use.
    # Ignored when MODE != 0; pass a 1-element placeholder.
    subset_idxs: LayoutTensor[
        DType.uint32, Layout.row_major(N_MIN), MutAnyOrigin
    ],
    gamma: Scalar[dtype],
    alpha_buf: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    rng_counter: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
):
    """Compute REDQ TD target from N stacked target-Q values.

    SAC: y = r + γ * (combined_Q(s', a') - α * log_π(a'|s')) * (1 - done)

    combined_Q depends on MODE:
      MODE=0 (min): min over q_next[subset_idxs[m], b] for m in 0..N_MIN
      MODE=1 (ave): mean over q_next[n, b] for n in 0..N_ENSEMBLE
      MODE=2 (rem): Σ_n w_n * q_next[n, b], w_n ~ Uniform(0,1), normalized

    One thread per batch sample. Reads alpha from GPU memory.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var one = Scalar[dtype](1.0)
    var combined = Scalar[dtype](0.0)

    comptime if MODE == 0:
        # --- min over N_MIN subset ---
        var first_idx = Int(subset_idxs.ptr[0])
        var v0 = q_next.ptr[first_idx * BATCH + b]
        if v0 != v0:
            v0 = Scalar[dtype](0.0)
        combined = v0
        comptime for m in range(1, N_MIN):
            var idx = Int(subset_idxs.ptr[m])
            var v = q_next.ptr[idx * BATCH + b]
            if v != v:
                v = Scalar[dtype](0.0)
            if v < combined:
                combined = v
    comptime if MODE == 1:
        # --- mean over all N_ENSEMBLE ---
        var acc = Scalar[dtype](0.0)
        comptime for n in range(N_ENSEMBLE):
            var v = q_next.ptr[n * BATCH + b]
            if v != v:
                v = Scalar[dtype](0.0)
            acc = acc + v
        combined = acc / Scalar[dtype](N_ENSEMBLE)
    comptime if MODE == 2:
        # --- REM: random convex combination over all N_ENSEMBLE ---
        # Philox-generated uniforms per (sample, critic) pair, normalized.
        var seed = UInt64(rng_counter.ptr[0]) + UInt64(b) * UInt64(N_ENSEMBLE)
        var ws = InlineArray[Scalar[dtype], N_ENSEMBLE](uninitialized=True)
        var sum_w = Scalar[dtype](0.0)
        comptime for n in range(N_ENSEMBLE):
            var philox = PhiloxRandom(
                seed=seed + UInt64(n), offset=0
            )
            var r = philox.step_uniform()
            var w = Scalar[dtype](Float32(r[0]) + Float32(1e-8))
            ws[n] = w
            sum_w = sum_w + w
        var inv_sum = Scalar[dtype](1.0) / sum_w
        var acc = Scalar[dtype](0.0)
        comptime for n in range(N_ENSEMBLE):
            var v = q_next.ptr[n * BATCH + b]
            if v != v:
                v = Scalar[dtype](0.0)
            acc = acc + ws[n] * inv_sum * v
        combined = acc

    # Subtract entropy bonus and build TD target.
    var alpha = alpha_buf.ptr[0]
    var lp = log_probs.ptr[b]
    var tgt = Scalar[dtype](
        rewards.ptr[b] + gamma * (combined - alpha * lp) * (one - dones.ptr[b])
    )
    if tgt != tgt:
        tgt = Scalar[dtype](0.0)
    td_targets.ptr[b] = tgt
