"""REDQ ensemble-target kernel (CPU).

Phase R.1. Computes the REDQ TD target from a stacked block of N
target-Q evaluations:

    y[b] = r[b] + (1 - term[b]) * γ * (combined_Q(s', a')[b] - α * log_π(a'|s')[b])

`combined_Q` depends on `MODE`:
  MODE = REDQ_TARGET_MIN — min over `q_next[subset_idxs[m], b]` for m in 0..N_MIN
  MODE = REDQ_TARGET_AVE — mean over `q_next[n, b]` for n in 0..N_ENSEMBLE
  MODE = REDQ_TARGET_REM — random convex mixture (Philox draws per (b, n));
                          deferred to the GPU path

This file ports the legacy `deep_agents/redq/kernels.mojo`
`redq_ensemble_target_kernel` to the deep_agents2 idiom:
  - CPU version is a straight host loop (R.1).
  - Terminal mask is in the same kernel rather than a follow-up call
    to `apply_terminal_mask`. The legacy code did this; SAC's
    deep_agents2 TargetYBlock splits the bootstrap and the
    reward-add+mask into two passes (graph + helper). REDQ keeps it
    fused — one host loop over BATCH — because the combine step is
    already a loop and the cost of a separate mask pass would be
    pure overhead.

CleanRL natural-termination semantics: bootstrap is DROPPED on
`term=1` (real termination) and KEPT on `term=0` (time-limit
truncation). For truncation-only envs (`term ≡ 0`) the mask
reduces to `r + γ·(combined − α·lp)` — bit-identical to the
unmasked target.
"""

from mojo_rl.nn2.constants import DT


# =============================================================================
# MODE selectors — runtime ints carried as comptime params at the call site.
# Match the legacy `deep_agents/redq/config.mojo` constants exactly.
# =============================================================================


comptime REDQ_TARGET_MIN: Int = 0
comptime REDQ_TARGET_AVE: Int = 1
comptime REDQ_TARGET_REM: Int = 2


def redq_ensemble_target_cpu[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],
    q_next: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [N_ENSEMBLE, BATCH]
    terms: UnsafePointer[Scalar[DT], MutAnyOrigin],
    log_probs: UnsafePointer[Scalar[DT], MutAnyOrigin],
    subset_idxs: UnsafePointer[Int, MutAnyOrigin],     # [N_MIN]
    gamma: Scalar[DT],
    alpha: Scalar[DT],
    out_y: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Compute `out_y[b] = r[b] + (1 - term[b]) * γ * (combined_Q[b]
    - α * log_prob[b])` in-place, one host iteration per batch
    sample.

    `q_next` is the row-major [N_ENSEMBLE, BATCH] stack of target-Q
    evaluations; `q_next[n, b] = Q_target^n(s'[b], a'[b])`.

    `subset_idxs` (length N_MIN) names which target critics
    participate in MIN-mode. Set by the caller via host-side
    Fisher-Yates each step. Ignored when MODE != MIN.

    MODE = REM is not supported on CPU: the legacy GPU REM path uses
    Philox per-(sample, critic) draws and matching it on the CPU
    would just be a parity-tax with no model-side gain. Raise at the
    boundary so unsupported configs fail loudly."""
    comptime assert (
        MODE == REDQ_TARGET_MIN or MODE == REDQ_TARGET_AVE
    ), (
        "redq_ensemble_target_cpu: only MIN/AVE supported on CPU "
        "(REM is GPU-only; see kernels.mojo docstring)"
    )
    comptime assert N_MIN >= 1, "N_MIN must be >= 1"
    comptime assert N_MIN <= N_ENSEMBLE, "N_MIN must be <= N_ENSEMBLE"

    for b in range(BATCH):
        var combined: Scalar[DT] = Scalar[DT](0.0)
        comptime if MODE == REDQ_TARGET_MIN:
            var first_idx = subset_idxs[0]
            combined = q_next[first_idx * BATCH + b]
            for m in range(1, N_MIN):
                var v = q_next[subset_idxs[m] * BATCH + b]
                if v < combined:
                    combined = v
        else:
            # MODE == REDQ_TARGET_AVE
            var acc: Scalar[DT] = Scalar[DT](0.0)
            for n in range(N_ENSEMBLE):
                acc += q_next[n * BATCH + b]
            combined = acc / Scalar[DT](N_ENSEMBLE)

        var soft_v = combined - alpha * log_probs[b]
        var bootstrap = gamma * soft_v
        var nonterm = Scalar[DT](1.0) - terms[b]
        out_y[b] = rewards[b] + nonterm * bootstrap
