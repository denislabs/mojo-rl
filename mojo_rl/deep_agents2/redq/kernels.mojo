"""REDQ ensemble-target kernels (CPU + GPU).

Computes the REDQ TD target from a stacked block of N target-Q
evaluations:

    y[b] = r[b] + (1 - term[b]) * γ * (combined_Q(s', a')[b] - α * log_π(a'|s')[b])

`combined_Q` depends on `MODE`:
  MODE = REDQ_TARGET_MIN — min over `q_next[subset_idxs[m], b]` for m in 0..N_MIN
  MODE = REDQ_TARGET_AVE — mean over `q_next[n, b]` for n in 0..N_ENSEMBLE
  MODE = REDQ_TARGET_REM — random convex mixture (Philox draws per (b, n));
                          deferred (legacy GPU-only path; not on R.5's CPU+GPU
                          surface)

Ports the legacy `deep_agents/redq/kernels.mojo`
`redq_ensemble_target_kernel` to the deep_agents2 idiom:
  - CPU function is a straight host loop (R.1).
  - GPU kernel + host launcher mirror the same math (R.5).
  - Terminal mask is in the same kernel rather than a follow-up call
    to `apply_terminal_mask`. The legacy code did this; SAC's
    deep_agents2 TargetYBlock splits the bootstrap and the
    reward-add+mask into two passes (graph + helper). REDQ keeps it
    fused because the combine step is already a per-lane loop and a
    separate mask pass would be pure overhead.
  - α is passed as a Scalar (baked at kernel launch). REDQ does NOT
    capture under CUDA graphs (host control flow: subset sampling +
    policy-delay gating), so the device-α plumbing SAC needs for
    capture is unnecessary here.

CleanRL natural-termination semantics: bootstrap is DROPPED on
`term=1` (real termination) and KEPT on `term=0` (time-limit
truncation). For truncation-only envs (`term ≡ 0`) the mask
reduces to `r + γ·(combined − α·lp)` — bit-identical to the
unmasked target.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT, TPB


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


# =============================================================================
# GPU kernel — one thread per batch lane. Mirrors the CPU formula.
# =============================================================================


def _redq_ensemble_target_kernel[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    out_y: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    rewards: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    q_next: LayoutTensor[
        DT, Layout.row_major(N_ENSEMBLE, BATCH), MutAnyOrigin,
    ],
    terms: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    log_probs: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    subset_idxs: LayoutTensor[
        DType.uint32, Layout.row_major(N_MIN), MutAnyOrigin,
    ],
    gamma: Scalar[DT],
    alpha: Scalar[DT],
):
    """Compute the REDQ TD target on-device. Reads:
      - `q_next[N_ENSEMBLE, BATCH]` — stacked target-Q evaluations
      - `subset_idxs[N_MIN]` — host-uploaded subset for MODE=MIN
        (ignored when MODE=AVE; caller still allocates an N_MIN-sized
        buffer so the LayoutTensor type stays valid)

    Writes `out_y[b] = r[b] + (1 − term[b]) · γ · (combined_Q[b] −
    α · log_prob[b])`. NaN-guards both the per-critic Q read and the
    final target (match legacy semantics)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var combined: Scalar[DT] = Scalar[DT](0.0)
    comptime if MODE == REDQ_TARGET_MIN:
        var first_idx = Int(subset_idxs.ptr[0])
        var v0 = q_next.ptr[first_idx * BATCH + b]
        if v0 != v0:
            v0 = Scalar[DT](0.0)
        combined = v0
        comptime for m in range(1, N_MIN):
            var idx = Int(subset_idxs.ptr[m])
            var v = q_next.ptr[idx * BATCH + b]
            if v != v:
                v = Scalar[DT](0.0)
            if v < combined:
                combined = v
    comptime if MODE == REDQ_TARGET_AVE:
        var acc: Scalar[DT] = Scalar[DT](0.0)
        comptime for n in range(N_ENSEMBLE):
            var v = q_next.ptr[n * BATCH + b]
            if v != v:
                v = Scalar[DT](0.0)
            acc = acc + v
        combined = acc / Scalar[DT](N_ENSEMBLE)

    var lp = log_probs.ptr[b]
    var nonterm = Scalar[DT](1.0) - terms.ptr[b]
    var tgt = rewards.ptr[b] + nonterm * gamma * (combined - alpha * lp)
    if tgt != tgt:
        tgt = Scalar[DT](0.0)
    out_y.ptr[b] = tgt


def redq_ensemble_target_gpu[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    ctx: DeviceContext,
    out_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rewards_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    q_next_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    terms_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    log_probs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    subset_idxs_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    gamma: Scalar[DT],
    alpha: Scalar[DT],
) raises:
    """Host launcher for the REDQ ensemble-target kernel. Caller owns
    all the pointers as device storage. `subset_idxs_ptr` must point at
    a length-N_MIN uint32 device buffer (uploaded from host by the
    target_y block once per train_step)."""
    comptime assert (
        MODE == REDQ_TARGET_MIN or MODE == REDQ_TARGET_AVE
    ), (
        "redq_ensemble_target_gpu: only MIN/AVE supported (REM not"
        " ported; see kernels.mojo docstring)"
    )
    comptime assert N_MIN >= 1, "N_MIN must be >= 1"
    comptime assert N_MIN <= N_ENSEMBLE, "N_MIN must be <= N_ENSEMBLE"

    var out_y_lt = LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ](out_y_ptr)
    var rewards_lt = LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ](rewards_ptr)
    var q_next_lt = LayoutTensor[
        DT, Layout.row_major(N_ENSEMBLE, BATCH), MutAnyOrigin,
    ](q_next_ptr)
    var terms_lt = LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ](terms_ptr)
    var log_probs_lt = LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ](log_probs_ptr)
    var subset_lt = LayoutTensor[
        DType.uint32, Layout.row_major(N_MIN), MutAnyOrigin,
    ](subset_idxs_ptr)

    comptime n_blocks = (BATCH + TPB - 1) // TPB
    comptime kernel = _redq_ensemble_target_kernel[
        N_ENSEMBLE, N_MIN, MODE, BATCH,
    ]
    ctx.enqueue_function[kernel](
        out_y_lt,
        rewards_lt,
        q_next_lt,
        terms_lt,
        log_probs_lt,
        subset_lt,
        gamma,
        alpha,
        grid_dim=n_blocks,
        block_dim=TPB,
    )
