"""MuZero loss primitives — categorical soft cross-entropy (value/reward/policy).

All three MuZero heads are trained with **soft cross-entropy** against a soft
target distribution:

  * policy  — logits[ACT]  vs MCTS visit-count policy π
  * value   — logits[BINS] vs two-hot(h(value_target))   (categorical)
  * reward  — logits[BINS] vs two-hot(h(reward_target))  (categorical)

so one primitive serves all three. For a soft target ``q`` (a distribution),

    L = − Σ_i q_i · log softmax(z)_i
    ∂L/∂z = softmax(z) − q

which is the same analytic gradient AlphaZero's `AZLossOp` uses for its policy
term. The MuZero unroll (`blocks.mojo`) computes each head's grad-output
**slice** with this function, scales by the per-unroll-step factor ``1/(K+1)``
and ``1/BATCH``, packs the slices into the net's output-gradient tile, and runs
``Module.vjp`` — the DreamerV3-style manual forward/reverse scan, no monolithic
graph. Keeping the loss math here (analytic, gradchecked) is what makes that
scan trustworthy.
"""

from std.math import log, exp

from mojo_rl.nn.constants import DT


def softmax_row[
    NBINS: Int,
](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    base: Int,
    mut probs: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Numerically-stable softmax of ``logits[base .. base+NBINS)`` into
    ``probs[base .. base+NBINS)``."""
    var m = logits[base]
    for i in range(1, NBINS):
        var v = logits[base + i]
        if v > m:
            m = v
    var s = Scalar[DT](0.0)
    for i in range(NBINS):
        var e = exp(logits[base + i] - m)
        probs[base + i] = e
        s += e
    var inv = Scalar[DT](1.0) / s
    for i in range(NBINS):
        probs[base + i] = probs[base + i] * inv


def soft_ce_loss_and_grad[
    BATCH: Int, NBINS: Int,
](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_scale: Scalar[DT],
    mut grad_logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    """Soft cross-entropy over ``BATCH`` rows of ``NBINS`` logits vs soft
    targets. Returns the **summed** loss ``Σ_b −Σ_i q·log softmax(z)`` and writes
    ``grad_logits[b,i] = grad_scale · (softmax(z)_{b,i} − q_{b,i})``.

    ``grad_scale`` folds the caller's ``1/(K+1)`` step weight and ``1/BATCH``
    normalization (and any carry factor) into the analytic gradient in one place.
    """
    var total = Scalar[DT](0.0)
    for b in range(BATCH):
        var base = b * NBINS
        # stable log-softmax + accumulate loss, then grad = softmax − target.
        var m = logits[base]
        for i in range(1, NBINS):
            var v = logits[base + i]
            if v > m:
                m = v
        var s = Scalar[DT](0.0)
        for i in range(NBINS):
            s += exp(logits[base + i] - m)
        var log_s = log(s)
        for i in range(NBINS):
            var q = target[base + i]
            var log_sm = (logits[base + i] - m) - log_s
            total += -q * log_sm
            var sm = exp(log_sm)
            grad_logits[base + i] = grad_scale * (sm - q)
    return total


def soft_ce_slice_loss_and_grad[
    BATCH: Int, ROW: Int, OFF: Int, NBINS: Int,
](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_scale: Scalar[DT],
    mut grad_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    """Soft-CE over the ``[OFF, OFF+NBINS)`` column slice of a ``[BATCH, ROW]``
    logit tile (the MuZero heads pack policy / value / reward as adjacent slices
    of one network-output row). ``target`` is a contiguous ``[BATCH, NBINS]``
    distribution. Writes grad into the same ``[OFF, OFF+NBINS)`` slice of
    ``grad_out``; other columns of ``grad_out`` are left untouched (the caller
    fills the rest, e.g. the dynamics latent slice with the carry gradient).
    """
    var total = Scalar[DT](0.0)
    for b in range(BATCH):
        var base = b * ROW + OFF
        var m = logits[base]
        for i in range(1, NBINS):
            var v = logits[base + i]
            if v > m:
                m = v
        var s = Scalar[DT](0.0)
        for i in range(NBINS):
            s += exp(logits[base + i] - m)
        var log_s = log(s)
        var tb = b * NBINS
        for i in range(NBINS):
            var q = target[tb + i]
            var log_sm = (logits[base + i] - m) - log_s
            total += -q * log_sm
            var sm = exp(log_sm)
            grad_out[base + i] = grad_scale * (sm - q)
    return total
