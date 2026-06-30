"""MuZero categorical value/reward targets — scalar transform + two-hot encode.

MuZero trains the value and reward heads as **categorical** distributions: the
raw scalar target ``x`` is first compressed by the MuZero scalar transform

    h(x) = sign(x)·(√(|x|+1) − 1) + ε·x          (ε = 0.001)

then two-hot encoded over ``NUM_BINS`` evenly-spaced bins in ``[v_min, v_max]``.
The network emits logits; the loss is soft cross-entropy against the two-hot
target. Decoding inverts: softmax → expectation over the linear bins (recovers
``h(x)``) → ``h⁻¹`` back to the raw scalar.

**This convention is locked to the GPU MCTS kernel.** The learned-dynamics
backup (`mcts_gpu.mojo` `gpu_mcts_batched_expand_backup_muzero_kernel`) decodes
reward/value *inline* as `softmax · (v_min + i·step)` followed by the same
`h⁻¹` (ε=0.001). So the support ``[v_min, v_max]`` is in **transformed
(h-space)**, the bins are **linear** (NOT symlog — DreamerV3's `symlog` path is
a different transform and must not be used here), and the planner's
``v_min/v_max`` constructor args must equal the ones passed here. Getting any of
these inconsistent silently corrupts training.

Wraps the validated storage two-hot primitives (`compute_bins`,
`two_hot_encode` from `nn.storage.loss.two_hot`); the decode is inlined here
(softmax·linear-bins expectation, no pointer primitive) and the ``h``/``h⁻¹``
scalar transform is local. List-based (no raw pointers).
"""

from std.math import sqrt, exp

from mojo_rl.nn.constants import DT
from mojo_rl.nn.loss.two_hot import compute_bins, two_hot_encode


# ──────────────────────────────────────────────────────────────────────
# Scalar transform h(x) and its inverse (MuZero, ε = 0.001).
# ──────────────────────────────────────────────────────────────────────


@always_inline
def mz_scalar_transform(
    x: Scalar[DT], eps: Scalar[DT] = Scalar[DT](0.001)
) -> Scalar[DT]:
    """``h(x) = sign(x)·(√(|x|+1) − 1) + ε·x`` — compress a raw value/reward
    into the bounded h-space the categorical heads are trained on."""
    var sign = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var abs_x = x if x >= Scalar[DT](0.0) else -x
    return sign * (sqrt(abs_x + Scalar[DT](1.0)) - Scalar[DT](1.0)) + eps * x


@always_inline
def mz_inverse_scalar_transform(
    y: Scalar[DT], eps: Scalar[DT] = Scalar[DT](0.001)
) -> Scalar[DT]:
    """``h⁻¹(y)`` — closed-form inverse of `mz_scalar_transform`. Identical to
    the `h⁻¹` the GPU MCTS kernel applies after the categorical expectation."""
    var sign = Scalar[DT](1.0) if y >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var abs_y = y if y >= Scalar[DT](0.0) else -y
    var inner = sqrt(
        Scalar[DT](1.0)
        + Scalar[DT](4.0) * eps * (abs_y + Scalar[DT](1.0) + eps)
    )
    var f = (inner - Scalar[DT](1.0)) / (Scalar[DT](2.0) * eps)
    return sign * (f * f - Scalar[DT](1.0))


# ──────────────────────────────────────────────────────────────────────
# Two-hot target encoding (raw scalar → h(x) → two-hot over linear bins).
# ──────────────────────────────────────────────────────────────────────


def mz_two_hot_target_batch[
    BATCH: Int, NUM_BINS: Int,
](
    values: List[Scalar[DT]],
    val_off: Int,
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    mut targets: List[Scalar[DT]],
    tgt_off: Int,
):
    """Encode ``BATCH`` raw scalars (``values[val_off ..]``) into two-hot
    categorical targets written at ``targets[tgt_off + b·NUM_BINS ..]``.

    For each scalar: apply ``h(x)``, then two-hot over ``NUM_BINS`` evenly-spaced
    bins in ``[v_min, v_max]`` (h-space support; sums to 1 per row). Used for both
    value and reward targets (same machinery, different scalar streams).
    """
    var bins = compute_bins[NUM_BINS](v_min, v_max)
    for b in range(BATCH):
        var ht = mz_scalar_transform(values[val_off + b])
        var tgt = InlineArray[Scalar[DT], NUM_BINS](fill=0)
        two_hot_encode[NUM_BINS](ht, bins, tgt)
        var base = b * NUM_BINS
        for i in range(NUM_BINS):
            targets[tgt_off + base + i] = tgt[i]


def mz_two_hot_target_one[
    NUM_BINS: Int,
](
    value: Scalar[DT],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    mut target: List[Scalar[DT]],
    tgt_off: Int,
):
    """Single-sample `mz_two_hot_target_batch`; writes
    ``target[tgt_off .. tgt_off+NUM_BINS)``."""
    var bins = compute_bins[NUM_BINS](v_min, v_max)
    var ht = mz_scalar_transform(value)
    var tgt = InlineArray[Scalar[DT], NUM_BINS](fill=0)
    two_hot_encode[NUM_BINS](ht, bins, tgt)
    for i in range(NUM_BINS):
        target[tgt_off + i] = tgt[i]


# ──────────────────────────────────────────────────────────────────────
# Decode (logits → raw scalar): softmax · linear bins (→ h-space) → h⁻¹.
# ──────────────────────────────────────────────────────────────────────


def mz_decode_value_batch[
    BATCH: Int, NUM_BINS: Int,
](
    logits: List[Scalar[DT]],
    log_off: Int,
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    mut values: List[Scalar[DT]],
    val_off: Int,
):
    """Decode ``BATCH`` categorical logits (``logits[log_off + b·NUM_BINS ..]``)
    back to raw scalars written at ``values[val_off + b]`` — the inverse of
    `mz_two_hot_target_batch`, matching the GPU MCTS kernel's inline decode.

    softmax(logits) · (linear bins in [v_min, v_max]) recovers ``h(x)``; then
    ``h⁻¹`` recovers ``x``. Used by reanalyze / host-side sanity decoding. The
    softmax·linear-bins expectation is inlined here (no pointer primitive) and
    must stay bit-identical to the GPU MCTS kernel's h-space decode.
    """
    var bins = compute_bins[NUM_BINS](v_min, v_max)
    for b in range(BATCH):
        var base = log_off + b * NUM_BINS
        # numerically-stable softmax over the row.
        var m = logits[base]
        for i in range(1, NUM_BINS):
            var v = logits[base + i]
            if v > m:
                m = v
        var s = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            s += exp(logits[base + i] - m)
        var inv = Scalar[DT](1.0) / s
        # h-space expectation Σ softmax_i · bins[i].
        var ev = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            ev += (exp(logits[base + i] - m) * inv) * bins[i]
        values[val_off + b] = mz_inverse_scalar_transform(ev)
    _ = bins^
