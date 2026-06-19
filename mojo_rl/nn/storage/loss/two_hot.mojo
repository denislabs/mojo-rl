"""Two-hot encoding for distributional value heads (DreamerV3 / TD-MPC2).

Two-hot encoding converts a scalar `x` into a soft one-hot vector over
`NUM_BINS` evenly-spaced bin values:

  k = floor((x - v_min) / step),   step = (v_max - v_min) / (NUM_BINS - 1)
  target[k]     = (bins[k+1] - x) / step
  target[k + 1] = (x - bins[k])   / step
  target[i]     = 0  otherwise

The scalar value is decoded from logits as
  v = sum_i softmax(logits)_i * bins[i]
and (in DreamerV3's symlog form) `symexp(v)` recovers the real scale.

Storage-surface port of the legacy nn helpers
(`mojo_rl/nn/loss/two_hot.mojo`):
  * scalar helpers stay InlineArray-based (compile-time-sized inline use)
  * batched helpers operate on `Tensor` storages (CPU `.data` loops) instead
    of the legacy raw `UnsafePointer` form; they keep the `_ptr` suffix off
    their names since they no longer take pointers.
"""

from std.math import exp, log
from std.math import abs as math_abs

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor

# Scalar symlog / symexp live in the canonical shared math helper (DT-only)
# and are re-exported here so `from ...storage.loss.two_hot import symlog`
# call-sites keep working.
from mojo_rl.nn.primitives.ops.symlog_math import symlog, symexp


# ──────────────────────────────────────────────────────────────────────
# Bin construction (compile-time-sized InlineArray).
# ──────────────────────────────────────────────────────────────────────


def compute_bins[
    NUM_BINS: Int,
](v_min: Scalar[DT], v_max: Scalar[DT]) -> InlineArray[Scalar[DT], NUM_BINS]:
    """Evenly-spaced bins in [v_min, v_max]."""
    var bins = InlineArray[Scalar[DT], NUM_BINS](fill=0)
    if NUM_BINS == 1:
        bins[0] = (v_min + v_max) * Scalar[DT](0.5)
        return bins^
    var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins[i] = v_min + step * Scalar[DT](i)
    return bins^


def compute_symlog_bins[
    NUM_BINS: Int,
]() -> InlineArray[Scalar[DT], NUM_BINS]:
    """DreamerV3 default: bins evenly spaced in symlog space, range [-20, 20]."""
    var bins = InlineArray[Scalar[DT], NUM_BINS](fill=0)
    if NUM_BINS == 1:
        bins[0] = Scalar[DT](0.0)
        return bins^
    var step = Scalar[DT](40.0) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins[i] = Scalar[DT](-20.0) + step * Scalar[DT](i)
    return bins^


def fill_bins[
    NUM_BINS: Int,
](
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    mut bins: Tensor,
):
    """Tensor-form of `compute_bins` (writes the first NUM_BINS of `bins`)."""
    bins.ensure(NUM_BINS)
    if NUM_BINS == 1:
        bins.data[0] = (v_min + v_max) * Scalar[DT](0.5)
        return
    var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins.data[i] = v_min + step * Scalar[DT](i)


def fill_symlog_bins[
    NUM_BINS: Int,
](mut bins: Tensor):
    """Tensor-form of `compute_symlog_bins`."""
    bins.ensure(NUM_BINS)
    if NUM_BINS == 1:
        bins.data[0] = Scalar[DT](0.0)
        return
    var step = Scalar[DT](40.0) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins.data[i] = Scalar[DT](-20.0) + step * Scalar[DT](i)


# ──────────────────────────────────────────────────────────────────────
# Two-hot encoding (single-sample + batched).
# ──────────────────────────────────────────────────────────────────────


def two_hot_encode[
    NUM_BINS: Int,
](
    x: Scalar[DT],
    bins: InlineArray[Scalar[DT], NUM_BINS],
    mut target: InlineArray[Scalar[DT], NUM_BINS],
):
    """Encode scalar `x` into a two-hot distribution over `bins`."""
    for i in range(NUM_BINS):
        target[i] = Scalar[DT](0.0)

    var v_min = bins[0]
    var v_max = bins[NUM_BINS - 1]
    var x_clamped = x
    if x_clamped < v_min:
        x_clamped = v_min
    if x_clamped > v_max:
        x_clamped = v_max

    if NUM_BINS == 1:
        target[0] = Scalar[DT](1.0)
        return

    var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
    var k_float = (x_clamped - v_min) / step
    var k = Int(k_float)
    if k >= NUM_BINS - 1:
        k = NUM_BINS - 2

    var bin_low = bins[k]
    var bin_high = bins[k + 1]
    var width = bin_high - bin_low

    if math_abs(width) < Scalar[DT](1e-8):
        target[k] = Scalar[DT](1.0)
        return

    var upper_weight = (bin_high - x_clamped) / width
    var lower_weight = Scalar[DT](1.0) - upper_weight
    target[k] = upper_weight
    target[k + 1] = lower_weight


def two_hot_encode_batch[
    BATCH: Int, NUM_BINS: Int,
](
    ref values: Tensor,
    ref bins: Tensor,
    mut targets: Tensor,
):
    """Batch two-hot encode (CPU `.data` loops). Inputs:
        values  shape [BATCH]
        bins    shape [NUM_BINS]    (must be sorted ascending, evenly spaced)
        targets shape [BATCH * NUM_BINS] — written.
    """
    targets.ensure(BATCH * NUM_BINS)
    var v_min = bins.data[0]
    var v_max = bins.data[NUM_BINS - 1]

    for b in range(BATCH):
        var base = b * NUM_BINS
        var x = values.data[b]
        if x < v_min:
            x = v_min
        if x > v_max:
            x = v_max

        for i in range(NUM_BINS):
            targets.data[base + i] = Scalar[DT](0.0)

        if NUM_BINS == 1:
            targets.data[base] = Scalar[DT](1.0)
            continue

        var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
        var k_float = (x - v_min) / step
        var k = Int(k_float)
        if k >= NUM_BINS - 1:
            k = NUM_BINS - 2

        var bin_low = bins.data[k]
        var bin_high = bins.data[k + 1]
        var width = bin_high - bin_low

        if math_abs(width) < Scalar[DT](1e-8):
            targets.data[base + k] = Scalar[DT](1.0)
            continue

        var upper = (bin_high - x) / width
        targets.data[base + k] = upper
        targets.data[base + k + 1] = Scalar[DT](1.0) - upper


def two_hot_encode_symlog_batch[
    BATCH: Int, NUM_BINS: Int,
](
    ref values: Tensor,
    ref bins: Tensor,
    mut targets: Tensor,
):
    """DreamerV3-style: apply `symlog` to each value before encoding
    against symlog-spaced bins."""
    targets.ensure(BATCH * NUM_BINS)
    var v_min = bins.data[0]
    var v_max = bins.data[NUM_BINS - 1]

    for b in range(BATCH):
        var base = b * NUM_BINS
        var x = symlog(values.data[b])
        if x < v_min:
            x = v_min
        if x > v_max:
            x = v_max

        for i in range(NUM_BINS):
            targets.data[base + i] = Scalar[DT](0.0)

        if NUM_BINS == 1:
            targets.data[base] = Scalar[DT](1.0)
            continue

        var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
        var k_float = (x - v_min) / step
        var k = Int(k_float)
        if k >= NUM_BINS - 1:
            k = NUM_BINS - 2

        var bin_low = bins.data[k]
        var bin_high = bins.data[k + 1]
        var width = bin_high - bin_low

        if math_abs(width) < Scalar[DT](1e-8):
            targets.data[base + k] = Scalar[DT](1.0)
            continue

        var upper = (bin_high - x) / width
        targets.data[base + k] = upper
        targets.data[base + k + 1] = Scalar[DT](1.0) - upper


# ──────────────────────────────────────────────────────────────────────
# Decoding: logits → expected value.
# ──────────────────────────────────────────────────────────────────────


def decode_value[
    NUM_BINS: Int,
](
    logits: InlineArray[Scalar[DT], NUM_BINS],
    bins: InlineArray[Scalar[DT], NUM_BINS],
) -> Scalar[DT]:
    """Decode distributional value with symexp: returns
    `symexp(sum_i softmax(logits)_i * bins_i)` — bins live in symlog
    space, so the symexp recovers actual-value scale."""
    var max_val = logits[0]
    for i in range(1, NUM_BINS):
        if logits[i] > max_val:
            max_val = logits[i]

    var sum_exp = Scalar[DT](0.0)
    for i in range(NUM_BINS):
        sum_exp += exp(logits[i] - max_val)

    var value_symlog = Scalar[DT](0.0)
    for i in range(NUM_BINS):
        var prob = exp(logits[i] - max_val) / sum_exp
        value_symlog += prob * bins[i]

    return symexp(value_symlog)


def decode_value_batch[
    BATCH: Int, NUM_BINS: Int,
](
    ref logits: Tensor,
    ref bins: Tensor,
    mut values: Tensor,
):
    """Batch decode distributional values with symexp. `logits` shape
    [BATCH * NUM_BINS]; `bins` shape [NUM_BINS] (symlog space);
    `values` shape [BATCH] (actual-value space, written)."""
    values.ensure(BATCH)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var max_val = logits.data[base]
        for i in range(1, NUM_BINS):
            var lv = logits.data[base + i]
            if lv > max_val:
                max_val = lv

        var sum_exp = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            sum_exp += exp(logits.data[base + i] - max_val)

        var val_symlog = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            var prob = exp(logits.data[base + i] - max_val) / sum_exp
            val_symlog += prob * bins.data[i]

        values.data[b] = symexp(val_symlog)


def decode_value_batch_linear[
    BATCH: Int, NUM_BINS: Int,
](
    ref logits: Tensor,
    ref bins: Tensor,
    mut values: Tensor,
):
    """Like `decode_value_batch` but bins are in actual-value space
    (no symexp). For TD-MPC2 / pre-DreamerV3 distributional heads."""
    values.ensure(BATCH)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var max_val = logits.data[base]
        for i in range(1, NUM_BINS):
            var lv = logits.data[base + i]
            if lv > max_val:
                max_val = lv

        var sum_exp = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            sum_exp += exp(logits.data[base + i] - max_val)

        var val = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            var prob = exp(logits.data[base + i] - max_val) / sum_exp
            val += prob * bins.data[i]

        values.data[b] = val
