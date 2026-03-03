"""Two-Hot Encoding for Distributional RL (TDMPC2).

TDMPC2 uses distributional RL: rewards and values are represented as
distributions over num_bins evenly-spaced bins in [v_min, v_max].

Two-hot encoding converts a scalar x into a soft one-hot vector:
  - Find adjacent bins: bins[k] <= x < bins[k+1]
  - target[k]   = (bins[k+1] - x) / (bins[k+1] - bins[k])  (upper weight)
  - target[k+1] = (x - bins[k])   / (bins[k+1] - bins[k])  (lower weight)
  - target[i]   = 0  for all other i

The scalar value is decoded from logits as:
  value = sum_i(softmax(logits)_i * bins[i])

Typical values: num_bins=101, v_min=-10.0, v_max=10.0
"""

from math import exp, log
from ..constants import dtype


fn compute_bins[NUM_BINS: Int](
    v_min: Float32, v_max: Float32
) -> InlineArray[Float32, NUM_BINS]:
    """Compute evenly spaced bin values for distributional RL.

    Args:
        v_min: Minimum bin value.
        v_max: Maximum bin value.

    Returns:
        Array of NUM_BINS evenly spaced values from v_min to v_max.
    """
    var bins = InlineArray[Float32, NUM_BINS](fill=0)
    if NUM_BINS == 1:
        bins[0] = (v_min + v_max) * 0.5
        return bins^
    var step = (v_max - v_min) / Float32(NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins[i] = v_min + step * Float32(i)
    return bins^


fn two_hot_encode[NUM_BINS: Int](
    x: Float32,
    bins: InlineArray[Float32, NUM_BINS],
    mut target: InlineArray[Float32, NUM_BINS],
):
    """Encode scalar x as a two-hot distribution over bins.

    Args:
        x: Scalar value to encode.
        bins: Array of NUM_BINS bin values (evenly spaced).
        target: Output two-hot vector of size NUM_BINS (written).
    """
    # Zero out target
    for i in range(NUM_BINS):
        target[i] = Float32(0.0)

    # Clamp x to bin range
    var v_min = bins[0]
    var v_max = bins[NUM_BINS - 1]
    var x_clamped = x
    if x_clamped < v_min:
        x_clamped = v_min
    if x_clamped > v_max:
        x_clamped = v_max

    # Handle degenerate case
    if NUM_BINS == 1:
        target[0] = Float32(1.0)
        return

    # Find lower bin index k such that bins[k] <= x_clamped < bins[k+1]
    var step = (v_max - v_min) / Float32(NUM_BINS - 1)
    var k_float = (x_clamped - v_min) / step
    var k = Int(k_float)
    if k >= NUM_BINS - 1:
        k = NUM_BINS - 2  # clamp to last valid pair

    var bin_low = bins[k]
    var bin_high = bins[k + 1]
    var width = bin_high - bin_low

    if width < 1e-8:
        # Degenerate: put all weight on lower bin
        target[k] = Float32(1.0)
        return

    # Two-hot weights
    var upper_weight = (bin_high - x_clamped) / width
    var lower_weight = Float32(1.0) - upper_weight

    target[k] = upper_weight
    target[k + 1] = lower_weight


fn two_hot_encode_batch[BATCH: Int, NUM_BINS: Int](
    values: InlineArray[Float32, BATCH],
    bins: InlineArray[Float32, NUM_BINS],
    mut targets: InlineArray[Float32, BATCH * NUM_BINS],
):
    """Batch version of two_hot_encode.

    Args:
        values: Batch of scalar values to encode [BATCH].
        bins: Bin values [NUM_BINS].
        targets: Output two-hot vectors [BATCH * NUM_BINS] (written).
    """
    for b in range(BATCH):
        var v_min = bins[0]
        var v_max = bins[NUM_BINS - 1]
        var x = values[b]
        if x < v_min:
            x = v_min
        if x > v_max:
            x = v_max

        var base = b * NUM_BINS
        for i in range(NUM_BINS):
            targets[base + i] = Float32(0.0)

        if NUM_BINS == 1:
            targets[base] = Float32(1.0)
            continue

        var step = (v_max - v_min) / Float32(NUM_BINS - 1)
        var k_float = (x - v_min) / step
        var k = Int(k_float)
        if k >= NUM_BINS - 1:
            k = NUM_BINS - 2

        var bin_low = bins[k]
        var bin_high = bins[k + 1]
        var width = bin_high - bin_low

        if width < Float32(1e-8):
            targets[base + k] = Float32(1.0)
            continue

        var upper_weight = (bin_high - x) / width
        targets[base + k] = upper_weight
        targets[base + k + 1] = Float32(1.0) - upper_weight


fn decode_value[NUM_BINS: Int](
    logits: InlineArray[Float32, NUM_BINS],
    bins: InlineArray[Float32, NUM_BINS],
) -> Float32:
    """Decode distributional value: sum_i(softmax(logits)_i * bins_i).

    Args:
        logits: Raw logits over bins [NUM_BINS].
        bins: Bin values [NUM_BINS].

    Returns:
        Expected value under the distribution.
    """
    # Numerically stable softmax
    var max_val = logits[0]
    for i in range(1, NUM_BINS):
        if logits[i] > max_val:
            max_val = logits[i]

    var sum_exp = Float32(0.0)
    for i in range(NUM_BINS):
        sum_exp += exp(logits[i] - max_val)

    var value = Float32(0.0)
    for i in range(NUM_BINS):
        var prob = exp(logits[i] - max_val) / sum_exp
        value += prob * bins[i]

    return value


fn decode_value_batch[BATCH: Int, NUM_BINS: Int](
    logits: InlineArray[Float32, BATCH * NUM_BINS],
    bins: InlineArray[Float32, NUM_BINS],
    mut values: InlineArray[Float32, BATCH],
):
    """Batch decode distributional values.

    Args:
        logits: Batch of logits [BATCH * NUM_BINS].
        bins: Bin values [NUM_BINS].
        values: Output expected values [BATCH] (written).
    """
    for b in range(BATCH):
        var base = b * NUM_BINS
        # Find max for stability
        var max_val = logits[base]
        for i in range(1, NUM_BINS):
            if logits[base + i] > max_val:
                max_val = logits[base + i]
        # Compute expected value
        var sum_exp = Float32(0.0)
        for i in range(NUM_BINS):
            sum_exp += exp(logits[base + i] - max_val)
        var val = Float32(0.0)
        for i in range(NUM_BINS):
            var prob = exp(logits[base + i] - max_val) / sum_exp
            val += prob * bins[i]
        values[b] = val
