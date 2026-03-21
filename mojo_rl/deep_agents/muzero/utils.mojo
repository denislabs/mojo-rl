"""MuZero utility functions — scalar transforms and MinMax Q-value tracking.

Scalar transform: h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
Used to compress value and reward targets into a bounded range before
encoding as categorical distributions. The inverse is used to decode
network outputs back to scalar values.

MinMaxStats: Tracks min/max Q-values seen during MCTS search for
normalizing Q-values to [0, 1] in the PUCT formula.
"""

from std.math import sqrt, log, exp


# ═══════════════════════════════════════════════════════════════════════════
# Scalar Transform (MuZero value/reward encoding)
# ═══════════════════════════════════════════════════════════════════════════


fn scalar_transform(x: Float64, eps: Float64 = 0.001) -> Float64:
    """Apply MuZero scalar transform: h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x.

    Compresses large values into a bounded range for stable categorical
    distribution targets.

    Args:
        x: Raw scalar value (reward or n-step return).
        eps: Small linear term for invertibility (default: 0.001).

    Returns:
        Transformed scalar.
    """
    var sign = Float64(1.0) if x >= 0.0 else Float64(-1.0)
    var abs_x = x if x >= 0.0 else -x
    return sign * (sqrt(abs_x + 1.0) - 1.0) + eps * x


fn inverse_scalar_transform(y: Float64, eps: Float64 = 0.001) -> Float64:
    """Invert MuZero scalar transform: h^{-1}(y).

    Recovers the original scalar value from the transformed representation.
    Uses the closed-form inverse derived from the quadratic formula.

    Args:
        y: Transformed scalar value.
        eps: Same epsilon used in forward transform (default: 0.001).

    Returns:
        Original scalar value.
    """
    # Solve: y = sign(x)(sqrt(|x|+1) - 1) + eps*x
    # Let f = y - eps*x, then sign(x)(sqrt(|x|+1)-1) = f
    # After algebra: x = sign(y) * ((((1+4*eps*(|y|+1+eps))^0.5 - 1) / (2*eps))^2 - 1)
    var sign = Float64(1.0) if y >= 0.0 else Float64(-1.0)
    var abs_y = y if y >= 0.0 else -y

    # Closed-form inverse
    var inner = sqrt(1.0 + 4.0 * eps * (abs_y + 1.0 + eps))
    var f = ((inner - 1.0) / (2.0 * eps))
    return sign * (f * f - 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# MinMax Q-Value Normalization
# ═══════════════════════════════════════════════════════════════════════════


struct MinMaxStats(Movable, ImplicitlyCopyable):
    """Tracks min/max Q-values in the MCTS tree for PUCT normalization.

    Q-values are normalized to [0, 1] using the observed range, so the
    exploration constant c works consistently across different value scales.
    """

    var minimum: Float64
    var maximum: Float64

    fn __init__(out self):
        """Initialize with extreme values so first update always applies."""
        self.minimum = Float64(1e18)
        self.maximum = Float64(-1e18)

    fn __init__(out self, *, copy: Self):
        self.minimum = copy.minimum
        self.maximum = copy.maximum

    fn __init__(out self, *, deinit take: Self):
        self.minimum = take.minimum
        self.maximum = take.maximum

    fn update(mut self, value: Float64):
        """Update tracked range with a new Q-value.

        Args:
            value: New Q-value observed during MCTS backup.
        """
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    fn normalize(self, value: Float64) -> Float64:
        """Normalize a Q-value to [0, 1] using the tracked range.

        If the range is zero (no values seen, or all equal), returns the
        raw value unchanged.

        Args:
            value: Q-value to normalize.

        Returns:
            Normalized value in [0, 1].
        """
        var delta = self.maximum - self.minimum
        if delta > 0.0:
            return (value - self.minimum) / delta
        return value


# ═══════════════════════════════════════════════════════════════════════════
# Categorical Value Encoding (for distributional value/reward)
# ═══════════════════════════════════════════════════════════════════════════


fn compute_support[NUM_BINS: Int](v_min: Float64, v_max: Float64) -> InlineArray[
    Float64, NUM_BINS
]:
    """Compute evenly-spaced support bins for categorical value encoding.

    Args:
        v_min: Minimum value in the support.
        v_max: Maximum value in the support.

    Returns:
        InlineArray of NUM_BINS evenly-spaced values from v_min to v_max.
    """
    var bins = InlineArray[Float64, NUM_BINS](uninitialized=True)
    var step = (v_max - v_min) / Float64(NUM_BINS - 1) if NUM_BINS > 1 else Float64(
        0.0
    )
    for i in range(NUM_BINS):
        bins[i] = v_min + Float64(i) * step
    return bins


fn encode_categorical[
    NUM_BINS: Int,
    origin: MutOrigin,
](
    value: Float64,
    v_min: Float64,
    v_max: Float64,
    mut target: UnsafePointer[Float64, origin],
):
    """Encode a scalar as a soft two-hot categorical distribution.

    Distributes probability mass between the two nearest bins proportional
    to distance. Used for value and reward targets.

    Args:
        value: Scalar value to encode.
        v_min: Minimum support value.
        v_max: Maximum support value.
        target: Output buffer [NUM_BINS] to write probabilities into.
    """
    # Clamp to support range
    var clamped = value
    if clamped < v_min:
        clamped = v_min
    if clamped > v_max:
        clamped = v_max

    # Find position in support
    var step = (v_max - v_min) / Float64(NUM_BINS - 1) if NUM_BINS > 1 else Float64(
        1.0
    )
    var b = (clamped - v_min) / step

    # Integer bin indices
    var lo = Int(b)
    if lo >= NUM_BINS - 1:
        lo = NUM_BINS - 2
    if lo < 0:
        lo = 0
    var hi = lo + 1

    # Two-hot distribution
    var frac = b - Float64(lo)
    for i in range(NUM_BINS):
        target[i] = Float64(0.0)
    target[lo] = 1.0 - frac
    target[hi] = frac


fn decode_categorical[
    NUM_BINS: Int,
    origin: MutOrigin,
](
    logits: UnsafePointer[Float64, origin],
    v_min: Float64,
    v_max: Float64,
) -> Float64:
    """Decode a categorical distribution (logits) to a scalar value.

    Applies softmax to logits, then computes expectation over support.

    Args:
        logits: Raw network output [NUM_BINS].
        v_min: Minimum support value.
        v_max: Maximum support value.

    Returns:
        Expected scalar value.
    """
    var step = (v_max - v_min) / Float64(NUM_BINS - 1) if NUM_BINS > 1 else Float64(
        0.0
    )

    # Numerically stable softmax
    var max_val = logits[0]
    for i in range(1, NUM_BINS):
        if logits[i] > max_val:
            max_val = logits[i]

    var sum_exp = Float64(0.0)
    for i in range(NUM_BINS):
        sum_exp += exp(Float64(logits[i]) - max_val)

    # Expected value
    var result = Float64(0.0)
    for i in range(NUM_BINS):
        var prob = exp(Float64(logits[i]) - max_val) / sum_exp
        var bin_val = v_min + Float64(i) * step
        result += prob * bin_val

    return result


fn softmax_inplace[
    N: Int,
    origin: MutOrigin,
](mut data: UnsafePointer[Float64, origin]):
    """Apply softmax in-place to a buffer of N values.

    Args:
        data: Buffer of N logits, overwritten with probabilities.
    """
    var max_val = data[0]
    for i in range(1, N):
        if data[i] > max_val:
            max_val = data[i]

    var sum_exp = Float64(0.0)
    for i in range(N):
        data[i] = exp(data[i] - max_val)
        sum_exp += data[i]

    var inv_sum = 1.0 / sum_exp
    for i in range(N):
        data[i] *= inv_sum


fn cross_entropy_with_softmax[
    N: Int,
    o1: MutOrigin,
    o2: MutOrigin,
](
    logits: UnsafePointer[Float64, o1],
    target: UnsafePointer[Float64, o2],
) -> Float64:
    """Compute cross-entropy loss between softmax(logits) and target distribution.

    Args:
        logits: Raw network output [N].
        target: Target distribution [N] (probabilities summing to 1).

    Returns:
        Cross-entropy loss value.
    """
    # Numerically stable log-softmax
    var max_val = logits[0]
    for i in range(1, N):
        if logits[i] > max_val:
            max_val = logits[i]

    var sum_exp = Float64(0.0)
    for i in range(N):
        sum_exp += exp(logits[i] - max_val)
    var log_sum = log(sum_exp) + max_val

    var loss = Float64(0.0)
    for i in range(N):
        if target[i] > 0.0:
            loss -= target[i] * (logits[i] - log_sum)

    return loss
