"""Normalization utilities for RL algorithms.

Provides functions for normalizing advantages and other values to have
zero mean and unit variance, which is commonly used in policy gradient
methods like PPO to stabilize training.

Example usage:
    from core.utils.normalization import normalize, normalize_inline

    # For List-based code
    var advantages = List[Float64]()
    # ... fill advantages ...
    normalize(advantages)

    # For InlineArray-based code
    var advantages = InlineArray[Scalar[DType.float32], 2048](fill=0)
    normalize_inline(buffer_len, advantages)
"""


fn normalize(mut values: List[Float64], eps: Float64 = 1e-8):
    """Normalize values in-place to have zero mean and unit variance.

    Computes: values[i] = (values[i] - mean) / (std + eps)

    Args:
        values: List of values to normalize (modified in-place).
        eps: Small constant for numerical stability (default: 1e-8).

    Example:
        var advantages = List[Float64]()
        for i in range(100):
            advantages.append(some_value)
        normalize(advantages)
        # advantages now has mean ≈ 0 and std ≈ 1
    """
    var n = len(values)
    if n <= 1:
        return

    # Compute mean
    var mean: Float64 = 0.0
    for i in range(n):
        mean += values[i]
    mean /= Float64(n)

    # Compute variance
    var variance: Float64 = 0.0
    for i in range(n):
        var diff = values[i] - mean
        variance += diff * diff
    variance /= Float64(n)

    # Normalize
    var std = (variance + eps) ** 0.5
    for i in range(n):
        values[i] = (values[i] - mean) / std


fn normalize_inline[
    dtype: DType, N: Int
](n: Int, mut values: InlineArray[Scalar[dtype], N], eps: Float64 = 1e-8):
    """Normalize values in an InlineArray to have zero mean and unit variance.

    Computes: values[i] = (values[i] - mean) / (std + eps) for i in [0, n)

    Args:
        n: Number of elements to normalize (must be <= N).
        values: Array of values to normalize (modified in-place).
        eps: Small constant for numerical stability (default: 1e-8).

    Parameters:
        dtype: Data type of the values.
        N: Maximum capacity of the InlineArray.

    Example:
        var advantages = InlineArray[Scalar[DType.float32], 2048](fill=0)
        # ... fill advantages ...
        normalize_inline(buffer_len, advantages)
    """
    if n <= 1:
        return

    # Compute mean
    var mean = Scalar[dtype](0.0)
    for i in range(n):
        mean += values[i]
    mean /= Scalar[dtype](n)

    # Compute variance
    var variance = Scalar[dtype](0.0)
    for i in range(n):
        var diff = values[i] - mean
        variance += diff * diff
    variance /= Scalar[dtype](n)

    # Normalize
    var std = (variance + Scalar[dtype](eps)) ** 0.5
    for i in range(n):
        values[i] = (values[i] - mean) / std


fn compute_mean(values: List[Float64]) -> Float64:
    """Compute the mean of a list of values.

    Args:
        values: List of values.

    Returns:
        Mean of the values, or 0 if empty.
    """
    var n = len(values)
    if n == 0:
        return 0.0

    var sum: Float64 = 0.0
    for i in range(n):
        sum += values[i]
    return sum / Float64(n)


fn compute_std(values: List[Float64], mean: Float64) -> Float64:
    """Compute the standard deviation of a list of values.

    Args:
        values: List of values.
        mean: Pre-computed mean of the values.

    Returns:
        Standard deviation of the values, or 0 if empty/single element.
    """
    var n = len(values)
    if n <= 1:
        return 0.0

    var variance: Float64 = 0.0
    for i in range(n):
        var diff = values[i] - mean
        variance += diff * diff
    variance /= Float64(n)

    return variance ** 0.5


fn compute_mean_std(values: List[Float64]) -> Tuple[Float64, Float64]:
    """Compute both mean and standard deviation in one pass.

    More efficient than calling compute_mean and compute_std separately.

    Args:
        values: List of values.

    Returns:
        Tuple of (mean, std).

    Example:
        var (mean, std) = compute_mean_std(rewards)
    """
    var n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], 0.0)

    # Compute mean
    var sum: Float64 = 0.0
    for i in range(n):
        sum += values[i]
    var mean = sum / Float64(n)

    # Compute variance
    var variance: Float64 = 0.0
    for i in range(n):
        var diff = values[i] - mean
        variance += diff * diff
    variance /= Float64(n)

    return (mean, variance ** 0.5)
