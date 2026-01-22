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

from gpu.host import HostBuffer


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


# =============================================================================
# Running Mean/Std for Reward Normalization (CleanRL-style)
# =============================================================================


struct RunningMeanStd:
    """Running mean and standard deviation tracker using Welford's algorithm.

    This is used for reward normalization in PPO and other algorithms.
    The running statistics are updated incrementally with each batch,
    which provides stable normalization across the entire training run.

    Reference: CleanRL's VecNormalize implementation.

    Example:
        var rms = RunningMeanStd()
        # During training:
        rms.update_batch(rewards_host, n_rewards)
        # Normalize rewards:
        for i in range(n):
            rewards[i] = (rewards[i] - rms.mean) / (rms.std() + 1e-8)
    """

    var count: Float64
    var mean: Float64
    var var_sum: Float64  # Sum of squared differences (M2 in Welford's)

    fn __init__(out self):
        """Initialize with zero statistics."""
        self.count = 0.0
        self.mean = 0.0
        self.var_sum = 0.0

    fn variance(self) -> Float64:
        """Return the running variance."""
        if self.count < 2:
            return 1.0  # Default to 1 to avoid division issues
        return self.var_sum / self.count

    fn std(self) -> Float64:
        """Return the running standard deviation."""
        return (self.variance() + 1e-8) ** 0.5

    fn update[
        dt: DType
    ](mut self, buffer: HostBuffer[dt], n: Int):
        """Update running statistics with a batch of values.

        Uses Welford's online algorithm for numerically stable updates.
        See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        Args:
            buffer: HostBuffer containing the values.
            n: Number of values to process from the buffer.
        """
        if n == 0:
            return

        # Compute batch statistics first
        var batch_mean = Float64(0.0)
        for i in range(n):
            batch_mean += Float64(buffer[i])
        batch_mean /= Float64(n)

        var batch_var = Float64(0.0)
        for i in range(n):
            var diff = Float64(buffer[i]) - batch_mean
            batch_var += diff * diff
        batch_var /= Float64(n)

        # Combine with running statistics using parallel algorithm
        # Reference: Chan et al., "Updating Formulae and a Pairwise Algorithm"
        var new_count = self.count + Float64(n)

        if self.count == 0:
            # First batch
            self.mean = batch_mean
            self.var_sum = batch_var * Float64(n)
        else:
            # Combine running and batch statistics
            var delta = batch_mean - self.mean
            var new_mean = self.mean + delta * Float64(n) / new_count

            # Update variance using parallel algorithm
            self.var_sum = (
                self.var_sum
                + batch_var * Float64(n)
                + delta * delta * self.count * Float64(n) / new_count
            )
            self.mean = new_mean

        self.count = new_count

    fn normalize[
        dt: DType
    ](self, mut buffer: HostBuffer[dt], n: Int, eps: Float64 = 1e-8):
        """Normalize values in-place using running statistics.

        Computes: buffer[i] = (buffer[i] - mean) / (std + eps)

        Args:
            buffer: HostBuffer to normalize in-place.
            n: Number of values to normalize.
            eps: Small constant for numerical stability.
        """
        var std_val = self.std()
        for i in range(n):
            buffer[i] = Scalar[dt](
                (Float64(buffer[i]) - self.mean) / (std_val + eps)
            )
