"""Softmax utility functions for policy gradient algorithms.

Provides numerically stable softmax implementations for converting logits
to probability distributions. These are standalone functions (not neural
network layers) used for action sampling in policy gradient methods.

For neural network layers with backpropagation support, use
`deep_rl.model.Softmax[dim]` instead.

Example usage:
    from core.utils.softmax import softmax, softmax_inline

    # For List-based code (tile-coded agents)
    var logits = List[Float64]()
    # ... compute logits ...
    var probs = softmax(logits)

    # For InlineArray-based code (deep agents)
    var logits = InlineArray[Scalar[DType.float32], 4](fill=0)
    var probs = softmax_inline(logits)
"""

from math import exp


fn softmax(logits: List[Float64]) -> List[Float64]:
    """Compute numerically stable softmax probabilities.

    Computes: probs[i] = exp(logits[i] - max) / sum(exp(logits[j] - max))

    The max subtraction prevents overflow for large logits.

    Args:
        logits: Input logits (unnormalized log-probabilities).

    Returns:
        Probability distribution summing to 1.

    Example:
        var logits = List[Float64]()
        logits.append(1.0)
        logits.append(2.0)
        logits.append(3.0)
        var probs = softmax(logits)
        # probs ≈ [0.09, 0.24, 0.67]
    """
    var n = len(logits)
    if n == 0:
        return List[Float64]()

    # Find max for numerical stability
    var max_logit = logits[0]
    for i in range(1, n):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Compute exp(logits - max) and sum
    var exp_logits = List[Float64]()
    var sum_exp: Float64 = 0.0
    for i in range(n):
        var e = exp(logits[i] - max_logit)
        exp_logits.append(e)
        sum_exp += e

    # Normalize
    var probs = List[Float64]()
    for i in range(n):
        probs.append(exp_logits[i] / sum_exp)

    return probs^


fn softmax_inline[
    dtype: DType, N: Int
](logits: InlineArray[Scalar[dtype], N]) -> InlineArray[Scalar[dtype], N]:
    """Compute numerically stable softmax probabilities for InlineArray.

    Computes: probs[i] = exp(logits[i] - max) / sum(exp(logits[j] - max))

    Args:
        logits: Input logits (unnormalized log-probabilities).

    Returns:
        Probability distribution summing to 1.

    Parameters:
        dtype: Data type of the values.
        N: Number of elements (actions).

    Example:
        var logits = InlineArray[Scalar[DType.float32], 4](fill=0)
        logits[0] = 1.0
        logits[1] = 2.0
        var probs = softmax_inline(logits)
    """
    var probs = InlineArray[Scalar[dtype], N](uninitialized=True)

    # Find max for numerical stability
    var max_logit = logits[0]
    for i in range(1, N):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Compute exp(logits - max) and sum
    var sum_exp = Scalar[dtype](0.0)
    for i in range(N):
        var e = exp(logits[i] - max_logit)
        probs[i] = e
        sum_exp += e

    # Normalize
    for i in range(N):
        probs[i] /= sum_exp

    return probs


fn log_softmax(logits: List[Float64]) -> List[Float64]:
    """Compute numerically stable log-softmax.

    Computes: log_probs[i] = logits[i] - max - log(sum(exp(logits[j] - max)))

    More numerically stable than log(softmax(logits)) for computing
    log probabilities directly.

    Args:
        logits: Input logits (unnormalized log-probabilities).

    Returns:
        Log-probabilities (sum of exp equals 1).

    Example:
        var logits = List[Float64]()
        logits.append(1.0)
        logits.append(2.0)
        var log_probs = log_softmax(logits)
        # To get probability of action 0: exp(log_probs[0])
    """
    var n = len(logits)
    if n == 0:
        return List[Float64]()

    # Find max for numerical stability
    var max_logit = logits[0]
    for i in range(1, n):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Compute log(sum(exp(logits - max)))
    var sum_exp: Float64 = 0.0
    for i in range(n):
        sum_exp += exp(logits[i] - max_logit)
    var log_sum_exp = max_logit + log(sum_exp)

    # Compute log_softmax
    var log_probs = List[Float64]()
    for i in range(n):
        log_probs.append(logits[i] - log_sum_exp)

    return log_probs^


fn log(x: Float64) -> Float64:
    """Natural logarithm (wrapper for math.log)."""
    from math import log as math_log

    return math_log(x)


fn sample_from_probs(probs: List[Float64]) -> Int:
    """Sample an action index from a probability distribution.

    Args:
        probs: Probability distribution over actions.

    Returns:
        Sampled action index.

    Example:
        var probs = softmax(logits)
        var action = sample_from_probs(probs)
    """
    from random import random_float64

    var rand = random_float64()
    var cumsum: Float64 = 0.0
    var n = len(probs)

    for i in range(n):
        cumsum += probs[i]
        if rand < cumsum:
            return i

    return n - 1  # Fallback for numerical precision


fn sample_from_probs_inline[
    dtype: DType, N: Int
](probs: InlineArray[Scalar[dtype], N]) -> Int:
    """Sample an action index from a probability distribution (InlineArray).

    Args:
        probs: Probability distribution over actions.

    Returns:
        Sampled action index.

    Parameters:
        dtype: Data type of the probabilities.
        N: Number of actions.
    """
    from random import random_float64

    var rand = Scalar[dtype](random_float64())
    var cumsum = Scalar[dtype](0.0)

    for i in range(N):
        cumsum += probs[i]
        if rand < cumsum:
            return i

    return N - 1  # Fallback for numerical precision


fn argmax_probs(probs: List[Float64]) -> Int:
    """Get the action with highest probability (greedy selection).

    Args:
        probs: Probability distribution over actions.

    Returns:
        Index of action with highest probability.
    """
    var n = len(probs)
    if n == 0:
        return 0

    var best_idx = 0
    var best_prob = probs[0]
    for i in range(1, n):
        if probs[i] > best_prob:
            best_prob = probs[i]
            best_idx = i

    return best_idx


fn argmax_probs_inline[
    dtype: DType, N: Int
](probs: InlineArray[Scalar[dtype], N]) -> Int:
    """Get the action with highest probability (greedy selection, InlineArray).

    Args:
        probs: Probability distribution over actions.

    Returns:
        Index of action with highest probability.

    Parameters:
        dtype: Data type of the probabilities.
        N: Number of actions.
    """
    var best_idx = 0
    var best_prob = probs[0]
    for i in range(1, N):
        if probs[i] > best_prob:
            best_prob = probs[i]
            best_idx = i

    return best_idx
