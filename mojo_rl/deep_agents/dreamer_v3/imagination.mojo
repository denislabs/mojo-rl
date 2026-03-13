"""Imagination rollout utilities for DreamerV3.

Contains lambda return computation, return normalization, and tanh-normal
distribution utilities used during imagination-based policy optimization.

Reference: Hafner et al., 2023 — Mastering Diverse Domains through
World Models (DreamerV3)
"""

from std.math import exp, log, sqrt, abs
from mojo_rl.nn.constants import dtype


# =============================================================================
# Lambda Returns
# =============================================================================


fn compute_lambda_returns[
    HORIZON: Int,
    BATCH: Int,
](
    rewards: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    continues: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mut returns: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gamma: Float64,
    lambda_: Float64,
):
    """Compute generalized lambda returns via backward scan.

    ret[H-1] = values[H-1]  (bootstrap from last value)
    For h = H-2..0:
      ret[h] = rew[h] + gamma * cont[h] * ((1-lam)*val[h+1] + lam*ret[h+1])

    Args:
        rewards: Predicted rewards [HORIZON * BATCH].
        values: Critic value estimates [HORIZON * BATCH].
        continues: Continuation probabilities [HORIZON * BATCH].
        returns: Output buffer for lambda returns [HORIZON * BATCH].
        gamma: Discount factor.
        lambda_: Lambda parameter for mixing TD and MC returns.
    """
    # Last timestep: bootstrap from critic value
    for b in range(BATCH):
        returns[(HORIZON - 1) * BATCH + b] = values[(HORIZON - 1) * BATCH + b]

    # Backward scan
    for h in range(HORIZON - 2, -1, -1):
        for b in range(BATCH):
            var idx = h * BATCH + b
            var next_idx = (h + 1) * BATCH + b
            var r = Float64(rewards[idx])
            var c = Float64(continues[idx])
            var v_next = Float64(values[next_idx])
            var ret_next = Float64(returns[next_idx])
            var ret = r + gamma * c * (
                (1.0 - lambda_) * v_next + lambda_ * ret_next
            )
            returns[idx] = Scalar[dtype](ret)


# =============================================================================
# Return Normalization
# =============================================================================


fn normalize_returns[
    HORIZON: Int,
    BATCH: Int,
](
    returns: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mut return_ema_lo: Float64,
    mut return_ema_hi: Float64,
    rate: Float64 = 0.01,
) -> Float64:
    """Normalize returns using percentile-based EMA scaling.

    Tracks running 5th/95th percentile estimates via EMA and normalizes
    returns to the range [0, 1] based on these bounds. This is the
    DreamerV3 return normalization scheme.

    Args:
        returns: Lambda returns [HORIZON * BATCH], modified in-place.
        return_ema_lo: Running EMA of the low percentile (updated in-place).
        return_ema_hi: Running EMA of the high percentile (updated in-place).
        rate: EMA decay rate for percentile tracking.

    Returns:
        The scale used for normalization (hi - lo, clamped to >= 1.0).
    """
    var total = HORIZON * BATCH

    # Approximate 5th/95th percentiles with min/max of current batch.
    # A full percentile computation would require sorting; this is a
    # lightweight approximation that works well in practice.
    var lo = Float64(returns[0])
    var hi = Float64(returns[0])
    for i in range(1, total):
        var v = Float64(returns[i])
        if v < lo:
            lo = v
        if v > hi:
            hi = v

    # EMA update of percentile estimates
    return_ema_lo = (1.0 - rate) * return_ema_lo + rate * lo
    return_ema_hi = (1.0 - rate) * return_ema_hi + rate * hi

    # Normalize with clamped scale
    var scale = return_ema_hi - return_ema_lo
    if scale < 1.0:
        scale = 1.0

    for i in range(total):
        var v = Float64(returns[i])
        returns[i] = Scalar[dtype]((v - return_ema_lo) / scale)

    return scale


# =============================================================================
# Tanh-Normal Distribution
# =============================================================================


fn sample_tanh_normal(
    mean: Float64,
    log_std: Float64,
    noise: Float64,
) -> Float64:
    """Sample from a tanh-squashed normal distribution.

    Computes action = tanh(mean + std * noise) where noise ~ N(0,1).
    Used by the DreamerV3 actor for continuous action spaces.

    Args:
        mean: Mean of the underlying normal distribution.
        log_std: Log standard deviation of the underlying normal.
        noise: Pre-sampled standard normal variate.

    Returns:
        Sampled action in (-1, 1).
    """
    var std = exp(log_std)
    if std < 1e-6:
        std = 1e-6
    var pre_tanh = mean + std * noise
    # tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    var ep = exp(pre_tanh)
    var en = exp(-pre_tanh)
    return (ep - en) / (ep + en)


fn log_prob_tanh_normal(
    action: Float64,
    mean: Float64,
    log_std: Float64,
) -> Float64:
    """Log probability under a tanh-squashed normal distribution.

    Computes log p(action) = log N(atanh(action) | mean, std)
                            - log(1 - action^2 + eps)

    The second term is the log-det-Jacobian correction for the tanh
    change of variables.

    Args:
        action: Observed action in (-1, 1).
        mean: Mean of the underlying normal distribution.
        log_std: Log standard deviation of the underlying normal.

    Returns:
        Log probability of the action.
    """
    var std = exp(log_std)
    if std < 1e-6:
        std = 1e-6

    # Clamp action to valid atanh domain
    var eps = 1e-6
    var a_clamped = action
    if a_clamped > 1.0 - eps:
        a_clamped = 1.0 - eps
    if a_clamped < -1.0 + eps:
        a_clamped = -1.0 + eps

    # atanh(a) = 0.5 * ln((1+a)/(1-a))
    var pre_tanh = 0.5 * log((1.0 + a_clamped) / (1.0 - a_clamped))

    # Normal log probability: -0.5*z^2 - log(std) - 0.5*log(2*pi)
    var z = (pre_tanh - mean) / std
    var log_normal = -0.5 * z * z - log_std - 0.9189385332046727

    # Tanh Jacobian correction: -log(1 - tanh^2(pre_tanh))
    var log_det = log(1.0 - a_clamped * a_clamped + eps)

    return log_normal - log_det
