from std.math import cos, sin, log, sqrt, pi
from std.random import random_float64


# =============================================================================
# Gaussian Random Number Generation
# =============================================================================


fn gaussian_noise() -> Float64:
    """Generate standard Gaussian noise (mean=0, std=1) using Box-Muller transform.

    Uses the standard library's random_float64() for uniform samples.

    Returns:
        A sample from N(0, 1).

    Example:
        var noise = gaussian_noise()
        var scaled_noise = noise * std + mean  # For N(mean, std).
    """
    var u1 = random_float64()
    var u2 = random_float64()
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


fn gaussian_noise_pair() -> Tuple[Float64, Float64]:
    """Generate two independent standard Gaussian samples using Box-Muller.

    More efficient when you need multiple samples, as Box-Muller
    naturally produces two independent values.

    Returns:
        Tuple of two samples from N(0, 1).

    Example:
        var (z1, z2) = gaussian_noise_pair().
    """
    var u1 = random_float64()
    var u2 = random_float64()
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    var r = sqrt(-2.0 * log(u1))
    var theta = 2.0 * pi * u2
    return (r * cos(theta), r * sin(theta))
