from math import cos, sin, log, sqrt, pi
from random import random_float64


# =============================================================================
# GPU Random Number Generator (shared utility)
# =============================================================================


@always_inline
fn xorshift32(state: Scalar[DType.uint32]) -> Scalar[DType.uint32]:
    """Simple xorshift PRNG - fast and GPU-friendly."""
    var x = state
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


@always_inline
fn random_uniform[
    dtype: DType
](rng: Scalar[DType.uint32]) -> Tuple[Scalar[dtype], Scalar[DType.uint32]]:
    """Generate uniform random number in [0, 1) and return (value, new_rng_state).
    """
    var new_rng = xorshift32(rng)
    var value = Scalar[dtype](new_rng) / Scalar[dtype](Scalar[DType.uint32].MAX)
    return (value, new_rng)


@always_inline
fn random_range[
    dtype: DType
](rng: Scalar[DType.uint32], low: Scalar[dtype], high: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[DType.uint32]
]:
    """Generate uniform random number in [low, high) and return (value, new_rng_state).
    """
    var result = random_uniform[dtype](rng)
    var value = low + result[0] * (high - low)
    return (value, result[1])


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
        var scaled_noise = noise * std + mean  # For N(mean, std)
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
        var (z1, z2) = gaussian_noise_pair()
    """
    var u1 = random_float64()
    var u2 = random_float64()
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    var r = sqrt(-2.0 * log(u1))
    var theta = 2.0 * pi * u2
    return (r * cos(theta), r * sin(theta))


@always_inline
fn gaussian_noise_gpu[
    dtype: DType
](rng: Scalar[DType.uint32]) -> Tuple[Scalar[dtype], Scalar[DType.uint32]]:
    """Generate standard Gaussian noise on GPU using Box-Muller transform.

    GPU-friendly version that maintains RNG state for deterministic sequences.

    Args:
        rng: Current RNG state.

    Returns:
        Tuple of (gaussian_sample, new_rng_state).

    Example:
        var rng_state = Scalar[DType.uint32](12345)
        var (noise, rng_state) = gaussian_noise_gpu[DType.float32](rng_state)
    """
    # Generate two uniform samples
    var result1 = random_uniform[dtype](rng)
    var u1 = result1[0]
    var rng1 = result1[1]

    var result2 = random_uniform[dtype](rng1)
    var u2 = result2[0]
    var rng2 = result2[1]

    # Avoid log(0)
    if u1 < Scalar[dtype](1e-10):
        u1 = Scalar[dtype](1e-10)

    # Box-Muller transform (returns one of the two values)
    var r = sqrt(Scalar[dtype](-2.0) * log(u1))
    var theta = Scalar[dtype](2.0 * pi) * u2
    var z = r * cos(theta)

    return (z, rng2)


@always_inline
fn gaussian_noise_pair_gpu[
    dtype: DType
](rng: Scalar[DType.uint32]) -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[DType.uint32]
]:
    """Generate two independent Gaussian samples on GPU using Box-Muller.

    More efficient when you need multiple samples per thread.

    Args:
        rng: Current RNG state.

    Returns:
        Tuple of (z1, z2, new_rng_state).
    """
    # Generate two uniform samples
    var result1 = random_uniform[dtype](rng)
    var u1 = result1[0]
    var rng1 = result1[1]

    var result2 = random_uniform[dtype](rng1)
    var u2 = result2[0]
    var rng2 = result2[1]

    # Avoid log(0)
    if u1 < Scalar[dtype](1e-10):
        u1 = Scalar[dtype](1e-10)

    # Box-Muller transform
    var r = sqrt(Scalar[dtype](-2.0) * log(u1))
    var theta = Scalar[dtype](2.0 * pi) * u2
    var z1 = r * cos(theta)
    var z2 = r * sin(theta)

    return (z1, z2, rng2)
