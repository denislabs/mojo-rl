from math import cos, sin


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
