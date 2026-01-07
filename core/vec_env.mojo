"""Vectorized environment infrastructure for parallel environment execution.

This module provides core types and utilities for running multiple copies of
an environment in parallel using SIMD operations.

Key components:
- VecStepResult: Batched step results (observations, rewards, dones)
- random_simd: Helper for SIMD random number generation
- Native SIMD comparison methods for element-wise operations

Performance notes:
- Uses native SIMD methods (.eq(), .lt(), .gt(), etc.) for element-wise comparisons
- Uses @always_inline for zero function call overhead
- Uses @parameter for compile-time loop unrolling
"""

from random import random_float64


# ============================================================================
# Vectorized Step Result
# ============================================================================


struct VecStepResult[num_envs: Int]:
    """Result from stepping multiple environments in parallel.

    Contains batched observations, rewards, and done flags for all environments.

    Parameters:
        num_envs: Number of parallel environments (compile-time constant).
    """

    var observations: List[SIMD[DType.float64, 4]]
    """List of observations, one per environment. Each is a 4D SIMD vector."""

    var rewards: SIMD[DType.float64, Self.num_envs]
    """SIMD vector of rewards, one per environment."""

    var dones: SIMD[DType.bool, Self.num_envs]
    """SIMD vector of done flags, one per environment."""

    fn __init__(
        out self,
        var observations: List[SIMD[DType.float64, 4]],
        rewards: SIMD[DType.float64, Self.num_envs],
        dones: SIMD[DType.bool, Self.num_envs],
    ):
        """Initialize a VecStepResult with batched data.

        Args:
            observations: List of observations (length num_envs) - ownership transferred.
            rewards: SIMD vector of rewards.
            dones: SIMD vector of done flags.
        """
        self.observations = observations^
        self.rewards = rewards
        self.dones = dones


# ============================================================================
# SIMD Splat (broadcast scalar to all lanes)
# ============================================================================


@always_inline
fn simd_splat_f64[n: Int](value: Float64) -> SIMD[DType.float64, n]:
    """Create a SIMD vector with all elements set to value.

    Parameters:
        n: SIMD vector width.

    Args:
        value: The scalar value to broadcast.

    Returns:
        SIMD vector with all elements set to value.
    """
    var result = SIMD[DType.float64, n]()

    @parameter
    for i in range(n):
        result[i] = value
    return result


@always_inline
fn simd_splat_i32[n: Int](value: Int32) -> SIMD[DType.int32, n]:
    """Create a SIMD vector with all elements set to value.

    Parameters:
        n: SIMD vector width.

    Args:
        value: The scalar value to broadcast.

    Returns:
        SIMD vector with all elements set to value.
    """
    var result = SIMD[DType.int32, n]()

    @parameter
    for i in range(n):
        result[i] = value
    return result


# ============================================================================
# SIMD Element-wise Comparison (using native SIMD methods)
# ============================================================================


@always_inline
fn simd_eq_i32[n: Int](
    a: SIMD[DType.int32, n], b: SIMD[DType.int32, n]
) -> SIMD[DType.bool, n]:
    """Element-wise equality comparison of two int32 SIMD vectors.

    Uses native SIMD .eq() method for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        a: First SIMD vector.
        b: Second SIMD vector.

    Returns:
        Boolean SIMD vector where each element is True if a[i] == b[i].
    """
    return a.eq(b)


@always_inline
fn simd_ge_i32[n: Int](
    a: SIMD[DType.int32, n], b: SIMD[DType.int32, n]
) -> SIMD[DType.bool, n]:
    """Element-wise greater-than-or-equal comparison.

    Uses native SIMD .ge() method for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        a: First SIMD vector.
        b: Second SIMD vector.

    Returns:
        Boolean SIMD vector where each element is True if a[i] >= b[i].
    """
    return a.ge(b)


@always_inline
fn simd_lt_f64[n: Int](
    a: SIMD[DType.float64, n], b: Float64
) -> SIMD[DType.bool, n]:
    """Element-wise less-than comparison with scalar.

    Uses native SIMD .lt() method for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        a: SIMD vector.
        b: Scalar threshold.

    Returns:
        Boolean SIMD vector where each element is True if a[i] < b.
    """
    return a.lt(b)


@always_inline
fn simd_gt_f64[n: Int](
    a: SIMD[DType.float64, n], b: Float64
) -> SIMD[DType.bool, n]:
    """Element-wise greater-than comparison with scalar.

    Uses native SIMD .gt() method for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        a: SIMD vector.
        b: Scalar threshold.

    Returns:
        Boolean SIMD vector where each element is True if a[i] > b.
    """
    return a.gt(b)


# ============================================================================
# SIMD Random Number Generation
# ============================================================================


@always_inline
fn random_simd[n: Int](low: Float64, high: Float64) -> SIMD[DType.float64, n]:
    """Generate a SIMD vector of random values in [low, high).

    Parameters:
        n: SIMD vector width.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).

    Returns:
        SIMD vector with n random values uniformly distributed in [low, high).
    """
    var result = SIMD[DType.float64, n]()
    var range_size = high - low

    @parameter
    for i in range(n):
        result[i] = random_float64() * range_size + low
    return result


@always_inline
fn random_simd_centered[n: Int](half_range: Float64) -> SIMD[DType.float64, n]:
    """Generate a SIMD vector of random values in [-half_range, half_range).

    Convenience function for centered distributions (common in RL initialization).

    Parameters:
        n: SIMD vector width.

    Args:
        half_range: Half the range width. Values will be in [-half_range, half_range).

    Returns:
        SIMD vector with n random values centered around 0.
    """
    return random_simd[n](-half_range, half_range)


# ============================================================================
# SIMD Utilities
# ============================================================================


@always_inline
fn simd_or[n: Int](
    a: SIMD[DType.bool, n], b: SIMD[DType.bool, n]
) -> SIMD[DType.bool, n]:
    """Element-wise OR of two boolean SIMD vectors.

    Uses native bitwise OR operator for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        a: First boolean SIMD vector.
        b: Second boolean SIMD vector.

    Returns:
        Boolean SIMD vector with element-wise OR.
    """
    return a | b


@always_inline
fn simd_any[n: Int](mask: SIMD[DType.bool, n]) -> Bool:
    """Check if any element in the boolean SIMD vector is True.

    Uses SIMD reduce_or for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        mask: Boolean SIMD vector.

    Returns:
        True if any element is True, False otherwise.
    """
    return mask.reduce_or()


@always_inline
fn simd_all[n: Int](mask: SIMD[DType.bool, n]) -> Bool:
    """Check if all elements in the boolean SIMD vector are True.

    Uses SIMD reduce_and for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        mask: Boolean SIMD vector.

    Returns:
        True if all elements are True, False otherwise.
    """
    return mask.reduce_and()


@always_inline
fn simd_count_true[n: Int](mask: SIMD[DType.bool, n]) -> Int:
    """Count the number of True elements in a boolean SIMD vector.

    Uses SIMD cast and reduce for optimal performance.

    Parameters:
        n: SIMD vector width.

    Args:
        mask: Boolean SIMD vector.

    Returns:
        Number of True elements.
    """
    # Cast bool to int and sum
    return Int(mask.cast[DType.int32]().reduce_add())
