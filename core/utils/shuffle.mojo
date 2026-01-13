"""Shuffling utilities for minibatch sampling in RL algorithms.

Provides Fisher-Yates shuffle implementations for generating random
permutations of indices, commonly used in PPO and other algorithms
that perform multiple epochs over collected data.

Example usage:
    from core.utils.shuffle import shuffle_indices, shuffle_indices_inline

    # For List-based code
    var indices = List[Int]()
    shuffle_indices(100, indices)  # Shuffles indices 0..99

    # For InlineArray-based code
    var indices = InlineArray[Int, 100](fill=0)
    shuffle_indices_inline(100, indices)
"""

from random import random_float64


fn shuffle_indices(n: Int, mut indices: List[Int]):
    """Generate shuffled indices [0, n) using Fisher-Yates algorithm.

    Resizes the list and fills with a random permutation of [0, n).

    Args:
        n: Number of indices to generate.
        indices: Output list (will be cleared and resized).

    Example:
        var indices = List[Int]()
        shuffle_indices(64, indices)
        # indices now contains a random permutation of [0, 64)
    """
    indices.clear()
    for i in range(n):
        indices.append(i)

    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        var j = Int(random_float64() * Float64(i + 1))
        if j > i:
            j = i
        # Swap
        var temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp


fn shuffle_indices_inline[
    N: Int
](n: Int, mut indices: InlineArray[Int, N]):
    """Generate shuffled indices [0, n) into an InlineArray using Fisher-Yates.

    Fills the first `n` elements with a random permutation of [0, n).
    The InlineArray must have capacity >= n.

    Args:
        n: Number of indices to generate (must be <= N).
        indices: Output array (first n elements will be shuffled).

    Parameters:
        N: Maximum capacity of the InlineArray.

    Example:
        var indices = InlineArray[Int, 2048](fill=0)
        shuffle_indices_inline(buffer_len, indices)
    """
    # Initialize with sequential indices
    for i in range(n):
        indices[i] = i

    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        var j = Int(random_float64() * Float64(i + 1))
        if j > i:
            j = i
        # Swap
        var temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp
