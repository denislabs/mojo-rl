"""Tensor operations using LayoutTensor and InlineArray.

Uses compile-time dimensions for maximum performance:
- InlineArray for stack-allocated storage
- LayoutTensor for structured access
- All shapes known at compile time
- @always_inline for hot path functions
"""

from layout import Layout, LayoutTensor
from random import random_float64
from math import sqrt, exp, tanh as math_tanh
from sys import simd_width_of


# =============================================================================
# Matrix operations with compile-time dimensions
# =============================================================================


@always_inline
fn matmul[
    M: Int, K: Int, N: Int, dtype: DType = DType.float64
](
    A: InlineArray[Scalar[dtype], M * K],
    B: InlineArray[Scalar[dtype], K * N],
) -> InlineArray[Scalar[dtype], M * N]:
    """Matrix multiplication: A @ B - SIMD optimized with tiling.

    A: (M, K), B: (K, N) -> C: (M, N).
    Row-major layout.

    Optimizations:
    - SIMD vectorization along N dimension (contiguous B access)
    - Loop tiling for cache efficiency
    - Accumulation in SIMD registers
    """
    comptime width = simd_width_of[dtype]()
    var C = InlineArray[Scalar[dtype], M * N](fill=0)

    # Process each row of A
    for i in range(M):
        # For each row, compute dot products with all columns of B
        # Vectorize along N (columns of B and C) for contiguous access

        # SIMD loop: process `width` columns at a time
        var j = 0
        while j + width <= N:
            # Accumulate in SIMD register
            var acc = SIMD[dtype, width](0)

            # Dot product: sum over K
            for k in range(K):
                var a_val = SIMD[dtype, width](A[i * K + k])  # broadcast scalar
                # Load `width` elements from row k of B
                var b_vec = B.unsafe_ptr().offset(k * N + j).load[width=width]()
                # FMA: acc += a_val * b_vec
                acc = a_val.fma(b_vec, acc)

            # Store result
            C.unsafe_ptr().offset(i * N + j).store(acc)
            j += width

        # Scalar remainder for columns
        while j < N:
            var sum: Scalar[dtype] = 0
            for k in range(K):
                sum += A[i * K + k] * B[k * N + j]
            C[i * N + j] = sum
            j += 1

    return C^


@always_inline
fn matmul_add_bias[
    M: Int, K: Int, N: Int, dtype: DType = DType.float64
](
    A: InlineArray[Scalar[dtype], M * K],
    B: InlineArray[Scalar[dtype], K * N],
    bias: InlineArray[Scalar[dtype], N],
) -> InlineArray[Scalar[dtype], M * N]:
    """Matrix multiplication with bias: A @ B + bias (broadcast) - SIMD optimized.

    A: (M, K), B: (K, N), bias: (N,) -> C: (M, N).
    """
    comptime width = simd_width_of[dtype]()
    var C = InlineArray[Scalar[dtype], M * N](fill=0)

    for i in range(M):
        # SIMD loop: process `width` columns at a time
        var j = 0
        while j + width <= N:
            # Start with bias (vectorized load)
            var acc = bias.unsafe_ptr().offset(j).load[width=width]()

            # Dot product: sum over K
            for k in range(K):
                var a_val = SIMD[dtype, width](A[i * K + k])  # broadcast scalar
                var b_vec = B.unsafe_ptr().offset(k * N + j).load[width=width]()
                acc = a_val.fma(b_vec, acc)

            # Store result
            C.unsafe_ptr().offset(i * N + j).store(acc)
            j += width

        # Scalar remainder
        while j < N:
            var sum: Scalar[dtype] = bias[j]
            for k in range(K):
                sum += A[i * K + k] * B[k * N + j]
            C[i * N + j] = sum
            j += 1

    return C^


@always_inline
fn transpose[
    rows: Int, cols: Int, dtype: DType = DType.float64
](A: InlineArray[Scalar[dtype], rows * cols]) -> InlineArray[
    Scalar[dtype], cols * rows
]:
    """Transpose matrix: (rows, cols) -> (cols, rows)."""
    var result = InlineArray[Scalar[dtype], cols * rows](fill=0)

    for i in range(rows):
        for j in range(cols):
            result[j * rows + i] = A[i * cols + j]

    return result^


# =============================================================================
# Activation functions
# =============================================================================


@always_inline
fn relu[
    size: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], size]) -> InlineArray[Scalar[dtype], size]:
    """ReLU activation: max(0, x) - SIMD optimized."""
    comptime width = simd_width_of[dtype]()
    var result = InlineArray[Scalar[dtype], size](fill=0)
    var zero_vec = SIMD[dtype, width](0)

    # Vectorized loop
    var i = 0
    while i + width <= size:
        var vec = x.unsafe_ptr().offset(i).load[width=width]()
        # SIMD max(0, x): select positive values, else 0
        var mask = vec.gt(zero_vec)
        var result_vec = mask.select(vec, zero_vec)
        result.unsafe_ptr().offset(i).store(result_vec)
        i += width

    # Scalar remainder
    while i < size:
        result[i] = x[i] if x[i] > 0 else 0
        i += 1

    return result^


@always_inline
fn tanh_activation[
    size: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], size]) -> InlineArray[Scalar[dtype], size]:
    """Tanh activation."""
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        result[i] = math_tanh(x[i])
    return result^


@always_inline
fn sigmoid[
    size: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], size]) -> InlineArray[Scalar[dtype], size]:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        var val = x[i]
        if val > 20:
            result[i] = 1.0
        elif val < -20:
            result[i] = 0.0
        else:
            result[i] = 1.0 / (1.0 + exp(-val))
    return result^


# =============================================================================
# Activation gradients
# =============================================================================


@always_inline
fn relu_grad[
    size: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], size]) -> InlineArray[Scalar[dtype], size]:
    """Gradient of ReLU: 1 if x > 0, else 0 - SIMD optimized."""
    comptime width = simd_width_of[dtype]()
    var result = InlineArray[Scalar[dtype], size](fill=0)
    var zero_vec = SIMD[dtype, width](0)
    var one_vec = SIMD[dtype, width](1)

    # Vectorized loop
    var i = 0
    while i + width <= size:
        var vec = x.unsafe_ptr().offset(i).load[width=width]()
        var mask = vec.gt(zero_vec)
        var result_vec = mask.select(one_vec, zero_vec)
        result.unsafe_ptr().offset(i).store(result_vec)
        i += width

    # Scalar remainder
    while i < size:
        result[i] = Scalar[dtype](1.0) if x[i] > 0 else Scalar[dtype](0.0)
        i += 1

    return result^


@always_inline
fn tanh_grad[
    size: Int, dtype: DType = DType.float64
](activated: InlineArray[Scalar[dtype], size]) -> InlineArray[
    Scalar[dtype], size
]:
    """Gradient of tanh given the OUTPUT (not input) - SIMD optimized.

    If y = tanh(x), then dy/dx = 1 - y^2.
    """
    comptime width = simd_width_of[dtype]()
    var result = InlineArray[Scalar[dtype], size](fill=0)
    var one_vec = SIMD[dtype, width](1)

    # Vectorized loop
    var i = 0
    while i + width <= size:
        var y = activated.unsafe_ptr().offset(i).load[width=width]()
        var grad = one_vec - y * y
        result.unsafe_ptr().offset(i).store(grad)
        i += width

    # Scalar remainder
    while i < size:
        var y = activated[i]
        result[i] = 1.0 - y * y
        i += 1

    return result^


@always_inline
fn sigmoid_grad[
    size: Int, dtype: DType = DType.float64
](activated: InlineArray[Scalar[dtype], size]) -> InlineArray[
    Scalar[dtype], size
]:
    """Gradient of sigmoid given the OUTPUT (not input) - SIMD optimized.

    If y = sigmoid(x), then dy/dx = y * (1 - y).
    """
    comptime width = simd_width_of[dtype]()
    var result = InlineArray[Scalar[dtype], size](fill=0)
    var one_vec = SIMD[dtype, width](1)

    # Vectorized loop
    var i = 0
    while i + width <= size:
        var y = activated.unsafe_ptr().offset(i).load[width=width]()
        var grad = y * (one_vec - y)
        result.unsafe_ptr().offset(i).store(grad)
        i += width

    # Scalar remainder
    while i < size:
        var y = activated[i]
        result[i] = y * (1.0 - y)
        i += 1

    return result^


# =============================================================================
# Element-wise operations
# =============================================================================


@always_inline
fn elementwise_mul[
    size: Int, dtype: DType = DType.float64
](
    a: InlineArray[Scalar[dtype], size],
    b: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """Element-wise multiplication (Hadamard product) - SIMD optimized."""
    comptime width = simd_width_of[dtype]()
    var result = InlineArray[Scalar[dtype], size](fill=0)

    # Vectorized loop
    var i = 0
    while i + width <= size:
        var vec_a = a.unsafe_ptr().offset(i).load[width=width]()
        var vec_b = b.unsafe_ptr().offset(i).load[width=width]()
        result.unsafe_ptr().offset(i).store(vec_a * vec_b)
        i += width

    # Scalar remainder
    while i < size:
        result[i] = a[i] * b[i]
        i += 1

    return result^


@always_inline
fn elementwise_sub[
    size: Int, dtype: DType = DType.float64
](
    a: InlineArray[Scalar[dtype], size],
    b: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """Element-wise subtraction: a - b (SIMD optimized)."""
    comptime width = simd_width_of[dtype]()
    var result = InlineArray[Scalar[dtype], size](fill=0)

    # Vectorized loop
    var i = 0
    while i + width <= size:
        var vec_a = a.unsafe_ptr().offset(i).load[width=width]()
        var vec_b = b.unsafe_ptr().offset(i).load[width=width]()
        result.unsafe_ptr().offset(i).store(vec_a - vec_b)
        i += width

    # Scalar remainder
    while i < size:
        result[i] = a[i] - b[i]
        i += 1

    return result^


@always_inline
fn scale[
    size: Int, dtype: DType = DType.float64
](
    x: InlineArray[Scalar[dtype], size],
    scalar: Scalar[dtype],
) -> InlineArray[
    Scalar[dtype], size
]:
    """Scale all elements by a scalar (SIMD optimized)."""
    comptime width = simd_width_of[dtype]()
    var result = InlineArray[Scalar[dtype], size](fill=0)
    var scalar_vec = SIMD[dtype, width](scalar)

    # Vectorized loop
    var i = 0
    while i + width <= size:
        var vec = x.unsafe_ptr().offset(i).load[width=width]()
        result.unsafe_ptr().offset(i).store(vec * scalar_vec)
        i += width

    # Scalar remainder
    while i < size:
        result[i] = x[i] * scalar
        i += 1

    return result^


# =============================================================================
# Initialization
# =============================================================================


fn zeros[
    size: Int, dtype: DType = DType.float64
]() -> InlineArray[Scalar[dtype], size]:
    """Create zero-initialized array."""
    return InlineArray[Scalar[dtype], size](fill=0)


fn xavier_init[
    size: Int, fan_in: Int, fan_out: Int, dtype: DType = DType.float64
]() -> InlineArray[Scalar[dtype], size]:
    """Xavier/Glorot uniform initialization."""
    var result = InlineArray[Scalar[dtype], size](fill=0)
    var limit = sqrt(6.0 / Float64(fan_in + fan_out))
    for i in range(size):
        var val = (random_float64() * 2.0 - 1.0) * limit
        result[i] = val.cast[dtype]()
    return result^


fn random_uniform[
    size: Int, dtype: DType = DType.float64
](low: Scalar[dtype] = 0.0, high: Scalar[dtype] = 1.0) -> InlineArray[
    Scalar[dtype], size
]:
    """Uniform random initialization."""
    var result = InlineArray[Scalar[dtype], size](fill=0)
    var range_size = high - low
    for i in range(size):
        result[i] = low + Scalar[dtype](random_float64()) * range_size
    return result^


# =============================================================================
# Reductions
# =============================================================================


fn sum_all[
    size: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], size]) -> Scalar[dtype]:
    """Sum all elements."""
    var result: Scalar[dtype] = 0
    for i in range(size):
        result += x[i]
    return result


fn mean_all[
    size: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], size]) -> Scalar[dtype]:
    """Mean of all elements."""
    return sum_all[size, dtype](x) / size


fn sum_axis0[
    rows: Int, cols: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], rows * cols]) -> InlineArray[
    Scalar[dtype], cols
]:
    """Sum along axis 0 (rows), result shape: (cols,)."""
    var result = InlineArray[Scalar[dtype], cols](fill=0)
    for j in range(cols):
        for i in range(rows):
            result[j] += x[i * cols + j]
    return result^


# =============================================================================
# Utility
# =============================================================================


fn print_matrix[
    rows: Int, cols: Int, dtype: DType = DType.float64
](x: InlineArray[Scalar[dtype], rows * cols], name: String = "Matrix",):
    """Print matrix for debugging."""
    print(name + " [" + String(rows) + " x " + String(cols) + "]:")
    for i in range(rows):
        var row_str = String("  [")
        for j in range(cols):
            if j > 0:
                row_str += ", "
            row_str += String(x[i * cols + j])[:8]
        row_str += "]"
        print(row_str)


fn copy_array[
    size: Int, dtype: DType = DType.float64
](src: InlineArray[Scalar[dtype], size]) -> InlineArray[Scalar[dtype], size]:
    """Create a copy of an array."""
    var result = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        result[i] = src[i]
    return result^
