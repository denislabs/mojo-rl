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
    """Matrix multiplication: A @ B.

    A: (M, K), B: (K, N) -> C: (M, N).
    Row-major layout.
    """
    var C = InlineArray[Scalar[dtype], M * N](fill=0)

    for i in range(M):
        for j in range(N):
            var sum: Scalar[dtype] = 0

            for k in range(K):
                sum += A[i * K + k] * B[k * N + j]
            C[i * N + j] = sum

    return C^


@always_inline
fn matmul_add_bias[
    M: Int, K: Int, N: Int, dtype: DType = DType.float64
](
    A: InlineArray[Scalar[dtype], M * K],
    B: InlineArray[Scalar[dtype], K * N],
    bias: InlineArray[Scalar[dtype], N],
) -> InlineArray[Scalar[dtype], M * N]:
    """Matrix multiplication with bias: A @ B + bias (broadcast).

    A: (M, K), B: (K, N), bias: (N,) -> C: (M, N).
    """
    var C = InlineArray[Scalar[dtype], M * N](fill=0)

    for i in range(M):
        for j in range(N):
            var sum: Scalar[dtype] = bias[j]

            for k in range(K):
                sum += A[i * K + k] * B[k * N + j]
            C[i * N + j] = sum

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
    """ReLU activation: max(0, x)."""
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        if x[i] > 0:
            result[i] = x[i]
        else:
            result[i] = 0
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
    """Gradient of ReLU: 1 if x > 0, else 0."""
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        if x[i] > 0:
            result[i] = 1.0
        else:
            result[i] = 0.0
    return result^


@always_inline
fn tanh_grad[
    size: Int, dtype: DType = DType.float64
](activated: InlineArray[Scalar[dtype], size]) -> InlineArray[
    Scalar[dtype], size
]:
    """Gradient of tanh given the OUTPUT (not input).

    If y = tanh(x), then dy/dx = 1 - y^2.
    """
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        var y = activated[i]
        result[i] = 1.0 - y * y
    return result^


@always_inline
fn sigmoid_grad[
    size: Int, dtype: DType = DType.float64
](activated: InlineArray[Scalar[dtype], size]) -> InlineArray[
    Scalar[dtype], size
]:
    """Gradient of sigmoid given the OUTPUT (not input).

    If y = sigmoid(x), then dy/dx = y * (1 - y).
    """
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        var y = activated[i]
        result[i] = y * (1.0 - y)
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
    """Element-wise multiplication (Hadamard product)."""
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        result[i] = a[i] * b[i]
    return result^


@always_inline
fn elementwise_sub[
    size: Int, dtype: DType = DType.float64
](
    a: InlineArray[Scalar[dtype], size],
    b: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """Element-wise subtraction: a - b."""
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        result[i] = a[i] - b[i]
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
    """Scale all elements by a scalar."""
    var result = InlineArray[Scalar[dtype], size](fill=0)

    for i in range(size):
        result[i] = x[i] * scalar
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
        result[i] = low + random_float64() * range_size
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
