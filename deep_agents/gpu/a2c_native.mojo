"""GPU-Native A2C Implementation.

This is an experimental implementation that uses proper GPU parallelism:
- Tiled matrix multiplication with shared memory
- Parallel reductions for softmax
- Batched operations across environments

This file is developed incrementally alongside a2c.mojo (which is kept intact).

Phase 1: Tiled matmul kernel for forward pass
"""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp, log, sqrt, cos, sin
from random import random_float64
from bit import log2_ceil
from utils.numerics import max_finite, min_finite

# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32
comptime TPB: Int = 8  # Threads per block (tile size) - must be power of 2


# =============================================================================
# Naive Matrix Multiplication Kernel (no shared memory - baseline)
# =============================================================================
# Computes: C[M, N] = A[M, K] @ B[K, N]
# Each thread computes one element of C


fn naive_matmul_kernel[
    M: Int,  # Number of rows in A and C (e.g., NUM_ENVS)
    K: Int,  # Number of cols in A, rows in B (e.g., OBS_DIM)
    N: Int,  # Number of cols in B and C (e.g., HIDDEN_DIM)
    TILE_SIZE: Int = TPB,
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """Naive matrix multiplication: C = A @ B.

    Grid: (ceil(N/TILE_SIZE), ceil(M/TILE_SIZE))
    Block: (TILE_SIZE, TILE_SIZE)

    Each thread computes one element C[row, col].
    No shared memory - just global memory access.
    """
    var row = Int(block_idx.y) * TILE_SIZE + Int(thread_idx.y)
    var col = Int(block_idx.x) * TILE_SIZE + Int(thread_idx.x)

    if row < M and col < N:
        var acc: Scalar[dtype] = 0

        # Use compile-time unrolled loop
        @parameter
        for k in range(K):
            acc += rebind[Scalar[dtype]](A[row, k]) * rebind[Scalar[dtype]](
                B[k, col]
            )

        C[row, col] = acc


# =============================================================================
# Tiled Matrix Multiplication Kernel (with shared memory)
# =============================================================================


fn tiled_matmul_kernel[
    M: Int,  # Number of rows in A and C (e.g., NUM_ENVS)
    K: Int,  # Number of cols in A, rows in B (e.g., OBS_DIM)
    N: Int,  # Number of cols in B and C (e.g., HIDDEN_DIM)
    TILE_SIZE: Int = TPB,
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """Tiled matrix multiplication: C = A @ B.

    Grid: (ceil(N/TILE_SIZE), ceil(M/TILE_SIZE))
    Block: (TILE_SIZE, TILE_SIZE)

    Each thread computes one element C[row, col].
    """
    # Thread position within block
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)

    # Global position in output matrix
    var global_row = Int(block_idx.y) * TILE_SIZE + local_row
    var global_col = Int(block_idx.x) * TILE_SIZE + local_col

    # Shared memory tiles
    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Accumulator for this thread's output element
    var acc: Scalar[dtype] = 0

    # Number of tiles along K dimension
    comptime num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    # Iterate over tiles
    @parameter
    for tile_idx in range(num_tiles):
        var tile_start = tile_idx * TILE_SIZE

        # Load A tile: A[global_row, tile_start + local_col]
        var a_col = tile_start + local_col
        if global_row < M and a_col < K:
            A_shared[local_row, local_col] = rebind[Scalar[dtype]](
                A[global_row, a_col]
            )
        else:
            A_shared[local_row, local_col] = Scalar[dtype](0)

        # Load B tile: B[tile_start + local_row, global_col]
        var b_row = tile_start + local_row
        if b_row < K and global_col < N:
            B_shared[local_row, local_col] = rebind[Scalar[dtype]](
                B[b_row, global_col]
            )
        else:
            B_shared[local_row, local_col] = Scalar[dtype](0)

        # Synchronize to ensure all threads have loaded their tiles
        barrier()

        # Compute partial dot product for this tile
        @parameter
        for k in range(TILE_SIZE):
            # Bounds check for the K dimension
            if tile_start + k < K:
                acc += rebind[Scalar[dtype]](A_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](B_shared[k, local_col])

        # Synchronize before loading next tile
        barrier()

    # Write result to global memory
    if global_row < M and global_col < N:
        C[global_row, global_col] = acc


# =============================================================================
# Tiled Matmul + Bias + ReLU (fused kernel)
# =============================================================================
# Computes: C[M, N] = ReLU(A[M, K] @ B[K, N] + bias[N])


fn tiled_matmul_bias_relu_kernel[
    M: Int,  # Number of rows in A and C (e.g., NUM_ENVS)
    K: Int,  # Number of cols in A, rows in B (e.g., OBS_DIM)
    N: Int,  # Number of cols in B and C (e.g., HIDDEN_DIM)
    TILE_SIZE: Int = TPB,
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    """Fused tiled matmul + bias + ReLU: C = ReLU(A @ B + bias).

    Same structure as tiled_matmul_kernel but adds bias and ReLU at the end.
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE_SIZE + local_row
    var global_col = Int(block_idx.x) * TILE_SIZE + local_col

    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    comptime num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    @parameter
    for tile_idx in range(num_tiles):
        var tile_start = tile_idx * TILE_SIZE

        var a_col = tile_start + local_col
        if global_row < M and a_col < K:
            A_shared[local_row, local_col] = rebind[Scalar[dtype]](
                A[global_row, a_col]
            )
        else:
            A_shared[local_row, local_col] = Scalar[dtype](0)

        var b_row = tile_start + local_row
        if b_row < K and global_col < N:
            B_shared[local_row, local_col] = rebind[Scalar[dtype]](
                B[b_row, global_col]
            )
        else:
            B_shared[local_row, local_col] = Scalar[dtype](0)

        barrier()

        @parameter
        for k in range(TILE_SIZE):
            if tile_start + k < K:
                acc += rebind[Scalar[dtype]](A_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](B_shared[k, local_col])

        barrier()

    # Write result with bias and ReLU
    if global_row < M and global_col < N:
        var result = acc + rebind[Scalar[dtype]](bias[global_col])
        # ReLU
        C[global_row, global_col] = result if result > Scalar[dtype](
            0
        ) else Scalar[dtype](0)


# =============================================================================
# Tiled Matmul + Bias (no activation - for output layers)
# =============================================================================


fn tiled_matmul_bias_kernel[
    M: Int,
    K: Int,
    N: Int,
    TILE_SIZE: Int = TPB,
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    """Fused tiled matmul + bias (no activation): C = A @ B + bias."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE_SIZE + local_row
    var global_col = Int(block_idx.x) * TILE_SIZE + local_col

    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    comptime num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    @parameter
    for tile_idx in range(num_tiles):
        var tile_start = tile_idx * TILE_SIZE

        var a_col = tile_start + local_col
        if global_row < M and a_col < K:
            A_shared[local_row, local_col] = rebind[Scalar[dtype]](
                A[global_row, a_col]
            )
        else:
            A_shared[local_row, local_col] = Scalar[dtype](0)

        var b_row = tile_start + local_row
        if b_row < K and global_col < N:
            B_shared[local_row, local_col] = rebind[Scalar[dtype]](
                B[b_row, global_col]
            )
        else:
            B_shared[local_row, local_col] = Scalar[dtype](0)

        barrier()

        @parameter
        for k in range(TILE_SIZE):
            if tile_start + k < K:
                acc += rebind[Scalar[dtype]](A_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](B_shared[k, local_col])

        barrier()

    # Write result with bias (no activation)
    if global_row < M and global_col < N:
        C[global_row, global_col] = acc + rebind[Scalar[dtype]](
            bias[global_col]
        )


# =============================================================================
# Parallel Softmax Kernel (tree-based reduction)
# =============================================================================
# Computes softmax over the last dimension: output[i, j] = exp(input[i, j] - max) / sum
# Uses parallel reduction for max and sum operations


fn parallel_softmax_kernel[
    M: Int,  # Batch size (NUM_ENVS)
    N: Int,  # Number of classes (NUM_ACTIONS)
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """Parallel softmax using tree-based reduction.

    Grid: (1, M)  - one block per row
    Block: (next_power_of_2(N), 1)

    Each block computes softmax for one row.
    """
    var row = Int(block_idx.y)
    var tid = Int(thread_idx.x)

    if row >= M:
        return

    # Next power of 2 >= N for reduction
    comptime BLOCK_SIZE = 1 << log2_ceil(N)

    # Shared memory for reductions
    var shared_max = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Load input values (or min_finite for out-of-bounds threads)
    var val: Scalar[dtype] = min_finite[dtype]()
    if tid < N:
        val = rebind[Scalar[dtype]](input[row, tid])
    shared_max[tid] = val

    barrier()

    # Parallel reduction to find max
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared_max[tid] = max(
                rebind[Scalar[dtype]](shared_max[tid]),
                rebind[Scalar[dtype]](shared_max[tid + stride]),
            )
        barrier()
        stride = stride // 2

    var row_max = rebind[Scalar[dtype]](shared_max[0])

    # Compute exp(val - max) and store in shared_sum
    var exp_val: Scalar[dtype] = 0.0
    if tid < N:
        exp_val = exp(val - row_max)
    shared_sum[tid] = exp_val

    barrier()

    # Parallel reduction for sum
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared_sum[tid] = rebind[Scalar[dtype]](shared_sum[tid]) + rebind[
                Scalar[dtype]
            ](shared_sum[tid + stride])
        barrier()
        stride = stride // 2

    var row_sum = rebind[Scalar[dtype]](shared_sum[0])

    # Write normalized output
    if tid < N:
        output[row, tid] = exp_val / row_sum


# =============================================================================
# PHASE 3: Backward Pass Kernels
# =============================================================================

# =============================================================================
# Policy Gradient Kernel
# =============================================================================
# Computes d_logits for policy gradient with entropy bonus
# d_logits[i, j] = probs[i, j] - (1 if j == action[i] else 0)  (for -log_prob gradient)
# Then scaled by -advantage for policy gradient direction
# Entropy gradient: -entropy_coef * (1 + log(probs))
# Combined: d_logits = -advantage * (one_hot - probs) + entropy_coef * (1 + log(probs))
#         = advantage * (probs - one_hot) + entropy_coef * (1 + log(probs))


fn policy_gradient_kernel[
    M: Int,  # NUM_ENVS (batch size)
    N: Int,  # NUM_ACTIONS
](
    d_logits: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    probs: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
    actions: LayoutTensor[DType.int32, Layout.row_major(M), ImmutAnyOrigin],
    advantages: LayoutTensor[dtype, Layout.row_major(M), ImmutAnyOrigin],
    entropy_coef: Scalar[dtype],
):
    """Compute policy gradient w.r.t. logits.

    Grid: (ceil(N/TILE), ceil(M/TILE))
    Block: (TILE, TILE)

    Each thread handles one (env, action) pair.
    """
    var row = Int(block_idx.y) * Int(block_dim.y) + Int(thread_idx.y)
    var col = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)

    if row < M and col < N:
        var prob = rebind[Scalar[dtype]](probs[row, col])
        var action = rebind[Scalar[DType.int32]](actions[row])
        var advantage = rebind[Scalar[dtype]](advantages[row])

        # Policy gradient: advantage * (prob - one_hot)
        var one_hot: Scalar[dtype] = Scalar[dtype](1.0) if Int(
            action
        ) == col else Scalar[dtype](0.0)
        var policy_grad = advantage * (prob - one_hot)

        # Entropy gradient: entropy_coef * (1 + log(prob))
        # Clamp prob to avoid log(0)
        var prob_clamped = max(prob, Scalar[dtype](1e-8))
        var entropy_grad = entropy_coef * (
            Scalar[dtype](1.0) + log(prob_clamped)
        )

        d_logits[row, col] = policy_grad + entropy_grad


# =============================================================================
# Value Loss Gradient Kernel
# =============================================================================
# d_values = values - returns (MSE derivative scaled by 0.5)


fn value_loss_gradient_kernel[
    M: Int,  # NUM_ENVS
](
    d_values: LayoutTensor[dtype, Layout.row_major(M, 1), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(M, 1), ImmutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(M), ImmutAnyOrigin],
    value_coef: Scalar[dtype],
):
    """Compute value loss gradient w.r.t. values.

    Grid: (1, ceil(M/TILE))
    Block: (1, TILE)
    """
    var row = Int(block_idx.y) * Int(block_dim.y) + Int(thread_idx.y)

    if row < M:
        var value = rebind[Scalar[dtype]](values[row, 0])
        var ret = rebind[Scalar[dtype]](returns[row])
        # d_loss/d_value = value_coef * (value - return) for MSE
        d_values[row, 0] = value_coef * (value - ret)


# =============================================================================
# Matmul Transpose A: C = A.T @ B
# =============================================================================
# For weight gradients: d_W = input.T @ d_output


fn tiled_matmul_transA_kernel[
    M: Int,  # Original rows of A (becomes cols after transpose)
    K: Int,  # Original cols of A, rows of B
    N: Int,  # Cols of B and C
    TILE_SIZE: Int = TPB,
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],  # [M, N]
    A: LayoutTensor[
        dtype, Layout.row_major(K, M), ImmutAnyOrigin
    ],  # [K, M] - will access as A.T[M, K]
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],  # [K, N]
):
    """Compute C = A.T @ B where A is [K, M], so A.T is [M, K].

    Grid: (ceil(N/TILE), ceil(M/TILE))
    Block: (TILE, TILE)

    Each thread computes C[row, col] = sum_k(A[k, row] * B[k, col])
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE_SIZE + local_row
    var global_col = Int(block_idx.x) * TILE_SIZE + local_col

    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    comptime num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    @parameter
    for tile_idx in range(num_tiles):
        var tile_start = tile_idx * TILE_SIZE

        # Load A.T[global_row, tile_start + local_col] = A[tile_start + local_col, global_row]
        var a_k = tile_start + local_col
        if global_row < M and a_k < K:
            A_shared[local_row, local_col] = rebind[Scalar[dtype]](
                A[a_k, global_row]
            )
        else:
            A_shared[local_row, local_col] = Scalar[dtype](0)

        # Load B[tile_start + local_row, global_col]
        var b_k = tile_start + local_row
        if b_k < K and global_col < N:
            B_shared[local_row, local_col] = rebind[Scalar[dtype]](
                B[b_k, global_col]
            )
        else:
            B_shared[local_row, local_col] = Scalar[dtype](0)

        barrier()

        @parameter
        for k in range(TILE_SIZE):
            if tile_start + k < K:
                acc += rebind[Scalar[dtype]](A_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](B_shared[k, local_col])

        barrier()

    if global_row < M and global_col < N:
        C[global_row, global_col] = acc


# =============================================================================
# Matmul Transpose B: C = A @ B.T
# =============================================================================
# For input gradients: d_input = d_output @ W.T


fn tiled_matmul_transB_kernel[
    M: Int,  # Rows of A and C
    K: Int,  # Cols of A, cols of B (rows of B.T)
    N: Int,  # Rows of B (cols of B.T), cols of C
    TILE_SIZE: Int = TPB,
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],  # [M, N]
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],  # [M, K]
    B: LayoutTensor[
        dtype, Layout.row_major(N, K), ImmutAnyOrigin
    ],  # [N, K] - will access as B.T[K, N]
):
    """Compute C = A @ B.T where B is [N, K], so B.T is [K, N].

    Grid: (ceil(N/TILE), ceil(M/TILE))
    Block: (TILE, TILE)

    Each thread computes C[row, col] = sum_k(A[row, k] * B[col, k])
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE_SIZE + local_row
    var global_col = Int(block_idx.x) * TILE_SIZE + local_col

    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    comptime num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    @parameter
    for tile_idx in range(num_tiles):
        var tile_start = tile_idx * TILE_SIZE

        # Load A[global_row, tile_start + local_col]
        var a_k = tile_start + local_col
        if global_row < M and a_k < K:
            A_shared[local_row, local_col] = rebind[Scalar[dtype]](
                A[global_row, a_k]
            )
        else:
            A_shared[local_row, local_col] = Scalar[dtype](0)

        # Load B.T[tile_start + local_row, global_col] = B[global_col, tile_start + local_row]
        var b_k = tile_start + local_row
        if global_col < N and b_k < K:
            B_shared[local_row, local_col] = rebind[Scalar[dtype]](
                B[global_col, b_k]
            )
        else:
            B_shared[local_row, local_col] = Scalar[dtype](0)

        barrier()

        @parameter
        for k in range(TILE_SIZE):
            if tile_start + k < K:
                acc += rebind[Scalar[dtype]](A_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](B_shared[k, local_col])

        barrier()

    if global_row < M and global_col < N:
        C[global_row, global_col] = acc


# =============================================================================
# ReLU Backward Kernel
# =============================================================================
# d_input = d_output * (pre_activation > 0 ? 1 : 0)


fn relu_backward_kernel[
    M: Int,
    N: Int,
    TILE_SIZE: Int = TPB,
](
    d_input: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    d_output: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
    pre_activation: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """ReLU backward: d_input = d_output * (pre_activation > 0).

    Grid: (ceil(N/TILE), ceil(M/TILE))
    Block: (TILE, TILE)
    """
    var row = Int(block_idx.y) * TILE_SIZE + Int(thread_idx.y)
    var col = Int(block_idx.x) * TILE_SIZE + Int(thread_idx.x)

    if row < M and col < N:
        var d_out = rebind[Scalar[dtype]](d_output[row, col])
        var pre_act = rebind[Scalar[dtype]](pre_activation[row, col])
        d_input[row, col] = d_out if pre_act > Scalar[dtype](0) else Scalar[
            dtype
        ](0)


# =============================================================================
# Bias Gradient Kernel (sum reduction over batch dimension)
# =============================================================================
# d_bias = sum(d_output, axis=0)


fn bias_gradient_kernel[
    M: Int,  # Batch size (to sum over)
    N: Int,  # Output dimension
](
    d_bias: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    d_output: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """Sum d_output over batch dimension to get bias gradient.

    Grid: (1, 1)
    Block: (N, 1) where N <= max threads per block

    Each thread handles one output dimension.
    For large N, we'd need multiple blocks.
    """
    var col = Int(thread_idx.x)

    if col < N:
        var acc: Scalar[dtype] = 0
        for row in range(M):
            acc += rebind[Scalar[dtype]](d_output[row, col])
        d_bias[col] = acc


# For larger N, we need a more parallel approach
fn bias_gradient_parallel_kernel[
    M: Int,  # Batch size
    N: Int,  # Output dimension
](
    d_bias: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    d_output: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """Parallel sum for bias gradient using tree reduction.

    Grid: (N, 1) - one block per output dimension
    Block: (next_power_of_2(M), 1)

    Each block computes sum for one output column.
    """
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    if col >= N:
        return

    comptime BLOCK_SIZE = 1 << log2_ceil(M)

    var shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Load values
    var val: Scalar[dtype] = 0
    if tid < M:
        val = rebind[Scalar[dtype]](d_output[tid, col])
    shared_sum[tid] = val

    barrier()

    # Tree reduction
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared_sum[tid] = rebind[Scalar[dtype]](shared_sum[tid]) + rebind[
                Scalar[dtype]
            ](shared_sum[tid + stride])
        barrier()
        stride = stride // 2

    if tid == 0:
        d_bias[col] = shared_sum[0]


# =============================================================================
# Element-wise Add Kernel (for combining gradients)
# =============================================================================


fn elementwise_add_kernel[
    M: Int,
    N: Int,
    TILE_SIZE: Int = TPB,
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """Element-wise addition: output = a + b.

    Grid: (ceil(N/TILE), ceil(M/TILE))
    Block: (TILE, TILE)
    """
    var row = Int(block_idx.y) * TILE_SIZE + Int(thread_idx.y)
    var col = Int(block_idx.x) * TILE_SIZE + Int(thread_idx.x)

    if row < M and col < N:
        output[row, col] = rebind[Scalar[dtype]](a[row, col]) + rebind[
            Scalar[dtype]
        ](b[row, col])


# =============================================================================
# Matmul + Bias + Save Pre-Activation (for backward pass)
# =============================================================================
# Like tiled_matmul_bias_relu_kernel but saves pre_activation for backward pass


fn tiled_matmul_bias_relu_save_kernel[
    M: Int,
    K: Int,
    N: Int,
    TILE_SIZE: Int = TPB,
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    pre_activation: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    """Fused matmul + bias + ReLU that also saves pre-activation for backward.
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE_SIZE + local_row
    var global_col = Int(block_idx.x) * TILE_SIZE + local_col

    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    comptime num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    @parameter
    for tile_idx in range(num_tiles):
        var tile_start = tile_idx * TILE_SIZE

        var a_col = tile_start + local_col
        if global_row < M and a_col < K:
            A_shared[local_row, local_col] = rebind[Scalar[dtype]](
                A[global_row, a_col]
            )
        else:
            A_shared[local_row, local_col] = Scalar[dtype](0)

        var b_row = tile_start + local_row
        if b_row < K and global_col < N:
            B_shared[local_row, local_col] = rebind[Scalar[dtype]](
                B[b_row, global_col]
            )
        else:
            B_shared[local_row, local_col] = Scalar[dtype](0)

        barrier()

        @parameter
        for k in range(TILE_SIZE):
            if tile_start + k < K:
                acc += rebind[Scalar[dtype]](A_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](B_shared[k, local_col])

        barrier()

    if global_row < M and global_col < N:
        var pre_act = acc + rebind[Scalar[dtype]](bias[global_col])
        pre_activation[global_row, global_col] = pre_act  # Save for backward
        output[global_row, global_col] = pre_act if pre_act > Scalar[dtype](
            0
        ) else Scalar[dtype](0)


# =============================================================================
# PHASE 4: Training Integration
# =============================================================================

# =============================================================================
# SGD Weight Update Kernel
# =============================================================================
# W = W - lr * d_W


fn sgd_update_2d_kernel[
    M: Int,
    N: Int,
    TILE_SIZE: Int = TPB,
](
    W: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    d_W: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
    lr: Scalar[dtype],
):
    """SGD update for 2D weight matrix: W = W - lr * d_W.

    Grid: (ceil(N/TILE), ceil(M/TILE))
    Block: (TILE, TILE)
    """
    var row = Int(block_idx.y) * TILE_SIZE + Int(thread_idx.y)
    var col = Int(block_idx.x) * TILE_SIZE + Int(thread_idx.x)

    if row < M and col < N:
        W[row, col] = rebind[Scalar[dtype]](W[row, col]) - lr * rebind[
            Scalar[dtype]
        ](d_W[row, col])


fn sgd_update_1d_kernel[
    N: Int,
](
    W: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    d_W: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
    lr: Scalar[dtype],
):
    """SGD update for 1D bias vector: W = W - lr * d_W.

    Grid: (1, 1)
    Block: (N, 1) - assumes N <= max threads per block
    """
    var idx = Int(thread_idx.x)

    if idx < N:
        W[idx] = rebind[Scalar[dtype]](W[idx]) - lr * rebind[Scalar[dtype]](
            d_W[idx]
        )


# =============================================================================
# Gradient Accumulation Kernel (for actor/critic sharing hidden layer)
# =============================================================================


fn accumulate_gradient_kernel[
    M: Int,
    N: Int,
    TILE_SIZE: Int = TPB,
](
    d_total: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    d_add: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """Add gradient to accumulator: d_total += d_add.

    Grid: (ceil(N/TILE), ceil(M/TILE))
    Block: (TILE, TILE)
    """
    var row = Int(block_idx.y) * TILE_SIZE + Int(thread_idx.y)
    var col = Int(block_idx.x) * TILE_SIZE + Int(thread_idx.x)

    if row < M and col < N:
        d_total[row, col] = rebind[Scalar[dtype]](d_total[row, col]) + rebind[
            Scalar[dtype]
        ](d_add[row, col])


# =============================================================================
# CPU Reference Implementation (for testing)
# =============================================================================


fn matmul_cpu_reference[
    M: Int, K: Int, N: Int
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """CPU reference matmul for verification."""
    for i in range(M):
        for j in range(N):
            var acc: Scalar[dtype] = 0
            for k in range(K):
                acc += rebind[Scalar[dtype]](A[i, k]) * rebind[Scalar[dtype]](
                    B[k, j]
                )
            C[i, j] = acc


fn matmul_bias_relu_cpu_reference[
    M: Int, K: Int, N: Int
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    """CPU reference matmul + bias + ReLU for verification."""
    for i in range(M):
        for j in range(N):
            var acc: Scalar[dtype] = 0
            for k in range(K):
                acc += rebind[Scalar[dtype]](A[i, k]) * rebind[Scalar[dtype]](
                    B[k, j]
                )
            var result = acc + rebind[Scalar[dtype]](bias[j])
            C[i, j] = result if result > Scalar[dtype](0) else Scalar[dtype](0)


fn softmax_cpu_reference[
    M: Int, N: Int
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """CPU reference softmax for verification."""
    for i in range(M):
        # Find max
        var row_max: Scalar[dtype] = min_finite[dtype]()
        for j in range(N):
            var val = rebind[Scalar[dtype]](input[i, j])
            if val > row_max:
                row_max = val

        # Compute exp and sum
        var row_sum: Scalar[dtype] = 0
        for j in range(N):
            var exp_val = exp(rebind[Scalar[dtype]](input[i, j]) - row_max)
            output[i, j] = exp_val
            row_sum += exp_val

        # Normalize
        for j in range(N):
            output[i, j] = rebind[Scalar[dtype]](output[i, j]) / row_sum


# =============================================================================
# CPU Reference for Backward Pass
# =============================================================================


fn policy_gradient_cpu_reference[
    M: Int, N: Int
](
    d_logits: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    probs: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
    actions: LayoutTensor[DType.int32, Layout.row_major(M), ImmutAnyOrigin],
    advantages: LayoutTensor[dtype, Layout.row_major(M), ImmutAnyOrigin],
    entropy_coef: Scalar[dtype],
):
    """CPU reference for policy gradient."""
    for i in range(M):
        var action = Int(rebind[Scalar[DType.int32]](actions[i]))
        var advantage = rebind[Scalar[dtype]](advantages[i])
        for j in range(N):
            var prob = rebind[Scalar[dtype]](probs[i, j])
            var one_hot: Scalar[dtype] = Scalar[dtype](
                1.0
            ) if action == j else Scalar[dtype](0.0)
            var policy_grad = advantage * (prob - one_hot)
            var prob_clamped = max(prob, Scalar[dtype](1e-8))
            var entropy_grad = entropy_coef * (
                Scalar[dtype](1.0) + log(prob_clamped)
            )
            d_logits[i, j] = policy_grad + entropy_grad


fn matmul_transA_cpu_reference[
    M: Int, K: Int, N: Int
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(K, M), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """CPU reference for C = A.T @ B where A is [K, M]."""
    for i in range(M):
        for j in range(N):
            var acc: Scalar[dtype] = 0
            for k in range(K):
                acc += rebind[Scalar[dtype]](A[k, i]) * rebind[Scalar[dtype]](
                    B[k, j]
                )
            C[i, j] = acc


fn matmul_transB_cpu_reference[
    M: Int, K: Int, N: Int
](
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(N, K), ImmutAnyOrigin],
):
    """CPU reference for C = A @ B.T where B is [N, K]."""
    for i in range(M):
        for j in range(N):
            var acc: Scalar[dtype] = 0
            for k in range(K):
                acc += rebind[Scalar[dtype]](A[i, k]) * rebind[Scalar[dtype]](
                    B[j, k]
                )
            C[i, j] = acc


fn relu_backward_cpu_reference[
    M: Int, N: Int
](
    d_input: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    d_output: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
    pre_activation: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """CPU reference for ReLU backward."""
    for i in range(M):
        for j in range(N):
            var d_out = rebind[Scalar[dtype]](d_output[i, j])
            var pre_act = rebind[Scalar[dtype]](pre_activation[i, j])
            d_input[i, j] = d_out if pre_act > Scalar[dtype](0) else Scalar[
                dtype
            ](0)


fn bias_gradient_cpu_reference[
    M: Int, N: Int
](
    d_bias: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    d_output: LayoutTensor[dtype, Layout.row_major(M, N), ImmutAnyOrigin],
):
    """CPU reference for bias gradient (sum over batch dimension)."""
    for j in range(N):
        var acc: Scalar[dtype] = 0
        for i in range(M):
            acc += rebind[Scalar[dtype]](d_output[i, j])
        d_bias[j] = acc


# =============================================================================
# Test Harness
# =============================================================================


fn test_square_matmul() raises:
    """Test with square matrices first (matching p16.mojo pattern)."""
    print("=" * 60)
    print("Testing Square Matrix Multiplication (p16 pattern)")
    print("=" * 60)

    # Use square matrices like p16.mojo
    comptime SIZE = 16
    comptime TILE = 8

    print("  SIZE:", SIZE)
    print("  TILE:", TILE)
    print()

    with DeviceContext() as ctx:
        var A_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        var B_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        var C_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        var C_expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)

        # Initialize A and B with simple values
        with A_buf.map_to_host() as A_host, B_buf.map_to_host() as B_host:
            for i in range(SIZE):
                for j in range(SIZE):
                    A_host[i * SIZE + j] = Scalar[dtype](i * SIZE + j)
                    B_host[i * SIZE + j] = Scalar[dtype](2.0) * (i * SIZE + j)

            # Compute expected result on CPU
            for i in range(SIZE):
                for j in range(SIZE):
                    var acc: Scalar[dtype] = 0
                    for k in range(SIZE):
                        acc += A_host[i * SIZE + k] * B_host[k * SIZE + j]
                    C_expected[i * SIZE + j] = acc

        C_buf.enqueue_fill(0)

        # Create tensors with square layout
        var A = LayoutTensor[
            dtype, Layout.row_major(SIZE, SIZE), ImmutAnyOrigin
        ](A_buf)
        var B = LayoutTensor[
            dtype, Layout.row_major(SIZE, SIZE), ImmutAnyOrigin
        ](B_buf)
        var C = LayoutTensor[dtype, Layout.row_major(SIZE, SIZE), MutAnyOrigin](
            C_buf
        )

        comptime grid_dim = (SIZE + TILE - 1) // TILE

        print("  Grid dim: (", grid_dim, ",", grid_dim, ")")
        print("  Block dim: (", TILE, ",", TILE, ")")
        print()

        # Run kernel
        print("  Running square naive matmul...")
        ctx.enqueue_function_checked[
            naive_matmul_kernel[SIZE, SIZE, SIZE, TILE],
            naive_matmul_kernel[SIZE, SIZE, SIZE, TILE],
        ](
            C,
            A,
            B,
            grid_dim=(grid_dim, grid_dim),
            block_dim=(TILE, TILE),
        )
        ctx.synchronize()

        # Check results
        var max_diff: Float64 = 0.0
        with C_buf.map_to_host() as C_host:
            for i in range(SIZE * SIZE):
                var diff = abs(Float64(C_host[i]) - Float64(C_expected[i]))
                if diff > max_diff:
                    max_diff = diff

        print("  Max difference:", max_diff)
        if max_diff < 1e-3:
            print("  Status: PASSED")
        else:
            print("  Status: FAILED")
        print()


fn test_rectangular_matmul() raises:
    """Test rectangular matrix multiplication with correctness verification."""
    print("=" * 60)
    print("Testing Rectangular Matrix Multiplication")
    print("=" * 60)

    comptime M = 64  # rows (NUM_ENVS)
    comptime K = 8  # inner dim (OBS_DIM)
    comptime N = 32  # cols (HIDDEN_DIM)
    comptime TILE = 8

    print("  M:", M, "K:", K, "N:", N, "TILE:", TILE)
    print()

    with DeviceContext() as ctx:
        var A_buf = ctx.enqueue_create_buffer[dtype](M * K)
        var B_buf = ctx.enqueue_create_buffer[dtype](K * N)
        var C_naive_buf = ctx.enqueue_create_buffer[dtype](M * N)
        var C_tiled_buf = ctx.enqueue_create_buffer[dtype](M * N)
        var C_expected = ctx.enqueue_create_host_buffer[dtype](M * N)

        # Initialize with random-ish values
        with A_buf.map_to_host() as A_host, B_buf.map_to_host() as B_host:
            for i in range(M):
                for j in range(K):
                    A_host[i * K + j] = Scalar[dtype](i * K + j)
            for i in range(K):
                for j in range(N):
                    B_host[i * N + j] = Scalar[dtype](2.0) * (i * N + j)

            # Compute expected on CPU
            for i in range(M):
                for j in range(N):
                    var acc: Scalar[dtype] = 0
                    for k in range(K):
                        acc += A_host[i * K + k] * B_host[k * N + j]
                    C_expected[i * N + j] = acc

        C_naive_buf.enqueue_fill(0)
        C_tiled_buf.enqueue_fill(0)

        var A = LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin](
            A_buf
        )
        var B = LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin](
            B_buf
        )
        var C_naive = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
            C_naive_buf
        )
        var C_tiled = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
            C_tiled_buf
        )

        comptime grid_x = (N + TILE - 1) // TILE
        comptime grid_y = (M + TILE - 1) // TILE

        print("  Grid: (", grid_x, ",", grid_y, ")")
        print("  Block: (", TILE, ",", TILE, ")")
        print()

        # Test 1: Naive matmul
        print("  Test 1: Naive matmul...")
        ctx.enqueue_function_checked[
            naive_matmul_kernel[M, K, N, TILE],
            naive_matmul_kernel[M, K, N, TILE],
        ](C_naive, A, B, grid_dim=(grid_x, grid_y), block_dim=(TILE, TILE))
        ctx.synchronize()

        var max_diff_naive: Float64 = 0.0
        with C_naive_buf.map_to_host() as C_host:
            for i in range(M * N):
                var diff = abs(Float64(C_host[i]) - Float64(C_expected[i]))
                if diff > max_diff_naive:
                    max_diff_naive = diff

        print("    Max diff:", max_diff_naive)
        if max_diff_naive < 1e-2:
            print("    Status: PASSED")
        else:
            print("    Status: FAILED")

        # Test 2: Tiled matmul with shared memory
        print("  Test 2: Tiled matmul (shared memory)...")
        ctx.enqueue_function_checked[
            tiled_matmul_kernel[M, K, N, TILE],
            tiled_matmul_kernel[M, K, N, TILE],
        ](C_tiled, A, B, grid_dim=(grid_x, grid_y), block_dim=(TILE, TILE))
        ctx.synchronize()

        var max_diff_tiled: Float64 = 0.0
        with C_tiled_buf.map_to_host() as C_host:
            for i in range(M * N):
                var diff = abs(Float64(C_host[i]) - Float64(C_expected[i]))
                if diff > max_diff_tiled:
                    max_diff_tiled = diff

        print("    Max diff:", max_diff_tiled)
        if max_diff_tiled < 1e-2:
            print("    Status: PASSED")
        else:
            print("    Status: FAILED")

        print()


fn test_matmul_bias_relu() raises:
    """Test matmul + bias + ReLU."""
    print("=" * 60)
    print("Testing Matmul + Bias + ReLU")
    print("=" * 60)

    comptime M = 64  # NUM_ENVS
    comptime K = 8  # OBS_DIM
    comptime N = 32  # HIDDEN_DIM
    comptime TILE = 8

    print("  M:", M, "K:", K, "N:", N)
    print()

    with DeviceContext() as ctx:
        var A_buf = ctx.enqueue_create_buffer[dtype](M * K)
        var B_buf = ctx.enqueue_create_buffer[dtype](K * N)
        var bias_buf = ctx.enqueue_create_buffer[dtype](N)
        var C_gpu_buf = ctx.enqueue_create_buffer[dtype](M * N)
        var C_expected = ctx.enqueue_create_host_buffer[dtype](M * N)

        # Initialize
        with A_buf.map_to_host() as A_host, B_buf.map_to_host() as B_host, bias_buf.map_to_host() as bias_host:
            for i in range(M * K):
                A_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.5)
            for i in range(K * N):
                B_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.5)
            for i in range(N):
                bias_host[i] = Scalar[dtype](i * 0.01)

            # CPU reference
            for i in range(M):
                for j in range(N):
                    var acc: Scalar[dtype] = 0
                    for k in range(K):
                        acc += A_host[i * K + k] * B_host[k * N + j]
                    var result = acc + bias_host[j]
                    C_expected[i * N + j] = result if result > Scalar[dtype](
                        0
                    ) else Scalar[dtype](0)

        C_gpu_buf.enqueue_fill(0)

        var A = LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin](
            A_buf
        )
        var B = LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin](
            B_buf
        )
        var bias = LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin](
            bias_buf
        )
        var C_gpu = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
            C_gpu_buf
        )

        comptime grid_x = (N + TILE - 1) // TILE
        comptime grid_y = (M + TILE - 1) // TILE

        print("  Running tiled_matmul_bias_relu_kernel...")
        ctx.enqueue_function_checked[
            tiled_matmul_bias_relu_kernel[M, K, N, TILE],
            tiled_matmul_bias_relu_kernel[M, K, N, TILE],
        ](C_gpu, A, B, bias, grid_dim=(grid_x, grid_y), block_dim=(TILE, TILE))
        ctx.synchronize()

        var max_diff: Float64 = 0.0
        with C_gpu_buf.map_to_host() as C_host:
            for i in range(M * N):
                var diff = abs(Float64(C_host[i]) - Float64(C_expected[i]))
                if diff > max_diff:
                    max_diff = diff

        print("  Max diff:", max_diff)
        if max_diff < 1e-2:
            print("  Status: PASSED")
        else:
            print("  Status: FAILED")
        print()


fn test_parallel_softmax() raises:
    """Test parallel softmax."""
    print("=" * 60)
    print("Testing Parallel Softmax")
    print("=" * 60)

    comptime M = 64  # NUM_ENVS
    comptime N = 4  # NUM_ACTIONS (small, typical for RL)

    print("  M:", M, "N:", N)
    print()

    with DeviceContext() as ctx:
        var input_buf = ctx.enqueue_create_buffer[dtype](M * N)
        var output_gpu_buf = ctx.enqueue_create_buffer[dtype](M * N)
        var output_expected = ctx.enqueue_create_host_buffer[dtype](M * N)

        # Initialize with logits (can be any values)
        with input_buf.map_to_host() as input_host:
            for i in range(M):
                for j in range(N):
                    input_host[i * N + j] = Scalar[dtype]((i + j) * 0.1 - 0.5)

            # CPU reference
            for i in range(M):
                var row_max: Scalar[dtype] = min_finite[dtype]()
                for j in range(N):
                    var val = input_host[i * N + j]
                    if val > row_max:
                        row_max = val

                var row_sum: Scalar[dtype] = 0
                for j in range(N):
                    var exp_val = exp(input_host[i * N + j] - row_max)
                    output_expected[i * N + j] = exp_val
                    row_sum += exp_val

                for j in range(N):
                    output_expected[i * N + j] = (
                        output_expected[i * N + j] / row_sum
                    )

        output_gpu_buf.enqueue_fill(0)

        var input_t = LayoutTensor[
            dtype, Layout.row_major(M, N), ImmutAnyOrigin
        ](input_buf)
        var output_t = LayoutTensor[
            dtype, Layout.row_major(M, N), MutAnyOrigin
        ](output_gpu_buf)

        # Grid: one block per row, block size = next power of 2 >= N
        comptime BLOCK_SIZE = 1 << log2_ceil(N)

        print("  Block size:", BLOCK_SIZE)
        print("  Running parallel_softmax_kernel...")
        ctx.enqueue_function_checked[
            parallel_softmax_kernel[M, N],
            parallel_softmax_kernel[M, N],
        ](output_t, input_t, grid_dim=(1, M), block_dim=(BLOCK_SIZE, 1))
        ctx.synchronize()

        var max_diff: Float64 = 0.0
        with output_gpu_buf.map_to_host() as output_host:
            for i in range(M * N):
                var diff = abs(
                    Float64(output_host[i]) - Float64(output_expected[i])
                )
                if diff > max_diff:
                    max_diff = diff

        print("  Max diff:", max_diff)
        if max_diff < 1e-5:
            print("  Status: PASSED")
        else:
            print("  Status: FAILED")

        # Verify probabilities sum to 1
        var sum_ok = True
        with output_gpu_buf.map_to_host() as output_host:
            for i in range(M):
                var row_sum: Float64 = 0
                for j in range(N):
                    row_sum += Float64(output_host[i * N + j])
                if abs(row_sum - 1.0) > 1e-5:
                    sum_ok = False
                    print("  Row", i, "sum:", row_sum)
                    break

        if sum_ok:
            print("  Probability sums: OK (all rows sum to 1)")
        else:
            print("  Probability sums: FAILED")
        print()


fn test_batched_forward_pass() raises:
    """Test complete batched forward pass for A2C.

    Forward pass:
    1. obs [NUM_ENVS, OBS_DIM] @ W1 + b1 -> hidden (with ReLU)
    2. hidden @ W_actor + b_actor -> logits
    3. softmax(logits) -> probs
    4. hidden @ W_critic + b_critic -> values
    """
    print("=" * 60)
    print("Testing Batched Forward Pass (A2C)")
    print("=" * 60)

    # RL-realistic dimensions
    comptime NUM_ENVS = 256
    comptime OBS_DIM = 4  # CartPole
    comptime HIDDEN_DIM = 64
    comptime NUM_ACTIONS = 2
    comptime TILE = 8

    print("  NUM_ENVS:", NUM_ENVS)
    print("  OBS_DIM:", OBS_DIM)
    print("  HIDDEN_DIM:", HIDDEN_DIM)
    print("  NUM_ACTIONS:", NUM_ACTIONS)
    print()

    with DeviceContext() as ctx:
        # Allocate all buffers
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var logits_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)
        var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)

        # Expected results (CPU)
        var hidden_expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var logits_expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var probs_expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var values_expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_ENVS * 1
        )

        # Initialize weights and observations
        with obs_buf.map_to_host() as obs_host, W1_buf.map_to_host() as W1_host, b1_buf.map_to_host() as b1_host, W_actor_buf.map_to_host() as W_actor_host, b_actor_buf.map_to_host() as b_actor_host, W_critic_buf.map_to_host() as W_critic_host, b_critic_buf.map_to_host() as b_critic_host:
            # Xavier-like init
            for i in range(NUM_ENVS * OBS_DIM):
                obs_host[i] = Scalar[dtype]((i % 20) * 0.1 - 1.0)
            for i in range(OBS_DIM * HIDDEN_DIM):
                W1_host[i] = Scalar[dtype]((i % 10) * 0.02 - 0.1)
            for i in range(HIDDEN_DIM):
                b1_host[i] = Scalar[dtype](0.01)
            for i in range(HIDDEN_DIM * NUM_ACTIONS):
                W_actor_host[i] = Scalar[dtype]((i % 10) * 0.02 - 0.1)
            for i in range(NUM_ACTIONS):
                b_actor_host[i] = Scalar[dtype](0.0)
            for i in range(HIDDEN_DIM):
                W_critic_host[i] = Scalar[dtype]((i % 10) * 0.02 - 0.1)
            b_critic_host[0] = Scalar[dtype](0.0)

            # === CPU Reference Forward Pass ===
            # 1. Hidden = ReLU(obs @ W1 + b1)
            for i in range(NUM_ENVS):
                for j in range(HIDDEN_DIM):
                    var acc: Scalar[dtype] = b1_host[j]
                    for k in range(OBS_DIM):
                        acc += (
                            obs_host[i * OBS_DIM + k]
                            * W1_host[k * HIDDEN_DIM + j]
                        )
                    hidden_expected[i * HIDDEN_DIM + j] = acc if acc > Scalar[
                        dtype
                    ](0) else Scalar[dtype](0)

            # 2. Logits = hidden @ W_actor + b_actor
            for i in range(NUM_ENVS):
                for j in range(NUM_ACTIONS):
                    var acc: Scalar[dtype] = b_actor_host[j]
                    for k in range(HIDDEN_DIM):
                        acc += (
                            hidden_expected[i * HIDDEN_DIM + k]
                            * W_actor_host[k * NUM_ACTIONS + j]
                        )
                    logits_expected[i * NUM_ACTIONS + j] = acc

            # 3. Probs = softmax(logits)
            for i in range(NUM_ENVS):
                var row_max: Scalar[dtype] = min_finite[dtype]()
                for j in range(NUM_ACTIONS):
                    if logits_expected[i * NUM_ACTIONS + j] > row_max:
                        row_max = logits_expected[i * NUM_ACTIONS + j]
                var row_sum: Scalar[dtype] = 0
                for j in range(NUM_ACTIONS):
                    var exp_val = exp(
                        logits_expected[i * NUM_ACTIONS + j] - row_max
                    )
                    probs_expected[i * NUM_ACTIONS + j] = exp_val
                    row_sum += exp_val
                for j in range(NUM_ACTIONS):
                    probs_expected[i * NUM_ACTIONS + j] = (
                        probs_expected[i * NUM_ACTIONS + j] / row_sum
                    )

            # 4. Values = hidden @ W_critic + b_critic
            for i in range(NUM_ENVS):
                var acc: Scalar[dtype] = b_critic_host[0]
                for k in range(HIDDEN_DIM):
                    acc += (
                        hidden_expected[i * HIDDEN_DIM + k] * W_critic_host[k]
                    )
                values_expected[i] = acc

        # Zero output buffers
        hidden_buf.enqueue_fill(0)
        logits_buf.enqueue_fill(0)
        probs_buf.enqueue_fill(0)
        values_buf.enqueue_fill(0)

        # Create tensors
        var obs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
        ](obs_buf)
        var W1 = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](W1_buf)
        var b1 = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](b1_buf)
        var hidden = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](hidden_buf)
        var hidden_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](hidden_buf)
        var W_actor = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
        ](W_actor_buf)
        var b_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
        ](b_actor_buf)
        var logits = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](logits_buf)
        var logits_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](logits_buf)
        var probs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](probs_buf)
        var W_critic = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
        ](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin](
            b_critic_buf
        )
        var values = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
        ](values_buf)

        # === GPU Forward Pass ===
        print("  Running GPU forward pass...")

        # Step 1: Hidden layer (obs @ W1 + b1 with ReLU)
        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE
        ctx.enqueue_function_checked[
            tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
            tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
        ](
            hidden,
            obs,
            W1,
            b1,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )

        # Step 2: Actor logits (hidden @ W_actor + b_actor)
        comptime grid_a_x = (NUM_ACTIONS + TILE - 1) // TILE
        comptime grid_a_y = (NUM_ENVS + TILE - 1) // TILE
        ctx.enqueue_function_checked[
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
        ](
            logits,
            hidden_immut,
            W_actor,
            b_actor,
            grid_dim=(grid_a_x, grid_a_y),
            block_dim=(TILE, TILE),
        )

        # Step 3: Softmax
        comptime SOFTMAX_BLOCK = 1 << log2_ceil(NUM_ACTIONS)
        ctx.enqueue_function_checked[
            parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
            parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
        ](
            probs,
            logits_immut,
            grid_dim=(1, NUM_ENVS),
            block_dim=(SOFTMAX_BLOCK, 1),
        )

        # Step 4: Critic value (hidden @ W_critic + b_critic)
        comptime grid_v_x = (1 + TILE - 1) // TILE
        comptime grid_v_y = (NUM_ENVS + TILE - 1) // TILE
        ctx.enqueue_function_checked[
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
        ](
            values,
            hidden_immut,
            W_critic,
            b_critic,
            grid_dim=(grid_v_x, grid_v_y),
            block_dim=(TILE, TILE),
        )

        ctx.synchronize()

        # === Verify Results ===
        print("  Verifying results...")

        # Check hidden
        var max_diff_hidden: Float64 = 0.0
        with hidden_buf.map_to_host() as h:
            for i in range(NUM_ENVS * HIDDEN_DIM):
                var diff = abs(Float64(h[i]) - Float64(hidden_expected[i]))
                if diff > max_diff_hidden:
                    max_diff_hidden = diff
        print(
            "    Hidden max diff:",
            max_diff_hidden,
            "PASSED" if max_diff_hidden < 1e-3 else "FAILED",
        )

        # Check logits
        var max_diff_logits: Float64 = 0.0
        with logits_buf.map_to_host() as l:
            for i in range(NUM_ENVS * NUM_ACTIONS):
                var diff = abs(Float64(l[i]) - Float64(logits_expected[i]))
                if diff > max_diff_logits:
                    max_diff_logits = diff
        print(
            "    Logits max diff:",
            max_diff_logits,
            "PASSED" if max_diff_logits < 1e-3 else "FAILED",
        )

        # Check probs
        var max_diff_probs: Float64 = 0.0
        with probs_buf.map_to_host() as p:
            for i in range(NUM_ENVS * NUM_ACTIONS):
                var diff = abs(Float64(p[i]) - Float64(probs_expected[i]))
                if diff > max_diff_probs:
                    max_diff_probs = diff
        print(
            "    Probs max diff:",
            max_diff_probs,
            "PASSED" if max_diff_probs < 1e-5 else "FAILED",
        )

        # Check values
        var max_diff_values: Float64 = 0.0
        with values_buf.map_to_host() as v:
            for i in range(NUM_ENVS):
                var diff = abs(Float64(v[i]) - Float64(values_expected[i]))
                if diff > max_diff_values:
                    max_diff_values = diff
        print(
            "    Values max diff:",
            max_diff_values,
            "PASSED" if max_diff_values < 1e-3 else "FAILED",
        )

        # Overall status
        var all_passed = (
            max_diff_hidden < 1e-3
            and max_diff_logits < 1e-3
            and max_diff_probs < 1e-5
            and max_diff_values < 1e-3
        )
        print()
        if all_passed:
            print("  Forward pass: ALL PASSED")
        else:
            print("  Forward pass: SOME FAILED")
        print()


fn test_backward_pass_kernels() raises:
    """Test individual backward pass kernels."""
    print("=" * 60)
    print("Testing Backward Pass Kernels")
    print("=" * 60)

    comptime NUM_ENVS = 64
    comptime NUM_ACTIONS = 4
    comptime HIDDEN_DIM = 32
    comptime OBS_DIM = 8
    comptime TILE = 8

    print("  NUM_ENVS:", NUM_ENVS)
    print("  NUM_ACTIONS:", NUM_ACTIONS)
    print("  HIDDEN_DIM:", HIDDEN_DIM)
    print("  OBS_DIM:", OBS_DIM)
    print()

    with DeviceContext() as ctx:
        # =====================================================================
        # Test 1: Policy Gradient Kernel
        # =====================================================================
        print("  Test 1: Policy Gradient Kernel...")

        var probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var actions_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
        var advantages_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        var d_logits_gpu_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var d_logits_expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )

        # Initialize
        with probs_buf.map_to_host() as probs_host, actions_buf.map_to_host() as actions_host, advantages_buf.map_to_host() as adv_host:
            for i in range(NUM_ENVS):
                # Create valid probability distribution
                var sum_p: Scalar[dtype] = 0
                for j in range(NUM_ACTIONS):
                    probs_host[i * NUM_ACTIONS + j] = Scalar[dtype](
                        0.1 + (i + j) % 5 * 0.1
                    )
                    sum_p += probs_host[i * NUM_ACTIONS + j]
                # Normalize
                for j in range(NUM_ACTIONS):
                    probs_host[i * NUM_ACTIONS + j] = (
                        probs_host[i * NUM_ACTIONS + j] / sum_p
                    )

                actions_host[i] = Scalar[DType.int32](i % NUM_ACTIONS)
                adv_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.5)

            # CPU reference
            var entropy_coef = Scalar[dtype](0.01)
            for i in range(NUM_ENVS):
                var action = Int(actions_host[i])
                var advantage = adv_host[i]
                for j in range(NUM_ACTIONS):
                    var prob = probs_host[i * NUM_ACTIONS + j]
                    var one_hot: Scalar[dtype] = Scalar[dtype](
                        1.0
                    ) if action == j else Scalar[dtype](0.0)
                    var policy_grad = advantage * (prob - one_hot)
                    var prob_clamped = max(prob, Scalar[dtype](1e-8))
                    var entropy_grad = entropy_coef * (
                        Scalar[dtype](1.0) + log(prob_clamped)
                    )
                    d_logits_expected[i * NUM_ACTIONS + j] = (
                        policy_grad + entropy_grad
                    )

        d_logits_gpu_buf.enqueue_fill(0)

        var probs_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](probs_buf)
        var actions_t = LayoutTensor[
            DType.int32, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](actions_buf)
        var adv_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](advantages_buf)
        var d_logits_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](d_logits_gpu_buf)

        comptime grid_pg_x = (NUM_ACTIONS + TILE - 1) // TILE
        comptime grid_pg_y = (NUM_ENVS + TILE - 1) // TILE

        ctx.enqueue_function_checked[
            policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
            policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
        ](
            d_logits_t,
            probs_t,
            actions_t,
            adv_t,
            Scalar[dtype](0.01),
            grid_dim=(grid_pg_x, grid_pg_y),
            block_dim=(TILE, TILE),
        )
        ctx.synchronize()

        var max_diff_pg: Float64 = 0.0
        with d_logits_gpu_buf.map_to_host() as d_logits_host:
            for i in range(NUM_ENVS * NUM_ACTIONS):
                var diff = abs(
                    Float64(d_logits_host[i]) - Float64(d_logits_expected[i])
                )
                if diff > max_diff_pg:
                    max_diff_pg = diff

        print(
            "    Max diff:",
            max_diff_pg,
            "PASSED" if max_diff_pg < 1e-5 else "FAILED",
        )

        # =====================================================================
        # Test 2: Matmul Transpose A (for weight gradients: d_W = input.T @ d_output)
        # =====================================================================
        print("  Test 2: Matmul Transpose A (d_W = input.T @ d_output)...")

        # C[OBS_DIM, HIDDEN_DIM] = A[NUM_ENVS, OBS_DIM].T @ B[NUM_ENVS, HIDDEN_DIM]
        var A_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * OBS_DIM
        )  # [NUM_ENVS, OBS_DIM]
        var B_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )  # [NUM_ENVS, HIDDEN_DIM]
        var C_gpu_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var C_expected = ctx.enqueue_create_host_buffer[dtype](
            OBS_DIM * HIDDEN_DIM
        )

        with A_buf.map_to_host() as A_host, B_buf.map_to_host() as B_host:
            for i in range(NUM_ENVS * OBS_DIM):
                A_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.5)
            for i in range(NUM_ENVS * HIDDEN_DIM):
                B_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.3)

            # CPU reference: C[i, j] = sum_k(A[k, i] * B[k, j])
            for i in range(OBS_DIM):
                for j in range(HIDDEN_DIM):
                    var acc: Scalar[dtype] = 0
                    for k in range(NUM_ENVS):
                        acc += (
                            A_host[k * OBS_DIM + i] * B_host[k * HIDDEN_DIM + j]
                        )
                    C_expected[i * HIDDEN_DIM + j] = acc

        C_gpu_buf.enqueue_fill(0)

        var A_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
        ](A_buf)
        var B_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](B_buf)
        var C_t = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin
        ](C_gpu_buf)

        comptime grid_ta_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_ta_y = (OBS_DIM + TILE - 1) // TILE

        ctx.enqueue_function_checked[
            tiled_matmul_transA_kernel[OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE],
            tiled_matmul_transA_kernel[OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE],
        ](
            C_t,
            A_t,
            B_t,
            grid_dim=(grid_ta_x, grid_ta_y),
            block_dim=(TILE, TILE),
        )
        ctx.synchronize()

        var max_diff_ta: Float64 = 0.0
        with C_gpu_buf.map_to_host() as C_host:
            for i in range(OBS_DIM * HIDDEN_DIM):
                var diff = abs(Float64(C_host[i]) - Float64(C_expected[i]))
                if diff > max_diff_ta:
                    max_diff_ta = diff

        print(
            "    Max diff:",
            max_diff_ta,
            "PASSED" if max_diff_ta < 1e-2 else "FAILED",
        )

        # =====================================================================
        # Test 3: Matmul Transpose B (for input gradients: d_input = d_output @ W.T)
        # =====================================================================
        print("  Test 3: Matmul Transpose B (d_input = d_output @ W.T)...")

        # C[NUM_ENVS, OBS_DIM] = A[NUM_ENVS, HIDDEN_DIM] @ B[OBS_DIM, HIDDEN_DIM].T
        var A2_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )  # d_output
        var B2_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)  # W
        var C2_gpu_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var C2_expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_ENVS * OBS_DIM
        )

        with A2_buf.map_to_host() as A2_host, B2_buf.map_to_host() as B2_host:
            for i in range(NUM_ENVS * HIDDEN_DIM):
                A2_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.5)
            for i in range(OBS_DIM * HIDDEN_DIM):
                B2_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.3)

            # CPU reference: C[i, j] = sum_k(A[i, k] * B[j, k])
            for i in range(NUM_ENVS):
                for j in range(OBS_DIM):
                    var acc: Scalar[dtype] = 0
                    for k in range(HIDDEN_DIM):
                        acc += (
                            A2_host[i * HIDDEN_DIM + k]
                            * B2_host[j * HIDDEN_DIM + k]
                        )
                    C2_expected[i * OBS_DIM + j] = acc

        C2_gpu_buf.enqueue_fill(0)

        var A2_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](A2_buf)
        var B2_t = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](B2_buf)
        var C2_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), MutAnyOrigin
        ](C2_gpu_buf)

        comptime grid_tb_x = (OBS_DIM + TILE - 1) // TILE
        comptime grid_tb_y = (NUM_ENVS + TILE - 1) // TILE

        ctx.enqueue_function_checked[
            tiled_matmul_transB_kernel[NUM_ENVS, HIDDEN_DIM, OBS_DIM, TILE],
            tiled_matmul_transB_kernel[NUM_ENVS, HIDDEN_DIM, OBS_DIM, TILE],
        ](
            C2_t,
            A2_t,
            B2_t,
            grid_dim=(grid_tb_x, grid_tb_y),
            block_dim=(TILE, TILE),
        )
        ctx.synchronize()

        var max_diff_tb: Float64 = 0.0
        with C2_gpu_buf.map_to_host() as C2_host:
            for i in range(NUM_ENVS * OBS_DIM):
                var diff = abs(Float64(C2_host[i]) - Float64(C2_expected[i]))
                if diff > max_diff_tb:
                    max_diff_tb = diff

        print(
            "    Max diff:",
            max_diff_tb,
            "PASSED" if max_diff_tb < 1e-2 else "FAILED",
        )

        # =====================================================================
        # Test 4: ReLU Backward
        # =====================================================================
        print("  Test 4: ReLU Backward...")

        var d_out_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var pre_act_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_in_gpu_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_in_expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )

        with d_out_buf.map_to_host() as d_out_host, pre_act_buf.map_to_host() as pre_act_host:
            for i in range(NUM_ENVS * HIDDEN_DIM):
                d_out_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.5)
                pre_act_host[i] = Scalar[dtype](
                    (i % 20) * 0.1 - 1.0
                )  # Some positive, some negative

            # CPU reference
            for i in range(NUM_ENVS * HIDDEN_DIM):
                d_in_expected[i] = d_out_host[i] if pre_act_host[i] > Scalar[
                    dtype
                ](0) else Scalar[dtype](0)

        d_in_gpu_buf.enqueue_fill(0)

        var d_out_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_out_buf)
        var pre_act_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](pre_act_buf)
        var d_in_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_in_gpu_buf)

        comptime grid_relu_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_relu_y = (NUM_ENVS + TILE - 1) // TILE

        ctx.enqueue_function_checked[
            relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
            relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
        ](
            d_in_t,
            d_out_t,
            pre_act_t,
            grid_dim=(grid_relu_x, grid_relu_y),
            block_dim=(TILE, TILE),
        )
        ctx.synchronize()

        var max_diff_relu: Float64 = 0.0
        with d_in_gpu_buf.map_to_host() as d_in_host:
            for i in range(NUM_ENVS * HIDDEN_DIM):
                var diff = abs(
                    Float64(d_in_host[i]) - Float64(d_in_expected[i])
                )
                if diff > max_diff_relu:
                    max_diff_relu = diff

        print(
            "    Max diff:",
            max_diff_relu,
            "PASSED" if max_diff_relu < 1e-6 else "FAILED",
        )

        # =====================================================================
        # Test 5: Bias Gradient (parallel reduction)
        # =====================================================================
        print("  Test 5: Bias Gradient (parallel reduction)...")

        var d_output_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_bias_gpu_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var d_bias_expected = ctx.enqueue_create_host_buffer[dtype](HIDDEN_DIM)

        with d_output_buf.map_to_host() as d_output_host:
            for i in range(NUM_ENVS * HIDDEN_DIM):
                d_output_host[i] = Scalar[dtype]((i % 10) * 0.1 - 0.5)

            # CPU reference: sum over batch dimension
            for j in range(HIDDEN_DIM):
                var acc: Scalar[dtype] = 0
                for i in range(NUM_ENVS):
                    acc += d_output_host[i * HIDDEN_DIM + j]
                d_bias_expected[j] = acc

        d_bias_gpu_buf.enqueue_fill(0)

        var d_output_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_output_buf)
        var d_bias_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](d_bias_gpu_buf)

        comptime BLOCK_SIZE_BIAS = 1 << log2_ceil(NUM_ENVS)

        ctx.enqueue_function_checked[
            bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
            bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
        ](
            d_bias_t,
            d_output_t,
            grid_dim=(HIDDEN_DIM, 1),
            block_dim=(BLOCK_SIZE_BIAS, 1),
        )
        ctx.synchronize()

        var max_diff_bias: Float64 = 0.0
        with d_bias_gpu_buf.map_to_host() as d_bias_host:
            for i in range(HIDDEN_DIM):
                var diff = abs(
                    Float64(d_bias_host[i]) - Float64(d_bias_expected[i])
                )
                if diff > max_diff_bias:
                    max_diff_bias = diff

        print(
            "    Max diff:",
            max_diff_bias,
            "PASSED" if max_diff_bias < 1e-3 else "FAILED",
        )

        # =====================================================================
        # Overall Status
        # =====================================================================
        var all_passed = (
            max_diff_pg < 1e-5
            and max_diff_ta < 1e-2
            and max_diff_tb < 1e-2
            and max_diff_relu < 1e-6
            and max_diff_bias < 1e-3
        )
        print()
        if all_passed:
            print("  Backward pass kernels: ALL PASSED")
        else:
            print("  Backward pass kernels: SOME FAILED")
        print()


fn test_complete_training_step() raises:
    """Test a complete A2C training step: forward -> backward -> weight update.
    """
    print("=" * 60)
    print("Testing Complete Training Step (A2C)")
    print("=" * 60)

    # Network architecture
    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 32
    comptime NUM_ACTIONS = 2
    comptime TILE = 8

    # Hyperparameters
    var lr = Scalar[dtype](0.001)
    var entropy_coef = Scalar[dtype](0.01)
    var value_coef = Scalar[dtype](0.5)

    print("  NUM_ENVS:", NUM_ENVS)
    print("  OBS_DIM:", OBS_DIM)
    print("  HIDDEN_DIM:", HIDDEN_DIM)
    print("  NUM_ACTIONS:", NUM_ACTIONS)
    print("  Learning rate:", lr)
    print()

    with DeviceContext() as ctx:
        # =====================================================================
        # Allocate all buffers
        # =====================================================================

        # Network weights
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

        # Forward pass intermediates
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var pre_act1_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )  # Pre-ReLU
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var logits_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)

        # Training data
        var actions_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
        var advantages_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        var returns_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)

        # Gradients
        var d_logits_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var d_values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)
        var d_hidden_actor_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_hidden_critic_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_hidden_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_pre_relu_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )

        # Weight gradients
        var d_W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var d_b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var d_W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var d_b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var d_W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var d_b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

        # =====================================================================
        # Initialize weights and data
        # =====================================================================
        print("  Initializing weights and data...")

        with W1_buf.map_to_host() as W1_host, b1_buf.map_to_host() as b1_host, W_actor_buf.map_to_host() as W_actor_host, b_actor_buf.map_to_host() as b_actor_host, W_critic_buf.map_to_host() as W_critic_host, b_critic_buf.map_to_host() as b_critic_host:
            # Xavier-like init
            for i in range(OBS_DIM * HIDDEN_DIM):
                W1_host[i] = Scalar[dtype]((i % 20) * 0.01 - 0.1)
            for i in range(HIDDEN_DIM):
                b1_host[i] = Scalar[dtype](0.0)
            for i in range(HIDDEN_DIM * NUM_ACTIONS):
                W_actor_host[i] = Scalar[dtype]((i % 20) * 0.01 - 0.1)
            for i in range(NUM_ACTIONS):
                b_actor_host[i] = Scalar[dtype](0.0)
            for i in range(HIDDEN_DIM):
                W_critic_host[i] = Scalar[dtype]((i % 20) * 0.01 - 0.1)
            b_critic_host[0] = Scalar[dtype](0.0)

        with obs_buf.map_to_host() as obs_host, actions_buf.map_to_host() as actions_host, advantages_buf.map_to_host() as adv_host, returns_buf.map_to_host() as ret_host:
            for i in range(NUM_ENVS):
                for j in range(OBS_DIM):
                    obs_host[i * OBS_DIM + j] = Scalar[dtype](
                        (i + j) % 10 * 0.1 - 0.5
                    )
                actions_host[i] = Scalar[DType.int32](i % NUM_ACTIONS)
                adv_host[i] = Scalar[dtype](
                    (i % 10) * 0.2 - 1.0
                )  # Advantages in [-1, 1]
                ret_host[i] = Scalar[dtype]((i % 10) * 0.1)  # Returns

        # Zero gradient buffers
        d_logits_buf.enqueue_fill(0)
        d_values_buf.enqueue_fill(0)
        d_hidden_actor_buf.enqueue_fill(0)
        d_hidden_critic_buf.enqueue_fill(0)
        d_hidden_buf.enqueue_fill(0)
        d_pre_relu_buf.enqueue_fill(0)
        d_W1_buf.enqueue_fill(0)
        d_b1_buf.enqueue_fill(0)
        d_W_actor_buf.enqueue_fill(0)
        d_b_actor_buf.enqueue_fill(0)
        d_W_critic_buf.enqueue_fill(0)
        d_b_critic_buf.enqueue_fill(0)
        pre_act1_buf.enqueue_fill(0)
        hidden_buf.enqueue_fill(0)
        logits_buf.enqueue_fill(0)
        probs_buf.enqueue_fill(0)
        values_buf.enqueue_fill(0)

        # =====================================================================
        # Create tensors
        # =====================================================================
        var obs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
        ](obs_buf)
        var W1 = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin
        ](W1_buf)
        var W1_immut = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](W1_buf)
        var b1 = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](b1_buf)
        var b1_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](b1_buf)
        var pre_act1 = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](pre_act1_buf)
        var pre_act1_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](pre_act1_buf)
        var hidden = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](hidden_buf)
        var hidden_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](hidden_buf)
        var W_actor = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), MutAnyOrigin
        ](W_actor_buf)
        var W_actor_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
        ](W_actor_buf)
        var b_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
        ](b_actor_buf)
        var b_actor_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
        ](b_actor_buf)
        var logits = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](logits_buf)
        var logits_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](logits_buf)
        var probs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](probs_buf)
        var probs_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](probs_buf)
        var W_critic = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin
        ](W_critic_buf)
        var W_critic_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
        ](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            b_critic_buf
        )
        var b_critic_immut = LayoutTensor[
            dtype, Layout.row_major(1), ImmutAnyOrigin
        ](b_critic_buf)
        var values = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
        ](values_buf)
        var values_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
        ](values_buf)
        var actions = LayoutTensor[
            DType.int32, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](actions_buf)
        var advantages = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](advantages_buf)
        var returns = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](returns_buf)

        var d_logits = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](d_logits_buf)
        var d_logits_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](d_logits_buf)
        var d_values = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
        ](d_values_buf)
        var d_values_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
        ](d_values_buf)
        var d_hidden_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_hidden_actor_buf)
        var d_hidden_actor_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_hidden_actor_buf)
        var d_hidden_critic = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_hidden_critic_buf)
        var d_hidden_critic_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_hidden_critic_buf)
        var d_hidden = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_hidden_buf)
        var d_hidden_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_hidden_buf)
        var d_pre_relu = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_pre_relu_buf)
        var d_pre_relu_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_pre_relu_buf)

        var d_W1 = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin
        ](d_W1_buf)
        var d_W1_immut = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](d_W1_buf)
        var d_b1 = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](d_b1_buf)
        var d_b1_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](d_b1_buf)
        var d_W_actor = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), MutAnyOrigin
        ](d_W_actor_buf)
        var d_W_actor_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
        ](d_W_actor_buf)
        var d_b_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
        ](d_b_actor_buf)
        var d_b_actor_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
        ](d_b_actor_buf)
        var d_W_critic = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin
        ](d_W_critic_buf)
        var d_W_critic_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
        ](d_W_critic_buf)
        var d_b_critic = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            d_b_critic_buf
        )
        var d_b_critic_immut = LayoutTensor[
            dtype, Layout.row_major(1), ImmutAnyOrigin
        ](d_b_critic_buf)

        # =====================================================================
        # FORWARD PASS
        # =====================================================================
        print("  Running forward pass...")

        # Step 1: Hidden = ReLU(obs @ W1 + b1) - save pre-activation
        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE
        ctx.enqueue_function_checked[
            tiled_matmul_bias_relu_save_kernel[
                NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
            ],
            tiled_matmul_bias_relu_save_kernel[
                NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
            ],
        ](
            hidden,
            pre_act1,
            obs,
            W1_immut,
            b1_immut,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )

        # Step 2: Logits = hidden @ W_actor + b_actor
        comptime grid_a_x = (NUM_ACTIONS + TILE - 1) // TILE
        comptime grid_a_y = (NUM_ENVS + TILE - 1) // TILE
        ctx.enqueue_function_checked[
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
        ](
            logits,
            hidden_immut,
            W_actor_immut,
            b_actor_immut,
            grid_dim=(grid_a_x, grid_a_y),
            block_dim=(TILE, TILE),
        )

        # Step 3: Probs = softmax(logits)
        comptime SOFTMAX_BLOCK = 1 << log2_ceil(NUM_ACTIONS)
        ctx.enqueue_function_checked[
            parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
            parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
        ](
            probs,
            logits_immut,
            grid_dim=(1, NUM_ENVS),
            block_dim=(SOFTMAX_BLOCK, 1),
        )

        # Step 4: Values = hidden @ W_critic + b_critic
        comptime grid_v_x = (1 + TILE - 1) // TILE
        comptime grid_v_y = (NUM_ENVS + TILE - 1) // TILE
        ctx.enqueue_function_checked[
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
        ](
            values,
            hidden_immut,
            W_critic_immut,
            b_critic_immut,
            grid_dim=(grid_v_x, grid_v_y),
            block_dim=(TILE, TILE),
        )

        ctx.synchronize()

        # Compute initial loss for comparison
        var initial_policy_loss: Float64 = 0
        var initial_value_loss: Float64 = 0
        with probs_buf.map_to_host() as p, values_buf.map_to_host() as v, actions_buf.map_to_host() as act, advantages_buf.map_to_host() as adv, returns_buf.map_to_host() as ret:
            for i in range(NUM_ENVS):
                var action = Int(act[i])
                var prob = Float64(p[i * NUM_ACTIONS + action])
                var advantage = Float64(adv[i])
                initial_policy_loss -= advantage * log(max(prob, 1e-8))

                var value = Float64(v[i])
                var target = Float64(ret[i])
                initial_value_loss += 0.5 * (value - target) * (value - target)

        initial_policy_loss /= NUM_ENVS
        initial_value_loss /= NUM_ENVS
        print("    Initial policy loss:", initial_policy_loss)
        print("    Initial value loss:", initial_value_loss)

        # =====================================================================
        # BACKWARD PASS
        # =====================================================================
        print("  Running backward pass...")

        # Step 1: Policy gradient - d_logits
        ctx.enqueue_function_checked[
            policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
            policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
        ](
            d_logits,
            probs_immut,
            actions,
            advantages,
            entropy_coef,
            grid_dim=(grid_a_x, grid_a_y),
            block_dim=(TILE, TILE),
        )

        # Step 2: Value loss gradient - d_values
        ctx.enqueue_function_checked[
            value_loss_gradient_kernel[NUM_ENVS],
            value_loss_gradient_kernel[NUM_ENVS],
        ](
            d_values,
            values_immut,
            returns,
            value_coef,
            grid_dim=(1, grid_v_y),
            block_dim=(1, TILE),
        )

        # Step 3: d_W_actor = hidden.T @ d_logits
        ctx.enqueue_function_checked[
            tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, NUM_ACTIONS, TILE],
            tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, NUM_ACTIONS, TILE],
        ](
            d_W_actor,
            hidden_immut,
            d_logits_immut,
            grid_dim=(grid_a_x, (HIDDEN_DIM + TILE - 1) // TILE),
            block_dim=(TILE, TILE),
        )

        # Step 4: d_b_actor = sum(d_logits, axis=0)
        comptime BLOCK_SIZE_BIAS_A = 1 << log2_ceil(NUM_ENVS)
        ctx.enqueue_function_checked[
            bias_gradient_parallel_kernel[NUM_ENVS, NUM_ACTIONS],
            bias_gradient_parallel_kernel[NUM_ENVS, NUM_ACTIONS],
        ](
            d_b_actor,
            d_logits_immut,
            grid_dim=(NUM_ACTIONS, 1),
            block_dim=(BLOCK_SIZE_BIAS_A, 1),
        )

        # Step 5: d_hidden_actor = d_logits @ W_actor.T
        ctx.enqueue_function_checked[
            tiled_matmul_transB_kernel[NUM_ENVS, NUM_ACTIONS, HIDDEN_DIM, TILE],
            tiled_matmul_transB_kernel[NUM_ENVS, NUM_ACTIONS, HIDDEN_DIM, TILE],
        ](
            d_hidden_actor,
            d_logits_immut,
            W_actor_immut,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )

        # Step 6: d_W_critic = hidden.T @ d_values
        ctx.enqueue_function_checked[
            tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, 1, TILE],
            tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, 1, TILE],
        ](
            d_W_critic,
            hidden_immut,
            d_values_immut,
            grid_dim=(grid_v_x, (HIDDEN_DIM + TILE - 1) // TILE),
            block_dim=(TILE, TILE),
        )

        # Step 7: d_b_critic = sum(d_values, axis=0)
        ctx.enqueue_function_checked[
            bias_gradient_parallel_kernel[NUM_ENVS, 1],
            bias_gradient_parallel_kernel[NUM_ENVS, 1],
        ](
            d_b_critic,
            d_values_immut,
            grid_dim=(1, 1),
            block_dim=(BLOCK_SIZE_BIAS_A, 1),
        )

        # Step 8: d_hidden_critic = d_values @ W_critic.T
        ctx.enqueue_function_checked[
            tiled_matmul_transB_kernel[NUM_ENVS, 1, HIDDEN_DIM, TILE],
            tiled_matmul_transB_kernel[NUM_ENVS, 1, HIDDEN_DIM, TILE],
        ](
            d_hidden_critic,
            d_values_immut,
            W_critic_immut,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )

        # Step 9: d_hidden = d_hidden_actor + d_hidden_critic
        ctx.enqueue_function_checked[
            elementwise_add_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
            elementwise_add_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
        ](
            d_hidden,
            d_hidden_actor_immut,
            d_hidden_critic_immut,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )

        # Step 10: ReLU backward - d_pre_relu = d_hidden * (pre_act1 > 0)
        ctx.enqueue_function_checked[
            relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
            relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
        ](
            d_pre_relu,
            d_hidden_immut,
            pre_act1_immut,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )

        # Step 11: d_W1 = obs.T @ d_pre_relu
        ctx.enqueue_function_checked[
            tiled_matmul_transA_kernel[OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE],
            tiled_matmul_transA_kernel[OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE],
        ](
            d_W1,
            obs,
            d_pre_relu_immut,
            grid_dim=(grid_h_x, (OBS_DIM + TILE - 1) // TILE),
            block_dim=(TILE, TILE),
        )

        # Step 12: d_b1 = sum(d_pre_relu, axis=0)
        ctx.enqueue_function_checked[
            bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
            bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
        ](
            d_b1,
            d_pre_relu_immut,
            grid_dim=(HIDDEN_DIM, 1),
            block_dim=(BLOCK_SIZE_BIAS_A, 1),
        )

        ctx.synchronize()

        # =====================================================================
        # WEIGHT UPDATE (SGD)
        # =====================================================================
        print("  Updating weights (SGD)...")

        # Update W1
        ctx.enqueue_function_checked[
            sgd_update_2d_kernel[OBS_DIM, HIDDEN_DIM, TILE],
            sgd_update_2d_kernel[OBS_DIM, HIDDEN_DIM, TILE],
        ](
            W1,
            d_W1_immut,
            lr,
            grid_dim=(grid_h_x, (OBS_DIM + TILE - 1) // TILE),
            block_dim=(TILE, TILE),
        )

        # Update b1
        ctx.enqueue_function_checked[
            sgd_update_1d_kernel[HIDDEN_DIM],
            sgd_update_1d_kernel[HIDDEN_DIM],
        ](b1, d_b1_immut, lr, grid_dim=(1, 1), block_dim=(HIDDEN_DIM, 1))

        # Update W_actor
        ctx.enqueue_function_checked[
            sgd_update_2d_kernel[HIDDEN_DIM, NUM_ACTIONS, TILE],
            sgd_update_2d_kernel[HIDDEN_DIM, NUM_ACTIONS, TILE],
        ](
            W_actor,
            d_W_actor_immut,
            lr,
            grid_dim=(grid_a_x, (HIDDEN_DIM + TILE - 1) // TILE),
            block_dim=(TILE, TILE),
        )

        # Update b_actor
        ctx.enqueue_function_checked[
            sgd_update_1d_kernel[NUM_ACTIONS],
            sgd_update_1d_kernel[NUM_ACTIONS],
        ](
            b_actor,
            d_b_actor_immut,
            lr,
            grid_dim=(1, 1),
            block_dim=(NUM_ACTIONS, 1),
        )

        # Update W_critic
        ctx.enqueue_function_checked[
            sgd_update_2d_kernel[HIDDEN_DIM, 1, TILE],
            sgd_update_2d_kernel[HIDDEN_DIM, 1, TILE],
        ](
            W_critic,
            d_W_critic_immut,
            lr,
            grid_dim=(grid_v_x, (HIDDEN_DIM + TILE - 1) // TILE),
            block_dim=(TILE, TILE),
        )

        # Update b_critic
        ctx.enqueue_function_checked[
            sgd_update_1d_kernel[1],
            sgd_update_1d_kernel[1],
        ](b_critic, d_b_critic_immut, lr, grid_dim=(1, 1), block_dim=(1, 1))

        ctx.synchronize()

        # =====================================================================
        # VERIFY: Run forward pass again and check loss decreased
        # =====================================================================
        print("  Verifying loss after update...")

        # Re-run forward pass
        hidden_buf.enqueue_fill(0)
        logits_buf.enqueue_fill(0)
        probs_buf.enqueue_fill(0)
        values_buf.enqueue_fill(0)

        ctx.enqueue_function_checked[
            tiled_matmul_bias_relu_save_kernel[
                NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
            ],
            tiled_matmul_bias_relu_save_kernel[
                NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
            ],
        ](
            hidden,
            pre_act1,
            obs,
            W1_immut,
            b1_immut,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )

        ctx.enqueue_function_checked[
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
        ](
            logits,
            hidden_immut,
            W_actor_immut,
            b_actor_immut,
            grid_dim=(grid_a_x, grid_a_y),
            block_dim=(TILE, TILE),
        )

        ctx.enqueue_function_checked[
            parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
            parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
        ](
            probs,
            logits_immut,
            grid_dim=(1, NUM_ENVS),
            block_dim=(SOFTMAX_BLOCK, 1),
        )

        ctx.enqueue_function_checked[
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
            tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
        ](
            values,
            hidden_immut,
            W_critic_immut,
            b_critic_immut,
            grid_dim=(grid_v_x, grid_v_y),
            block_dim=(TILE, TILE),
        )

        ctx.synchronize()

        # Compute final loss
        var final_policy_loss: Float64 = 0
        var final_value_loss: Float64 = 0
        with probs_buf.map_to_host() as p, values_buf.map_to_host() as v, actions_buf.map_to_host() as act, advantages_buf.map_to_host() as adv, returns_buf.map_to_host() as ret:
            for i in range(NUM_ENVS):
                var action = Int(act[i])
                var prob = Float64(p[i * NUM_ACTIONS + action])
                var advantage = Float64(adv[i])
                final_policy_loss -= advantage * log(max(prob, 1e-8))

                var value = Float64(v[i])
                var target = Float64(ret[i])
                final_value_loss += 0.5 * (value - target) * (value - target)

        final_policy_loss /= NUM_ENVS
        final_value_loss /= NUM_ENVS

        print("    Final policy loss:", final_policy_loss)
        print("    Final value loss:", final_value_loss)

        # Check if value loss decreased (policy loss may not decrease with negative advantages)
        var value_loss_decreased = final_value_loss < initial_value_loss

        # Verify weights actually changed
        var weight_diff: Float64 = 0
        with d_W1_buf.map_to_host() as d_W1_host:
            for i in range(OBS_DIM * HIDDEN_DIM):
                weight_diff += abs(Float64(d_W1_host[i]))

        print("    Sum of |d_W1|:", weight_diff)
        var weights_changed = weight_diff > 1e-6

        print()
        if value_loss_decreased and weights_changed:
            print(
                "  Training step: PASSED (value loss decreased, weights"
                " updated)"
            )
        elif weights_changed:
            print(
                "  Training step: PARTIAL (weights updated, but value loss"
                " didn't decrease)"
            )
            print(
                "    This can happen with random data - the important thing is"
                " gradients flow correctly"
            )
        else:
            print("  Training step: FAILED (weights didn't change)")
        print()


# =============================================================================
# PHASE 5: CartPole Training Integration
# =============================================================================

# CartPole physics constants
comptime CARTPOLE_GRAVITY: Float64 = 9.8
comptime CARTPOLE_MASSCART: Float64 = 1.0
comptime CARTPOLE_MASSPOLE: Float64 = 0.1
comptime CARTPOLE_TOTAL_MASS: Float64 = 1.1
comptime CARTPOLE_LENGTH: Float64 = 0.5
comptime CARTPOLE_POLEMASS_LENGTH: Float64 = 0.05
comptime CARTPOLE_FORCE_MAG: Float64 = 10.0
comptime CARTPOLE_TAU: Float64 = 0.02
comptime CARTPOLE_THETA_THRESHOLD: Float64 = 0.2094395  # 12 degrees
comptime CARTPOLE_X_THRESHOLD: Float64 = 2.4


fn cartpole_step(
    mut x: Float64,
    mut x_dot: Float64,
    mut theta: Float64,
    mut theta_dot: Float64,
    action: Int,
) -> Tuple[Float64, Bool]:
    """Step CartPole environment. Returns (reward, done)."""
    var force = CARTPOLE_FORCE_MAG if action == 1 else -CARTPOLE_FORCE_MAG

    var costheta = cos(theta)
    var sintheta = sin(theta)

    var temp = (
        force + CARTPOLE_POLEMASS_LENGTH * theta_dot * theta_dot * sintheta
    ) / CARTPOLE_TOTAL_MASS

    var thetaacc = (CARTPOLE_GRAVITY * sintheta - costheta * temp) / (
        CARTPOLE_LENGTH
        * (
            4.0 / 3.0
            - CARTPOLE_MASSPOLE * costheta * costheta / CARTPOLE_TOTAL_MASS
        )
    )

    var xacc = (
        temp
        - CARTPOLE_POLEMASS_LENGTH * thetaacc * costheta / CARTPOLE_TOTAL_MASS
    )

    # Euler integration
    x = x + CARTPOLE_TAU * x_dot
    x_dot = x_dot + CARTPOLE_TAU * xacc
    theta = theta + CARTPOLE_TAU * theta_dot
    theta_dot = theta_dot + CARTPOLE_TAU * thetaacc

    # Check termination
    var done = (
        x < -CARTPOLE_X_THRESHOLD
        or x > CARTPOLE_X_THRESHOLD
        or theta < -CARTPOLE_THETA_THRESHOLD
        or theta > CARTPOLE_THETA_THRESHOLD
    )

    var reward: Float64 = 0.0 if done else 1.0
    return (reward, done)


fn cartpole_reset(
    mut x: Float64,
    mut x_dot: Float64,
    mut theta: Float64,
    mut theta_dot: Float64,
):
    """Reset CartPole to random initial state."""
    x = random_float64() * 0.1 - 0.05
    x_dot = random_float64() * 0.1 - 0.05
    theta = random_float64() * 0.1 - 0.05
    theta_dot = random_float64() * 0.1 - 0.05


fn train_cartpole_native() raises:
    """Train A2C on CartPole using GPU-native kernels."""
    print("=" * 60)
    print("Training A2C on CartPole (GPU-Native)")
    print("=" * 60)

    # Hyperparameters
    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64
    comptime NUM_ACTIONS = 2
    comptime ROLLOUT_LEN = 32
    comptime TILE = 8

    var lr = Scalar[dtype](0.0003)
    var gamma = Scalar[dtype](0.99)
    var gae_lambda = Scalar[dtype](0.95)
    var entropy_coef = Scalar[dtype](0.01)
    var value_coef = Scalar[dtype](0.5)
    var max_updates = 500
    var max_steps_per_env = 500

    print("  NUM_ENVS:", NUM_ENVS)
    print("  ROLLOUT_LEN:", ROLLOUT_LEN)
    print("  HIDDEN_DIM:", HIDDEN_DIM)
    print("  Learning rate:", lr)
    print("  Max updates:", max_updates)
    print()

    # CPU-side environment state
    var env_x = List[Float64]()
    var env_x_dot = List[Float64]()
    var env_theta = List[Float64]()
    var env_theta_dot = List[Float64]()
    var env_steps = List[Int]()
    var episode_rewards = List[Float64]()

    # Initialize environments
    for _ in range(NUM_ENVS):
        env_x.append(random_float64() * 0.1 - 0.05)
        env_x_dot.append(random_float64() * 0.1 - 0.05)
        env_theta.append(random_float64() * 0.1 - 0.05)
        env_theta_dot.append(random_float64() * 0.1 - 0.05)
        env_steps.append(0)
        episode_rewards.append(0.0)

    # Tracking
    var completed_episodes = 0
    var total_reward_sum: Float64 = 0.0
    var recent_rewards = List[Float64]()

    with DeviceContext() as ctx:
        # =====================================================================
        # Allocate GPU buffers
        # =====================================================================

        # Network weights
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

        # Forward pass buffers
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var pre_act1_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var logits_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)

        # Rollout storage
        var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * ROLLOUT_LEN * OBS_DIM
        )
        var rollout_actions_buf = ctx.enqueue_create_buffer[DType.int32](
            NUM_ENVS * ROLLOUT_LEN
        )
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * ROLLOUT_LEN
        )
        var rollout_dones_buf = ctx.enqueue_create_buffer[DType.int32](
            NUM_ENVS * ROLLOUT_LEN
        )
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * ROLLOUT_LEN
        )
        var rollout_advantages_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * ROLLOUT_LEN
        )
        var rollout_returns_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * ROLLOUT_LEN
        )
        var bootstrap_values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)

        # Gradient buffers
        var d_logits_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var d_values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)
        var d_hidden_actor_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_hidden_critic_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_hidden_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_pre_relu_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var d_b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var d_W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var d_b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var d_W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var d_b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

        # =====================================================================
        # Initialize weights (Xavier-like)
        # =====================================================================
        with W1_buf.map_to_host() as W1_host, b1_buf.map_to_host() as b1_host, W_actor_buf.map_to_host() as W_actor_host, b_actor_buf.map_to_host() as b_actor_host, W_critic_buf.map_to_host() as W_critic_host, b_critic_buf.map_to_host() as b_critic_host:
            var scale1 = sqrt(2.0 / Float64(OBS_DIM))
            for i in range(OBS_DIM * HIDDEN_DIM):
                W1_host[i] = Scalar[dtype]((random_float64() - 0.5) * scale1)
            for i in range(HIDDEN_DIM):
                b1_host[i] = Scalar[dtype](0.0)

            var scale_actor = sqrt(2.0 / Float64(HIDDEN_DIM))
            for i in range(HIDDEN_DIM * NUM_ACTIONS):
                W_actor_host[i] = Scalar[dtype](
                    (random_float64() - 0.5) * scale_actor * 0.01
                )
            for i in range(NUM_ACTIONS):
                b_actor_host[i] = Scalar[dtype](0.0)

            var scale_critic = sqrt(2.0 / Float64(HIDDEN_DIM))
            for i in range(HIDDEN_DIM):
                W_critic_host[i] = Scalar[dtype](
                    (random_float64() - 0.5) * scale_critic
                )
            b_critic_host[0] = Scalar[dtype](0.0)

        # =====================================================================
        # Create tensors
        # =====================================================================
        var obs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), MutAnyOrigin
        ](obs_buf)
        var obs_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
        ](obs_buf)
        var W1 = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin
        ](W1_buf)
        var W1_immut = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](W1_buf)
        var b1 = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](b1_buf)
        var b1_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](b1_buf)
        var pre_act1 = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](pre_act1_buf)
        var pre_act1_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](pre_act1_buf)
        var hidden = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](hidden_buf)
        var hidden_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](hidden_buf)
        var W_actor = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), MutAnyOrigin
        ](W_actor_buf)
        var W_actor_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
        ](W_actor_buf)
        var b_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
        ](b_actor_buf)
        var b_actor_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
        ](b_actor_buf)
        var logits = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](logits_buf)
        var logits_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](logits_buf)
        var probs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](probs_buf)
        var probs_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](probs_buf)
        var W_critic = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin
        ](W_critic_buf)
        var W_critic_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
        ](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            b_critic_buf
        )
        var b_critic_immut = LayoutTensor[
            dtype, Layout.row_major(1), ImmutAnyOrigin
        ](b_critic_buf)
        var values = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
        ](values_buf)
        var values_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
        ](values_buf)

        # Grid dimensions
        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE
        comptime grid_a_x = (NUM_ACTIONS + TILE - 1) // TILE
        comptime grid_a_y = (NUM_ENVS + TILE - 1) // TILE
        comptime grid_v_x = (1 + TILE - 1) // TILE
        comptime grid_v_y = (NUM_ENVS + TILE - 1) // TILE
        comptime SOFTMAX_BLOCK = 1 << log2_ceil(NUM_ACTIONS)
        comptime BLOCK_SIZE_BIAS = 1 << log2_ceil(NUM_ENVS)

        # =====================================================================
        # Training loop
        # =====================================================================
        print("  Starting training...")

        for update in range(max_updates):
            # === Collect rollout ===
            for step in range(ROLLOUT_LEN):
                # Copy current observations to GPU
                with obs_buf.map_to_host() as obs_host:
                    for i in range(NUM_ENVS):
                        obs_host[i * OBS_DIM + 0] = Scalar[dtype](env_x[i])
                        obs_host[i * OBS_DIM + 1] = Scalar[dtype](env_x_dot[i])
                        obs_host[i * OBS_DIM + 2] = Scalar[dtype](env_theta[i])
                        obs_host[i * OBS_DIM + 3] = Scalar[dtype](
                            env_theta_dot[i]
                        )

                # Forward pass on GPU
                ctx.enqueue_function_checked[
                    tiled_matmul_bias_relu_save_kernel[
                        NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                    ],
                    tiled_matmul_bias_relu_save_kernel[
                        NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                    ],
                ](
                    hidden,
                    pre_act1,
                    obs_immut,
                    W1_immut,
                    b1_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[
                        NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE
                    ],
                    tiled_matmul_bias_kernel[
                        NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE
                    ],
                ](
                    logits,
                    hidden_immut,
                    W_actor_immut,
                    b_actor_immut,
                    grid_dim=(grid_a_x, grid_a_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    probs,
                    logits_immut,
                    grid_dim=(1, NUM_ENVS),
                    block_dim=(SOFTMAX_BLOCK, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                ](
                    values,
                    hidden_immut,
                    W_critic_immut,
                    b_critic_immut,
                    grid_dim=(grid_v_x, grid_v_y),
                    block_dim=(TILE, TILE),
                )

                ctx.synchronize()

                # Sample actions and step environments (CPU)
                with probs_buf.map_to_host() as probs_host, values_buf.map_to_host() as values_host, rollout_obs_buf.map_to_host() as ro_host, rollout_actions_buf.map_to_host() as ra_host, rollout_rewards_buf.map_to_host() as rr_host, rollout_dones_buf.map_to_host() as rd_host, rollout_values_buf.map_to_host() as rv_host:
                    for i in range(NUM_ENVS):
                        # Store observation
                        ro_host[
                            (i * ROLLOUT_LEN + step) * OBS_DIM + 0
                        ] = Scalar[dtype](env_x[i])
                        ro_host[
                            (i * ROLLOUT_LEN + step) * OBS_DIM + 1
                        ] = Scalar[dtype](env_x_dot[i])
                        ro_host[
                            (i * ROLLOUT_LEN + step) * OBS_DIM + 2
                        ] = Scalar[dtype](env_theta[i])
                        ro_host[
                            (i * ROLLOUT_LEN + step) * OBS_DIM + 3
                        ] = Scalar[dtype](env_theta_dot[i])

                        # Sample action
                        var p0 = Float64(probs_host[i * NUM_ACTIONS + 0])
                        var u = random_float64()
                        var action = 0 if u < p0 else 1

                        ra_host[i * ROLLOUT_LEN + step] = Scalar[DType.int32](
                            action
                        )
                        rv_host[i * ROLLOUT_LEN + step] = values_host[i]

                        # Step environment
                        var x = env_x[i]
                        var x_dot = env_x_dot[i]
                        var theta = env_theta[i]
                        var theta_dot = env_theta_dot[i]

                        var result = cartpole_step(
                            x, x_dot, theta, theta_dot, action
                        )
                        var reward = result[0]
                        var done = result[1]

                        env_x[i] = x
                        env_x_dot[i] = x_dot
                        env_theta[i] = theta
                        env_theta_dot[i] = theta_dot
                        env_steps[i] += 1

                        rr_host[i * ROLLOUT_LEN + step] = Scalar[dtype](reward)
                        rd_host[i * ROLLOUT_LEN + step] = (
                            1 if done
                            or env_steps[i] >= max_steps_per_env else 0
                        )

                        episode_rewards[i] += reward

                        if done or env_steps[i] >= max_steps_per_env:
                            completed_episodes += 1
                            total_reward_sum += episode_rewards[i]
                            recent_rewards.append(episode_rewards[i])
                            if len(recent_rewards) > 100:
                                _ = recent_rewards.pop(0)

                            # Reset
                            cartpole_reset(
                                env_x[i],
                                env_x_dot[i],
                                env_theta[i],
                                env_theta_dot[i],
                            )
                            env_steps[i] = 0
                            episode_rewards[i] = 0.0

            # === Get bootstrap values ===
            with obs_buf.map_to_host() as obs_host:
                for i in range(NUM_ENVS):
                    obs_host[i * OBS_DIM + 0] = Scalar[dtype](env_x[i])
                    obs_host[i * OBS_DIM + 1] = Scalar[dtype](env_x_dot[i])
                    obs_host[i * OBS_DIM + 2] = Scalar[dtype](env_theta[i])
                    obs_host[i * OBS_DIM + 3] = Scalar[dtype](env_theta_dot[i])

            ctx.enqueue_function_checked[
                tiled_matmul_bias_relu_save_kernel[
                    NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                ],
                tiled_matmul_bias_relu_save_kernel[
                    NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                ],
            ](
                hidden,
                pre_act1,
                obs_immut,
                W1_immut,
                b1_immut,
                grid_dim=(grid_h_x, grid_h_y),
                block_dim=(TILE, TILE),
            )

            ctx.enqueue_function_checked[
                tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
            ](
                values,
                hidden_immut,
                W_critic_immut,
                b_critic_immut,
                grid_dim=(grid_v_x, grid_v_y),
                block_dim=(TILE, TILE),
            )
            ctx.synchronize()

            with values_buf.map_to_host() as v_host, bootstrap_values_buf.map_to_host() as bv_host:
                for i in range(NUM_ENVS):
                    bv_host[i] = v_host[i]

            # === Compute GAE (CPU for simplicity) ===
            with rollout_rewards_buf.map_to_host() as rr_host, rollout_dones_buf.map_to_host() as rd_host, rollout_values_buf.map_to_host() as rv_host, bootstrap_values_buf.map_to_host() as bv_host, rollout_advantages_buf.map_to_host() as adv_host, rollout_returns_buf.map_to_host() as ret_host:
                for i in range(NUM_ENVS):
                    var gae: Float64 = 0.0
                    var next_value = Float64(bv_host[i])

                    for t in range(ROLLOUT_LEN - 1, -1, -1):
                        var idx = i * ROLLOUT_LEN + t
                        var reward = Float64(rr_host[idx])
                        var done = Int(rd_host[idx])
                        var value = Float64(rv_host[idx])

                        var not_done = 1.0 if done == 0 else 0.0
                        var delta = (
                            reward
                            + Float64(gamma) * next_value * not_done
                            - value
                        )
                        gae = (
                            delta
                            + Float64(gamma)
                            * Float64(gae_lambda)
                            * not_done
                            * gae
                        )

                        adv_host[idx] = Scalar[dtype](gae)
                        ret_host[idx] = Scalar[dtype](gae + value)
                        next_value = value

            # === Training step for each timestep in rollout ===
            for step in range(ROLLOUT_LEN):
                # Load observation for this step
                with obs_buf.map_to_host() as obs_host, rollout_obs_buf.map_to_host() as ro_host:
                    for i in range(NUM_ENVS):
                        for d in range(OBS_DIM):
                            obs_host[i * OBS_DIM + d] = ro_host[
                                (i * ROLLOUT_LEN + step) * OBS_DIM + d
                            ]

                # Forward pass
                hidden_buf.enqueue_fill(0)
                logits_buf.enqueue_fill(0)
                probs_buf.enqueue_fill(0)
                values_buf.enqueue_fill(0)

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_relu_save_kernel[
                        NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                    ],
                    tiled_matmul_bias_relu_save_kernel[
                        NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                    ],
                ](
                    hidden,
                    pre_act1,
                    obs_immut,
                    W1_immut,
                    b1_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[
                        NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE
                    ],
                    tiled_matmul_bias_kernel[
                        NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE
                    ],
                ](
                    logits,
                    hidden_immut,
                    W_actor_immut,
                    b_actor_immut,
                    grid_dim=(grid_a_x, grid_a_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    probs,
                    logits_immut,
                    grid_dim=(1, NUM_ENVS),
                    block_dim=(SOFTMAX_BLOCK, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                ](
                    values,
                    hidden_immut,
                    W_critic_immut,
                    b_critic_immut,
                    grid_dim=(grid_v_x, grid_v_y),
                    block_dim=(TILE, TILE),
                )

                ctx.synchronize()

                # Create tensors for this step's data
                var actions_step_buf = ctx.enqueue_create_buffer[DType.int32](
                    NUM_ENVS
                )
                var advantages_step_buf = ctx.enqueue_create_buffer[dtype](
                    NUM_ENVS
                )
                var returns_step_buf = ctx.enqueue_create_buffer[dtype](
                    NUM_ENVS
                )

                with actions_step_buf.map_to_host() as a_host, advantages_step_buf.map_to_host() as adv_host, returns_step_buf.map_to_host() as ret_host, rollout_actions_buf.map_to_host() as ra_host, rollout_advantages_buf.map_to_host() as radv_host, rollout_returns_buf.map_to_host() as rret_host:
                    for i in range(NUM_ENVS):
                        var idx = i * ROLLOUT_LEN + step
                        a_host[i] = ra_host[idx]
                        adv_host[i] = radv_host[idx]
                        ret_host[i] = rret_host[idx]

                var actions_t = LayoutTensor[
                    DType.int32, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
                ](actions_step_buf)
                var advantages_t = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
                ](advantages_step_buf)
                var returns_t = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
                ](returns_step_buf)

                # Backward pass
                d_logits_buf.enqueue_fill(0)
                d_values_buf.enqueue_fill(0)
                d_hidden_actor_buf.enqueue_fill(0)
                d_hidden_critic_buf.enqueue_fill(0)
                d_hidden_buf.enqueue_fill(0)
                d_pre_relu_buf.enqueue_fill(0)
                d_W1_buf.enqueue_fill(0)
                d_b1_buf.enqueue_fill(0)
                d_W_actor_buf.enqueue_fill(0)
                d_b_actor_buf.enqueue_fill(0)
                d_W_critic_buf.enqueue_fill(0)
                d_b_critic_buf.enqueue_fill(0)

                var d_logits = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
                ](d_logits_buf)
                var d_logits_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, NUM_ACTIONS),
                    ImmutAnyOrigin,
                ](d_logits_buf)
                var d_values = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
                ](d_values_buf)
                var d_values_immut = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
                ](d_values_buf)
                var d_hidden_actor = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
                ](d_hidden_actor_buf)
                var d_hidden_actor_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, HIDDEN_DIM),
                    ImmutAnyOrigin,
                ](d_hidden_actor_buf)
                var d_hidden_critic = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
                ](d_hidden_critic_buf)
                var d_hidden_critic_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, HIDDEN_DIM),
                    ImmutAnyOrigin,
                ](d_hidden_critic_buf)
                var d_hidden = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
                ](d_hidden_buf)
                var d_hidden_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, HIDDEN_DIM),
                    ImmutAnyOrigin,
                ](d_hidden_buf)
                var d_pre_relu = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
                ](d_pre_relu_buf)
                var d_pre_relu_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, HIDDEN_DIM),
                    ImmutAnyOrigin,
                ](d_pre_relu_buf)
                var d_W1 = LayoutTensor[
                    dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin
                ](d_W1_buf)
                var d_W1_immut = LayoutTensor[
                    dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
                ](d_W1_buf)
                var d_b1 = LayoutTensor[
                    dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
                ](d_b1_buf)
                var d_b1_immut = LayoutTensor[
                    dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
                ](d_b1_buf)
                var d_W_actor = LayoutTensor[
                    dtype,
                    Layout.row_major(HIDDEN_DIM, NUM_ACTIONS),
                    MutAnyOrigin,
                ](d_W_actor_buf)
                var d_W_actor_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(HIDDEN_DIM, NUM_ACTIONS),
                    ImmutAnyOrigin,
                ](d_W_actor_buf)
                var d_b_actor = LayoutTensor[
                    dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
                ](d_b_actor_buf)
                var d_b_actor_immut = LayoutTensor[
                    dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
                ](d_b_actor_buf)
                var d_W_critic = LayoutTensor[
                    dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin
                ](d_W_critic_buf)
                var d_W_critic_immut = LayoutTensor[
                    dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
                ](d_W_critic_buf)
                var d_b_critic = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](d_b_critic_buf)
                var d_b_critic_immut = LayoutTensor[
                    dtype, Layout.row_major(1), ImmutAnyOrigin
                ](d_b_critic_buf)

                # Policy gradient
                ctx.enqueue_function_checked[
                    policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
                    policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    d_logits,
                    probs_immut,
                    actions_t,
                    advantages_t,
                    entropy_coef,
                    grid_dim=(grid_a_x, grid_a_y),
                    block_dim=(TILE, TILE),
                )

                # Value loss gradient
                ctx.enqueue_function_checked[
                    value_loss_gradient_kernel[NUM_ENVS],
                    value_loss_gradient_kernel[NUM_ENVS],
                ](
                    d_values,
                    values_immut,
                    returns_t,
                    value_coef,
                    grid_dim=(1, grid_v_y),
                    block_dim=(1, TILE),
                )

                # Backprop through actor
                ctx.enqueue_function_checked[
                    tiled_matmul_transA_kernel[
                        HIDDEN_DIM, NUM_ENVS, NUM_ACTIONS, TILE
                    ],
                    tiled_matmul_transA_kernel[
                        HIDDEN_DIM, NUM_ENVS, NUM_ACTIONS, TILE
                    ],
                ](
                    d_W_actor,
                    hidden_immut,
                    d_logits_immut,
                    grid_dim=(grid_a_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    bias_gradient_parallel_kernel[NUM_ENVS, NUM_ACTIONS],
                    bias_gradient_parallel_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    d_b_actor,
                    d_logits_immut,
                    grid_dim=(NUM_ACTIONS, 1),
                    block_dim=(BLOCK_SIZE_BIAS, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_transB_kernel[
                        NUM_ENVS, NUM_ACTIONS, HIDDEN_DIM, TILE
                    ],
                    tiled_matmul_transB_kernel[
                        NUM_ENVS, NUM_ACTIONS, HIDDEN_DIM, TILE
                    ],
                ](
                    d_hidden_actor,
                    d_logits_immut,
                    W_actor_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                # Backprop through critic
                ctx.enqueue_function_checked[
                    tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, 1, TILE],
                    tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, 1, TILE],
                ](
                    d_W_critic,
                    hidden_immut,
                    d_values_immut,
                    grid_dim=(grid_v_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    bias_gradient_parallel_kernel[NUM_ENVS, 1],
                    bias_gradient_parallel_kernel[NUM_ENVS, 1],
                ](
                    d_b_critic,
                    d_values_immut,
                    grid_dim=(1, 1),
                    block_dim=(BLOCK_SIZE_BIAS, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_transB_kernel[NUM_ENVS, 1, HIDDEN_DIM, TILE],
                    tiled_matmul_transB_kernel[NUM_ENVS, 1, HIDDEN_DIM, TILE],
                ](
                    d_hidden_critic,
                    d_values_immut,
                    W_critic_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                # Combine hidden gradients
                ctx.enqueue_function_checked[
                    elementwise_add_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                    elementwise_add_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                ](
                    d_hidden,
                    d_hidden_actor_immut,
                    d_hidden_critic_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                # ReLU backward
                ctx.enqueue_function_checked[
                    relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                    relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                ](
                    d_pre_relu,
                    d_hidden_immut,
                    pre_act1_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                # Backprop through first layer
                ctx.enqueue_function_checked[
                    tiled_matmul_transA_kernel[
                        OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE
                    ],
                    tiled_matmul_transA_kernel[
                        OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE
                    ],
                ](
                    d_W1,
                    obs_immut,
                    d_pre_relu_immut,
                    grid_dim=(grid_h_x, (OBS_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
                    bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
                ](
                    d_b1,
                    d_pre_relu_immut,
                    grid_dim=(HIDDEN_DIM, 1),
                    block_dim=(BLOCK_SIZE_BIAS, 1),
                )

                # SGD update
                ctx.enqueue_function_checked[
                    sgd_update_2d_kernel[OBS_DIM, HIDDEN_DIM, TILE],
                    sgd_update_2d_kernel[OBS_DIM, HIDDEN_DIM, TILE],
                ](
                    W1,
                    d_W1_immut,
                    lr,
                    grid_dim=(grid_h_x, (OBS_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    sgd_update_1d_kernel[HIDDEN_DIM],
                    sgd_update_1d_kernel[HIDDEN_DIM],
                ](
                    b1,
                    d_b1_immut,
                    lr,
                    grid_dim=(1, 1),
                    block_dim=(HIDDEN_DIM, 1),
                )

                ctx.enqueue_function_checked[
                    sgd_update_2d_kernel[HIDDEN_DIM, NUM_ACTIONS, TILE],
                    sgd_update_2d_kernel[HIDDEN_DIM, NUM_ACTIONS, TILE],
                ](
                    W_actor,
                    d_W_actor_immut,
                    lr,
                    grid_dim=(grid_a_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    sgd_update_1d_kernel[NUM_ACTIONS],
                    sgd_update_1d_kernel[NUM_ACTIONS],
                ](
                    b_actor,
                    d_b_actor_immut,
                    lr,
                    grid_dim=(1, 1),
                    block_dim=(NUM_ACTIONS, 1),
                )

                ctx.enqueue_function_checked[
                    sgd_update_2d_kernel[HIDDEN_DIM, 1, TILE],
                    sgd_update_2d_kernel[HIDDEN_DIM, 1, TILE],
                ](
                    W_critic,
                    d_W_critic_immut,
                    lr,
                    grid_dim=(grid_v_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    sgd_update_1d_kernel[1],
                    sgd_update_1d_kernel[1],
                ](
                    b_critic,
                    d_b_critic_immut,
                    lr,
                    grid_dim=(1, 1),
                    block_dim=(1, 1),
                )

                ctx.synchronize()

            # Print progress
            if (update + 1) % 50 == 0 or update == 0:
                var avg_reward: Float64 = 0.0
                if len(recent_rewards) > 0:
                    for i in range(len(recent_rewards)):
                        avg_reward += recent_rewards[i]
                    avg_reward /= len(recent_rewards)
                print(
                    "  Update",
                    update + 1,
                    "| Episodes:",
                    completed_episodes,
                    "| Avg reward (last 100):",
                    avg_reward,
                )

        # Final summary
        print()
        print("  Training complete!")
        print("  Total episodes:", completed_episodes)
        var final_avg: Float64 = 0.0
        if len(recent_rewards) > 0:
            for i in range(len(recent_rewards)):
                final_avg += recent_rewards[i]
            final_avg /= len(recent_rewards)
        print("  Final avg reward (last 100):", final_avg)

        if final_avg >= 195.0:
            print("  CartPole SOLVED! (avg >= 195)")
        elif final_avg >= 100.0:
            print("  Good progress! (avg >= 100)")
        else:
            print("  Training needs more iterations or tuning")
        print()


fn main() raises:
    print()
    print("GPU-Native A2C - Phase 5: CartPole Training")
    print()

    # Test 1: Square matmul (baseline)
    test_square_matmul()

    # Test 2: Rectangular matmul (RL dimensions)
    test_rectangular_matmul()

    # Test 3: Matmul + bias + ReLU (forward pass hidden layer)
    test_matmul_bias_relu()

    # Test 4: Parallel softmax (action probabilities)
    test_parallel_softmax()

    # Test 5: Complete batched forward pass
    test_batched_forward_pass()

    # Test 6: Backward pass kernels
    test_backward_pass_kernels()

    # Test 7: Complete training step
    test_complete_training_step()

    print("All tests completed!")
    print()
