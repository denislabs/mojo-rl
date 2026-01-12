"""Neural Network Module System for GPU.

This module provides GPU-powered neural network operations using the same patterns
as test_mlp.mojo. It provides reusable kernels and helper functions.

Usage:
    from deep_rl.nn_gpu import (
        linear_forward, linear_backward_dx, linear_backward_dW, linear_backward_db,
        relu_forward, relu_backward, adam_step, gpu_mse_loss_backward
    )

    with DeviceContext() as ctx:
        # Create buffers
        var W_buf = ctx.enqueue_create_buffer[dtype](IN_DIM * OUT_DIM)
        # ... use the functions with buffers
"""

from math import sqrt, exp

from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace

# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32
comptime TILE = 16  # Tile size for matmul kernels
comptime TPB = 256  # Threads per block for elementwise ops


# =============================================================================
# GPU Kernels - Reusable Building Blocks
# =============================================================================


fn generic_matmul_kernel[
    dtype: DType,
    M: Int,  # Output rows
    K: Int,  # Inner dimension
    N: Int,  # Output cols
    TILE: Int,
    TRANSPOSE_A: Bool,
    TRANSPOSE_B: Bool,
    HAS_BIAS: Bool,
    ACTIVATION: StringLiteral,  # "none", "relu", "dual_relu", "tanh"
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    output2: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    A: LayoutTensor[
        dtype,
        Layout.row_major(M, K) if not TRANSPOSE_A else Layout.row_major(K, M),
        ImmutAnyOrigin,
    ],
    B: LayoutTensor[
        dtype,
        Layout.row_major(K, N) if not TRANSPOSE_B else Layout.row_major(N, K),
        ImmutAnyOrigin,
    ],
    bias: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    """Generic tiled matmul with compile-time options.

    Supports:
    - Forward:     C = A @ B + bias  (TRANSPOSE_A=False, TRANSPOSE_B=False)
    - Backward dW: C = A.T @ B       (TRANSPOSE_A=True,  TRANSPOSE_B=False)
    - Backward dx: C = A @ B.T       (TRANSPOSE_A=False, TRANSPOSE_B=True)
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0

    @parameter
    if HAS_BIAS:
        if global_col < N:
            acc = bias[global_col]

    comptime num_tiles = (K + TILE - 1) // TILE

    for tile_idx in range(num_tiles):

        @parameter
        if TRANSPOSE_A:
            var k_idx = tile_idx * TILE + local_col
            if global_row < M and k_idx < K:
                A_shared[local_row, local_col] = A[k_idx, global_row]
            else:
                A_shared[local_row, local_col] = 0
        else:
            var k_idx = tile_idx * TILE + local_col
            if global_row < M and k_idx < K:
                A_shared[local_row, local_col] = A[global_row, k_idx]
            else:
                A_shared[local_row, local_col] = 0

        @parameter
        if TRANSPOSE_B:
            var k_idx = tile_idx * TILE + local_row
            if k_idx < K and global_col < N:
                B_shared[local_row, local_col] = B[global_col, k_idx]
            else:
                B_shared[local_row, local_col] = 0
        else:
            var k_idx = tile_idx * TILE + local_row
            if k_idx < K and global_col < N:
                B_shared[local_row, local_col] = B[k_idx, global_col]
            else:
                B_shared[local_row, local_col] = 0

        barrier()

        # Compute with 4x unrolling for better ILP
        @parameter
        for k_base in range(0, TILE, 4):
            acc += (
                A_shared[local_row, k_base + 0]
                * B_shared[k_base + 0, local_col]
            )
            acc += (
                A_shared[local_row, k_base + 1]
                * B_shared[k_base + 1, local_col]
            )
            acc += (
                A_shared[local_row, k_base + 2]
                * B_shared[k_base + 2, local_col]
            )
            acc += (
                A_shared[local_row, k_base + 3]
                * B_shared[k_base + 3, local_col]
            )

        barrier()

    if global_row < M and global_col < N:

        @parameter
        if ACTIVATION == "relu":
            output[global_row, global_col] = max(acc, 0)
        elif ACTIVATION == "dual_relu":
            output[global_row, global_col] = acc  # pre-ReLU
            output2[global_row, global_col] = max(acc, 0)  # post-ReLU
        else:  # "none"
            output[global_row, global_col] = acc


# =============================================================================
# Linear Forward Kernels
# =============================================================================


fn linear_forward_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Forward pass: y = x @ W + b (no activation)."""
    generic_matmul_kernel[
        dtype, BATCH, IN_DIM, OUT_DIM, TILE, False, False, True, "none"
    ](output, output, x, W, b)


fn linear_forward_relu_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward: y = max(0, x @ W + b)."""
    generic_matmul_kernel[
        dtype, BATCH, IN_DIM, OUT_DIM, TILE, False, False, True, "relu"
    ](output, output, x, W, b)


fn linear_forward_relu_dual_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    output_pre: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    output_relu: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward: outputs both pre-ReLU and post-ReLU in ONE matmul."""
    generic_matmul_kernel[
        dtype, BATCH, IN_DIM, OUT_DIM, TILE, False, False, True, "dual_relu"
    ](output_pre, output_relu, x, W, b)


# =============================================================================
# ReLU Kernels (standalone)
# =============================================================================


fn relu_forward_kernel[
    BATCH: Int,
    DIM: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    """ReLU forward: y = max(0, x), caches input for backward."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return

    var row = idx // DIM
    var col = idx % DIM
    var val = input[row, col]
    cache[row, col] = val
    output[row, col] = max(val, 0)


fn relu_backward_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin
    ],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin],
):
    """ReLU backward: dx = dy * (x > 0)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return

    var row = idx // DIM
    var col = idx % DIM
    var x_val = cache[row, col]
    var dy_val = grad_output[row, col]
    grad_input[row, col] = dy_val if x_val > 0 else 0


# =============================================================================
# Tanh Kernels (standalone)
# =============================================================================


fn tanh_forward_kernel[
    BATCH: Int,
    DIM: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    """Tanh forward: y = tanh(x), caches output for backward."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return

    var row = idx // DIM
    var col = idx % DIM
    var val = input[row, col]
    var val_f32 = rebind[Scalar[DType.float32]](val)
    var exp_val = exp(val_f32)
    var exp_neg_val = exp(-val_f32)
    var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
    var result = rebind[output.element_type](tanh_val)
    cache[row, col] = result
    output[row, col] = result


fn tanh_backward_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin
    ],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin],
):
    """Tanh backward: dx = dy * (1 - tanh(x)^2)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return

    var row = idx // DIM
    var col = idx % DIM
    var t = cache[row, col]
    var dy = grad_output[row, col]
    grad_input[row, col] = dy * (1 - t * t)


# =============================================================================
# Linear Backward Kernels
# =============================================================================


fn linear_backward_dx_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dx: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
):
    """Input gradient: dx = dy @ W.T."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var W_T_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: dx.element_type = 0
    comptime num_tiles = (OUT_DIM + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        var dy_col = tile_idx * TILE + local_col
        if global_row < BATCH and dy_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[global_row, dy_col]
        else:
            dy_shared[local_row, local_col] = 0

        var W_col = tile_idx * TILE + local_row
        if W_col < OUT_DIM and global_col < IN_DIM:
            W_T_shared[local_row, local_col] = W[global_col, W_col]
        else:
            W_T_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += dy_shared[local_row, k] * W_T_shared[k, local_col]

        barrier()

    if global_row < BATCH and global_col < IN_DIM:
        dx[global_row, global_col] = acc


fn linear_backward_dx_relu_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dx: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    pre_activation: LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin
    ],
):
    """Fused: dx = (dy @ W.T) * (pre_activation > 0)."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var W_T_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: dx.element_type = 0
    comptime num_tiles = (OUT_DIM + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        var dy_col = tile_idx * TILE + local_col
        if global_row < BATCH and dy_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[global_row, dy_col]
        else:
            dy_shared[local_row, local_col] = 0

        var W_col = tile_idx * TILE + local_row
        if W_col < OUT_DIM and global_col < IN_DIM:
            W_T_shared[local_row, local_col] = W[global_col, W_col]
        else:
            W_T_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k_base in range(0, TILE, 4):
            acc += (
                dy_shared[local_row, k_base + 0]
                * W_T_shared[k_base + 0, local_col]
            )
            acc += (
                dy_shared[local_row, k_base + 1]
                * W_T_shared[k_base + 1, local_col]
            )
            acc += (
                dy_shared[local_row, k_base + 2]
                * W_T_shared[k_base + 2, local_col]
            )
            acc += (
                dy_shared[local_row, k_base + 3]
                * W_T_shared[k_base + 3, local_col]
            )

        barrier()

    if global_row < BATCH and global_col < IN_DIM:
        var pre_act = pre_activation[global_row, global_col]
        dx[global_row, global_col] = acc if pre_act > 0 else 0


fn linear_backward_dW_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Weight gradient: dW = x.T @ dy."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var x_T_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: dW.element_type = 0
    comptime num_tiles = (BATCH + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        var batch_idx = tile_idx * TILE + local_col
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = x[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        var dy_row = tile_idx * TILE + local_row
        if dy_row < BATCH and global_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[dy_row, global_col]
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += x_T_shared[local_row, k] * dy_shared[k, local_col]

        barrier()

    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = acc


fn linear_backward_db_kernel[
    BATCH: Int,
    OUT_DIM: Int,
](
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Bias gradient: db = sum(dy, axis=0)."""
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    if col >= OUT_DIM:
        return

    var acc: db.element_type = 0
    for batch in range(BATCH):
        acc += dy[batch, col]
    db[col] = acc


# =============================================================================
# Fused dW + db Kernel
# =============================================================================


fn linear_backward_dW_db_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Fused backward: dW = x.T @ dy, db = sum(dy, axis=0)."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var x_T_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var dW_acc: dW.element_type = 0
    var db_acc: dW.element_type = 0

    comptime num_tiles = (BATCH + TILE - 1) // TILE
    var compute_db = block_idx.y == 0

    for tile_idx in range(num_tiles):
        var batch_idx = tile_idx * TILE + local_col
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = x[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        var dy_row = tile_idx * TILE + local_row
        var dy_val: dy_shared.element_type = 0
        if dy_row < BATCH and global_col < OUT_DIM:
            dy_val = dy[dy_row, global_col]
            dy_shared[local_row, local_col] = dy_val
            if compute_db:
                db_acc += dy_val
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k_base in range(0, TILE, 4):
            dW_acc += (
                x_T_shared[local_row, k_base + 0]
                * dy_shared[k_base + 0, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 1]
                * dy_shared[k_base + 1, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 2]
                * dy_shared[k_base + 2, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 3]
                * dy_shared[k_base + 3, local_col]
            )

        barrier()

    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = dW_acc

    if compute_db:
        dy_shared[local_row, local_col] = db_acc
        barrier()

        var stride = TILE // 2
        while stride > 0:
            if local_row < stride:
                dy_shared[local_row, local_col] += dy_shared[
                    local_row + stride, local_col
                ]
            barrier()
            stride //= 2

        if local_row == 0 and global_col < OUT_DIM:
            db[global_col] = dy_shared[0, local_col]


# =============================================================================
# Xavier Initialization Kernel
# =============================================================================


fn xavier_init_kernel[
    SIZE: Int,
    FAN_IN: Int,
    FAN_OUT: Int,
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    rng_seed: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
):
    """Xavier/Glorot initialization on GPU."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return

    var state = rebind[Scalar[DType.uint32]](rng_seed[0]) + UInt32(
        idx * 1099087573
    )
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5

    var u = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )
    var scale = sqrt(Scalar[dtype](6.0) / Scalar[dtype](FAN_IN + FAN_OUT))
    weights[idx] = u * scale


# =============================================================================
# Zero Buffer Kernel
# =============================================================================


fn zero_buffer_kernel[
    SIZE: Int
](buf: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],):
    """Zero out a buffer."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    buf[idx] = 0


# =============================================================================
# Adam Optimizer Kernel
# =============================================================================


fn adam_update_kernel[
    SIZE: Int
](
    params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    m: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    v: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    bc1: Scalar[dtype],
    bc2: Scalar[dtype],
):
    """Adam optimizer update kernel."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return

    var g = grads[idx]
    var m_val = m[idx]
    var v_val = v[idx]

    var m_new = beta1 * m_val + (1 - beta1) * g
    var v_new = beta2 * v_val + (1 - beta2) * g * g

    var m_hat = m_new / bc1
    var v_hat = v_new / bc2

    params[idx] = params[idx] - lr * m_hat / (sqrt(v_hat) + eps)

    m[idx] = m_new
    v[idx] = v_new


# =============================================================================
# Data Generation Kernel
# =============================================================================


fn generate_data_kernel[
    BATCH: Int,
    INPUT_DIM: Int,
    OUTPUT_DIM: Int,
](
    x_data: LayoutTensor[
        dtype, Layout.row_major(BATCH, INPUT_DIM), MutAnyOrigin
    ],
    y_data: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUTPUT_DIM), MutAnyOrigin
    ],
    rng_seeds: LayoutTensor[
        DType.uint32, Layout.row_major(BATCH), MutAnyOrigin
    ],
):
    """Generate XOR-like data on GPU: y = x1 * x2."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH:
        return

    var state = rebind[Scalar[DType.uint32]](rng_seeds[idx])

    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    var x1 = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )

    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    var x2 = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )

    x_data[idx, 0] = x1
    x_data[idx, 1] = x2
    y_data[idx, 0] = x1 * x2

    rng_seeds[idx] = state


# =============================================================================
# Copy Kernel
# =============================================================================


fn copy_kernel[
    SIZE: Int
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Copy src to dst."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    dst[idx] = src[idx]
