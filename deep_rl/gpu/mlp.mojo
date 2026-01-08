"""GPU Multi-Layer Perceptron for deep RL.

This module provides GPU-accelerated MLP operations using
tiled matrix multiplication and block primitives.

Operations:
- linear_forward_tanh_kernel: y = tanh(x @ W + b) - fused for hidden layers
- tanh_grad_kernel: grad = 1 - tanh(x)^2 (given tanh output)
- elementwise_mul_kernel: output = a * b (for gradient chaining)

MLP Architectures:
- MLP2: input -> hidden (tanh) -> output
- MLP3: input -> hidden1 (tanh) -> hidden2 (tanh) -> output
"""

from gpu import thread_idx, block_idx, block_dim, barrier, block
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import tanh as math_tanh


# ============================================================================
# Fused Linear + Tanh Kernel (for hidden layers)
# ============================================================================

fn linear_forward_tanh_kernel[
    dtype: DType,
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward pass with tanh: y = tanh(x @ W + b).

    Used for hidden layers in MLP.
    """
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    x_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    if global_col < OUT_DIM:
        acc = rebind[Scalar[dtype]](b[global_col])

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dtype]](x_shared[local_row, k]) * rebind[Scalar[dtype]](W_shared[k, local_col])

        barrier()

    # Apply tanh activation
    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = math_tanh(acc)


fn linear_forward_kernel[
    dtype: DType,
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Forward pass without activation: y = x @ W + b.

    Used for output layers in MLP.
    """
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    x_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    if global_col < OUT_DIM:
        acc = rebind[Scalar[dtype]](b[global_col])

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dtype]](x_shared[local_row, k]) * rebind[Scalar[dtype]](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc


# ============================================================================
# Backward Pass Kernels
# ============================================================================

fn linear_backward_dW_kernel[
    dtype: DType,
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Weight gradient: dW = x.T @ dy."""
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    x_T_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    dy_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    comptime num_tiles = (BATCH + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        batch_idx = tile_idx * TILE + local_col
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = x[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        dy_row = tile_idx * TILE + local_row
        if dy_row < BATCH and global_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[dy_row, global_col]
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dtype]](x_T_shared[local_row, k]) * rebind[Scalar[dtype]](dy_shared[k, local_col])

        barrier()

    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = acc


fn linear_backward_db_kernel[
    dtype: DType,
    BATCH: Int,
    OUT_DIM: Int,
    TPB: Int,
](
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Bias gradient: db = sum(dy, axis=0)."""
    col = Int(block_idx.x)
    local_i = thread_idx.x

    if col >= OUT_DIM:
        return

    var my_value: Scalar[dtype] = 0
    batch_idx = Int(local_i)
    if batch_idx < BATCH:
        my_value = rebind[Scalar[dtype]](dy[batch_idx, col])

    total = block.sum[block_size=TPB, broadcast=False](
        val=SIMD[dtype, 1](my_value)
    )

    if local_i == 0:
        db[col] = total[0]


fn linear_backward_dx_kernel[
    dtype: DType,
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int,
](
    dx: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
):
    """Input gradient: dx = dy @ W.T."""
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    dy_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_T_shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0
    comptime num_tiles = (OUT_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        dy_col = tile_idx * TILE + local_col
        if global_row < BATCH and dy_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[global_row, dy_col]
        else:
            dy_shared[local_row, local_col] = 0

        W_col = tile_idx * TILE + local_row
        if W_col < OUT_DIM and global_col < IN_DIM:
            W_T_shared[local_row, local_col] = W[global_col, W_col]
        else:
            W_T_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dtype]](dy_shared[local_row, k]) * rebind[Scalar[dtype]](W_T_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < IN_DIM:
        dx[global_row, global_col] = acc


# ============================================================================
# Activation Gradient Kernels
# ============================================================================

fn tanh_grad_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    tanh_output: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Compute tanh gradient given tanh OUTPUT: grad = 1 - y^2 where y = tanh(x)."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        y = rebind[Scalar[dtype]](tanh_output[global_i])
        output[global_i] = 1 - y * y


fn elementwise_mul_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Element-wise multiplication: output = a * b."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        output[global_i] = rebind[Scalar[dtype]](a[global_i]) * rebind[Scalar[dtype]](b[global_i])


# ============================================================================
# Soft Update Kernel (for target networks)
# ============================================================================

fn soft_update_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    target: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    source: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    tau: Scalar[dtype],
):
    """Soft update: target = tau * source + (1 - tau) * target."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        src_val = rebind[Scalar[dtype]](source[global_i])
        tgt_val = rebind[Scalar[dtype]](target[global_i])
        target[global_i] = tau * src_val + (1 - tau) * tgt_val


# ============================================================================
# Adam Update Kernel
# ============================================================================

fn adam_update_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    m: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    v: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    bias_correction1: Scalar[dtype],
    bias_correction2: Scalar[dtype],
):
    """Adam optimizer update kernel."""
    from math import sqrt

    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        g = rebind[Scalar[dtype]](grads[global_i])
        m_val = rebind[Scalar[dtype]](m[global_i])
        v_val = rebind[Scalar[dtype]](v[global_i])

        m_new = beta1 * m_val + (1 - beta1) * g
        v_new = beta2 * v_val + (1 - beta2) * g * g

        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2

        params[global_i] = rebind[Scalar[dtype]](params[global_i]) - lr * m_hat / (sqrt(v_hat) + eps)

        m[global_i] = m_new
        v[global_i] = v_new
