"""GPU Linear Layer for deep RL.

This module provides GPU-accelerated linear layer operations using
tiled matrix multiplication (P16 pattern) and block primitives (P27 pattern).

Operations:
- Forward: y = x @ W + b (fused matmul + bias add)
- Backward: dW = x.T @ dy, db = sum(dy), dx = dy @ W.T
- Adam update: elementwise Adam optimizer step
- Soft update: target network update

Uses:
- Tiled matmul for forward/backward (P16 pattern)
- Block.sum() for bias gradient reduction (P27 pattern)
- Elementwise for Adam and soft update (P23 pattern)
"""

from gpu import thread_idx, block_idx, block_dim, barrier, block
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import sqrt


# ============================================================================
# Forward Pass: y = x @ W + b
# ============================================================================


fn linear_forward_kernel[
    dtype: DType,
    BATCH: Int,  # Batch size (M)
    IN_DIM: Int,  # Input features (K)
    OUT_DIM: Int,  # Output features (N)
    TILE: Int,  # Tile size
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Forward pass kernel: y = x @ W + b using tiled matmul.

    x: (BATCH, IN_DIM), W: (IN_DIM, OUT_DIM), b: (OUT_DIM,) -> y: (BATCH, OUT_DIM)
    """
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    # Shared memory for tiles
    x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Start with bias
    var acc: output.element_type = 0
    if global_col < OUT_DIM:
        acc = b[global_col]

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        # Load x tile
        x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        # Load W tile
        W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        # Compute partial dot product
        @parameter
        for k in range(TILE):
            acc += x_shared[local_row, k] * W_shared[k, local_col]

        barrier()

    # Write result
    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc


fn linear_forward_relu_kernel[
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
    """Fused forward pass with ReLU: y = max(0, x @ W + b)."""
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0
    if global_col < OUT_DIM:
        acc = b[global_col]

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
            acc += x_shared[local_row, k] * W_shared[k, local_col]

        barrier()

    # Write result with ReLU
    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = max(acc, 0)


# ============================================================================
# Backward Pass: Compute gradients
# ============================================================================


fn linear_backward_dW_kernel[
    dtype: DType,
    BATCH: Int,  # M
    IN_DIM: Int,  # K (rows of dW)
    OUT_DIM: Int,  # N (cols of dW)
    TILE: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Compute weight gradient: dW = x.T @ dy.

    x: (BATCH, IN_DIM), dy: (BATCH, OUT_DIM) -> dW: (IN_DIM, OUT_DIM)
    This is matmul of x.T (IN_DIM, BATCH) @ dy (BATCH, OUT_DIM).
    """
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row  # IN_DIM dimension
    global_col = Int(block_idx.x) * TILE + local_col  # OUT_DIM dimension

    # Shared memory for tiles
    # x_T_shared stores a tile of x.T, which is column-major access of x
    x_T_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: dW.element_type = 0
    comptime num_tiles = (BATCH + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        # Load x.T tile: x_T[global_row, tile_idx*TILE + local_col] = x[tile_idx*TILE + local_col, global_row]
        batch_idx = tile_idx * TILE + local_col
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = x[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        # Load dy tile
        dy_row = tile_idx * TILE + local_row
        if dy_row < BATCH and global_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[dy_row, global_col]
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += x_T_shared[local_row, k] * dy_shared[k, local_col]

        barrier()

    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = acc


fn linear_backward_db_kernel[
    dtype: DType,
    BATCH: Int,
    OUT_DIM: Int,
    TPB: Int,  # Threads per block
](
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Compute bias gradient: db = sum(dy, axis=0).

    dy: (BATCH, OUT_DIM) -> db: (OUT_DIM,)
    Each thread block handles one output dimension.
    """
    col = Int(block_idx.x)
    local_i = thread_idx.x

    if col >= OUT_DIM:
        return

    # Each thread loads one element from its batch position
    var my_value: dy.element_type = 0
    batch_idx = Int(local_i)
    if batch_idx < BATCH:
        my_value = dy[batch_idx, col]

    # Sum across threads using block.sum
    total = block.sum[block_size=TPB, broadcast=False](val=my_value)

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
    """Compute input gradient: dx = dy @ W.T.

    dy: (BATCH, OUT_DIM), W: (IN_DIM, OUT_DIM) -> dx: (BATCH, IN_DIM)
    This is matmul of dy (BATCH, OUT_DIM) @ W.T (OUT_DIM, IN_DIM).
    """
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row  # BATCH dimension
    global_col = Int(block_idx.x) * TILE + local_col  # IN_DIM dimension

    dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_T_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: dx.element_type = 0
    comptime num_tiles = (OUT_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        # Load dy tile
        dy_col = tile_idx * TILE + local_col
        if global_row < BATCH and dy_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[global_row, dy_col]
        else:
            dy_shared[local_row, local_col] = 0

        # Load W.T tile: W_T[tile_idx*TILE + local_row, global_col] = W[global_col, tile_idx*TILE + local_row]
        W_col = tile_idx * TILE + local_row
        if W_col < OUT_DIM and global_col < IN_DIM:
            W_T_shared[local_row, local_col] = W[global_col, W_col]
        else:
            W_T_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += dy_shared[local_row, k] * W_T_shared[k, local_col]

        barrier()

    if global_row < BATCH and global_col < IN_DIM:
        dx[global_row, global_col] = acc


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
    """Adam optimizer update kernel.

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    params -= lr * (m / bias_correction1) / (sqrt(v / bias_correction2) + eps)
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        g = grads[global_i]
        m_val = m[global_i]
        v_val = v[global_i]

        # Update moments
        m_new = beta1 * m_val + (1 - beta1) * g
        v_new = beta2 * v_val + (1 - beta2) * g * g

        # Bias-corrected estimates
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2

        # Update parameters
        params[global_i] = params[global_i] - lr * m_hat / (sqrt(v_hat) + eps)

        # Store updated moments
        m[global_i] = m_new
        v[global_i] = v_new


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
        src_val = source[global_i]
        tgt_val = target[global_i]
        target[global_i] = tau * src_val + (1 - tau) * tgt_val
