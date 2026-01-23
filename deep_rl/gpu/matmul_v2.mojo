"""GPU matrix multiplication using idiomatic Mojo patterns.

This module provides GPU-accelerated matrix multiplication using the modern
LayoutTensor API with tile(), copy_dram_to_sram_async, and thread layouts.

Based on Mojo GPU Puzzles P16 idiomatic solution:
https://puzzles.modular.com/puzzle_16/tiled.html

Operations:
- tiled_matmul_kernel: Optimized matmul using tile() and async copies
- tiled_matmul_kernel_padded: Handles non-tile-aligned dimensions

For neural networks: C = A @ B where A is (M, K) and B is (K, N)
"""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace, async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async

# =============================================================================
# Idiomatic Tiled Matrix Multiplication
# =============================================================================


@always_inline
fn tiled_matmul_kernel_idiomatic[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    TILE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """Idiomatic tiled matmul using tile() and async memory copies.

    This version assumes M, N, K are all divisible by TILE.
    For arbitrary dimensions, use tiled_matmul_kernel_general.

    Grid: ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var block_row = Int(block_idx.y)
    var block_col = Int(block_idx.x)

    # Get output tile that this block is responsible for
    var out_tile = output.tile[TILE, TILE](block_row, block_col)

    # Shared memory for tiles
    var a_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0

    # Thread layout for coalesced memory access
    # Each thread loads one element, threads arranged in row-major order
    comptime NUM_THREADS = TILE * TILE
    comptime BLOCK_DIM_COUNT = 2
    comptime load_layout = Layout.row_major(1, TILE)

    # Number of tiles along K dimension
    comptime num_k_tiles = K // TILE

    # Use runtime loop to avoid compile-time explosion with large K
    for tile_idx in range(num_k_tiles):
        # Get tiles from A and B
        var a_tile = a.tile[TILE, TILE](block_row, tile_idx)
        var b_tile = b.tile[TILE, TILE](tile_idx, block_col)

        # Async copy tiles to shared memory with coalesced access
        copy_dram_to_sram_async[
            thread_layout=load_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        ](a_shared, a_tile)

        copy_dram_to_sram_async[
            thread_layout=load_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        ](b_shared, b_tile)

        # Wait for async copies to complete
        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        @parameter
        for k in range(TILE):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write result to output tile
    out_tile[local_row, local_col] = acc


@always_inline
fn tiled_matmul_kernel_general[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    TILE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """General tiled matmul that handles arbitrary dimensions.

    Uses tile() for clean indexing but adds bounds checking for edge tiles.
    This is the version to use for neural networks with arbitrary layer sizes.

    Grid: ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    # Shared memory for tiles
    var a_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0

    # Number of tiles along K dimension (ceiling division)
    comptime num_k_tiles = (K + TILE - 1) // TILE

    # Use runtime loop to avoid compile-time explosion with large K
    for tile_idx in range(num_k_tiles):
        # Compute global indices for this tile
        var a_col = tile_idx * TILE + local_col
        var b_row = tile_idx * TILE + local_row

        # Load A tile with bounds check
        if global_row < M and a_col < K:
            a_shared[local_row, local_col] = a[global_row, a_col]
        else:
            a_shared[local_row, local_col] = 0

        # Load B tile with bounds check
        if b_row < K and global_col < N:
            b_shared[local_row, local_col] = b[b_row, global_col]
        else:
            b_shared[local_row, local_col] = 0

        barrier()

        # Compute partial product - use @parameter for inner loop (small, fixed size)
        @parameter
        for k in range(TILE):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write result with bounds check
    if global_row < M and global_col < N:
        output[global_row, global_col] = acc


# =============================================================================
# Linear Layer Specific: y = x @ W + b
# =============================================================================


@always_inline
fn linear_forward_kernel[
    dtype: DType,
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
):
    """Linear layer forward: output = input @ W + b with input caching.

    Combines tiled matmul with bias addition and input caching for backward pass.

    Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row  # Batch index
    var global_col = Int(block_idx.x) * TILE + local_col  # Output index

    # Shared memory
    var x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var W_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Start with bias
    var acc: output.element_type = 0
    if global_col < OUT_DIM:
        acc = b[global_col]

    # Tiled matrix multiplication
    for tile_idx in range((IN_DIM + TILE - 1) // TILE):
        var x_col = tile_idx * TILE + local_col
        var W_row = tile_idx * TILE + local_row

        # Load input tile and cache it
        if global_row < BATCH and x_col < IN_DIM:
            var x_val = input[global_row, x_col]
            x_shared[local_row, local_col] = x_val
            cache[global_row, x_col] = x_val  # Cache for backward
        else:
            x_shared[local_row, local_col] = 0

        # Load weight tile
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += x_shared[local_row, k] * W_shared[k, local_col]

        barrier()

    # Write result
    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc


@always_inline
fn linear_backward_dx_kernel[
    dtype: DType,
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int,
](
    grad_input: LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ],
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
):
    """Linear layer backward for input gradient: dx = dy @ W.T.

    Grid: ((IN_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row  # Batch index
    var global_col = Int(block_idx.x) * TILE + local_col  # Input index

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

    var acc: grad_input.element_type = 0

    # dx = dy @ W.T means we iterate over OUT_DIM
    for tile_idx in range((OUT_DIM + TILE - 1) // TILE):
        var dy_col = tile_idx * TILE + local_col
        var W_col = tile_idx * TILE + local_row  # W.T row = W col

        # Load grad_output tile
        if global_row < BATCH and dy_col < OUT_DIM:
            dy_shared[local_row, local_col] = grad_output[global_row, dy_col]
        else:
            dy_shared[local_row, local_col] = 0

        # Load W.T tile (transpose: W_T[i,j] = W[j,i])
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
        grad_input[global_row, global_col] = acc


@always_inline
fn linear_backward_dW_kernel[
    dtype: DType,
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Linear layer backward for weight gradient: dW = x.T @ dy.

    Grid: ((OUT_DIM + TILE - 1) // TILE, (IN_DIM + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row  # IN_DIM index
    var global_col = Int(block_idx.x) * TILE + local_col  # OUT_DIM index

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

    # dW = x.T @ dy means we iterate over BATCH
    for tile_idx in range((BATCH + TILE - 1) // TILE):
        var batch_idx = tile_idx * TILE + local_col  # For x.T
        var dy_row = tile_idx * TILE + local_row

        # Load x.T tile (transpose: x_T[i,j] = x[j,i])
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = cache[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        # Load grad_output tile
        if dy_row < BATCH and global_col < OUT_DIM:
            dy_shared[local_row, local_col] = grad_output[dy_row, global_col]
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += x_T_shared[local_row, k] * dy_shared[k, local_col]

        barrier()

    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = acc
