"""GPU matrix multiplication for deep RL.

This module provides GPU-accelerated matrix multiplication using patterns
from Mojo GPU puzzles (P16).

Operations:
- naive_matmul: Simple matmul, each thread computes one output element
- tiled_matmul: Optimized matmul using shared memory tiling

For neural networks: C = A @ B where A is (M, K) and B is (K, N)
"""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace, async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async


@always_inline
fn tiled_matmul_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    TILE: Int,  # Tile size (threads per block dimension)
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """Tiled matmul kernel using shared memory.

    Uses shared memory tiles to reduce global memory accesses.
    Each thread block computes a TILE x TILE portion of output.
    """
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    # Allocate shared memory for tiles
    a_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dtype] = 0

    # Number of tiles along K dimension
    comptime num_tiles = (K + TILE - 1) // TILE

    # Iterate over tiles
    for tile_idx in range(num_tiles):
        # Load A tile: A[global_row, tile_idx * TILE + local_col]
        a_col = tile_idx * TILE + local_col
        if global_row < M and a_col < K:
            a_shared[local_row, local_col] = a[global_row, a_col]
        else:
            a_shared[local_row, local_col] = 0

        # Load B tile: B[tile_idx * TILE + local_row, global_col]
        b_row = tile_idx * TILE + local_row
        if b_row < K and global_col < N:
            b_shared[local_row, local_col] = b[b_row, global_col]
        else:
            b_shared[local_row, local_col] = 0

        barrier()

        # Compute partial product for this tile
        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dtype]](a_shared[local_row, k]) * rebind[
                Scalar[dtype]
            ](b_shared[k, local_col])

        barrier()

    # Write result
    if global_row < M and global_col < N:
        output[global_row, global_col] = acc


@always_inline
fn matmul_kernel[
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
fn matmul_bias_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    TILE: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    """Matmul with bias kernel using shared memory tiling.

    This version assumes M, N, K are all divisible by TILE.
    For arbitrary dimensions, use tiled_matmul_kernel_general.

    Grid: ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var block_row = Int(block_idx.y)
    var block_col = Int(block_idx.x)
    var global_col = Int(block_idx.x) * TILE + local_col
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
    if global_col < N:
        acc = bias[global_col]

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
