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
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor


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
    @parameter
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
