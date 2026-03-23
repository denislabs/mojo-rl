"""GPU matrix multiplication for deep RL.

This module provides GPU-accelerated matrix multiplication using patterns
from Mojo GPU puzzles (P16).

Operations:
- tiled_matmul_kernel: Shared memory tiled matmul (all GPUs)
- matmul_kernel: Idiomatic tiled matmul with async copies (all GPUs)
- mma_matmul_kernel: Tensor core matmul via MMA intrinsics (NVIDIA only)
- gpu_matmul: Auto-dispatching launcher (MMA on NVIDIA, tiled on Apple)

For neural networks: C = A @ B where A is (M, K) and B is (K, N)
"""

from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.primitives import lane_id
from std.sys import is_nvidia_gpu
from std.gpu.compute.mma import mma
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async

from ..constants import (
    MMA_M,
    MMA_N,
    MMA_K,
    MMA_BLOCK_M,
    MMA_BLOCK_N,
    MMA_WARPS_M,
    MMA_WARPS_N,
    MMA_NUM_WARPS,
    MMA_BLOCK_THREADS,
)


@always_inline
def tiled_matmul_kernel[
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
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
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
        comptime for k in range(TILE):
            acc += rebind[Scalar[dtype]](a_shared[local_row, k]) * rebind[
                Scalar[dtype]
            ](b_shared[k, local_col])

        barrier()

    # Write result
    if global_row < M and global_col < N:
        output[global_row, global_col] = acc


@always_inline
def matmul_kernel[
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
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
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
        comptime for k in range(TILE):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write result to output tile
    out_tile[local_row, local_col] = acc


# =============================================================================
# MMA (Tensor Core) Matmul — NVIDIA only
# =============================================================================


@always_inline
def mma_matmul_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """MMA-based matmul using NVIDIA tensor cores (m16n8k8 TF32).

    Each warp computes a 16×8 output tile. 8 warps per block
    arranged 2×4 produce a 32×32 output tile.

    Block: (256, 1)
    Grid: ((N + 31) // 32, (M + 31) // 32)
    """
    # Guarded so this only compiles on NVIDIA targets
    comptime if is_nvidia_gpu():
        # Thread and warp identification
        var tid = Int(thread_idx.x)
        var warp_id = tid // 32
        var warp_m = warp_id // MMA_WARPS_N  # 0 or 1
        var warp_n = warp_id % MMA_WARPS_N  # 0..3

        # Block position in the output matrix
        var block_row = Int(block_idx.y) * MMA_BLOCK_M
        var block_col = Int(block_idx.x) * MMA_BLOCK_N

        # Shared memory for one K-step: A[32, 8] and B[8, 32]
        var a_smem = LayoutTensor[
            dtype,
            Layout.row_major(MMA_BLOCK_M, MMA_K),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var b_smem = LayoutTensor[
            dtype,
            Layout.row_major(MMA_K, MMA_BLOCK_N),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Accumulator — 4 FP32 values per thread (m16n8k8 output fragment)
        var acc = SIMD[DType.float32, 4](0)

        # Lane-level indices for MMA fragment layout
        var lid = lane_id()
        var group_id = lid >> 2  # 0..7 — selects row within 16-row tile
        var group_lane = lid % 4  # 0..3 — selects col pair

        comptime num_k_tiles = (K + MMA_K - 1) // MMA_K

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * MMA_K

            # --- Load A[32, 8] into shared memory (256 threads, 256 elems) ---
            var a_r = tid // MMA_K  # row 0..31
            var a_c = tid % MMA_K  # col 0..7
            var ga_r = block_row + a_r
            var ga_c = k_off + a_c
            if ga_r < M and ga_c < K:
                a_smem[a_r, a_c] = a[ga_r, ga_c]
            else:
                a_smem[a_r, a_c] = 0

            # --- Load B[8, 32] into shared memory ---
            var b_r = tid // MMA_BLOCK_N  # row 0..7
            var b_c = tid % MMA_BLOCK_N  # col 0..31
            var gb_r = k_off + b_r
            var gb_c = block_col + b_c
            if gb_r < K and gb_c < N:
                b_smem[b_r, b_c] = b[gb_r, gb_c]
            else:
                b_smem[b_r, b_c] = 0

            barrier()

            # --- Construct A fragment for this warp (16×8 → 4 values/thread) ---
            # Layout: [0]=(group_id, group_lane)
            #         [1]=(group_id+8, group_lane)
            #         [2]=(group_id, group_lane+4)
            #         [3]=(group_id+8, group_lane+4)
            var warp_row = warp_m * MMA_M
            var a_frag = SIMD[DType.float32, 4](
                rebind[Scalar[DType.float32]](
                    a_smem[warp_row + Int(group_id), Int(group_lane)]
                ),
                rebind[Scalar[DType.float32]](
                    a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]
                ),
                rebind[Scalar[DType.float32]](
                    a_smem[warp_row + Int(group_id), Int(group_lane) + 4]
                ),
                rebind[Scalar[DType.float32]](
                    a_smem[warp_row + Int(group_id) + 8, Int(group_lane) + 4]
                ),
            )

            # --- Construct B fragment for this warp (8×8 → 2 values/thread) ---
            # Layout: [0]=(group_lane, group_id)
            #         [1]=(group_lane+4, group_id)
            var warp_col = warp_n * MMA_N
            var b_frag = SIMD[DType.float32, 2](
                rebind[Scalar[DType.float32]](
                    b_smem[Int(group_lane), warp_col + Int(group_id)]
                ),
                rebind[Scalar[DType.float32]](
                    b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                ),
            )

            # --- Warp-level MMA: acc = a_frag × b_frag + acc ---
            mma(acc, a_frag, b_frag, acc)

            barrier()

        # --- Store result with bounds checking ---
        # Output layout for m16n8k8:
        # [0]=(group_id, group_lane*2)     [1]=(group_id, group_lane*2+1)
        # [2]=(group_id+8, group_lane*2)   [3]=(group_id+8, group_lane*2+1)
        var r0 = block_row + warp_m * MMA_M + Int(group_id)
        var r1 = r0 + 8
        var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
        var c1 = c0 + 1

        if r0 < M and c0 < N:
            output[r0, c0] = rebind[Scalar[dtype]](acc[0])
        if r0 < M and c1 < N:
            output[r0, c1] = rebind[Scalar[dtype]](acc[1])
        if r1 < M and c0 < N:
            output[r1, c0] = rebind[Scalar[dtype]](acc[2])
        if r1 < M and c1 < N:
            output[r1, c1] = rebind[Scalar[dtype]](acc[3])


# =============================================================================
# Register-tiled 2×2 scalar kernel (Apple fallback for 32×32 blocks)
# =============================================================================


@always_inline
def _tiled_2x2_kernel[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
):
    """Register-tiled matmul: each thread computes 2×2 output elements.

    256 threads cover 32×32 = 1024 output elements (4 per thread).
    Shared memory tiles: A[32, SMEM_K] + B[SMEM_K, 32] per K-step.

    Block: (256, 1)
    Grid: ((N + 31) // 32, (M + 31) // 32)
    """
    comptime BT = 32  # Block tile size
    comptime SK = 16  # Shared memory K-tile

    var tid = Int(thread_idx.x)
    var sub_r = tid // 16  # 0..15 → output rows 2*sub_r, 2*sub_r+1
    var sub_c = tid % 16  # 0..15 → output cols 2*sub_c, 2*sub_c+1

    var block_row = Int(block_idx.y) * BT
    var block_col = Int(block_idx.x) * BT

    var a_smem = LayoutTensor[
        dtype,
        Layout.row_major(BT, SK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_smem = LayoutTensor[
        dtype,
        Layout.row_major(SK, BT),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var acc00: Scalar[dtype] = 0
    var acc01: Scalar[dtype] = 0
    var acc10: Scalar[dtype] = 0
    var acc11: Scalar[dtype] = 0

    comptime num_k_tiles = (K + SK - 1) // SK

    for k_tile in range(num_k_tiles):
        var k_off = k_tile * SK

        # Load A[32, SK] = 512 elements with 256 threads → 2 per thread
        var a_r0 = tid // SK
        var a_c0 = tid % SK
        var a_r1 = (tid + 256) // SK
        var a_c1 = (tid + 256) % SK
        var ga_r0 = block_row + a_r0
        var ga_r1 = block_row + a_r1

        if ga_r0 < M and k_off + a_c0 < K:
            a_smem[a_r0, a_c0] = a[ga_r0, k_off + a_c0]
        else:
            a_smem[a_r0, a_c0] = 0
        if ga_r1 < M and k_off + a_c1 < K:
            a_smem[a_r1, a_c1] = a[ga_r1, k_off + a_c1]
        else:
            a_smem[a_r1, a_c1] = 0

        # Load B[SK, 32] = 512 elements with 256 threads → 2 per thread
        var b_r0 = tid // BT
        var b_c0 = tid % BT
        var b_r1 = (tid + 256) // BT
        var b_c1 = (tid + 256) % BT
        var gb_c0 = block_col + b_c0
        var gb_c1 = block_col + b_c1

        if k_off + b_r0 < K and gb_c0 < N:
            b_smem[b_r0, b_c0] = b[k_off + b_r0, gb_c0]
        else:
            b_smem[b_r0, b_c0] = 0
        if k_off + b_r1 < K and gb_c1 < N:
            b_smem[b_r1, b_c1] = b[k_off + b_r1, gb_c1]
        else:
            b_smem[b_r1, b_c1] = 0

        barrier()

        # Compute 2×2 register tile
        for k in range(SK):
            if k_off + k < K:
                var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                acc00 += a0 * b0
                acc01 += a0 * b1
                acc10 += a1 * b0
                acc11 += a1 * b1

        barrier()

    # Store 2×2 results with bounds checking
    var gr0 = block_row + sub_r * 2
    var gc0 = block_col + sub_c * 2
    if gr0 < M and gc0 < N:
        output[gr0, gc0] = acc00
    if gr0 < M and gc0 + 1 < N:
        output[gr0, gc0 + 1] = acc01
    if gr0 + 1 < M and gc0 < N:
        output[gr0 + 1, gc0] = acc10
    if gr0 + 1 < M and gc0 + 1 < N:
        output[gr0 + 1, gc0 + 1] = acc11


# =============================================================================
# Auto-dispatching GPU matmul launcher
# =============================================================================


def gpu_matmul[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    TILE: Int,
](
    ctx: DeviceContext,
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
) raises:
    """Auto-dispatching GPU matmul: MMA on NVIDIA, register-tiled on Apple.

    Both paths use 256 threads and 32×32 output tiles:
    - NVIDIA: 8 warps × m16n8k8 tensor core MMA (all warps active)
    - Apple/other: register-tiled 2×2 scalar (each thread computes 4 elements)
    """
    comptime grid_x = (N + MMA_BLOCK_N - 1) // MMA_BLOCK_N
    comptime grid_y = (M + MMA_BLOCK_M - 1) // MMA_BLOCK_M

    @always_inline
    def kernel(
        output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
        a: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
        b: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
    ):
        comptime if is_nvidia_gpu():
            mma_matmul_kernel[dtype, M, N, K](output, a, b)
        else:
            _tiled_2x2_kernel[dtype, M, N, K](output, a, b)

    ctx.enqueue_function[kernel, kernel](
        output,
        a,
        b,
        grid_dim=(grid_x, grid_y),
        block_dim=(MMA_BLOCK_THREADS, 1),
    )


@always_inline
def matmul_bias_kernel[
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
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
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
        comptime for k in range(TILE):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write result to output tile
    out_tile[local_row, local_col] = acc
