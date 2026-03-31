from ...constants import (
    dtype,
    TILE,
    TPB,
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
from ...autodiff.op import DiffOp, FusedOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceStream
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block, lane_id
from std.sys import is_nvidia_gpu, has_nvidia_gpu_accelerator
from std.gpu.compute.mma import mma
from linalg.matmul import matmul as max_matmul


struct FusedMatMulBias[in_dim: Int, out_dim: Int](FusedOp):
    """Fused y = x @ W + b in a single operation.

    PARAM_SIZE = in_dim * out_dim + out_dim  (W then b)
    CACHE_SIZE = in_dim  (caches input for dW)
    """

    comptime OP_ID: Int = OpID.FUSED_MATMUL_BIAS._value
    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim + Self.out_dim
    comptime CACHE_SIZE: Int = Self.in_dim
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0
    comptime FUSED_COUNT: Int = 2

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    def eval[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward: y = x @ W + b, cache input."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](params.ptr + Self.in_dim * Self.out_dim)

        for ba in range(BATCH):
            for i in range(Self.in_dim):
                cache[ba, i] = input[ba, i]
            for j in range(Self.out_dim):
                var acc: output.element_type = b[j]
                for k in range(Self.in_dim):
                    acc += input[ba, k] * W[k, j]
                output[ba, j] = acc

    @staticmethod
    def vjp[
        BATCH: Int
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward: dx = dy @ W.T, dW += x.T @ dy, db += sum(dy, axis=0)."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grad_params.ptr)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grad_params.ptr + Self.in_dim * Self.out_dim)

        for ba in range(BATCH):
            # dx = dy @ W.T
            for i in range(Self.in_dim):
                var acc: grad_output.element_type = 0
                for j in range(Self.out_dim):
                    acc += grad_output[ba, j] * W[i, j]
                grad_input[ba, i] = acc

            # dW += x.T @ dy
            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    dW[i, j] = dW[i, j] + cache[ba, i] * grad_output[ba, j]

            # db += sum(dy, axis=0)
            for j in range(Self.out_dim):
                db[j] = db[j] + grad_output[ba, j]

    # =========================================================================
    # GPU kernel implementations — tiled (Apple fallback)
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_tiled[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """Fused forward: y = x @ W + b with tiled matmul.

        Grid: ((out_dim + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        var input_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var W_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Init accumulator with bias
        var acc: Scalar[dtype] = 0
        if global_col < Self.out_dim:
            acc = rebind[Scalar[dtype]](b[global_col])

        comptime num_tiles = (Self.in_dim + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var in_col = tile_idx * TILE + local_col
            if global_row < BATCH and in_col < Self.in_dim:
                var x_val = input[global_row, in_col]
                input_shared[local_row, local_col] = x_val
                if Int(block_idx.x) == 0:
                    cache[global_row, in_col] = x_val
            else:
                input_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.in_dim and global_col < Self.out_dim:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](
                    input_shared[local_row, k]
                ) * rebind[Scalar[dtype]](W_shared[k, local_col])

            barrier()

        if global_row < BATCH and global_col < Self.out_dim:
            output[global_row, global_col] = acc

    @always_inline
    @staticmethod
    def backward_kernel_tiled[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        db: LayoutTensor[dtype, Layout.row_major(Self.out_dim), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
    ):
        """Fused backward with dual-region grid: dx region + dW/db region.

        Grid: (max(dx_grid_x, dW_grid_x), dx_grid_y + dW_grid_y)
        Block: (TILE, TILE)
        """
        comptime dx_grid_x = (Self.in_dim + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE
        comptime dW_grid_y = (Self.in_dim + TILE - 1) // TILE

        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var block_y = Int(block_idx.y)

        var shared_A = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var shared_B = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        if block_y < dx_grid_y:
            # Region 1: dx = grad_output @ W.T
            var global_row = block_y * TILE + local_row
            var global_col = Int(block_idx.x) * TILE + local_col

            var acc: Scalar[dtype] = 0
            comptime num_tiles = (Self.out_dim + TILE - 1) // TILE

            for tile_idx in range(num_tiles):
                var dy_col = tile_idx * TILE + local_col
                if global_row < BATCH and dy_col < Self.out_dim:
                    shared_A[local_row, local_col] = grad_output[
                        global_row, dy_col
                    ]
                else:
                    shared_A[local_row, local_col] = 0

                var WT_row = tile_idx * TILE + local_row
                if global_col < Self.in_dim and WT_row < Self.out_dim:
                    shared_B[local_row, local_col] = W[global_col, WT_row]
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    acc += rebind[Scalar[dtype]](
                        shared_A[local_row, k]
                    ) * rebind[Scalar[dtype]](shared_B[k, local_col])

                barrier()

            if global_row < BATCH and global_col < Self.in_dim:
                grad_input[global_row, global_col] = acc
        else:
            # Region 2: dW = cache.T @ grad_output, db = sum(grad_output, axis=0)
            var dW_block_y = block_y - dx_grid_y
            var global_row = dW_block_y * TILE + local_row  # in_dim axis
            var global_col = Int(block_idx.x) * TILE + local_col  # out_dim axis

            var acc: Scalar[dtype] = 0
            var db_acc: Scalar[dtype] = 0
            comptime num_tiles = (BATCH + TILE - 1) // TILE

            for tile_idx in range(num_tiles):
                var batch_col = tile_idx * TILE + local_col
                if batch_col < BATCH and global_row < Self.in_dim:
                    shared_A[local_row, local_col] = cache[
                        batch_col, global_row
                    ]
                else:
                    shared_A[local_row, local_col] = 0

                var batch_row = tile_idx * TILE + local_row
                if batch_row < BATCH and global_col < Self.out_dim:
                    var grad_val = grad_output[batch_row, global_col]
                    shared_B[local_row, local_col] = grad_val
                    if dW_block_y == 0:
                        db_acc += rebind[Scalar[dtype]](grad_val)
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    acc += rebind[Scalar[dtype]](
                        shared_A[local_row, k]
                    ) * rebind[Scalar[dtype]](shared_B[k, local_col])

                barrier()

            if global_row < Self.in_dim and global_col < Self.out_dim:
                dW[global_row, global_col] = acc

            # db reduction (first row of dW blocks only)
            if dW_block_y == 0 and global_col < Self.out_dim:
                shared_A[local_row, local_col] = db_acc
                barrier()
                if local_row == 0:
                    var total: Scalar[dtype] = 0
                    for r in range(TILE):
                        total += rebind[Scalar[dtype]](shared_A[r, local_col])
                    db[global_col] = total

    # =========================================================================
    # GPU kernel implementations — MMA (NVIDIA tensor cores)
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_mma[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """MMA forward: y = x @ W + b with tensor cores.

        Grid: ((out_dim + 31) // 32, (BATCH + 31) // 32)
        Block: (256, 1)
        """
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            # Cache input (block_idx.x == 0 only)
            if Int(block_idx.x) == 0:
                for i in range(0, Self.in_dim, MMA_BLOCK_THREADS):
                    var col = i + tid
                    if col < Self.in_dim:
                        for r in range(MMA_BLOCK_M):
                            var gr = block_row + r
                            if gr < BATCH:
                                cache[gr, col] = input[gr, col]

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

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4

            comptime num_k_tiles = (Self.in_dim + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_r = block_row + a_r
                var ga_c = k_off + a_c
                if ga_r < BATCH and ga_c < Self.in_dim:
                    a_smem[a_r, a_c] = input[ga_r, ga_c]
                else:
                    a_smem[a_r, a_c] = 0

                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                var gb_r = k_off + br
                var gb_c = block_col + bc
                if gb_r < Self.in_dim and gb_c < Self.out_dim:
                    b_smem[br, bc] = W[gb_r, gb_c]
                else:
                    b_smem[br, bc] = 0

                barrier()

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
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )

                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            # Store results with bias added
            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < BATCH and c0 < Self.out_dim:
                output[r0, c0] = rebind[Scalar[dtype]](acc[0]) + rebind[
                    Scalar[dtype]
                ](b[c0])
            if r0 < BATCH and c1 < Self.out_dim:
                output[r0, c1] = rebind[Scalar[dtype]](acc[1]) + rebind[
                    Scalar[dtype]
                ](b[c1])
            if r1 < BATCH and c0 < Self.out_dim:
                output[r1, c0] = rebind[Scalar[dtype]](acc[2]) + rebind[
                    Scalar[dtype]
                ](b[c0])
            if r1 < BATCH and c1 < Self.out_dim:
                output[r1, c1] = rebind[Scalar[dtype]](acc[3]) + rebind[
                    Scalar[dtype]
                ](b[c1])

    @always_inline
    @staticmethod
    def eval_kernel_2x2[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """Register-tiled 2×2 forward: y = x @ W + b.

        Grid: ((out_dim + 31) // 32, (BATCH + 31) // 32)
        Block: (256, 1)
        """
        comptime BT = 32
        comptime SK = 16

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_row = Int(block_idx.y) * BT
        var block_col = Int(block_idx.x) * BT

        # Cache input
        if Int(block_idx.x) == 0:
            for i in range(0, Self.in_dim, MMA_BLOCK_THREADS):
                var col = i + tid
                if col < Self.in_dim:
                    for r in range(BT):
                        var gr = block_row + r
                        if gr < BATCH:
                            cache[gr, col] = input[gr, col]

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

        comptime num_k_tiles = (Self.in_dim + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if block_row + a_r0 < BATCH and k_off + a_c0 < Self.in_dim:
                a_smem[a_r0, a_c0] = input[block_row + a_r0, k_off + a_c0]
            else:
                a_smem[a_r0, a_c0] = 0
            if block_row + a_r1 < BATCH and k_off + a_c1 < Self.in_dim:
                a_smem[a_r1, a_c1] = input[block_row + a_r1, k_off + a_c1]
            else:
                a_smem[a_r1, a_c1] = 0

            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < Self.in_dim and block_col + b_c0 < Self.out_dim:
                b_smem[b_r0, b_c0] = W[k_off + b_r0, block_col + b_c0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < Self.in_dim and block_col + b_c1 < Self.out_dim:
                b_smem[b_r1, b_c1] = W[k_off + b_r1, block_col + b_c1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < Self.in_dim:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                    var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    acc10 += a1 * b0
                    acc11 += a1 * b1

            barrier()

        # Store with bias
        var gr0 = block_row + sub_r * 2
        var gc0 = block_col + sub_c * 2
        if gr0 < BATCH and gc0 < Self.out_dim:
            output[gr0, gc0] = acc00 + rebind[Scalar[dtype]](b[gc0])
        if gr0 < BATCH and gc0 + 1 < Self.out_dim:
            output[gr0, gc0 + 1] = acc01 + rebind[Scalar[dtype]](b[gc0 + 1])
        if gr0 + 1 < BATCH and gc0 < Self.out_dim:
            output[gr0 + 1, gc0] = acc10 + rebind[Scalar[dtype]](b[gc0])
        if gr0 + 1 < BATCH and gc0 + 1 < Self.out_dim:
            output[gr0 + 1, gc0 + 1] = acc11 + rebind[Scalar[dtype]](b[gc0 + 1])

    @always_inline
    @staticmethod
    def backward_dx_kernel_mma[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
    ):
        """MMA backward: dx = grad_output @ W.T."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

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

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4

            comptime num_k_tiles = (Self.out_dim + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if block_row + a_r < BATCH and k_off + a_c < Self.out_dim:
                    a_smem[a_r, a_c] = grad_output[block_row + a_r, k_off + a_c]
                else:
                    a_smem[a_r, a_c] = 0

                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                var w_row = block_col + bc
                var w_col = k_off + br
                if w_row < Self.in_dim and w_col < Self.out_dim:
                    b_smem[br, bc] = W[w_row, w_col]
                else:
                    b_smem[br, bc] = 0

                barrier()

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
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < BATCH and c0 < Self.in_dim:
                grad_input[r0, c0] = rebind[Scalar[dtype]](acc[0])
            if r0 < BATCH and c1 < Self.in_dim:
                grad_input[r0, c1] = rebind[Scalar[dtype]](acc[1])
            if r1 < BATCH and c0 < Self.in_dim:
                grad_input[r1, c0] = rebind[Scalar[dtype]](acc[2])
            if r1 < BATCH and c1 < Self.in_dim:
                grad_input[r1, c1] = rebind[Scalar[dtype]](acc[3])

    @always_inline
    @staticmethod
    def backward_dx_kernel_2x2[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
    ):
        """Register-tiled 2×2 backward: dx = grad_output @ W.T."""
        comptime BT = 32
        comptime SK = 16

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
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

        comptime num_k_tiles = (Self.out_dim + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if block_row + a_r0 < BATCH and k_off + a_c0 < Self.out_dim:
                a_smem[a_r0, a_c0] = grad_output[block_row + a_r0, k_off + a_c0]
            else:
                a_smem[a_r0, a_c0] = 0
            if block_row + a_r1 < BATCH and k_off + a_c1 < Self.out_dim:
                a_smem[a_r1, a_c1] = grad_output[block_row + a_r1, k_off + a_c1]
            else:
                a_smem[a_r1, a_c1] = 0

            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < Self.out_dim and block_col + b_c0 < Self.in_dim:
                b_smem[b_r0, b_c0] = W[block_col + b_c0, k_off + b_r0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < Self.out_dim and block_col + b_c1 < Self.in_dim:
                b_smem[b_r1, b_c1] = W[block_col + b_c1, k_off + b_r1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < Self.out_dim:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                    var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    acc10 += a1 * b0
                    acc11 += a1 * b1

            barrier()

        var gr0 = block_row + sub_r * 2
        var gc0 = block_col + sub_c * 2
        if gr0 < BATCH and gc0 < Self.in_dim:
            grad_input[gr0, gc0] = acc00
        if gr0 < BATCH and gc0 + 1 < Self.in_dim:
            grad_input[gr0, gc0 + 1] = acc01
        if gr0 + 1 < BATCH and gc0 < Self.in_dim:
            grad_input[gr0 + 1, gc0] = acc10
        if gr0 + 1 < BATCH and gc0 + 1 < Self.in_dim:
            grad_input[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    def backward_dW_kernel_mma[
        BATCH: Int
    ](
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
    ):
        """MMA backward: dW = cache.T @ grad_output."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M  # in_dim
            var block_col = Int(block_idx.x) * MMA_BLOCK_N  # out_dim

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

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4

            comptime num_k_tiles = (BATCH + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if k_off + a_c < BATCH and block_row + a_r < Self.in_dim:
                    a_smem[a_r, a_c] = cache[k_off + a_c, block_row + a_r]
                else:
                    a_smem[a_r, a_c] = 0

                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                if k_off + br < BATCH and block_col + bc < Self.out_dim:
                    b_smem[br, bc] = grad_output[k_off + br, block_col + bc]
                else:
                    b_smem[br, bc] = 0

                barrier()

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
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < Self.in_dim and c0 < Self.out_dim:
                dW[r0, c0] = rebind[Scalar[dtype]](acc[0])
            if r0 < Self.in_dim and c1 < Self.out_dim:
                dW[r0, c1] = rebind[Scalar[dtype]](acc[1])
            if r1 < Self.in_dim and c0 < Self.out_dim:
                dW[r1, c0] = rebind[Scalar[dtype]](acc[2])
            if r1 < Self.in_dim and c1 < Self.out_dim:
                dW[r1, c1] = rebind[Scalar[dtype]](acc[3])

    @always_inline
    @staticmethod
    def backward_dW_kernel_2x2[
        BATCH: Int
    ](
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
    ):
        """Register-tiled 2×2 backward: dW = cache.T @ grad_output."""
        comptime BT = 32
        comptime SK = 16

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
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

        comptime num_k_tiles = (BATCH + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if k_off + a_c0 < BATCH and block_row + a_r0 < Self.in_dim:
                a_smem[a_r0, a_c0] = cache[k_off + a_c0, block_row + a_r0]
            else:
                a_smem[a_r0, a_c0] = 0
            if k_off + a_c1 < BATCH and block_row + a_r1 < Self.in_dim:
                a_smem[a_r1, a_c1] = cache[k_off + a_c1, block_row + a_r1]
            else:
                a_smem[a_r1, a_c1] = 0

            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < BATCH and block_col + b_c0 < Self.out_dim:
                b_smem[b_r0, b_c0] = grad_output[k_off + b_r0, block_col + b_c0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < BATCH and block_col + b_c1 < Self.out_dim:
                b_smem[b_r1, b_c1] = grad_output[k_off + b_r1, block_col + b_c1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < BATCH:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                    var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    acc10 += a1 * b0
                    acc11 += a1 * b1

            barrier()

        var gr0 = block_row + sub_r * 2
        var gc0 = block_col + sub_c * 2
        if gr0 < Self.in_dim and gc0 < Self.out_dim:
            dW[gr0, gc0] = acc00
        if gr0 < Self.in_dim and gc0 + 1 < Self.out_dim:
            dW[gr0, gc0 + 1] = acc01
        if gr0 + 1 < Self.in_dim and gc0 < Self.out_dim:
            dW[gr0 + 1, gc0] = acc10
        if gr0 + 1 < Self.in_dim and gc0 + 1 < Self.out_dim:
            dW[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    def backward_db_kernel[
        BATCH: Int
    ](
        db: LayoutTensor[dtype, Layout.row_major(Self.out_dim), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
    ):
        """Formula: db = sum(grad_output, axis=0). Simple elementwise kernel.

        Grid: ((out_dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var col = Int(block_idx.x) * TPB + Int(thread_idx.x)
        if col < Self.out_dim:
            var acc: Scalar[dtype] = 0
            for b in range(BATCH):
                acc += rebind[Scalar[dtype]](grad_output[b, col])
            db[col] = acc

    # =========================================================================
    # GPU launchers (auto-dispatching: MMA on NVIDIA, 2x2 on Apple)
    # =========================================================================

    @staticmethod
    def eval_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
        ](params.ptr + Self.in_dim * Self.out_dim)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](input.ptr)

        comptime if has_nvidia_gpu_accelerator():
            # 1. Cache input (contiguous copy since cache = [BATCH, in_dim])
            comptime cache_elems = BATCH * Self.in_dim
            comptime cache_blocks = (cache_elems + TPB - 1) // TPB

            @always_inline
            def cache_input_wrapper(
                cache: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.in_dim),
                    MutAnyOrigin,
                ],
                input: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.in_dim),
                    ImmutAnyOrigin,
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= cache_elems:
                    return
                cache[idx // Self.in_dim, idx % Self.in_dim] = input[
                    idx // Self.in_dim, idx % Self.in_dim
                ]

            ctx.enqueue_function[cache_input_wrapper, cache_input_wrapper](
                cache,
                input_immut,
                grid_dim=(cache_blocks,),
                block_dim=(TPB,),
            )

            # 2. Matmul: output = input @ W
            comptime if has_nvidia_gpu_accelerator() and Self.out_dim < 64:
                from ...gpu.matmul_ops import matmul_kernel as safe_mm

                comptime MM_TILE = 8
                comptime MM_GRID = (
                    (Self.out_dim + MM_TILE - 1) // MM_TILE,
                    (BATCH + MM_TILE - 1) // MM_TILE,
                )

                @always_inline
                def _safe_mm_wrapper(
                    mm_out: LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.out_dim),
                        MutAnyOrigin,
                    ],
                    a: LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.in_dim),
                        ImmutAnyOrigin,
                    ],
                    b: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.in_dim, Self.out_dim),
                        ImmutAnyOrigin,
                    ],
                ):
                    safe_mm[
                        dtype, BATCH, Self.out_dim, Self.in_dim, MM_TILE
                    ](mm_out, a, b)

                ctx.enqueue_function[_safe_mm_wrapper, _safe_mm_wrapper](
                    output,
                    input_immut,
                    W,
                    grid_dim=MM_GRID,
                    block_dim=(MM_TILE, MM_TILE),
                )
            else:
                max_matmul[target="gpu"](output, input_immut, W, ctx)

            # 3. Bias add
            comptime bias_elems = BATCH * Self.out_dim
            comptime bias_blocks = (bias_elems + TPB - 1) // TPB

            @always_inline
            def bias_add_wrapper(
                output: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.out_dim),
                    MutAnyOrigin,
                ],
                b: LayoutTensor[
                    dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= bias_elems:
                    return
                var col = idx % Self.out_dim
                output[idx // Self.out_dim, col] = (
                    output[idx // Self.out_dim, col] + b[col]
                )

            ctx.enqueue_function[bias_add_wrapper, bias_add_wrapper](
                output,
                b,
                grid_dim=(bias_blocks,),
                block_dim=(TPB,),
            )
        else:
            comptime grid_x = (Self.out_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
            comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

            @always_inline
            def wrapper(
                output: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.out_dim),
                    MutAnyOrigin,
                ],
                input: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.in_dim),
                    ImmutAnyOrigin,
                ],
                W: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.in_dim, Self.out_dim),
                    ImmutAnyOrigin,
                ],
                b: LayoutTensor[
                    dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
                ],
                cache: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.in_dim),
                    MutAnyOrigin,
                ],
            ):
                Self.eval_kernel_2x2[BATCH](output, input, W, b, cache)

            ctx.enqueue_function[wrapper, wrapper](
                output,
                input_immut,
                W,
                b,
                cache,
                grid_dim=(grid_x, grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

    @staticmethod
    def eval_gpu_on_stream[
        BATCH: Int
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
        ](params.ptr + Self.in_dim * Self.out_dim)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](input.ptr)

        comptime grid_x = (Self.out_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.out_dim),
                ImmutAnyOrigin,
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
            ],
        ):
            comptime if is_nvidia_gpu():
                Self.eval_kernel_mma[BATCH](output, input, W, b, cache)
            else:
                Self.eval_kernel_2x2[BATCH](output, input, W, b, cache)

        var compiled = ctx.compile_function[wrapper, wrapper]()
        stream.enqueue_function(
            compiled,
            output,
            input_immut,
            W,
            b,
            cache,
            grid_dim=(grid_x, grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )

    @staticmethod
    def vjp_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](cache.ptr)
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grad_params.ptr)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grad_params.ptr + Self.in_dim * Self.out_dim)

        # Kernel 1: dx = grad_output @ W.T
        comptime dx_grid_x = (Self.in_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime dx_grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        def dx_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.out_dim),
                ImmutAnyOrigin,
            ],
        ):
            comptime if is_nvidia_gpu():
                Self.backward_dx_kernel_mma[BATCH](grad_input, grad_output, W)
            else:
                Self.backward_dx_kernel_2x2[BATCH](grad_input, grad_output, W)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            W,
            grid_dim=(dx_grid_x, dx_grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )

        # Kernel 2: dW = cache.T @ grad_output
        comptime dW_grid_x = (Self.out_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime dW_grid_y = (Self.in_dim + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        def dW_wrapper(
            dW: LayoutTensor[
                dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
            ],
        ):
            comptime if is_nvidia_gpu():
                Self.backward_dW_kernel_mma[BATCH](dW, cache, grad_output)
            else:
                Self.backward_dW_kernel_2x2[BATCH](dW, cache, grad_output)

        ctx.enqueue_function[dW_wrapper, dW_wrapper](
            dW,
            cache_immut,
            grad_output_immut,
            grid_dim=(dW_grid_x, dW_grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )

        # Kernel 3: db = sum(grad_output, axis=0)
        comptime db_grid_x = (Self.out_dim + TPB - 1) // TPB

        @always_inline
        def db_wrapper(
            db: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_db_kernel[BATCH](db, grad_output)

        ctx.enqueue_function[db_wrapper, db_wrapper](
            db,
            grad_output_immut,
            grid_dim=(db_grid_x,),
            block_dim=(TPB,),
        )
