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
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block, lane_id
from std.sys import is_nvidia_gpu, has_nvidia_gpu_accelerator
from std.gpu.compute.mma import mma
from linalg.matmul import matmul as _max_matmul
from layout.tile_tensor import lt_to_tt


struct MatMul[
    in_dim: Int,
    out_dim: Int,
    USE_MAX_KERNELS: Bool = True,
](DiffOp):
    """MatMul : y = x @ W  where x:(B, in_dim), W:(in_dim, out_dim), y:(B, out_dim).

    Pure matrix multiply without bias. BiasAdd is a separate DiffOp.

    PARAM_SIZE = in_dim * out_dim (W only)
    CACHE_SIZE = in_dim (caches input for dW computation in backward)

    USE_MAX_KERNELS (NVIDIA only): when True, route forward and backward through
    `linalg.matmul.matmul` (the optimized max_matmul GEMM). When False (default),
    use the custom MMA tensor-core kernel. Apple is unaffected — it always uses
    the 2x2 tiled fallback regardless of this flag.
    """

    comptime OP_ID: Int = OpID.MATMUL._value
    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim
    comptime CACHE_SIZE: Int = Self.in_dim
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

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
        BATCH: Int, dtype: DType = DType.float32
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
        """Forward: output = input @ W, cache input."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)

        # Cache input for backward (needed for dW)
        for b in range(BATCH):
            for i in range(Self.in_dim):
                cache[b, i] = input[b, i]

        # output = input @ W
        for b in range(BATCH):
            for j in range(Self.out_dim):
                var acc: output.element_type = 0
                for k in range(Self.in_dim):
                    acc += input[b, k] * W[k, j]
                output[b, j] = acc

    @staticmethod
    def vjp[
        BATCH: Int, dtype: DType = DType.float32
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
        """Backward: grad_input = grad_out @ W.T, dW += input.T @ grad_out."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grad_params.ptr)

        for b in range(BATCH):
            # grad_input = grad_output @ W.T
            for i in range(Self.in_dim):
                var acc: grad_output.element_type = 0
                for j in range(Self.out_dim):
                    acc += grad_output[b, j] * W[i, j]
                grad_input[b, i] = acc

            # dW += input.T @ grad_output (ACCUMULATE)
            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    dW[i, j] = dW[i, j] + cache[b, i] * grad_output[b, j]

    # =========================================================================
    # GPU kernel implementations — tiled (Apple fallback)
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_tiled[
        BATCH: Int, dtype: DType = DType.float32
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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """Tiled matmul forward: output = input @ W, stores input to cache.

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

        var acc: Scalar[dtype] = 0
        comptime num_tiles = (Self.in_dim + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var in_col = tile_idx * TILE + local_col
            if global_row < BATCH and in_col < Self.in_dim:
                input_shared[local_row, local_col] = rebind[Scalar[dtype]](
                    input[global_row, in_col]
                )
                # Cache input (only first x-block to avoid races)
                if Int(block_idx.x) == 0:
                    cache[global_row, in_col] = rebind[Scalar[dtype]](
                        input[global_row, in_col]
                    )
            else:
                input_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.in_dim and global_col < Self.out_dim:
                W_shared[local_row, local_col] = rebind[Scalar[dtype]](
                    W[W_row, global_col]
                )
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
    def backward_dx_kernel_tiled[
        BATCH: Int, dtype: DType = DType.float32
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
        """Tiled backward: grad_input = grad_output @ W.T.

        Grid: ((in_dim + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        var dy_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var WT_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var acc: Scalar[dtype] = 0
        comptime num_tiles = (Self.out_dim + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var dy_col = tile_idx * TILE + local_col
            if global_row < BATCH and dy_col < Self.out_dim:
                dy_shared[local_row, local_col] = rebind[Scalar[dtype]](
                    grad_output[global_row, dy_col]
                )
            else:
                dy_shared[local_row, local_col] = 0

            # W.T[tile_idx*TILE+local_row, global_col] = W[global_col, tile_idx*TILE+local_row]
            var WT_row = tile_idx * TILE + local_row
            if global_col < Self.in_dim and WT_row < Self.out_dim:
                WT_shared[local_row, local_col] = rebind[Scalar[dtype]](
                    W[global_col, WT_row]
                )
            else:
                WT_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](dy_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](WT_shared[k, local_col])

            barrier()

        if global_row < BATCH and global_col < Self.in_dim:
            grad_input[global_row, global_col] = acc

    @always_inline
    @staticmethod
    def backward_dW_kernel_tiled[
        BATCH: Int, dtype: DType = DType.float32
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
        """Tiled backward: dW = cache.T @ grad_output.

        Grid: ((out_dim + TILE - 1) // TILE, (in_dim + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # in_dim
        var global_col = Int(block_idx.x) * TILE + local_col  # out_dim

        var cacheT_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var dy_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var acc: Scalar[dtype] = 0
        comptime num_tiles = (BATCH + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            # cache.T[global_row, tile_idx*TILE+local_col] = cache[tile_idx*TILE+local_col, global_row]
            var batch_col = tile_idx * TILE + local_col
            if batch_col < BATCH and global_row < Self.in_dim:
                cacheT_shared[local_row, local_col] = rebind[Scalar[dtype]](
                    cache[batch_col, global_row]
                )
            else:
                cacheT_shared[local_row, local_col] = 0

            var batch_row = tile_idx * TILE + local_row
            if batch_row < BATCH and global_col < Self.out_dim:
                dy_shared[local_row, local_col] = rebind[Scalar[dtype]](
                    grad_output[batch_row, global_col]
                )
            else:
                dy_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](
                    cacheT_shared[local_row, k]
                ) * rebind[Scalar[dtype]](dy_shared[k, local_col])

            barrier()

        if global_row < Self.in_dim and global_col < Self.out_dim:
            # Accumulate (+=) into dW so multi-call backward (MuZero K-step
            # unroll, DreamerV3/TD-MPC2 BPTT) sums gradients across calls
            # instead of overwriting. Caller pre-zeros via zero_grads.
            dW[global_row, global_col] = dW[global_row, global_col] + acc

    # =========================================================================
    # GPU kernel implementations — MMA (NVIDIA tensor cores)
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_mma[
        BATCH: Int, dtype: DType = DType.float32
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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """MMA forward: output = input @ W with tensor cores.

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

            # Cache input (block_idx.x == 0 only, to avoid races)
            if Int(block_idx.x) == 0:
                # 256 threads cache 32 rows × in_dim
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

                # Load A[32, 8] — input tile
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_r = block_row + a_r
                var ga_c = k_off + a_c
                if ga_r < BATCH and ga_c < Self.in_dim:
                    a_smem[a_r, a_c] = input[ga_r, ga_c]
                else:
                    a_smem[a_r, a_c] = 0

                # Load B[8, 32] — W tile
                var b_r = tid // MMA_BLOCK_N
                var b_c = tid % MMA_BLOCK_N
                var gb_r = k_off + b_r
                var gb_c = block_col + b_c
                if gb_r < Self.in_dim and gb_c < Self.out_dim:
                    b_smem[b_r, b_c] = W[gb_r, gb_c]
                else:
                    b_smem[b_r, b_c] = 0

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

            # Store results
            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < BATCH and c0 < Self.out_dim:
                output[r0, c0] = acc[0].cast[dtype]()
            if r0 < BATCH and c1 < Self.out_dim:
                output[r0, c1] = acc[1].cast[dtype]()
            if r1 < BATCH and c0 < Self.out_dim:
                output[r1, c0] = acc[2].cast[dtype]()
            if r1 < BATCH and c1 < Self.out_dim:
                output[r1, c1] = acc[3].cast[dtype]()

    @always_inline
    @staticmethod
    def eval_kernel_2x2[
        BATCH: Int, dtype: DType = DType.float32
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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """Register-tiled 2×2 forward: output = input @ W.

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

        # Cache input (block_idx.x == 0 only)
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

        var gr0 = block_row + sub_r * 2
        var gc0 = block_col + sub_c * 2
        if gr0 < BATCH and gc0 < Self.out_dim:
            output[gr0, gc0] = acc00
        if gr0 < BATCH and gc0 + 1 < Self.out_dim:
            output[gr0, gc0 + 1] = acc01
        if gr0 + 1 < BATCH and gc0 < Self.out_dim:
            output[gr0 + 1, gc0] = acc10
        if gr0 + 1 < BATCH and gc0 + 1 < Self.out_dim:
            output[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    def backward_dx_kernel_mma[
        BATCH: Int, dtype: DType = DType.float32
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
        """MMA backward: grad_input = grad_output @ W.T.

        Computes (BATCH, out_dim) @ (out_dim, in_dim) → (BATCH, in_dim).
        W.T is loaded transposed from W.

        Grid: ((in_dim + 31) // 32, (BATCH + 31) // 32)
        Block: (256, 1)
        """
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M  # BATCH axis
            var block_col = Int(block_idx.x) * MMA_BLOCK_N  # in_dim axis

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

                # Load A = grad_output[32, 8]
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_r = block_row + a_r
                var ga_c = k_off + a_c
                if ga_r < BATCH and ga_c < Self.out_dim:
                    a_smem[a_r, a_c] = grad_output[ga_r, ga_c]
                else:
                    a_smem[a_r, a_c] = 0

                # Load B = W.T[8, 32]: B[k, c] = W[block_col + c, k_off + k]
                var b_r = tid // MMA_BLOCK_N
                var b_c = tid % MMA_BLOCK_N
                var w_row = block_col + b_c  # in_dim axis
                var w_col = k_off + b_r  # out_dim axis
                if w_row < Self.in_dim and w_col < Self.out_dim:
                    b_smem[b_r, b_c] = W[w_row, w_col]
                else:
                    b_smem[b_r, b_c] = 0

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
                grad_input[r0, c0] = acc[0].cast[dtype]()
            if r0 < BATCH and c1 < Self.in_dim:
                grad_input[r0, c1] = acc[1].cast[dtype]()
            if r1 < BATCH and c0 < Self.in_dim:
                grad_input[r1, c0] = acc[2].cast[dtype]()
            if r1 < BATCH and c1 < Self.in_dim:
                grad_input[r1, c1] = acc[3].cast[dtype]()

    @always_inline
    @staticmethod
    def backward_dx_kernel_2x2[
        BATCH: Int, dtype: DType = DType.float32
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
        """Register-tiled 2×2 backward: grad_input = grad_output @ W.T.

        Grid: ((in_dim + 31) // 32, (BATCH + 31) // 32)
        Block: (256, 1)
        """
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

            # Load A = grad_output
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

            # Load B = W.T: B[k, c] = W[block_col + c, k_off + k]
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
        BATCH: Int, dtype: DType = DType.float32
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
        """MMA backward: dW = cache.T @ grad_output.

        Computes (in_dim, BATCH) @ (BATCH, out_dim) → (in_dim, out_dim).

        Grid: ((out_dim + 31) // 32, (in_dim + 31) // 32)
        Block: (256, 1)
        """
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M  # in_dim axis
            var block_col = Int(block_idx.x) * MMA_BLOCK_N  # out_dim axis

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

                # Load A = cache.T[32, 8]: A[r, k] = cache[k_off + k, block_row + r]
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var cache_batch = k_off + a_c
                var cache_feat = block_row + a_r
                if cache_batch < BATCH and cache_feat < Self.in_dim:
                    a_smem[a_r, a_c] = cache[cache_batch, cache_feat]
                else:
                    a_smem[a_r, a_c] = 0

                # Load B = grad_output[8, 32]
                var b_r = tid // MMA_BLOCK_N
                var b_c = tid % MMA_BLOCK_N
                var gb_r = k_off + b_r
                var gb_c = block_col + b_c
                if gb_r < BATCH and gb_c < Self.out_dim:
                    b_smem[b_r, b_c] = grad_output[gb_r, gb_c]
                else:
                    b_smem[b_r, b_c] = 0

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

            # Accumulate (+=) into dW. Multi-call backward (MuZero K-step
            # unroll, RSSM/world-model BPTT) requires accumulation across
            # calls. Caller pre-zeros grad_params via zero_grads.
            if r0 < Self.in_dim and c0 < Self.out_dim:
                dW[r0, c0] = dW[r0, c0] + acc[0].cast[dtype]()
            if r0 < Self.in_dim and c1 < Self.out_dim:
                dW[r0, c1] = dW[r0, c1] + acc[1].cast[dtype]()
            if r1 < Self.in_dim and c0 < Self.out_dim:
                dW[r1, c0] = dW[r1, c0] + acc[2].cast[dtype]()
            if r1 < Self.in_dim and c1 < Self.out_dim:
                dW[r1, c1] = dW[r1, c1] + acc[3].cast[dtype]()

    @always_inline
    @staticmethod
    def backward_dW_kernel_2x2[
        BATCH: Int, dtype: DType = DType.float32
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
        """Register-tiled 2×2 backward: dW = cache.T @ grad_output.

        Grid: ((out_dim + 31) // 32, (in_dim + 31) // 32)
        Block: (256, 1)
        """
        comptime BT = 32
        comptime SK = 16

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_row = Int(block_idx.y) * BT  # in_dim axis
        var block_col = Int(block_idx.x) * BT  # out_dim axis

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

            # Load A = cache.T: A[r, k] = cache[k_off + k, block_row + r]
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

            # Load B = grad_output
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
        # Accumulate (+=) into dW so multiple backward calls in a single
        # update (MuZero K-step unroll, DreamerV3/TD-MPC2 BPTT) sum
        # gradients instead of overwriting. Caller pre-zeros via zero_grads.
        if gr0 < Self.in_dim and gc0 < Self.out_dim:
            dW[gr0, gc0] = dW[gr0, gc0] + acc00
        if gr0 < Self.in_dim and gc0 + 1 < Self.out_dim:
            dW[gr0, gc0 + 1] = dW[gr0, gc0 + 1] + acc01
        if gr0 + 1 < Self.in_dim and gc0 < Self.out_dim:
            dW[gr0 + 1, gc0] = dW[gr0 + 1, gc0] + acc10
        if gr0 + 1 < Self.in_dim and gc0 + 1 < Self.out_dim:
            dW[gr0 + 1, gc0 + 1] = dW[gr0 + 1, gc0 + 1] + acc11

    # =========================================================================
    # GPU launchers (auto-dispatching: MMA on NVIDIA, 2x2 on Apple)
    # =========================================================================

    @staticmethod
    def eval_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](input.ptr)

        comptime if Self.USE_MAX_KERNELS and has_nvidia_gpu_accelerator():
            # max_matmul path: separate cache-input kernel + linalg matmul.
            # max_matmul has no notion of a cache buffer, so we copy
            # input → cache in its own pass before invoking the GEMM.
            comptime cache_elems = BATCH * Self.in_dim
            comptime cache_blocks = (cache_elems + TPB - 1) // TPB

            @parameter
            @always_inline
            def cache_input_wrapper(
                cache: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
                ],
                input: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx < cache_elems:
                    cache.ptr[idx] = input.ptr[idx]

            ctx.enqueue_function[cache_input_wrapper, cache_input_wrapper](
                cache,
                input_immut,
                grid_dim=(cache_blocks,),
                block_dim=(TPB,),
            )

            var input_mm = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
            ](input.ptr)
            var W_mm = LayoutTensor[
                dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
            ](params.ptr)
            _max_matmul[target="gpu"](
                lt_to_tt(output),
                lt_to_tt(input_mm),
                lt_to_tt(W_mm),
                DeviceContextPtr(ctx),
            )
        else:
            comptime grid_x = (Self.out_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
            comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

            @parameter
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
                cache: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
                ],
            ):
                comptime if is_nvidia_gpu():
                    Self.eval_kernel_mma[BATCH, dtype](output, input, W, cache)
                else:
                    Self.eval_kernel_2x2[BATCH, dtype](output, input, W, cache)

            ctx.enqueue_function[wrapper, wrapper](
                output,
                input_immut,
                W,
                cache,
                grid_dim=(grid_x, grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

    @staticmethod
    def vjp_gpu[
        BATCH: Int, dtype: DType = DType.float32
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

        comptime if Self.USE_MAX_KERNELS and has_nvidia_gpu_accelerator():
            # dx = grad_output @ W^T via max_matmul (transpose_b)
            var W_for_dx = LayoutTensor[
                dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
            ](params.ptr)
            var grad_input_mm = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
            ](grad_input.ptr)
            var grad_output_mm = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
            ](grad_output.ptr)
            _max_matmul[transpose_b=True, target="gpu"](
                lt_to_tt(grad_input_mm),
                lt_to_tt(grad_output_mm),
                lt_to_tt(W_for_dx),
                DeviceContextPtr(ctx),
            )

            # dW = cache^T @ grad_output. Routed through the MMA kernel (not
            # max_matmul) because max_matmul has no accumulate mode and would
            # overwrite grad_params on each backward call — broken for
            # multi-call BPTT-style unrolls (MuZero K-step, RSSM, world-model
            # BPTT). The MMA kernel uses += so multiple backward calls
            # correctly accumulate. Caller pre-zeros via zero_grads.
            comptime dW_grid_x_max = (
                Self.out_dim + MMA_BLOCK_N - 1
            ) // MMA_BLOCK_N
            comptime dW_grid_y_max = (
                Self.in_dim + MMA_BLOCK_M - 1
            ) // MMA_BLOCK_M

            @parameter
            @always_inline
            def dW_wrapper_max(
                dW: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.in_dim, Self.out_dim),
                    MutAnyOrigin,
                ],
                cache: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
                ],
                grad_output: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.out_dim),
                    ImmutAnyOrigin,
                ],
            ):
                Self.backward_dW_kernel_mma[BATCH, dtype](
                    dW, cache, grad_output
                )

            ctx.enqueue_function[dW_wrapper_max, dW_wrapper_max](
                dW,
                cache_immut,
                grad_output_immut,
                grid_dim=(dW_grid_x_max, dW_grid_y_max),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )
        else:
            # Kernel 1: dx = grad_output @ W.T
            comptime dx_grid_x = (Self.in_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
            comptime dx_grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

            @parameter
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
                    Self.backward_dx_kernel_mma[BATCH, dtype](grad_input, grad_output, W)
                else:
                    Self.backward_dx_kernel_2x2[BATCH, dtype](grad_input, grad_output, W)

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

            @parameter
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
                    Self.backward_dW_kernel_mma[BATCH, dtype](dW, cache, grad_output)
                else:
                    Self.backward_dW_kernel_2x2[BATCH, dtype](dW, cache, grad_output)

            ctx.enqueue_function[dW_wrapper, dW_wrapper](
                dW,
                cache_immut,
                grad_output_immut,
                grid_dim=(dW_grid_x, dW_grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )
