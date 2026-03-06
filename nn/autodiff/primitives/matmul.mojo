from ...constants import dtype, TILE, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block


struct MatMul[in_dim: Int, out_dim: Int](DiffOp):
    """MatMul : y = x @ W  where x:(B, in_dim), W:(in_dim, out_dim), y:(B, out_dim).

    Pure matrix multiply without bias. BiasAdd is a separate DiffOp.

    PARAM_SIZE = in_dim * out_dim (W only)
    CACHE_SIZE = in_dim (caches input for dW computation in backward)
    """

    comptime OP_ID: Int = OpID.MATMUL._value
    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim
    comptime CACHE_SIZE: Int = Self.in_dim

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    fn eval[
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
    fn vjp[
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
    # GPU kernel implementations (@always_inline for fusion)
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
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
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var W_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var acc: Scalar[dtype] = 0
        comptime num_tiles = (Self.in_dim + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var in_col = tile_idx * TILE + local_col
            if global_row < BATCH and in_col < Self.in_dim:
                input_shared[local_row, local_col] = input[global_row, in_col]
                # Cache input (only first x-block to avoid races)
                if Int(block_idx.x) == 0:
                    cache[global_row, in_col] = input[global_row, in_col]
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
    fn backward_dx_kernel_impl[
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
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var WT_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var acc: Scalar[dtype] = 0
        comptime num_tiles = (Self.out_dim + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var dy_col = tile_idx * TILE + local_col
            if global_row < BATCH and dy_col < Self.out_dim:
                dy_shared[local_row, local_col] = grad_output[
                    global_row, dy_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            # W.T[tile_idx*TILE+local_row, global_col] = W[global_col, tile_idx*TILE+local_row]
            var WT_row = tile_idx * TILE + local_row
            if global_col < Self.in_dim and WT_row < Self.out_dim:
                WT_shared[local_row, local_col] = W[global_col, WT_row]
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
    fn backward_dW_kernel_impl[
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
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var dy_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var acc: Scalar[dtype] = 0
        comptime num_tiles = (BATCH + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            # cache.T[global_row, tile_idx*TILE+local_col] = cache[tile_idx*TILE+local_col, global_row]
            var batch_col = tile_idx * TILE + local_col
            if batch_col < BATCH and global_row < Self.in_dim:
                cacheT_shared[local_row, local_col] = cache[
                    batch_col, global_row
                ]
            else:
                cacheT_shared[local_row, local_col] = 0

            var batch_row = tile_idx * TILE + local_row
            if batch_row < BATCH and global_col < Self.out_dim:
                dy_shared[local_row, local_col] = grad_output[
                    batch_row, global_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](
                    cacheT_shared[local_row, k]
                ) * rebind[Scalar[dtype]](dy_shared[k, local_col])

            barrier()

        if global_row < Self.in_dim and global_col < Self.out_dim:
            dW[global_row, global_col] = acc

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
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
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](input.ptr)

        comptime grid_x = (Self.out_dim + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn wrapper(
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
            Self.eval_kernel_impl[BATCH](output, input, W, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            W,
            cache,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
        )

    @staticmethod
    fn vjp_gpu[
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

        # Kernel 1: dx = grad_output @ W.T
        comptime dx_grid_x = (Self.in_dim + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn dx_wrapper(
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
            Self.backward_dx_kernel_impl[BATCH](grad_input, grad_output, W)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            W,
            grid_dim=(dx_grid_x, dx_grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 2: dW = cache.T @ grad_output
        comptime dW_grid_x = (Self.out_dim + TILE - 1) // TILE
        comptime dW_grid_y = (Self.in_dim + TILE - 1) // TILE

        @always_inline
        fn dW_wrapper(
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
            Self.backward_dW_kernel_impl[BATCH](dW, cache, grad_output)

        ctx.enqueue_function[dW_wrapper, dW_wrapper](
            dW,
            cache_immut,
            grad_output_immut,
            grid_dim=(dW_grid_x, dW_grid_y),
            block_dim=(TILE, TILE),
        )
