from ..constants import dtype, TILE, TPB
from .model import Model
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block


struct Linear[in_dim: Int, out_dim: Int](Model):
    """Linear layer: y = x @ W + b (stateless).

    This is a stateless layer - all parameters and gradients are managed externally.
    The caller allocates and passes:
    - params: [W_flat (in_dim * out_dim) | b (out_dim)]
    - grads: [dW_flat (in_dim * out_dim) | db (out_dim)]

    PARAM_SIZE = in_dim * out_dim + out_dim (W flattened + b)
    CACHE_SIZE = in_dim (caches input for weight gradient computation)
    WORKSPACE_SIZE_PER_SAMPLE = 0 (leaf layer, no intermediate buffers needed)
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.IN_DIM * Self.OUT_DIM + Self.OUT_DIM
    comptime CACHE_SIZE: Int = Self.IN_DIM  # Cache input for dW computation
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0  # Leaf layer, no workspace needed

    fn __init__(out self):
        """Initialize stateless Linear layer."""
        pass

    fn __init__(out self, *, deinit take: Self):
        """Move constructor for Sequential composition."""
        pass

    fn __init__(out self, *, copy: Self):
        """Copy constructor for Copyable trait."""
        pass

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        INIT.init[Self.PARAM_SIZE, Self.IN_DIM, Self.OUT_DIM](params)

    @staticmethod
    fn forward[
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
        """Forward pass: output = input @ W + b.

        Caches the input for backward pass (needed for weight gradients).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [W_flat | b].
            cache: Cache buffer [BATCH, IN_DIM] for backward pass (written).
        """
        # Create 2D view of W from params (first in_dim * out_dim elements)
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        # b starts after W in params
        var b_offset = Self.in_dim * Self.out_dim

        # Cache input for backward
        for batch in range(BATCH):
            for i in range(Self.in_dim):
                cache[batch, i] = input[batch, i]

        # Compute y = x @ W + b
        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = params[b_offset + j]  # bias
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                output[batch, j] = acc

    @staticmethod
    fn forward[
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
    ):
        """Forward pass without caching (for inference).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [W_flat | b].
        """
        # Create 2D view of W from params
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var b_offset = Self.in_dim * Self.out_dim

        # Compute y = x @ W + b (no caching)
        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = params[b_offset + j]  # bias
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                output[batch, j] = acc

    # =========================================================================
    # GPU Kernel Implementations (@always_inline for fusion)
    # =========================================================================
    #
    # These are the core GPU computations that can be inlined into fused kernels.
    # They use thread_idx/block_idx and shared memory directly.
    #
    # For fusion, create a new kernel that calls multiple _kernel_impl functions.
    # =========================================================================

    @always_inline
    @staticmethod
    fn forward_kernel_impl[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        """Tiled matmul forward: output = input @ W + b, stores input to cache.

        Only threads in x-block 0 write the cache (to avoid race conditions).

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # batch
        var global_col = Int(block_idx.x) * TILE + local_col  # out_dim

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

        var acc: b.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        comptime num_tiles = (Self.IN_DIM + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var in_col = tile_idx * TILE + local_col
            if global_row < BATCH and in_col < Self.IN_DIM:
                input_shared[local_row, local_col] = input[global_row, in_col]
                if Int(block_idx.x) == 0:
                    cache[global_row, in_col] = input[global_row, in_col]
            else:
                input_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](
                    input_shared[local_row, k]
                ) * rebind[Scalar[dtype]](W_shared[k, local_col])

            barrier()

        if global_row < BATCH and global_col < Self.OUT_DIM:
            output[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn forward_kernel_impl_no_cache[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin],
    ):
        """Tiled matmul forward: output = input @ W + b (no cache, for inference).

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # batch
        var global_col = Int(block_idx.x) * TILE + local_col  # out_dim

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

        var acc: b.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        comptime num_tiles = (Self.IN_DIM + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var in_col = tile_idx * TILE + local_col
            if global_row < BATCH and in_col < Self.IN_DIM:
                input_shared[local_row, local_col] = input[global_row, in_col]
            else:
                input_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](
                    input_shared[local_row, k]
                ) * rebind[Scalar[dtype]](W_shared[k, local_col])

            barrier()

        if global_row < BATCH and global_col < Self.OUT_DIM:
            output[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn backward_dx_kernel_impl[
        BATCH: Int,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Tiled backward: grad_input = grad_output @ W.T.

        Grid: ((IN_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # batch
        var global_col = Int(block_idx.x) * TILE + local_col  # in_dim

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

        comptime num_tiles = (Self.OUT_DIM + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var dy_col = tile_idx * TILE + local_col
            if global_row < BATCH and dy_col < Self.OUT_DIM:
                dy_shared[local_row, local_col] = grad_output[
                    global_row, dy_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            # W.T[tile_idx*TILE+local_row, global_col] = W[global_col, tile_idx*TILE+local_row]
            var WT_row = tile_idx * TILE + local_row
            if global_col < Self.IN_DIM and WT_row < Self.OUT_DIM:
                WT_shared[local_row, local_col] = W[global_col, WT_row]
            else:
                WT_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](dy_shared[local_row, k]) * rebind[
                    Scalar[dtype]
                ](WT_shared[k, local_col])

            barrier()

        if global_row < BATCH and global_col < Self.IN_DIM:
            grad_input[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn backward_dW_kernel_impl[
        BATCH: Int,
    ](
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Tiled backward: dW = cache.T @ grad_output.

        Grid: ((OUT_DIM + TILE - 1) // TILE, (IN_DIM + TILE - 1) // TILE)
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
            if batch_col < BATCH and global_row < Self.IN_DIM:
                cacheT_shared[local_row, local_col] = cache[
                    batch_col, global_row
                ]
            else:
                cacheT_shared[local_row, local_col] = 0

            var batch_row = tile_idx * TILE + local_row
            if batch_row < BATCH and global_col < Self.OUT_DIM:
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

        if global_row < Self.IN_DIM and global_col < Self.OUT_DIM:
            dW[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn backward_db_kernel_impl[
        BATCH: Int,
    ](
        db: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Backward pass kernel implementation: db = sum(dy, axis=0).

        Grid: (OUT_DIM,)
        Block: (TPB,)

        Each block handles one output dimension and reduces across batch.
        """

        var col = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if col >= Self.OUT_DIM:
            return

        # Each thread loads elements strided by TPB
        var my_sum: db.element_type = 0
        var batch_idx = local_i
        while batch_idx < BATCH:
            my_sum += grad_output[batch_idx, col]
            batch_idx += TPB

        # Reduce across threads
        var total = block.sum[block_size=TPB, broadcast=False](val=my_sum)

        if local_i == 0:
            db[col] = total[0]

    # =========================================================================
    # GPU Launchers (with DeviceContext)
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU with caching."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self.IN_DIM * Self.OUT_DIM)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        comptime grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn forward_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH](output, input, W, b, cache)

        ctx.enqueue_function[forward_wrapper, forward_wrapper](
            output,
            input_immut,
            W,
            b,
            cache,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
        )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU without caching (for inference)."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self.IN_DIM * Self.OUT_DIM)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        comptime grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl_no_cache[BATCH](output, input, W, b)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            W,
            b,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
        )

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """Launch backward pass on GPU using three tiled kernels.

        Computes: grad_input = grad_output @ W.T, dW = cache.T @ grad_output, db = sum(dy, axis=0).
        """
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](cache.ptr)
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr + Self.IN_DIM * Self.OUT_DIM)

        # Kernel 1: dx = grad_output @ W.T  (tiled 2D)
        comptime dx_grid_x = (Self.IN_DIM + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn backward_dx_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_dx_kernel_impl[BATCH](grad_input, grad_output, W)

        ctx.enqueue_function[backward_dx_wrapper, backward_dx_wrapper](
            grad_input,
            grad_output_immut,
            W,
            grid_dim=(dx_grid_x, dx_grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 2: dW = cache.T @ grad_output  (tiled 2D)
        comptime dW_grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime dW_grid_y = (Self.IN_DIM + TILE - 1) // TILE

        @always_inline
        fn backward_dW_wrapper(
            dW: LayoutTensor[
                dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_dW_kernel_impl[BATCH](dW, cache, grad_output)

        ctx.enqueue_function[backward_dW_wrapper, backward_dW_wrapper](
            dW,
            cache_immut,
            grad_output_immut,
            grid_dim=(dW_grid_x, dW_grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 3: db = sum(grad_output, axis=0)
        @always_inline
        fn backward_db_wrapper(
            db: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_db_kernel_impl[BATCH](db, grad_output)

        ctx.enqueue_function[backward_db_wrapper, backward_db_wrapper](
            db,
            grad_output_immut,
            grid_dim=(Self.OUT_DIM,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn backward[
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
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass: compute grad_input and accumulate dW, db.

        Uses cached input from forward pass to compute weight gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH, OUT_DIM].
            grad_input: Gradient of loss w.r.t. input [BATCH, IN_DIM] (written).
            params: Model parameters [W_flat | b].
            cache: Cached input from forward pass [BATCH, IN_DIM].
            grads: Parameter gradients [dW_flat | db] (accumulated, not overwritten).
        """
        # Create 2D views of W and dW from 1D params/grads
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grads.ptr)
        var db_offset = Self.in_dim * Self.out_dim

        for batch in range(BATCH):
            # dx = dy @ W.T
            for i in range(Self.in_dim):
                var acc: grad_input.element_type = 0
                for j in range(Self.out_dim):
                    acc += grad_output[batch, j] * W[i, j]
                grad_input[batch, i] = acc

            # dW += x.T @ dy (accumulated)
            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    dW[i, j] = (
                        dW[i, j] + cache[batch, i] * grad_output[batch, j]
                    )

            # db += sum(dy, axis=0)
            for j in range(Self.out_dim):
                grads[db_offset + j] = (
                    grads[db_offset + j] + grad_output[batch, j]
                )
