from ..constants import dtype, TILE, TPB
from .model import Model
from layout import LayoutTensor, Layout
from layout.layout_tensor import copy_dram_to_sram_async
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace, async_copy_wait_all


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

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor for Sequential composition."""
        pass

    fn __copyinit__(out self, other: Self):
        """Copy constructor for Copyable trait."""
        pass

    fn forward[
        BATCH: Int
    ](
        self,
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

    fn forward[
        BATCH: Int
    ](
        self,
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
        """Forward pass kernel implementation: y = x @ W + b with input caching.

        This is the core GPU computation that can be inlined into fused kernels.
        Uses tiled matrix multiplication with shared memory.

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)

        Args:
            output: Output tensor [BATCH, OUT_DIM] (written).
            input: Input tensor [BATCH, IN_DIM].
            W: Weight matrix [IN_DIM, OUT_DIM].
            b: Bias vector [OUT_DIM].
            cache: Cache buffer [BATCH, IN_DIM] for backward pass (written).
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        # Shared memory for tiles
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
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        # Cache input (each thread caches one element if in bounds)
        # We do this during the first tile load to overlap with computation
        # Runtime loop to avoid compile-time explosion with large IN_DIM
        for tile_idx in range((Self.IN_DIM + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col

            # Load x tile and cache
            if global_row < BATCH and x_col < Self.IN_DIM:
                var x_val = input[global_row, x_col]
                x_shared[local_row, local_col] = x_val
                # Cache the input value
                cache[global_row, x_col] = x_val
            else:
                x_shared[local_row, local_col] = 0

            # Load W tile
            var W_row = tile_idx * TILE + local_row
            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
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
        """Forward pass kernel implementation without caching (for inference).

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

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

        var acc: output.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        var tile_x = (Self.IN_DIM + TILE - 1) // TILE
        for tile_idx in range(0, tile_x):
            var x_col = tile_idx * TILE + local_col
            if global_row < BATCH and x_col < Self.IN_DIM:
                x_shared[local_row, local_col] = input[global_row, x_col]
            else:
                x_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        if global_row < BATCH and global_col < Self.OUT_DIM:
            output[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn forward_kernel_impl_async[
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
        """Forward pass kernel using idiomatic async copy: y = x @ W + b.

        Uses copy_dram_to_sram_async for efficient memory transfers that:
        - Bypass registers (reduced register pressure)
        - Use dedicated copy engines
        - Enable compute-memory overlap

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        # Shared memory for tiles
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

        # Constants for async copy
        comptime NUM_THREADS = TILE * TILE
        comptime BLOCK_DIM_COUNT = 2  # 2D thread block
        comptime load_layout = Layout.row_major(1, TILE)  # Coalesced loading

        # Start with bias
        var acc: output.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        # Number of full tiles and check for remainder
        comptime num_full_tiles = Self.IN_DIM // TILE
        comptime has_remainder = (Self.IN_DIM % TILE) != 0

        # Process full tiles using async copy
        @parameter
        for tile_idx in range(num_full_tiles):
            # Get tiles using the tile API
            var x_tile = input.tile[TILE, TILE](
                Int(block_idx.y), tile_idx
            )
            var W_tile = W.tile[TILE, TILE](
                tile_idx, Int(block_idx.x)
            )

            # Async copy tiles to shared memory
            copy_dram_to_sram_async[
                thread_layout = load_layout,
                num_threads = NUM_THREADS,
                block_dim_count = BLOCK_DIM_COUNT,
            ](x_shared, x_tile)

            copy_dram_to_sram_async[
                thread_layout = load_layout,
                num_threads = NUM_THREADS,
                block_dim_count = BLOCK_DIM_COUNT,
            ](W_shared, W_tile)

            # Wait for async copies to complete
            async_copy_wait_all()
            barrier()

            # Cache input during first set of tiles (overlap with compute)
            @parameter
            if tile_idx == 0:
                # Cache one element per thread from the first tile
                var cache_col = local_col
                if global_row < BATCH and cache_col < Self.IN_DIM:
                    cache[global_row, cache_col] = x_shared[local_row, local_col]

            # Compute partial dot product
            @parameter
            for k in range(TILE):
                acc += x_shared[local_row, k] * W_shared[k, local_col]

            # Cache remaining elements from this tile
            @parameter
            if tile_idx > 0:
                var cache_col = tile_idx * TILE + local_col
                if global_row < BATCH and cache_col < Self.IN_DIM:
                    cache[global_row, cache_col] = x_shared[local_row, local_col]

            barrier()

        # Handle remainder tile with manual loading (bounds checking needed)
        @parameter
        if has_remainder:
            comptime remainder_tile_idx = num_full_tiles
            var x_col = remainder_tile_idx * TILE + local_col
            var W_row = remainder_tile_idx * TILE + local_row

            # Manual load with bounds checking for remainder
            if global_row < BATCH and x_col < Self.IN_DIM:
                var x_val = input[global_row, x_col]
                x_shared[local_row, local_col] = x_val
                cache[global_row, x_col] = x_val
            else:
                x_shared[local_row, local_col] = 0

            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            # Compute remainder partial dot product
            comptime remainder_size = Self.IN_DIM - remainder_tile_idx * TILE
            @parameter
            for k in range(remainder_size):
                acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        # Write result
        if global_row < BATCH and global_col < Self.OUT_DIM:
            output[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn forward_kernel_impl_async_no_cache[
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
        """Forward pass kernel using idiomatic async copy (inference, no cache).

        Uses copy_dram_to_sram_async for efficient memory transfers.

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        # Shared memory for tiles
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

        # Constants for async copy
        comptime NUM_THREADS = TILE * TILE
        comptime BLOCK_DIM_COUNT = 2
        comptime load_layout = Layout.row_major(1, TILE)

        # Start with bias
        var acc: output.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        comptime num_full_tiles = Self.IN_DIM // TILE
        comptime has_remainder = (Self.IN_DIM % TILE) != 0

        # Process full tiles using async copy
        @parameter
        for tile_idx in range(num_full_tiles):
            var x_tile = input.tile[TILE, TILE](
                Int(block_idx.y), tile_idx
            )
            var W_tile = W.tile[TILE, TILE](
                tile_idx, Int(block_idx.x)
            )

            copy_dram_to_sram_async[
                thread_layout = load_layout,
                num_threads = NUM_THREADS,
                block_dim_count = BLOCK_DIM_COUNT,
            ](x_shared, x_tile)

            copy_dram_to_sram_async[
                thread_layout = load_layout,
                num_threads = NUM_THREADS,
                block_dim_count = BLOCK_DIM_COUNT,
            ](W_shared, W_tile)

            async_copy_wait_all()
            barrier()

            @parameter
            for k in range(TILE):
                acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        # Handle remainder tile with manual loading
        @parameter
        if has_remainder:
            comptime remainder_tile_idx = num_full_tiles
            var x_col = remainder_tile_idx * TILE + local_col
            var W_row = remainder_tile_idx * TILE + local_row

            if global_row < BATCH and x_col < Self.IN_DIM:
                x_shared[local_row, local_col] = input[global_row, x_col]
            else:
                x_shared[local_row, local_col] = 0

            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            comptime remainder_size = Self.IN_DIM - remainder_tile_idx * TILE
            @parameter
            for k in range(remainder_size):
                acc += x_shared[local_row, k] * W_shared[k, local_col]

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
        """Backward pass kernel implementation: dx = dy @ W.T.

        Grid: ((IN_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # BATCH dimension
        var global_col = Int(block_idx.x) * TILE + local_col  # IN_DIM dimension

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

        # Runtime loop to avoid compile-time explosion with large OUT_DIM
        for tile_idx in range((Self.OUT_DIM + TILE - 1) // TILE):
            # Load dy tile
            var dy_col = tile_idx * TILE + local_col
            if global_row < BATCH and dy_col < Self.OUT_DIM:
                dy_shared[local_row, local_col] = grad_output[
                    global_row, dy_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            # Load W.T tile
            var W_col = tile_idx * TILE + local_row
            if W_col < Self.OUT_DIM and global_col < Self.IN_DIM:
                W_T_shared[local_row, local_col] = W[global_col, W_col]
            else:
                W_T_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                acc += dy_shared[local_row, k] * W_T_shared[k, local_col]

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
        """Backward pass kernel implementation: dW = x.T @ dy.

        Uses cached input from forward pass.

        Grid: ((OUT_DIM + TILE - 1) // TILE, (IN_DIM + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # IN_DIM dimension
        var global_col = (
            Int(block_idx.x) * TILE + local_col
        )  # OUT_DIM dimension

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

        # Runtime loop to avoid compile-time explosion with large BATCH
        var tile_x = (BATCH + TILE - 1) // TILE
        for tile_idx in range(tile_x):
            # Load x.T tile
            var batch_idx = tile_idx * TILE + local_col
            if global_row < Self.IN_DIM and batch_idx < BATCH:
                x_T_shared[local_row, local_col] = cache[batch_idx, global_row]
            else:
                x_T_shared[local_row, local_col] = 0

            # Load dy tile
            var dy_row = tile_idx * TILE + local_row
            if dy_row < BATCH and global_col < Self.OUT_DIM:
                dy_shared[local_row, local_col] = grad_output[
                    dy_row, global_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                acc += x_T_shared[local_row, k] * dy_shared[k, local_col]

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
        from gpu import block

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

    @always_inline
    @staticmethod
    fn backward_fused_kernel_impl[
        BATCH: Int,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ],
        db: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
    ):
        """Fused backward kernel: computes dx, dW, and db in a single launch.

        This eliminates 2 kernel launches by combining:
        - dx = grad_output @ W.T
        - dW = cache.T @ grad_output
        - db = sum(grad_output, axis=0)

        Grid partitioning (2D grid):
        - Rows [0, dx_grid_y): blocks compute grad_input (dx)
        - Rows [dx_grid_y, dx_grid_y + dW_grid_y): blocks compute dW
        - db is computed by dW blocks in the first row (dW_block_y == 0)

        Grid: (max(dx_grid_x, dW_grid_x), dx_grid_y + dW_grid_y)
        Block: (TILE, TILE)
        """
        # Thread indices
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var block_y = Int(block_idx.y)
        var block_x = Int(block_idx.x)

        # Grid dimensions for dx computation: grad_input[BATCH, IN_DIM]
        comptime dx_grid_x = (Self.IN_DIM + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE

        # Grid dimensions for dW computation: dW[IN_DIM, OUT_DIM]
        comptime dW_grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime dW_grid_y = (Self.IN_DIM + TILE - 1) // TILE

        # Allocate shared memory (reused for both dx and dW computations)
        var shared_A = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var shared_B = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        # Determine which computation this block handles based on block_y
        if block_y < dx_grid_y:
            # ================================================================
            # dx computation: grad_input = grad_output @ W.T
            # ================================================================
            if block_x >= dx_grid_x:
                return  # This block is outside dx grid bounds

            var global_row = block_y * TILE + local_row  # BATCH dimension
            var global_col = block_x * TILE + local_col  # IN_DIM dimension

            var acc: grad_input.element_type = 0

            # Tile over OUT_DIM (the k dimension for dx = dy @ W.T)
            for tile_idx in range((Self.OUT_DIM + TILE - 1) // TILE):
                # Load grad_output tile into shared_A
                var dy_col = tile_idx * TILE + local_col
                if global_row < BATCH and dy_col < Self.OUT_DIM:
                    shared_A[local_row, local_col] = grad_output[
                        global_row, dy_col
                    ]
                else:
                    shared_A[local_row, local_col] = 0

                # Load W.T tile into shared_B (W[global_col, W_col] for transpose)
                var W_col = tile_idx * TILE + local_row
                if W_col < Self.OUT_DIM and global_col < Self.IN_DIM:
                    shared_B[local_row, local_col] = W[global_col, W_col]
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                # Compute partial dot product
                @parameter
                for k in range(TILE):
                    acc += shared_A[local_row, k] * shared_B[k, local_col]

                barrier()

            # Write result
            if global_row < BATCH and global_col < Self.IN_DIM:
                grad_input[global_row, global_col] = acc

        else:
            # ================================================================
            # dW computation: dW = cache.T @ grad_output
            # Also computes db for the first row of dW blocks
            # ================================================================
            var dW_block_y = block_y - dx_grid_y
            var dW_block_x = block_x

            if dW_block_y >= dW_grid_y or dW_block_x >= dW_grid_x:
                return  # This block is outside dW grid bounds

            var global_row = dW_block_y * TILE + local_row  # IN_DIM dimension
            var global_col = dW_block_x * TILE + local_col  # OUT_DIM dimension

            var dW_acc: dW.element_type = 0
            var db_acc: db.element_type = 0  # Used only if dW_block_y == 0

            # Tile over BATCH (the k dimension for dW = cache.T @ dy)
            var num_tiles = (BATCH + TILE - 1) // TILE
            for tile_idx in range(num_tiles):
                # Load cache.T tile into shared_A
                var batch_idx = tile_idx * TILE + local_col
                if global_row < Self.IN_DIM and batch_idx < BATCH:
                    shared_A[local_row, local_col] = cache[batch_idx, global_row]
                else:
                    shared_A[local_row, local_col] = 0

                # Load grad_output tile into shared_B
                var dy_row = tile_idx * TILE + local_row
                if dy_row < BATCH and global_col < Self.OUT_DIM:
                    var dy_val = grad_output[dy_row, global_col]
                    shared_B[local_row, local_col] = dy_val
                    # Accumulate for db (only first row of dW blocks)
                    if dW_block_y == 0:
                        db_acc += dy_val
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                # Compute partial dot product
                @parameter
                for k in range(TILE):
                    dW_acc += shared_A[local_row, k] * shared_B[k, local_col]

                barrier()

            # Write dW result
            if global_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                dW[global_row, global_col] = dW_acc

            # Compute and write db using shared memory reduction
            # Only the first row of dW blocks (dW_block_y == 0) computes db
            if dW_block_y == 0 and global_col < Self.OUT_DIM:
                # Store partial sums in shared memory for reduction
                shared_A[local_row, local_col] = db_acc
                barrier()

                # Reduce across local_row dimension (thread with local_row==0 sums)
                if local_row == 0:
                    var total = shared_A[0, local_col]
                    @parameter
                    for r in range(1, TILE):
                        total += shared_A[r, local_col]
                    db[global_col] = total

    # =========================================================================
    # GPU Launchers (with DeviceContext)
    # =========================================================================
    #
    # These functions handle buffer-to-tensor conversion, grid/block config,
    # and kernel launch. They call the _kernel_impl functions.
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU with caching.

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [W_flat | b].
            cache_buf: Cache buffer [BATCH * IN_DIM] for backward pass.
        """
        # Create tensor views from buffers
        var params_ptr = params_buf.unsafe_ptr()
        var b_ptr = params_ptr + Self.IN_DIM * Self.OUT_DIM
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params_buf.unsafe_ptr())
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](b_ptr)
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](cache_buf.unsafe_ptr())

        # Configure grid and block
        comptime grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        # Define kernel wrapper that calls the impl
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
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
        ):
            # Use async kernel with copy_dram_to_sram_async for better performance
            Self.forward_kernel_impl_async[BATCH](
                output,
                input,
                W,
                b,
                cache,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
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
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU without caching (for inference).

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [W_flat | b].

        """
        var params_ptr = params_buf.unsafe_ptr()
        var b_ptr = params_ptr + Self.IN_DIM * Self.OUT_DIM
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params_buf.unsafe_ptr())
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](b_ptr)

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
            # Use async kernel with copy_dram_to_sram_async for better performance
            Self.forward_kernel_impl_async_no_cache[BATCH](
                output,
                input,
                W,
                b,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
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
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch backward pass on GPU using fused kernel.

        Computes all gradients in a SINGLE kernel launch:
        - grad_input = grad_output @ W.T
        - dW = cache.T @ grad_output
        - db = sum(grad_output, axis=0)

        This fused implementation eliminates 2 kernel launches compared to
        the separate dx, dW, db kernels, reducing GPU synchronization overhead.

        Args:
            ctx: GPU device context.
            grad_input_buf: Gradient w.r.t. input [BATCH * IN_DIM] (written).
            grad_output_buf: Gradient w.r.t. output [BATCH * OUT_DIM].
            params_buf: Parameters buffer [PARAM_SIZE] = [W_flat | b].
            cache_buf: Cached input from forward pass [BATCH * IN_DIM].
            grads_buf: Parameter gradients [PARAM_SIZE] = [dW_flat | db] (written).
        """
        # Create tensor views
        var grad_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](grad_input_buf.unsafe_ptr())
        var grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params_buf.unsafe_ptr())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var grads_ptr = grads_buf.unsafe_ptr()
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ](grads_ptr)
        var db_ptr = grads_ptr + Self.IN_DIM * Self.OUT_DIM
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ](db_ptr)

        # Grid dimensions for fused kernel:
        # - Rows [0, dx_grid_y): dx blocks
        # - Rows [dx_grid_y, dx_grid_y + dW_grid_y): dW blocks (also compute db)
        comptime dx_grid_x = (Self.IN_DIM + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE
        comptime dW_grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime dW_grid_y = (Self.IN_DIM + TILE - 1) // TILE

        # Combined grid: max width, sum of heights
        comptime fused_grid_x = dx_grid_x if dx_grid_x > dW_grid_x else dW_grid_x
        comptime fused_grid_y = dx_grid_y + dW_grid_y

        @always_inline
        fn fused_backward_kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            dW: LayoutTensor[
                dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
            ],
            db: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_fused_kernel_impl[BATCH](
                grad_input,
                dW,
                db,
                grad_output,
                W,
                cache,
            )

        # Single kernel launch for all backward computations
        ctx.enqueue_function[
            fused_backward_kernel_wrapper, fused_backward_kernel_wrapper
        ](
            grad_input,
            dW,
            db,
            grad_output,
            W,
            cache,
            grid_dim=(fused_grid_x, fused_grid_y),
            block_dim=(TILE, TILE),
        )

    # =========================================================================
    # GPU Workspace Methods (for Sequential compatibility)
    # Linear is a leaf layer, so workspace is unused - just delegate to regular methods.
    # =========================================================================

    @staticmethod
    fn forward_gpu_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],  # Unused for Linear
    ) raises:
        """GPU forward with workspace (workspace unused for Linear)."""
        Self.forward_gpu[BATCH](
            ctx,
            output_buf,
            input_buf,
            params_buf,
            cache_buf,
        )

    @staticmethod
    fn forward_gpu_no_cache_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],  # Unused for Linear
    ) raises:
        """GPU forward without cache, with workspace (workspace unused for Linear).
        """
        Self.forward_gpu_no_cache[BATCH](
            ctx,
            output_buf,
            input_buf,
            params_buf,
        )

    @staticmethod
    fn backward_gpu_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],  # Unused for Linear
    ) raises:
        """GPU backward with workspace (workspace unused for Linear)."""
        Self.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )

    fn backward[
        BATCH: Int
    ](
        self,
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
