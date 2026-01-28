from math import tanh
from ..constants import dtype, TILE, TPB
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace


struct LinearTanh[in_dim: Int, out_dim: Int](Model):
    """Fused Linear + Tanh layer: y = tanh(x @ W + b).

    This fused layer eliminates:
    - 1 kernel launch (Linear + Tanh -> single kernel)
    - 1 global memory write/read (Linear output -> Tanh input)

    Parameters and gradients layout (same as Linear):
    - params: [W_flat (in_dim * out_dim) | b (out_dim)]
    - grads: [dW_flat (in_dim * out_dim) | db (out_dim)]

    Cache layout:
    - cache: [input (in_dim) | output (out_dim)] per sample
    - input is needed for dW computation
    - output is needed for Tanh backward: d/dx tanh(x) = 1 - tanh²(x) = 1 - output²

    PARAM_SIZE = in_dim * out_dim + out_dim
    CACHE_SIZE = in_dim + out_dim (input + output per sample)
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.IN_DIM * Self.OUT_DIM + Self.OUT_DIM
    comptime CACHE_SIZE: Int = Self.IN_DIM + Self.OUT_DIM  # input + output
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0  # Leaf layer, no workspace needed

    fn __init__(out self):
        """Initialize stateless LinearTanh layer."""
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
        """Forward pass: output = tanh(input @ W + b).

        Caches input and output for backward pass.

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [W_flat | b].
            cache: Cache buffer [BATCH, IN_DIM + OUT_DIM] for backward pass.
        """
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var b_offset = Self.in_dim * Self.out_dim

        for batch in range(BATCH):
            # Cache input for dW computation
            for i in range(Self.in_dim):
                cache[batch, i] = input[batch, i]

            # Compute y = tanh(x @ W + b)
            for j in range(Self.out_dim):
                var acc = params[b_offset + j]  # bias
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                # Apply Tanh and cache output for backward
                var tanh_out = tanh(acc)
                cache[batch, Self.in_dim + j] = tanh_out
                output[batch, j] = tanh_out

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
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var b_offset = Self.in_dim * Self.out_dim

        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = params[b_offset + j]  # bias
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                # Apply Tanh inline (no caching)
                output[batch, j] = tanh(acc)

    # =========================================================================
    # GPU Kernel Implementations
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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Fused forward kernel: y = tanh(x @ W + b) with caching.

        Uses tiled matrix multiplication with shared memory.
        Tanh is applied inline after computing each output element.

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

        # Start with bias
        var acc: output.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        # Tiled matmul with input caching
        for tile_idx in range((Self.IN_DIM + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col

            # Load x tile and cache input
            if global_row < BATCH and x_col < Self.IN_DIM:
                var x_val = input[global_row, x_col]
                x_shared[local_row, local_col] = x_val
                # Cache input for dW computation (first IN_DIM elements of cache)
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

        # Write result with fused Tanh
        if global_row < BATCH and global_col < Self.OUT_DIM:
            # Apply Tanh
            var tanh_out = tanh(acc)
            # Cache output for Tanh backward (after IN_DIM in cache)
            cache[global_row, Self.IN_DIM + global_col] = tanh_out
            output[global_row, global_col] = tanh_out

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
        """Fused forward kernel without caching (for inference).

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

        for tile_idx in range((Self.IN_DIM + TILE - 1) // TILE):
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

        # Apply Tanh inline (no caching)
        if global_row < BATCH and global_col < Self.OUT_DIM:
            output[global_row, global_col] = tanh(acc)

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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Fused backward kernel: computes dx, dW, and db with Tanh gradient.

        The Tanh gradient is: d/dx tanh(x) = 1 - tanh²(x) = 1 - output²
        We cached the output, so we compute (1 - output²) * grad_output.

        Cache layout: [input (IN_DIM) | output (OUT_DIM)] per sample

        Grid partitioning (same as Linear fused backward):
        - Rows [0, dx_grid_y): blocks compute grad_input
        - Rows [dx_grid_y, dx_grid_y + dW_grid_y): blocks compute dW
        - db is computed by dW blocks in the first row

        Grid: (max(dx_grid_x, dW_grid_x), dx_grid_y + dW_grid_y)
        Block: (TILE, TILE)
        """
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

        # Shared memory
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

        if block_y < dx_grid_y:
            # ================================================================
            # dx computation: grad_input = (grad_output * tanh_grad) @ W.T
            # where tanh_grad = 1 - output²
            # ================================================================
            if block_x >= dx_grid_x:
                return

            var global_row = block_y * TILE + local_row  # BATCH dimension
            var global_col = block_x * TILE + local_col  # IN_DIM dimension

            var acc: grad_input.element_type = 0

            for tile_idx in range((Self.OUT_DIM + TILE - 1) // TILE):
                # Load grad_output * tanh_grad into shared_A
                var dy_col = tile_idx * TILE + local_col
                if global_row < BATCH and dy_col < Self.OUT_DIM:
                    var grad_val = grad_output[global_row, dy_col]
                    # Get cached output (after IN_DIM)
                    var tanh_out = cache[global_row, Self.IN_DIM + dy_col]
                    # Tanh gradient: 1 - tanh²(x) = 1 - output²
                    var tanh_grad = 1 - tanh_out * tanh_out
                    shared_A[local_row, local_col] = grad_val * tanh_grad
                else:
                    shared_A[local_row, local_col] = 0

                # Load W.T tile
                var W_col = tile_idx * TILE + local_row
                if W_col < Self.OUT_DIM and global_col < Self.IN_DIM:
                    shared_B[local_row, local_col] = W[global_col, W_col]
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                @parameter
                for k in range(TILE):
                    acc += shared_A[local_row, k] * shared_B[k, local_col]

                barrier()

            if global_row < BATCH and global_col < Self.IN_DIM:
                grad_input[global_row, global_col] = acc

        else:
            # ================================================================
            # dW computation: dW = input.T @ (grad_output * tanh_grad)
            # Also computes db for the first row of dW blocks
            # ================================================================
            var dW_block_y = block_y - dx_grid_y
            var dW_block_x = block_x

            if dW_block_y >= dW_grid_y or dW_block_x >= dW_grid_x:
                return

            var global_row = dW_block_y * TILE + local_row  # IN_DIM dimension
            var global_col = dW_block_x * TILE + local_col  # OUT_DIM dimension

            var dW_acc: dW.element_type = 0
            var db_acc: db.element_type = 0

            var num_tiles = (BATCH + TILE - 1) // TILE
            for tile_idx in range(num_tiles):
                # Load input.T tile (from first IN_DIM elements of cache)
                var batch_idx = tile_idx * TILE + local_col
                if global_row < Self.IN_DIM and batch_idx < BATCH:
                    shared_A[local_row, local_col] = cache[batch_idx, global_row]
                else:
                    shared_A[local_row, local_col] = 0

                # Load grad_output * tanh_grad tile
                var dy_row = tile_idx * TILE + local_row
                if dy_row < BATCH and global_col < Self.OUT_DIM:
                    var grad_val = grad_output[dy_row, global_col]
                    var tanh_out = cache[dy_row, Self.IN_DIM + global_col]
                    var tanh_grad = 1 - tanh_out * tanh_out
                    var scaled_grad = grad_val * tanh_grad
                    shared_B[local_row, local_col] = scaled_grad
                    # Accumulate for db (only first row of dW blocks)
                    if dW_block_y == 0:
                        db_acc += scaled_grad
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                @parameter
                for k in range(TILE):
                    dW_acc += shared_A[local_row, k] * shared_B[k, local_col]

                barrier()

            # Write dW result
            if global_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                dW[global_row, global_col] = dW_acc

            # Compute and write db using shared memory reduction
            if dW_block_y == 0 and global_col < Self.OUT_DIM:
                shared_A[local_row, local_col] = db_acc
                barrier()

                if local_row == 0:
                    var total = shared_A[0, local_col]
                    @parameter
                    for r in range(1, TILE):
                        total += shared_A[r, local_col]
                    db[global_col] = total

    # =========================================================================
    # GPU Launchers
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
        """Launch fused forward pass on GPU with caching."""
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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ](cache_buf.unsafe_ptr())

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
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH](output, input, W, b, cache)

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
        """Launch fused forward pass on GPU without caching (inference)."""
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
            Self.forward_kernel_impl_no_cache[BATCH](output, input, W, b)

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
        """Launch fused backward pass on GPU.

        Computes all gradients in a SINGLE kernel launch with Tanh gradient.
        """
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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var grads_ptr = grads_buf.unsafe_ptr()
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ](grads_ptr)
        var db_ptr = grads_ptr + Self.IN_DIM * Self.OUT_DIM
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ](db_ptr)

        comptime dx_grid_x = (Self.IN_DIM + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE
        comptime dW_grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime dW_grid_y = (Self.IN_DIM + TILE - 1) // TILE

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
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_fused_kernel_impl[BATCH](
                grad_input, dW, db, grad_output, W, cache
            )

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
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward with workspace (workspace unused for LinearTanh)."""
        Self.forward_gpu[BATCH](
            ctx, output_buf, input_buf, params_buf, cache_buf
        )

    @staticmethod
    fn forward_gpu_no_cache_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward without cache, with workspace."""
        Self.forward_gpu_no_cache[BATCH](ctx, output_buf, input_buf, params_buf)

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
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward with workspace (workspace unused for LinearTanh)."""
        Self.backward_gpu[BATCH](
            ctx, grad_input_buf, grad_output_buf, params_buf, cache_buf, grads_buf
        )

    # =========================================================================
    # CPU Backward (for reference/testing)
    # =========================================================================

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
        """Backward pass with fused Tanh gradient.

        Cache layout: [input (IN_DIM) | output (OUT_DIM)] per sample
        Tanh gradient: d/dx tanh(x) = 1 - tanh²(x) = 1 - output²
        """
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grads.ptr)
        var db_offset = Self.in_dim * Self.out_dim

        for batch in range(BATCH):
            # dx = (dy * tanh_grad) @ W.T
            for i in range(Self.in_dim):
                var acc: grad_input.element_type = 0
                for j in range(Self.out_dim):
                    var tanh_out = cache[batch, Self.in_dim + j]
                    var tanh_grad = 1 - tanh_out * tanh_out
                    var scaled_grad = grad_output[batch, j] * tanh_grad
                    acc += scaled_grad * W[i, j]
                grad_input[batch, i] = acc

            # dW += input.T @ (dy * tanh_grad)
            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    var tanh_out = cache[batch, Self.in_dim + j]
                    var tanh_grad = 1 - tanh_out * tanh_out
                    var scaled_grad = grad_output[batch, j] * tanh_grad
                    var cached_input = cache[batch, i]
                    dW[i, j] = dW[i, j] + cached_input * scaled_grad

            # db += sum(dy * tanh_grad, axis=0)
            for j in range(Self.out_dim):
                var tanh_out = cache[batch, Self.in_dim + j]
                var tanh_grad = 1 - tanh_out * tanh_out
                var scaled_grad = grad_output[batch, j] * tanh_grad
                grads[db_offset + j] = grads[db_offset + j] + scaled_grad
