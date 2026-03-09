"""LinearTanh layer using optimized common matmul building blocks.

This is a refactored version of LinearTanh that uses the shared matmul
kernels from nn.gpu.matmul_ops for better maintainability and
optimized performance on Apple Silicon.
"""

from std.math import tanh
from ..constants import dtype, TILE
from .model import Model

# Minimum tanh gradient to prevent gradient death at saturation.
# At full saturation tanh(x)=±1, (1-tanh²)=0 → actor can never recover.
# Floor of 0.01 ensures ~1% gradient signal even when fully saturated.
comptime TANH_GRAD_FLOOR: Scalar[dtype] = 0.01
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace

# Import optimized matmul building blocks
from ..gpu.matmul_ops import (
    TILE_APPLE,
    matmul_bias_tanh_cached_kernel,
    matmul_bias_tanh_kernel,
)


struct LinearTanh[in_dim: Int, out_dim: Int](Model):
    """Fused Linear + Tanh layer using optimized matmul ops: y = tanh(x @ W + b).

    This version uses shared matmul kernels from matmul_ops.mojo for:
    - Better maintainability (single source of truth for matmul)
    - Optimized 8x8 tiles for Apple Silicon
    - Consistent performance across all model layers

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

    fn __init__(out self, *, deinit take: Self):
        """Move constructor for Sequential composition."""
        pass

    fn __init__(out self, *, copy: Self):
        """Copy constructor for Copyable trait."""
        pass

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
        """Forward pass: output = tanh(input @ W + b).

        Caches input and output for backward pass.
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
                var tanh_out = tanh(acc)
                cache[batch, Self.in_dim + j] = tanh_out
                output[batch, j] = tanh_out

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
        """Forward pass without caching (for inference)."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var b_offset = Self.in_dim * Self.out_dim

        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = params[b_offset + j]
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                output[batch, j] = tanh(acc)

    # =========================================================================
    # GPU Kernel Implementations - Using Common Matmul Ops
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
        """Fused forward kernel using optimized matmul_bias_tanh_cached_kernel.

        Grid: ((OUT_DIM + TILE_APPLE - 1) // TILE_APPLE, (BATCH + TILE_APPLE - 1) // TILE_APPLE)
        Block: (TILE_APPLE, TILE_APPLE)
        """
        # Delegate to the optimized common kernel
        matmul_bias_tanh_cached_kernel[
            BATCH, Self.IN_DIM, Self.OUT_DIM, Self.CACHE_SIZE, TILE_APPLE
        ](output, input, W, b, cache)

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
        """Fused forward kernel using optimized matmul_bias_tanh_kernel.

        Grid: ((OUT_DIM + TILE_APPLE - 1) // TILE_APPLE, (BATCH + TILE_APPLE - 1) // TILE_APPLE)
        Block: (TILE_APPLE, TILE_APPLE)
        """
        # Delegate to the optimized common kernel
        matmul_bias_tanh_kernel[BATCH, Self.IN_DIM, Self.OUT_DIM, TILE_APPLE](
            output, input, W, b
        )

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
        """Fused backward kernel with Tanh gradient.

        Note: This kernel cannot use the base matmul ops directly because
        it needs to apply the tanh gradient inline. However, it uses the
        same 8x8 tile size for Apple Silicon optimization.

        Grid: (max(dx_grid_x, dW_grid_x), dx_grid_y + dW_grid_y)
        Block: (TILE_APPLE, TILE_APPLE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var block_y = Int(block_idx.y)
        var block_x = Int(block_idx.x)

        # Use optimized tile size for Apple Silicon
        comptime TILE = TILE_APPLE

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
            # dx computation: grad_input = (grad_output * tanh_grad) @ W.T
            if block_x >= dx_grid_x:
                return

            var global_row = block_y * TILE + local_row
            var global_col = block_x * TILE + local_col

            var acc: grad_input.element_type = 0

            for tile_idx in range((Self.OUT_DIM + TILE - 1) // TILE):
                var dy_col = tile_idx * TILE + local_col
                if global_row < BATCH and dy_col < Self.OUT_DIM:
                    var grad_val = grad_output[global_row, dy_col]
                    var tanh_out = cache[global_row, Self.IN_DIM + dy_col]
                    var tanh_grad = 1 - tanh_out * tanh_out
                    if tanh_grad < TANH_GRAD_FLOOR:
                        tanh_grad = TANH_GRAD_FLOOR
                    shared_A[local_row, local_col] = grad_val * tanh_grad
                else:
                    shared_A[local_row, local_col] = 0

                var W_col = tile_idx * TILE + local_row
                if W_col < Self.OUT_DIM and global_col < Self.IN_DIM:
                    shared_B[local_row, local_col] = W[global_col, W_col]
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    acc += rebind[grad_input.element_type](
                        shared_A[local_row, k]
                    ) * rebind[grad_input.element_type](shared_B[k, local_col])

                barrier()

            if global_row < BATCH and global_col < Self.IN_DIM:
                grad_input[global_row, global_col] = acc

        else:
            # dW computation: dW = input.T @ (grad_output * tanh_grad)
            var dW_block_y = block_y - dx_grid_y
            var dW_block_x = block_x

            if dW_block_y >= dW_grid_y or dW_block_x >= dW_grid_x:
                return

            var global_row = dW_block_y * TILE + local_row
            var global_col = dW_block_x * TILE + local_col

            var dW_acc: dW.element_type = 0
            var db_acc: db.element_type = 0

            var num_tiles = (BATCH + TILE - 1) // TILE
            for tile_idx in range(num_tiles):
                var batch_idx = tile_idx * TILE + local_col
                if global_row < Self.IN_DIM and batch_idx < BATCH:
                    shared_A[local_row, local_col] = cache[
                        batch_idx, global_row
                    ]
                else:
                    shared_A[local_row, local_col] = 0

                var dy_row = tile_idx * TILE + local_row
                if dy_row < BATCH and global_col < Self.OUT_DIM:
                    var grad_val = grad_output[dy_row, global_col]
                    var tanh_out = cache[dy_row, Self.IN_DIM + global_col]
                    var tanh_grad = 1 - tanh_out * tanh_out
                    if tanh_grad < TANH_GRAD_FLOOR:
                        tanh_grad = TANH_GRAD_FLOOR
                    var scaled_grad = grad_val * tanh_grad
                    shared_B[local_row, local_col] = scaled_grad
                    if dW_block_y == 0:
                        db_acc += scaled_grad
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    dW_acc += rebind[dW.element_type](
                        shared_A[local_row, k]
                    ) * rebind[dW.element_type](shared_B[k, local_col])

                barrier()

            if global_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                dW[global_row, global_col] = dW_acc

            if dW_block_y == 0 and global_col < Self.OUT_DIM:
                shared_A[local_row, local_col] = db_acc
                barrier()

                if local_row == 0:
                    var total = shared_A[0, local_col]

                    for r in range(1, TILE):
                        total += rebind[db.element_type](shared_A[r, local_col])
                    db[global_col] = total

    # =========================================================================
    # GPU Launchers - Using TILE_APPLE for grid calculations
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
        """Launch fused forward pass on GPU with caching."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self.IN_DIM * Self.OUT_DIM)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        comptime grid_x = (Self.OUT_DIM + TILE_APPLE - 1) // TILE_APPLE
        comptime grid_y = (BATCH + TILE_APPLE - 1) // TILE_APPLE

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
            input_immut,
            W,
            b,
            cache,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE_APPLE, TILE_APPLE),
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
        """Launch fused forward pass on GPU without caching (inference)."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self.IN_DIM * Self.OUT_DIM)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        comptime grid_x = (Self.OUT_DIM + TILE_APPLE - 1) // TILE_APPLE
        comptime grid_y = (BATCH + TILE_APPLE - 1) // TILE_APPLE

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
            block_dim=(TILE_APPLE, TILE_APPLE),
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
        """Launch fused backward pass on GPU."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
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

        comptime dx_grid_x = (Self.IN_DIM + TILE_APPLE - 1) // TILE_APPLE
        comptime dx_grid_y = (BATCH + TILE_APPLE - 1) // TILE_APPLE
        comptime dW_grid_x = (Self.OUT_DIM + TILE_APPLE - 1) // TILE_APPLE
        comptime dW_grid_y = (Self.IN_DIM + TILE_APPLE - 1) // TILE_APPLE

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
            grad_output_immut,
            W,
            cache_immut,
            grid_dim=(fused_grid_x, fused_grid_y),
            block_dim=(TILE_APPLE, TILE_APPLE),
        )

    # =========================================================================
    # CPU Backward (for reference/testing)
    # =========================================================================

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
        """Backward pass with fused Tanh gradient."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grads.ptr)
        var db_offset = Self.in_dim * Self.out_dim

        for batch in range(BATCH):
            for i in range(Self.in_dim):
                var acc: grad_input.element_type = 0
                for j in range(Self.out_dim):
                    var tanh_out = cache[batch, Self.in_dim + j]
                    var tanh_grad = 1 - tanh_out * tanh_out
                    if tanh_grad < TANH_GRAD_FLOOR:
                        tanh_grad = TANH_GRAD_FLOOR
                    var scaled_grad = grad_output[batch, j] * tanh_grad
                    acc += scaled_grad * W[i, j]
                grad_input[batch, i] = acc

            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    var tanh_out = cache[batch, Self.in_dim + j]
                    var tanh_grad = 1 - tanh_out * tanh_out
                    if tanh_grad < TANH_GRAD_FLOOR:
                        tanh_grad = TANH_GRAD_FLOOR
                    var scaled_grad = grad_output[batch, j] * tanh_grad
                    var cached_input = cache[batch, i]
                    dW[i, j] = dW[i, j] + cached_input * scaled_grad

            for j in range(Self.out_dim):
                var tanh_out = cache[batch, Self.in_dim + j]
                var tanh_grad = 1 - tanh_out * tanh_out
                if tanh_grad < TANH_GRAD_FLOOR:
                    tanh_grad = TANH_GRAD_FLOOR
                var scaled_grad = grad_output[batch, j] * tanh_grad
                grads[db_offset + j] = grads[db_offset + j] + scaled_grad
