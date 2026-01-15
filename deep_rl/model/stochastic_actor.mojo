from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from math import exp, log, sqrt

# =============================================================================
# GPU Constants
# =============================================================================

comptime TILE = 16  # Tile size for matmul kernels
comptime TPB = 256  # Threads per block for elementwise ops

# =============================================================================
# Constants for numerical stability
# =============================================================================

comptime LOG_STD_MIN: Float64 = -20.0  # Minimum log_std to prevent exp underflow
comptime LOG_STD_MAX: Float64 = 2.0  # Maximum log_std to prevent exp overflow
comptime EPS: Float64 = 1e-6  # Small constant for numerical stability


struct StochasticActor[in_dim: Int, action_dim: Int](Model):
    """Stochastic Actor (Gaussian Policy) for continuous action spaces.

    This layer implements a diagonal Gaussian policy with learned mean and log_std.
    It's designed for use with SAC, PPO, and other policy gradient algorithms.

    Architecture:
        - Input: features [BATCH, in_dim] (typically from a backbone network)
        - Output: [mean, log_std] concatenated as [BATCH, action_dim * 2]
        - Two separate linear heads: mean_head and log_std_head

    Parameters layout:
        [W_mean (in_dim * action_dim) | b_mean (action_dim) |
         W_log_std (in_dim * action_dim) | b_log_std (action_dim)]

    Cache layout:
        [input (in_dim)] - caches input for backward pass

    Usage with reparameterization trick:
        1. Forward pass: output = [mean, log_std]
        2. Sample noise: epsilon ~ N(0, 1)
        3. Compute pre-tanh action: z = mean + exp(log_std) * epsilon
        4. Apply tanh squashing: action = tanh(z)
        5. Compute log_prob with squashing correction:
           log_prob = log_normal(z; mean, std) - sum(log(1 - tanh(z)^2 + eps))

    Utility methods are provided for sampling and log probability computation.
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.action_dim * 2  # mean and log_std concatenated
    # Two linear layers: mean_head and log_std_head, each with W and b
    comptime PARAM_SIZE: Int = 2 * (
        Self.in_dim * Self.action_dim + Self.action_dim
    )
    comptime CACHE_SIZE: Int = Self.in_dim  # Cache input for backward pass
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0  # Leaf layer

    fn __init__(out self):
        """Initialize StochasticActor."""
        pass

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor for Sequential composition."""
        pass

    fn __copyinit__(out self, other: Self):
        """Copy constructor for Copyable trait."""
        pass

    # =========================================================================
    # Helper: Get parameter offsets
    # =========================================================================

    @always_inline
    @staticmethod
    fn _mean_W_offset() -> Int:
        """Offset for mean head weights."""
        return 0

    @always_inline
    @staticmethod
    fn _mean_b_offset() -> Int:
        """Offset for mean head bias."""
        return Self.in_dim * Self.action_dim

    @always_inline
    @staticmethod
    fn _log_std_W_offset() -> Int:
        """Offset for log_std head weights."""
        return Self.in_dim * Self.action_dim + Self.action_dim

    @always_inline
    @staticmethod
    fn _log_std_b_offset() -> Int:
        """Offset for log_std head bias."""
        return 2 * Self.in_dim * Self.action_dim + Self.action_dim

    # =========================================================================
    # Forward passes
    # =========================================================================

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
        """Forward pass: compute mean and log_std from input features.

        Output layout: [mean (action_dim) | log_std (action_dim)]
        log_std is clamped to [LOG_STD_MIN, LOG_STD_MAX] for numerical stability.

        Args:
            input: Input features [BATCH, in_dim].
            output: Output tensor [BATCH, action_dim * 2] = [mean | log_std].
            params: Model parameters.
            cache: Cache buffer [BATCH, in_dim] for backward pass.
        """
        # Create views for weights
        var W_mean = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](params.ptr + Self._mean_W_offset())
        var W_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](params.ptr + Self._log_std_W_offset())

        for batch in range(BATCH):
            # Cache input for backward
            for i in range(Self.in_dim):
                cache[batch, i] = input[batch, i]

            # Compute mean = input @ W_mean + b_mean
            for j in range(Self.action_dim):
                var mean_acc = params[Self._mean_b_offset() + j]  # bias
                for i in range(Self.in_dim):
                    mean_acc += input[batch, i] * W_mean[i, j]
                output[batch, j] = mean_acc

            # Compute log_std = clamp(input @ W_log_std + b_log_std)
            for j in range(Self.action_dim):
                var log_std_acc = params[Self._log_std_b_offset() + j]  # bias
                for i in range(Self.in_dim):
                    log_std_acc += input[batch, i] * W_log_std[i, j]
                # Clamp log_std for numerical stability
                var log_std_val = Float64(rebind[Scalar[dtype]](log_std_acc))
                if log_std_val < LOG_STD_MIN:
                    log_std_val = LOG_STD_MIN
                elif log_std_val > LOG_STD_MAX:
                    log_std_val = LOG_STD_MAX
                output[batch, Self.action_dim + j] = Scalar[dtype](log_std_val)

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
        """Forward pass without caching (for inference)."""
        var W_mean = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](params.ptr + Self._mean_W_offset())
        var W_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](params.ptr + Self._log_std_W_offset())

        for batch in range(BATCH):
            # Compute mean
            for j in range(Self.action_dim):
                var mean_acc = params[Self._mean_b_offset() + j]
                for i in range(Self.in_dim):
                    mean_acc += input[batch, i] * W_mean[i, j]
                output[batch, j] = mean_acc

            # Compute clamped log_std
            for j in range(Self.action_dim):
                var log_std_acc = params[Self._log_std_b_offset() + j]
                for i in range(Self.in_dim):
                    log_std_acc += input[batch, i] * W_log_std[i, j]
                var log_std_val = Float64(rebind[Scalar[dtype]](log_std_acc))
                if log_std_val < LOG_STD_MIN:
                    log_std_val = LOG_STD_MIN
                elif log_std_val > LOG_STD_MAX:
                    log_std_val = LOG_STD_MAX
                output[batch, Self.action_dim + j] = Scalar[dtype](log_std_val)

    # =========================================================================
    # Backward pass
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
        """Backward pass: compute gradients for both mean and log_std heads.

        Args:
            grad_output: Gradient w.r.t. output [BATCH, action_dim * 2].
            grad_input: Gradient w.r.t. input [BATCH, in_dim] (written).
            params: Model parameters.
            cache: Cached input from forward pass [BATCH, in_dim].
            grads: Parameter gradients (accumulated).
        """
        # Create views for weights
        var W_mean = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](params.ptr + Self._mean_W_offset())
        var W_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](params.ptr + Self._log_std_W_offset())

        # Create views for gradient storage
        var dW_mean = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](grads.ptr + Self._mean_W_offset())
        var dW_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](grads.ptr + Self._log_std_W_offset())

        for batch in range(BATCH):
            # Initialize grad_input to zero
            for i in range(Self.in_dim):
                grad_input[batch, i] = 0

            # Backward through mean head: dx += dy_mean @ W_mean.T
            for i in range(Self.in_dim):
                for j in range(Self.action_dim):
                    grad_input[batch, i] = (
                        grad_input[batch, i]
                        + grad_output[batch, j] * W_mean[i, j]
                    )

            # Backward through log_std head: dx += dy_log_std @ W_log_std.T
            for i in range(Self.in_dim):
                for j in range(Self.action_dim):
                    grad_input[batch, i] = (
                        grad_input[batch, i]
                        + grad_output[batch, Self.action_dim + j]
                        * W_log_std[i, j]
                    )

            # Accumulate dW_mean = x.T @ dy_mean
            for i in range(Self.in_dim):
                for j in range(Self.action_dim):
                    dW_mean[i, j] = (
                        dW_mean[i, j] + cache[batch, i] * grad_output[batch, j]
                    )

            # Accumulate db_mean
            for j in range(Self.action_dim):
                grads[Self._mean_b_offset() + j] = (
                    grads[Self._mean_b_offset() + j] + grad_output[batch, j]
                )

            # Accumulate dW_log_std = x.T @ dy_log_std
            for i in range(Self.in_dim):
                for j in range(Self.action_dim):
                    dW_log_std[i, j] = (
                        dW_log_std[i, j]
                        + cache[batch, i]
                        * grad_output[batch, Self.action_dim + j]
                    )

            # Accumulate db_log_std
            for j in range(Self.action_dim):
                grads[Self._log_std_b_offset() + j] = (
                    grads[Self._log_std_b_offset() + j]
                    + grad_output[batch, Self.action_dim + j]
                )

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
        W_mean: LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ],
        b_mean: LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ],
        W_log_std: LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ],
        b_log_std: LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """Forward kernel: compute mean and log_std with tiled matmul.

        Uses separate tiles for mean and log_std computation.
        Grid: ((action_dim + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
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

        # Compute mean = input @ W_mean + b_mean
        var mean_acc: output.element_type = 0
        if global_col < Self.action_dim:
            mean_acc = b_mean[global_col]

        for tile_idx in range(0, (Self.in_dim + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col

            # Load input tile and cache
            if global_row < BATCH and x_col < Self.in_dim:
                var x_val = input[global_row, x_col]
                x_shared[local_row, local_col] = x_val
                cache[global_row, x_col] = x_val
            else:
                x_shared[local_row, local_col] = 0

            # Load W_mean tile
            var W_row = tile_idx * TILE + local_row
            if W_row < Self.in_dim and global_col < Self.action_dim:
                W_shared[local_row, local_col] = W_mean[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                mean_acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        # Compute log_std = input @ W_log_std + b_log_std
        var log_std_acc: output.element_type = 0
        if global_col < Self.action_dim:
            log_std_acc = b_log_std[global_col]

        for tile_idx in range(0, (Self.in_dim + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col

            # Load input tile (already in x_shared from mean computation in some cases)
            if global_row < BATCH and x_col < Self.in_dim:
                x_shared[local_row, local_col] = input[global_row, x_col]
            else:
                x_shared[local_row, local_col] = 0

            # Load W_log_std tile
            var W_row = tile_idx * TILE + local_row
            if W_row < Self.in_dim and global_col < Self.action_dim:
                W_shared[local_row, local_col] = W_log_std[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                log_std_acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        # Write results with clamping for log_std
        if global_row < BATCH and global_col < Self.action_dim:
            output[global_row, global_col] = mean_acc

            # Clamp log_std
            var log_std_val = rebind[Scalar[DType.float32]](log_std_acc)
            if log_std_val < Scalar[DType.float32](LOG_STD_MIN):
                log_std_val = Scalar[DType.float32](LOG_STD_MIN)
            elif log_std_val > Scalar[DType.float32](LOG_STD_MAX):
                log_std_val = Scalar[DType.float32](LOG_STD_MAX)
            output[global_row, Self.action_dim + global_col] = rebind[
                output.element_type
            ](log_std_val)

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
        W_mean: LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ],
        b_mean: LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ],
        W_log_std: LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ],
        b_log_std: LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ],
    ):
        """Forward kernel without caching (for inference)."""
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

        # Compute mean
        var mean_acc: output.element_type = 0
        if global_col < Self.action_dim:
            mean_acc = b_mean[global_col]

        for tile_idx in range(0, (Self.in_dim + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col
            if global_row < BATCH and x_col < Self.in_dim:
                x_shared[local_row, local_col] = input[global_row, x_col]
            else:
                x_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.in_dim and global_col < Self.action_dim:
                W_shared[local_row, local_col] = W_mean[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                mean_acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        # Compute log_std
        var log_std_acc: output.element_type = 0
        if global_col < Self.action_dim:
            log_std_acc = b_log_std[global_col]

        for tile_idx in range(0, (Self.in_dim + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col
            if global_row < BATCH and x_col < Self.in_dim:
                x_shared[local_row, local_col] = input[global_row, x_col]
            else:
                x_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.in_dim and global_col < Self.action_dim:
                W_shared[local_row, local_col] = W_log_std[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                log_std_acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        if global_row < BATCH and global_col < Self.action_dim:
            output[global_row, global_col] = mean_acc

            var log_std_val = rebind[Scalar[DType.float32]](log_std_acc)
            if log_std_val < Scalar[DType.float32](LOG_STD_MIN):
                log_std_val = Scalar[DType.float32](LOG_STD_MIN)
            elif log_std_val > Scalar[DType.float32](LOG_STD_MAX):
                log_std_val = Scalar[DType.float32](LOG_STD_MAX)
            output[global_row, Self.action_dim + global_col] = rebind[
                output.element_type
            ](log_std_val)

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
        """Launch forward pass on GPU with caching."""
        var params_ptr = params_buf.unsafe_ptr()

        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var W_mean = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ](params_ptr + Self._mean_W_offset())
        var b_mean = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ](params_ptr + Self._mean_b_offset())
        var W_log_std = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ](params_ptr + Self._log_std_W_offset())
        var b_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ](params_ptr + Self._log_std_b_offset())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ](cache_buf.unsafe_ptr())

        comptime grid_x = (Self.action_dim + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            W_mean: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                ImmutAnyOrigin,
            ],
            b_mean: LayoutTensor[
                dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
            ],
            W_log_std: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                ImmutAnyOrigin,
            ],
            b_log_std: LayoutTensor[
                dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
            ],
            batch_size: Int,
        ):
            Self.forward_kernel_impl[BATCH](
                output,
                input,
                W_mean,
                b_mean,
                W_log_std,
                b_log_std,
                cache,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            W_mean,
            b_mean,
            W_log_std,
            b_log_std,
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
        """Launch forward pass on GPU without caching (for inference)."""
        var params_ptr = params_buf.unsafe_ptr()

        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var W_mean = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ](params_ptr + Self._mean_W_offset())
        var b_mean = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ](params_ptr + Self._mean_b_offset())
        var W_log_std = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ](params_ptr + Self._log_std_W_offset())
        var b_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
        ](params_ptr + Self._log_std_b_offset())

        comptime grid_x = (Self.action_dim + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            W_mean: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                ImmutAnyOrigin,
            ],
            b_mean: LayoutTensor[
                dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
            ],
            W_log_std: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                ImmutAnyOrigin,
            ],
            b_log_std: LayoutTensor[
                dtype, Layout.row_major(Self.action_dim), ImmutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl_no_cache[BATCH](
                output, input, W_mean, b_mean, W_log_std, b_log_std
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            W_mean,
            b_mean,
            W_log_std,
            b_log_std,
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
        """Launch backward pass on GPU.

        Computes gradients for both mean and log_std heads.
        """
        from gpu import block

        var params_ptr = params_buf.unsafe_ptr()
        var grads_ptr = grads_buf.unsafe_ptr()

        var grad_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](grad_input_buf.unsafe_ptr())
        var grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var W_mean = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ](params_ptr + Self._mean_W_offset())
        var W_log_std = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ](params_ptr + Self._log_std_W_offset())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var dW_mean = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](grads_ptr + Self._mean_W_offset())
        var db_mean = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
        ](grads_ptr + Self._mean_b_offset())
        var dW_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ](grads_ptr + Self._log_std_W_offset())
        var db_log_std = LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
        ](grads_ptr + Self._log_std_b_offset())

        # Kernel 1: dx = dy_mean @ W_mean.T + dy_log_std @ W_log_std.T
        comptime dx_grid_x = (Self.IN_DIM + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn dx_kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            W_mean: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                ImmutAnyOrigin,
            ],
            W_log_std: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_dx_kernel_impl[BATCH](
                grad_input, grad_output, W_mean, W_log_std
            )

        ctx.enqueue_function[dx_kernel_wrapper, dx_kernel_wrapper](
            grad_input,
            grad_output,
            W_mean,
            W_log_std,
            grid_dim=(dx_grid_x, dx_grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 2: dW_mean = x.T @ dy_mean
        comptime dW_grid_x = (Self.action_dim + TILE - 1) // TILE
        comptime dW_grid_y = (Self.in_dim + TILE - 1) // TILE

        @always_inline
        fn dW_mean_kernel_wrapper(
            dW_mean: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                MutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_dW_mean_kernel_impl[BATCH](
                dW_mean, cache, grad_output
            )

        ctx.enqueue_function[dW_mean_kernel_wrapper, dW_mean_kernel_wrapper](
            dW_mean,
            cache,
            grad_output,
            grid_dim=(dW_grid_x, dW_grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 3: dW_log_std = x.T @ dy_log_std
        @always_inline
        fn dW_log_std_kernel_wrapper(
            dW_log_std: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.action_dim),
                MutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_dW_log_std_kernel_impl[BATCH](
                dW_log_std, cache, grad_output
            )

        ctx.enqueue_function[
            dW_log_std_kernel_wrapper, dW_log_std_kernel_wrapper
        ](
            dW_log_std,
            cache,
            grad_output,
            grid_dim=(dW_grid_x, dW_grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 4: db_mean = sum(dy_mean, axis=0)
        @always_inline
        fn db_mean_kernel_wrapper(
            db_mean: LayoutTensor[
                dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_db_mean_kernel_impl[BATCH](db_mean, grad_output)

        ctx.enqueue_function[db_mean_kernel_wrapper, db_mean_kernel_wrapper](
            db_mean,
            grad_output,
            grid_dim=(Self.action_dim,),
            block_dim=(TPB,),
        )

        # Kernel 5: db_log_std = sum(dy_log_std, axis=0)
        @always_inline
        fn db_log_std_kernel_wrapper(
            db_log_std: LayoutTensor[
                dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_db_log_std_kernel_impl[BATCH](db_log_std, grad_output)

        ctx.enqueue_function[
            db_log_std_kernel_wrapper, db_log_std_kernel_wrapper
        ](
            db_log_std,
            grad_output,
            grid_dim=(Self.action_dim,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Backward GPU Kernel Implementations
    # =========================================================================

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
        W_mean: LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ],
        W_log_std: LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.action_dim),
            ImmutAnyOrigin,
        ],
    ):
        """Compute dx = dy_mean @ W_mean.T + dy_log_std @ W_log_std.T."""
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # BATCH
        var global_col = Int(block_idx.x) * TILE + local_col  # IN_DIM

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

        # First: dy_mean @ W_mean.T
        var acc: grad_input.element_type = 0

        for tile_idx in range(0, (Self.action_dim + TILE - 1) // TILE):
            var dy_col = tile_idx * TILE + local_col
            if global_row < BATCH and dy_col < Self.action_dim:
                dy_shared[local_row, local_col] = grad_output[
                    global_row, dy_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            var W_col = tile_idx * TILE + local_row
            if W_col < Self.action_dim and global_col < Self.IN_DIM:
                W_T_shared[local_row, local_col] = W_mean[global_col, W_col]
            else:
                W_T_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                acc += dy_shared[local_row, k] * W_T_shared[k, local_col]

            barrier()

        # Second: add dy_log_std @ W_log_std.T

        for tile_idx in range(0, (Self.action_dim + TILE - 1) // TILE):
            var dy_col = tile_idx * TILE + local_col
            if global_row < BATCH and dy_col < Self.action_dim:
                dy_shared[local_row, local_col] = grad_output[
                    global_row, Self.action_dim + dy_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            var W_col = tile_idx * TILE + local_row
            if W_col < Self.action_dim and global_col < Self.IN_DIM:
                W_T_shared[local_row, local_col] = W_log_std[global_col, W_col]
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
    fn backward_dW_mean_kernel_impl[
        BATCH: Int,
    ](
        dW_mean: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Compute dW_mean = x.T @ dy_mean."""
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # IN_DIM
        var global_col = Int(block_idx.x) * TILE + local_col  # action_dim

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

        var acc: dW_mean.element_type = 0

        for tile_idx in range(0, (BATCH + TILE - 1) // TILE):
            var batch_idx = tile_idx * TILE + local_col
            if global_row < Self.in_dim and batch_idx < BATCH:
                x_T_shared[local_row, local_col] = cache[batch_idx, global_row]
            else:
                x_T_shared[local_row, local_col] = 0

            var dy_row = tile_idx * TILE + local_row
            if dy_row < BATCH and global_col < Self.action_dim:
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

        if global_row < Self.in_dim and global_col < Self.action_dim:
            dW_mean[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn backward_dW_log_std_kernel_impl[
        BATCH: Int,
    ](
        dW_log_std: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.action_dim), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Compute dW_log_std = x.T @ dy_log_std."""
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row  # IN_DIM
        var global_col = Int(block_idx.x) * TILE + local_col  # action_dim

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

        var acc: dW_log_std.element_type = 0

        @parameter
        for tile_idx in range(0, (BATCH + TILE - 1) // TILE):
            var batch_idx = tile_idx * TILE + local_col
            if global_row < Self.in_dim and batch_idx < BATCH:
                x_T_shared[local_row, local_col] = cache[batch_idx, global_row]
            else:
                x_T_shared[local_row, local_col] = 0

            var dy_row = tile_idx * TILE + local_row
            if dy_row < BATCH and global_col < Self.action_dim:
                # Log_std gradient is in second half of output
                dy_shared[local_row, local_col] = grad_output[
                    dy_row, Self.action_dim + global_col
                ]
            else:
                dy_shared[local_row, local_col] = 0

            barrier()

            @parameter
            for k in range(TILE):
                acc += x_T_shared[local_row, k] * dy_shared[k, local_col]

            barrier()

        if global_row < Self.in_dim and global_col < Self.action_dim:
            dW_log_std[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn backward_db_mean_kernel_impl[
        BATCH: Int,
    ](
        db_mean: LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Compute db_mean = sum(dy_mean, axis=0)."""
        from gpu import block

        var col = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if col >= Self.action_dim:
            return

        var my_sum: db_mean.element_type = 0
        var batch_idx = local_i
        while batch_idx < BATCH:
            my_sum += grad_output[batch_idx, col]
            batch_idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_sum)

        if local_i == 0:
            db_mean[col] = total[0]

    @always_inline
    @staticmethod
    fn backward_db_log_std_kernel_impl[
        BATCH: Int,
    ](
        db_log_std: LayoutTensor[
            dtype, Layout.row_major(Self.action_dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Compute db_log_std = sum(dy_log_std, axis=0)."""
        from gpu import block

        var col = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if col >= Self.action_dim:
            return

        var my_sum: db_log_std.element_type = 0
        var batch_idx = local_i
        while batch_idx < BATCH:
            # Log_std gradient is in second half of output
            my_sum += grad_output[batch_idx, Self.action_dim + col]
            batch_idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_sum)

        if local_i == 0:
            db_log_std[col] = total[0]

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
        """GPU forward with workspace (workspace unused for StochasticActor)."""
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
        """GPU backward with workspace (workspace unused for StochasticActor).
        """
        Self.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )


# =============================================================================
# Utility Functions for Reparameterization Trick
# =============================================================================


fn rsample[
    BATCH: Int, action_dim: Int
](
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    noise: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut log_prob: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    """Reparameterized sample with log probability computation.

    Implements the reparameterization trick for Gaussian policies with tanh squashing:
        z = mean + exp(log_std) * noise
        action = tanh(z)
        log_prob = log_normal(z; mean, std) - sum(log(1 - tanh(z)^2 + eps))

    Args:
        mean: Mean of Gaussian [BATCH, action_dim].
        log_std: Log standard deviation [BATCH, action_dim].
        noise: Pre-sampled noise ~ N(0, 1) [BATCH, action_dim].
        action: Output actions after tanh [BATCH, action_dim] (written).
        log_prob: Output log probabilities [BATCH, 1] (written).
    """
    comptime LOG_2PI: Float64 = 1.8378770664093453  # log(2 * pi)

    for batch in range(BATCH):
        var total_log_prob: Float64 = 0.0

        for j in range(action_dim):
            var m = Float64(rebind[Scalar[dtype]](mean[batch, j]))
            var ls = Float64(rebind[Scalar[dtype]](log_std[batch, j]))
            var n = Float64(rebind[Scalar[dtype]](noise[batch, j]))

            # Compute std and pre-tanh action
            var std = exp(ls)
            var z = m + std * n

            # Apply tanh squashing
            var exp_z = exp(z)
            var exp_neg_z = exp(-z)
            var tanh_z = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)
            action[batch, j] = Scalar[dtype](tanh_z)

            # Log probability of Gaussian: -0.5 * (log(2*pi) + 2*log_std + ((z - mean)/std)^2)
            var z_normalized = (
                n  # (z - mean) / std = n since z = mean + std * n
            )
            var log_gaussian = -0.5 * (
                LOG_2PI + 2.0 * ls + z_normalized * z_normalized
            )

            # Squashing correction: -log(1 - tanh(z)^2 + eps)
            var squash_correction = log(1.0 - tanh_z * tanh_z + EPS)

            total_log_prob += log_gaussian - squash_correction

        log_prob[batch, 0] = Scalar[dtype](total_log_prob)


fn sample_action[
    BATCH: Int, action_dim: Int
](
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    noise: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
):
    """Sample actions using reparameterization trick (without log_prob).

    Args:
        mean: Mean of Gaussian [BATCH, action_dim].
        log_std: Log standard deviation [BATCH, action_dim].
        noise: Pre-sampled noise ~ N(0, 1) [BATCH, action_dim].
        action: Output actions after tanh [BATCH, action_dim] (written).
    """
    for batch in range(BATCH):
        for j in range(action_dim):
            var m = Float64(rebind[Scalar[dtype]](mean[batch, j]))
            var ls = Float64(rebind[Scalar[dtype]](log_std[batch, j]))
            var n = Float64(rebind[Scalar[dtype]](noise[batch, j]))

            var std = exp(ls)
            var z = m + std * n

            # Tanh squashing
            var exp_z = exp(z)
            var exp_neg_z = exp(-z)
            var tanh_z = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)
            action[batch, j] = Scalar[dtype](tanh_z)


fn compute_log_prob[
    BATCH: Int, action_dim: Int
](
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut log_prob: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    """Compute log probability of actions under the Gaussian policy.

    Note: This requires inverting tanh to get the pre-tanh value z.
    z = atanh(action) = 0.5 * log((1 + action) / (1 - action))

    Args:
        mean: Mean of Gaussian [BATCH, action_dim].
        log_std: Log standard deviation [BATCH, action_dim].
        action: Actions (must be in (-1, 1)) [BATCH, action_dim].
        log_prob: Output log probabilities [BATCH, 1] (written).
    """
    comptime LOG_2PI: Float64 = 1.8378770664093453

    for batch in range(BATCH):
        var total_log_prob: Float64 = 0.0

        for j in range(action_dim):
            var m = Float64(rebind[Scalar[dtype]](mean[batch, j]))
            var ls = Float64(rebind[Scalar[dtype]](log_std[batch, j]))
            var a = Float64(rebind[Scalar[dtype]](action[batch, j]))

            # Clamp action to valid range for atanh
            if a >= 1.0 - EPS:
                a = 1.0 - EPS
            elif a <= -1.0 + EPS:
                a = -1.0 + EPS

            # Inverse tanh: z = atanh(action)
            var z = 0.5 * log((1.0 + a) / (1.0 - a))

            var std = exp(ls)
            var z_normalized = (z - m) / std

            # Log probability of Gaussian
            var log_gaussian = -0.5 * (
                LOG_2PI + 2.0 * ls + z_normalized * z_normalized
            )

            # Squashing correction
            var squash_correction = log(1.0 - a * a + EPS)

            total_log_prob += log_gaussian - squash_correction

        log_prob[batch, 0] = Scalar[dtype](total_log_prob)


fn get_deterministic_action[
    BATCH: Int, action_dim: Int
](
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
):
    """Get deterministic action by applying tanh to the mean.

    Useful for evaluation where we want the most likely action.

    Args:
        mean: Mean of Gaussian [BATCH, action_dim].
        action: Output actions after tanh [BATCH, action_dim] (written).
    """
    for batch in range(BATCH):
        for j in range(action_dim):
            var m = Float64(rebind[Scalar[dtype]](mean[batch, j]))
            var exp_m = exp(m)
            var exp_neg_m = exp(-m)
            var tanh_m = (exp_m - exp_neg_m) / (exp_m + exp_neg_m)
            action[batch, j] = Scalar[dtype](tanh_m)


# =============================================================================
# Reparameterization Backward Pass Utilities
# =============================================================================


fn rsample_with_cache[
    BATCH: Int, action_dim: Int
](
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    noise: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut log_prob: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    mut z_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
):
    """Reparameterized sample with caching for backward pass.

    Same as rsample() but also caches the pre-tanh values (z) needed for
    rsample_backward(). Use this when you need to backpropagate through
    the reparameterization trick.

    Args:
        mean: Mean of Gaussian [BATCH, action_dim].
        log_std: Log standard deviation [BATCH, action_dim].
        noise: Pre-sampled noise ~ N(0, 1) [BATCH, action_dim].
        action: Output actions after tanh [BATCH, action_dim] (written).
        log_prob: Output log probabilities [BATCH, 1] (written).
        z_cache: Pre-tanh values for backward [BATCH, action_dim] (written).
    """
    comptime LOG_2PI: Float64 = 1.8378770664093453  # log(2 * pi)

    for batch in range(BATCH):
        var total_log_prob: Float64 = 0.0

        for j in range(action_dim):
            var m = Float64(rebind[Scalar[dtype]](mean[batch, j]))
            var ls = Float64(rebind[Scalar[dtype]](log_std[batch, j]))
            var n = Float64(rebind[Scalar[dtype]](noise[batch, j]))

            # Compute std and pre-tanh action
            var std = exp(ls)
            var z = m + std * n

            # Cache z for backward pass
            z_cache[batch, j] = Scalar[dtype](z)

            # Apply tanh squashing
            var exp_z = exp(z)
            var exp_neg_z = exp(-z)
            var tanh_z = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)
            action[batch, j] = Scalar[dtype](tanh_z)

            # Log probability of Gaussian
            var z_normalized = n  # (z - mean) / std = n
            var log_gaussian = -0.5 * (
                LOG_2PI + 2.0 * ls + z_normalized * z_normalized
            )

            # Squashing correction
            var squash_correction = log(1.0 - tanh_z * tanh_z + EPS)

            total_log_prob += log_gaussian - squash_correction

        log_prob[batch, 0] = Scalar[dtype](total_log_prob)


fn rsample_backward[
    BATCH: Int, action_dim: Int
](
    grad_action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    grad_log_prob: LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ],
    action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    noise: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut grad_mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut grad_log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
):
    """Backward pass through the reparameterization trick.

    Computes gradients of the loss w.r.t. mean and log_std given gradients
    w.r.t. action and log_prob.

    Forward equations:
        z = mean + exp(log_std) * noise
        action = tanh(z)
        log_prob = sum_j(-0.5*(log(2π) + 2*log_std_j + noise_j²) - log(1 - action_j² + ε))

    Backward equations:
        d(action)/d(z) = 1 - action²
        d(z)/d(mean) = 1
        d(z)/d(log_std) = std * noise

        d(log_prob)/d(z) = 2*action / (1 - action² + ε)  [from squash correction]
        d(log_prob)/d(log_std) = -1  [from Gaussian term]

    Args:
        grad_action: Gradient w.r.t. action [BATCH, action_dim] (e.g., -dQ/da from critic).
        grad_log_prob: Gradient w.r.t. log_prob [BATCH, 1] (e.g., -alpha for entropy term).
        action: Cached action values (tanh(z)) [BATCH, action_dim].
        log_std: Log standard deviation [BATCH, action_dim].
        noise: Cached noise values [BATCH, action_dim].
        grad_mean: Output gradient w.r.t. mean [BATCH, action_dim] (written).
        grad_log_std: Output gradient w.r.t. log_std [BATCH, action_dim] (written).
    """
    for batch in range(BATCH):
        var glp = Float64(rebind[Scalar[dtype]](grad_log_prob[batch, 0]))

        for j in range(action_dim):
            var ga = Float64(rebind[Scalar[dtype]](grad_action[batch, j]))
            var a = Float64(rebind[Scalar[dtype]](action[batch, j]))
            var ls = Float64(rebind[Scalar[dtype]](log_std[batch, j]))
            var n = Float64(rebind[Scalar[dtype]](noise[batch, j]))

            var std = exp(ls)

            # d(action)/d(z) = 1 - tanh²(z) = 1 - action²
            var dtanh_dz = 1.0 - a * a

            # Gradient of log_prob w.r.t. z (from squash correction)
            # d(-log(1 - a² + ε))/dz = d(-log(1 - tanh²(z) + ε))/dz
            #                        = 2*tanh(z) / (1 - tanh²(z) + ε) * dtanh/dz
            #                        = 2*a / (1 - a² + ε) * (1 - a²)
            #                        = 2*a * (1 - a²) / (1 - a² + ε)
            var dlogprob_dz = 2.0 * a * dtanh_dz / (1.0 - a * a + EPS)

            # Total gradient of z
            # grad_z = grad_action * d(action)/d(z) + grad_log_prob * d(log_prob)/d(z)
            var grad_z = ga * dtanh_dz + glp * dlogprob_dz

            # d(z)/d(mean) = 1
            grad_mean[batch, j] = Scalar[dtype](grad_z)

            # d(z)/d(log_std) = std * noise
            # d(log_prob)/d(log_std) = -1 (from Gaussian: -0.5 * 2 * log_std term)
            var grad_ls = grad_z * std * n + glp * (-1.0)
            grad_log_std[batch, j] = Scalar[dtype](grad_ls)


# =============================================================================
# GPU Kernels for Reparameterization Backward
# =============================================================================


@always_inline
fn rsample_with_cache_kernel_impl[
    BATCH: Int, action_dim: Int
](
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ],
    noise: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ],
    mut action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    mut log_prob: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    mut z_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
):
    """GPU kernel for rsample_with_cache.

    Grid: (BATCH,)
    Block: (action_dim,) or (TPB,) with loop if action_dim > TPB
    """
    var batch = Int(block_idx.x)
    var j = Int(thread_idx.x)

    if batch >= BATCH or j >= action_dim:
        return

    comptime LOG_2PI: Float64 = 1.8378770664093453

    var m = Float64(rebind[Scalar[dtype]](mean[batch, j]))
    var ls = Float64(rebind[Scalar[dtype]](log_std[batch, j]))
    var n = Float64(rebind[Scalar[dtype]](noise[batch, j]))

    # Compute std and pre-tanh action
    var std = exp(ls)
    var z = m + std * n

    # Cache z
    z_cache[batch, j] = Scalar[dtype](z)

    # Apply tanh squashing
    var exp_z = exp(z)
    var exp_neg_z = exp(-z)
    var tanh_z = (exp_z - exp_neg_z) / (exp_z + exp_neg_z)
    action[batch, j] = Scalar[dtype](tanh_z)

    # Compute per-dimension log prob contribution
    var z_normalized = n
    var log_gaussian = -0.5 * (LOG_2PI + 2.0 * ls + z_normalized * z_normalized)
    var squash_correction = log(1.0 - tanh_z * tanh_z + EPS)
    var dim_log_prob = log_gaussian - squash_correction

    # Use atomic add for log_prob reduction (or use block reduction)
    # For simplicity, we'll use a separate reduction kernel
    # Here we just store per-dim values and reduce later
    # This is a simplified implementation - in practice use block.sum


@always_inline
fn rsample_backward_kernel_impl[
    BATCH: Int, action_dim: Int
](
    grad_action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ],
    grad_log_prob: LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
    ],
    action: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ],
    noise: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ],
    grad_mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
    grad_log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ],
):
    """GPU kernel for rsample_backward.

    Elementwise kernel - one thread per (batch, action_dim) element.
    Grid: ((BATCH * action_dim + TPB - 1) // TPB,)
    Block: (TPB,)
    """
    var idx = Int(block_idx.x) * TPB + Int(thread_idx.x)
    comptime total_size = BATCH * action_dim

    if idx >= total_size:
        return

    var batch = idx // action_dim
    var j = idx % action_dim

    var ga = Float64(rebind[Scalar[dtype]](grad_action[batch, j]))
    var glp = Float64(rebind[Scalar[dtype]](grad_log_prob[batch, 0]))
    var a = Float64(rebind[Scalar[dtype]](action[batch, j]))
    var ls = Float64(rebind[Scalar[dtype]](log_std[batch, j]))
    var n = Float64(rebind[Scalar[dtype]](noise[batch, j]))

    var std = exp(ls)

    # d(action)/d(z) = 1 - action²
    var dtanh_dz = 1.0 - a * a

    # Gradient of log_prob w.r.t. z
    var dlogprob_dz = 2.0 * a * dtanh_dz / (1.0 - a * a + EPS)

    # Total gradient of z
    var grad_z = ga * dtanh_dz + glp * dlogprob_dz

    # d(z)/d(mean) = 1
    grad_mean[batch, j] = Scalar[dtype](grad_z)

    # d(z)/d(log_std) = std * noise, d(log_prob)/d(log_std) = -1
    var grad_ls = grad_z * std * n + glp * (-1.0)
    grad_log_std[batch, j] = Scalar[dtype](grad_ls)


fn rsample_backward_gpu[
    BATCH: Int, action_dim: Int
](
    ctx: DeviceContext,
    grad_action_buf: DeviceBuffer[dtype],
    grad_log_prob_buf: DeviceBuffer[dtype],
    action_buf: DeviceBuffer[dtype],
    log_std_buf: DeviceBuffer[dtype],
    noise_buf: DeviceBuffer[dtype],
    mut grad_mean_buf: DeviceBuffer[dtype],
    mut grad_log_std_buf: DeviceBuffer[dtype],
) raises:
    """Launch rsample_backward on GPU.

    Args:
        ctx: GPU device context.
        grad_action_buf: Gradient w.r.t. action [BATCH * action_dim].
        grad_log_prob_buf: Gradient w.r.t. log_prob [BATCH].
        action_buf: Cached action values [BATCH * action_dim].
        log_std_buf: Log std values [BATCH * action_dim].
        noise_buf: Cached noise values [BATCH * action_dim].
        grad_mean_buf: Output gradient w.r.t. mean [BATCH * action_dim] (written).
        grad_log_std_buf: Output gradient w.r.t. log_std [BATCH * action_dim] (written).
    """
    var grad_action = LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ](grad_action_buf)
    var grad_log_prob = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
    ](grad_log_prob_buf)
    var action = LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ](action_buf)
    var log_std = LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ](log_std_buf)
    var noise = LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
    ](noise_buf)
    var grad_mean = LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ](grad_mean_buf)
    var grad_log_std = LayoutTensor[
        dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
    ](grad_log_std_buf)

    comptime total_size = BATCH * action_dim
    comptime grid_size = (total_size + TPB - 1) // TPB

    @always_inline
    fn kernel_wrapper(
        grad_action: LayoutTensor[
            dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
        ],
        grad_log_prob: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ],
        action: LayoutTensor[
            dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
        ],
        log_std: LayoutTensor[
            dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
        ],
        noise: LayoutTensor[
            dtype, Layout.row_major(BATCH, action_dim), ImmutAnyOrigin
        ],
        grad_mean: LayoutTensor[
            dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
        ],
        grad_log_std: LayoutTensor[
            dtype, Layout.row_major(BATCH, action_dim), MutAnyOrigin
        ],
    ):
        rsample_backward_kernel_impl[BATCH, action_dim](
            grad_action,
            grad_log_prob,
            action,
            log_std,
            noise,
            grad_mean,
            grad_log_std,
        )

    ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
        grad_action,
        grad_log_prob,
        action,
        log_std,
        noise,
        grad_mean,
        grad_log_std,
        grid_dim=(grid_size,),
        block_dim=(TPB,),
    )
