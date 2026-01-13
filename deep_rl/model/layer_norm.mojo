from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from math import sqrt

# GPU constant
comptime TPB = 256  # Threads per block


struct LayerNorm[dim: Int](Model):
    """Layer Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.

    Normalizes across the feature dimension (last dimension).

    Parameters (stored in params tensor):
    - gamma [dim]: Scale parameter, initialized to 1.0
    - beta [dim]: Shift parameter, initialized to 0.0

    PARAM_SIZE = 2 * dim (gamma + beta)
    CACHE_SIZE = dim + 2 (normalized values + inv_std + mean per sample)

    Layout:
    - params: [gamma (dim) | beta (dim)]
    - cache: [normalized (dim) | inv_std (1) | mean (1)] per sample
    """

    var eps: Float64

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 2 * Self.dim  # gamma + beta
    comptime CACHE_SIZE: Int = Self.dim + 2  # normalized + inv_std + mean
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0  # Leaf layer

    fn __init__(out self, eps: Float64 = 1e-5):
        """Initialize LayerNorm with epsilon for numerical stability."""
        self.eps = eps

    fn __moveinit__(out self, deinit other: Self):
        self.eps = other.eps

    fn __copyinit__(out self, other: Self):
        self.eps = other.eps

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
        """Forward: y = gamma * normalize(x) + beta.

        Caches normalized values, inv_std, and mean for backward.
        """
        var eps = Scalar[dtype](self.eps)
        var n = Scalar[dtype](Self.dim)

        for batch in range(BATCH):
            # Compute mean
            var mean = rebind[Scalar[dtype]](input[batch, 0])
            for i in range(1, Self.dim):
                mean = mean + rebind[Scalar[dtype]](input[batch, i])
            mean = mean / n

            # Compute variance
            var diff0 = rebind[Scalar[dtype]](input[batch, 0]) - mean
            var var_ = diff0 * diff0
            for i in range(1, Self.dim):
                var diff = rebind[Scalar[dtype]](input[batch, i]) - mean
                var_ = var_ + diff * diff
            var_ = var_ / n

            # Compute inv_std
            var inv_std = Scalar[dtype](1.0 / sqrt(Float64(var_ + eps)))

            # Normalize and apply affine transform
            for i in range(Self.dim):
                var x_val = rebind[Scalar[dtype]](input[batch, i])
                var normalized = (x_val - mean) * inv_std
                # Cache normalized value
                cache[batch, i] = normalized
                # gamma at offset 0, beta at offset dim
                var gamma = rebind[Scalar[dtype]](params[i])
                var beta = rebind[Scalar[dtype]](params[Self.dim + i])
                output[batch, i] = gamma * normalized + beta

            # Cache inv_std and mean
            cache[batch, Self.dim] = inv_std
            cache[batch, Self.dim + 1] = mean

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
        var eps = Scalar[dtype](self.eps)
        var n = Scalar[dtype](Self.dim)

        for batch in range(BATCH):
            # Compute mean
            var mean = rebind[Scalar[dtype]](input[batch, 0])
            for i in range(1, Self.dim):
                mean = mean + rebind[Scalar[dtype]](input[batch, i])
            mean = mean / n

            # Compute variance
            var diff0 = rebind[Scalar[dtype]](input[batch, 0]) - mean
            var var_ = diff0 * diff0
            for i in range(1, Self.dim):
                var diff = rebind[Scalar[dtype]](input[batch, i]) - mean
                var_ = var_ + diff * diff
            var_ = var_ / n

            # Compute inv_std
            var inv_std = Scalar[dtype](1.0 / sqrt(Float64(var_ + eps)))

            # Normalize and apply affine transform
            for i in range(Self.dim):
                var x_val = rebind[Scalar[dtype]](input[batch, i])
                var normalized = (x_val - mean) * inv_std
                var gamma = rebind[Scalar[dtype]](params[i])
                var beta = rebind[Scalar[dtype]](params[Self.dim + i])
                output[batch, i] = gamma * normalized + beta

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
        """Backward pass for LayerNorm.

        Computes gradients for:
        - gamma (grads[0:dim])
        - beta (grads[dim:2*dim])
        - input (grad_input)
        """
        var n = Scalar[dtype](Self.dim)

        # First, accumulate parameter gradients across batch
        for batch in range(BATCH):
            for i in range(Self.dim):
                var normalized = rebind[Scalar[dtype]](cache[batch, i])
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])

                # dgamma += dy * normalized
                var old_dgamma = rebind[Scalar[dtype]](grads[i])
                grads[i] = old_dgamma + dy * normalized
                # dbeta += dy
                var old_dbeta = rebind[Scalar[dtype]](grads[Self.dim + i])
                grads[Self.dim + i] = old_dbeta + dy

        # Then compute input gradients
        for batch in range(BATCH):
            var inv_std = rebind[Scalar[dtype]](cache[batch, Self.dim])

            # Compute intermediate values for this sample
            var sum_dy_gamma: Scalar[dtype] = 0.0
            var sum_dy_gamma_norm: Scalar[dtype] = 0.0

            for i in range(Self.dim):
                var gamma = rebind[Scalar[dtype]](params[i])
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                var normalized = rebind[Scalar[dtype]](cache[batch, i])
                sum_dy_gamma = sum_dy_gamma + dy * gamma
                sum_dy_gamma_norm = sum_dy_gamma_norm + dy * gamma * normalized

            # Compute input gradients
            for i in range(Self.dim):
                var gamma = rebind[Scalar[dtype]](params[i])
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                var normalized = rebind[Scalar[dtype]](cache[batch, i])

                var dx = inv_std * (
                    dy * gamma - sum_dy_gamma / n - normalized * sum_dy_gamma_norm / n
                )
                grad_input[batch, i] = dx

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================
    #
    # LayerNorm requires per-sample statistics (mean, var), so we parallelize
    # over batches. Each block handles one sample.
    # =========================================================================

    @always_inline
    @staticmethod
    fn forward_kernel_impl[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), MutAnyOrigin
        ],
        eps: Scalar[dtype],
    ):
        """Forward pass kernel with caching.

        Grid: (BATCH,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        if thread_idx.x != 0:
            return

        var n = Scalar[dtype](Self.dim)

        # Compute mean
        var mean: Scalar[dtype] = 0.0
        for i in range(Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            mean = mean + val
        mean = mean / n

        # Compute variance
        var var_: Scalar[dtype] = 0.0
        for i in range(Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            var diff = val - mean
            var_ = var_ + diff * diff
        var_ = var_ / n

        # Compute inv_std
        var inv_std = Scalar[dtype](1.0 / sqrt(Float64(var_ + eps)))

        # Normalize and apply affine transform
        for i in range(Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            var normalized = (val - mean) * inv_std
            cache[batch_idx, i] = normalized
            var gamma = rebind[Scalar[dtype]](params[i])
            var beta = rebind[Scalar[dtype]](params[Self.dim + i])
            output[batch_idx, i] = gamma * normalized + beta

        cache[batch_idx, Self.dim] = inv_std
        cache[batch_idx, Self.dim + 1] = mean

    @always_inline
    @staticmethod
    fn forward_kernel_impl_no_cache[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin],
        eps: Scalar[dtype],
    ):
        """Forward pass kernel without caching.

        Grid: (BATCH,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        if thread_idx.x != 0:
            return

        var n = Scalar[dtype](Self.dim)

        # Compute mean
        var mean: Scalar[dtype] = 0.0
        for i in range(Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            mean = mean + val
        mean = mean / n

        # Compute variance
        var var_: Scalar[dtype] = 0.0
        for i in range(Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            var diff = val - mean
            var_ = var_ + diff * diff
        var_ = var_ / n

        # Compute inv_std
        var inv_std = Scalar[dtype](1.0 / sqrt(Float64(var_ + eps)))

        # Normalize and apply affine transform
        for i in range(Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            var normalized = (val - mean) * inv_std
            var gamma = rebind[Scalar[dtype]](params[i])
            var beta = rebind[Scalar[dtype]](params[Self.dim + i])
            output[batch_idx, i] = gamma * normalized + beta

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(2 * Self.dim), MutAnyOrigin],
    ):
        """Backward pass kernel.

        Note: Parameter gradients need atomic adds across batch samples,
        which is complex on GPU. This simple version processes sequentially.

        Grid: (BATCH,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        if thread_idx.x != 0:
            return

        var inv_std = rebind[Scalar[dtype]](cache[batch_idx, Self.dim])
        var n = Scalar[dtype](Self.dim)

        # Compute intermediate sums
        var sum_dy_gamma: Scalar[dtype] = 0.0
        var sum_dy_gamma_norm: Scalar[dtype] = 0.0

        for i in range(Self.dim):
            var gamma = rebind[Scalar[dtype]](params[i])
            var dy = rebind[Scalar[dtype]](grad_output[batch_idx, i])
            var normalized = rebind[Scalar[dtype]](cache[batch_idx, i])
            sum_dy_gamma = sum_dy_gamma + dy * gamma
            sum_dy_gamma_norm = sum_dy_gamma_norm + dy * gamma * normalized

        # Compute input gradients
        for i in range(Self.dim):
            var gamma = rebind[Scalar[dtype]](params[i])
            var dy = rebind[Scalar[dtype]](grad_output[batch_idx, i])
            var normalized = rebind[Scalar[dtype]](cache[batch_idx, i])

            var dx = inv_std * (
                dy * gamma - sum_dy_gamma / n - normalized * sum_dy_gamma_norm / n
            )
            grad_input[batch_idx, i] = dx

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
        """Launch forward pass on GPU with caching (trait-compatible)."""
        Self._forward_gpu_impl[BATCH](ctx, output_buf, input_buf, params_buf, cache_buf, 1e-5)

    @staticmethod
    fn _forward_gpu_impl[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        eps: Float64,
    ) raises:
        """Launch forward pass on GPU with caching."""
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var params = LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ](params_buf.unsafe_ptr())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), MutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var eps_scalar = Scalar[dtype](eps)

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim + 2), MutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_kernel_impl[BATCH](output, input, params, cache, eps)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            params,
            cache,
            eps_scalar,
            grid_dim=(BATCH,),
            block_dim=(1,),
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
        """Launch forward pass on GPU without caching (trait-compatible)."""
        Self._forward_gpu_no_cache_impl[BATCH](ctx, output_buf, input_buf, params_buf, 1e-5)

    @staticmethod
    fn _forward_gpu_no_cache_impl[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        eps: Float64,
    ) raises:
        """Launch forward pass on GPU without caching."""
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var params = LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ](params_buf.unsafe_ptr())
        var eps_scalar = Scalar[dtype](eps)

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_kernel_impl_no_cache[BATCH](output, input, params, eps)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            params,
            eps_scalar,
            grid_dim=(BATCH,),
            block_dim=(1,),
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
        """Launch backward pass on GPU."""
        var grad_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](grad_input_buf.unsafe_ptr())
        var grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var params = LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ](params_buf.unsafe_ptr())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var grads = LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), MutAnyOrigin
        ](grads_buf.unsafe_ptr())

        @always_inline
        fn kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), MutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](
                grad_input, grad_output, params, cache, grads
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_input,
            grad_output,
            params,
            cache,
            grads,
            grid_dim=(BATCH,),
            block_dim=(1,),
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
        """GPU forward with workspace (trait-compatible)."""
        Self._forward_gpu_impl[BATCH](
            ctx, output_buf, input_buf, params_buf, cache_buf, 1e-5
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
        """GPU forward without cache, with workspace (trait-compatible)."""
        Self._forward_gpu_no_cache_impl[BATCH](
            ctx, output_buf, input_buf, params_buf, 1e-5
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
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward with workspace (unused for LayerNorm)."""
        Self.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )
