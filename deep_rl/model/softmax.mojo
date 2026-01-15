from ..constants import dtype, TPB
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from math import exp


struct Softmax[dim: Int](Model):
    """Softmax activation: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x))).

    Uses numerically stable softmax (subtract max before exp).

    CACHE_SIZE = dim (caches softmax output for backward pass)
    WORKSPACE_SIZE_PER_SAMPLE = 0 (leaf layer, no intermediate buffers needed)

    Backward: dx_i = y_i * (dy_i - sum(dy_j * y_j))
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim  # Cache softmax output for backward
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0  # Leaf layer, no workspace needed

    fn __init__(out self):
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
        """Forward: y = softmax(x) with numerical stability.

        Caches softmax output for backward pass.
        Note: params is unused (Softmax has no parameters).
        """
        for batch in range(BATCH):
            # Find max for numerical stability
            var max_val = rebind[Scalar[dtype]](input[batch, 0])
            for i in range(1, Self.dim):
                var val = rebind[Scalar[dtype]](input[batch, i])
                if val > max_val:
                    max_val = val

            # Compute exp(x - max) and sum
            var sum_exp: Scalar[dtype] = 0.0
            for i in range(Self.dim):
                var val = rebind[Scalar[dtype]](input[batch, i])
                var exp_val = Scalar[dtype](exp(Float64(val - max_val)))
                output[batch, i] = exp_val
                sum_exp = sum_exp + exp_val

            # Normalize
            for i in range(Self.dim):
                var exp_val = rebind[Scalar[dtype]](output[batch, i])
                var softmax_val = exp_val / sum_exp
                output[batch, i] = softmax_val
                cache[batch, i] = softmax_val  # Cache for backward

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

        Note: params is unused (Softmax has no parameters).
        """
        for batch in range(BATCH):
            # Find max for numerical stability
            var max_val = rebind[Scalar[dtype]](input[batch, 0])
            for i in range(1, Self.dim):
                var val = rebind[Scalar[dtype]](input[batch, i])
                if val > max_val:
                    max_val = val

            # Compute exp(x - max) and sum
            var sum_exp: Scalar[dtype] = 0.0
            for i in range(Self.dim):
                var val = rebind[Scalar[dtype]](input[batch, i])
                var exp_val = Scalar[dtype](exp(Float64(val - max_val)))
                output[batch, i] = exp_val
                sum_exp = sum_exp + exp_val

            # Normalize
            for i in range(Self.dim):
                var exp_val = rebind[Scalar[dtype]](output[batch, i])
                output[batch, i] = exp_val / sum_exp

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
        """Backward: dx_i = y_i * (dy_i - sum(dy_j * y_j)).

        Uses cached softmax output from forward pass.
        Note: params and grads are unused (Softmax has no parameters).

        The full Jacobian is: J_ij = y_i * (delta_ij - y_j)
        So: dx_i = sum_j(J_ij * dy_j) = y_i * dy_i - y_i * sum_j(y_j * dy_j)
                = y_i * (dy_i - sum_j(y_j * dy_j))
        """
        for batch in range(BATCH):
            # Compute sum(y * dy)
            var sum_y_dy: Scalar[dtype] = 0.0
            for i in range(Self.dim):
                var y = rebind[Scalar[dtype]](cache[batch, i])
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                sum_y_dy = sum_y_dy + y * dy

            # Compute gradient: dx_i = y_i * (dy_i - sum(y * dy))
            for i in range(Self.dim):
                var y = rebind[Scalar[dtype]](cache[batch, i])
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                grad_input[batch, i] = y * (dy - sum_y_dy)

    # =========================================================================
    # GPU Kernel Implementations (@always_inline for fusion)
    # =========================================================================
    #
    # Softmax requires per-sample reduction (max, sum), which is different
    # from elementwise ops like ReLU/Sigmoid. We parallelize over batches
    # and do the reduction within each sample.
    #
    # Grid: (BATCH,)
    # Block: (min(dim, TPB),) - each block handles one sample
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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        """Forward pass kernel: y = softmax(x) with caching.

        Each block handles one sample in the batch.
        Thread 0 does all the work (simple version for small dim).

        Grid: (batch_size,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        # Only thread 0 does the work (sequential per sample)
        if thread_idx.x != 0:
            return

        # Find max for numerical stability
        var max_val = rebind[Scalar[dtype]](input[batch_idx, 0])
        for i in range(1, Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            if val > max_val:
                max_val = val

        # Compute exp(x - max) and sum
        var sum_exp: Scalar[dtype] = 0.0
        for i in range(Self.dim):
            var in_val = rebind[Scalar[dtype]](input[batch_idx, i])
            var exp_val = exp(in_val - max_val)
            output[batch_idx, i] = exp_val
            sum_exp = sum_exp + exp_val

        # Normalize and cache
        for i in range(Self.dim):
            var exp_val = rebind[Scalar[dtype]](output[batch_idx, i])
            var softmax_val = exp_val / sum_exp
            output[batch_idx, i] = softmax_val
            cache[batch_idx, i] = softmax_val

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
    ):
        """Forward pass kernel without caching (for inference).

        Grid: (batch_size,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        if thread_idx.x != 0:
            return

        # Find max for numerical stability
        var max_val = rebind[Scalar[dtype]](input[batch_idx, 0])
        for i in range(1, Self.dim):
            var val = rebind[Scalar[dtype]](input[batch_idx, i])
            if val > max_val:
                max_val = val

        # Compute exp(x - max) and sum
        var sum_exp: Scalar[dtype] = 0.0
        for i in range(Self.dim):
            var in_val = rebind[Scalar[dtype]](input[batch_idx, i])
            var exp_val = exp(in_val - max_val)
            output[batch_idx, i] = exp_val
            sum_exp = sum_exp + exp_val

        # Normalize
        for i in range(Self.dim):
            var exp_val = rebind[Scalar[dtype]](output[batch_idx, i])
            output[batch_idx, i] = exp_val / sum_exp

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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        """Backward pass kernel: dx_i = y_i * (dy_i - sum(y * dy)).

        Grid: (batch_size,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        if thread_idx.x != 0:
            return

        # Compute sum(y * dy)
        var sum_y_dy: Scalar[dtype] = 0.0
        for i in range(Self.dim):
            var y = rebind[Scalar[dtype]](cache[batch_idx, i])
            var dy = rebind[Scalar[dtype]](grad_output[batch_idx, i])
            sum_y_dy = sum_y_dy + y * dy

        # Compute gradient
        for i in range(Self.dim):
            var y = rebind[Scalar[dtype]](cache[batch_idx, i])
            var dy = rebind[Scalar[dtype]](grad_output[batch_idx, i])
            grad_input[batch_idx, i] = y * (dy - sum_y_dy)

    # =========================================================================
    # GPU Launchers (with DeviceContext)
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],  # Unused for Softmax
        cache_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU with caching."""
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](cache_buf.unsafe_ptr())

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH](output, input, cache)

        # One block per sample, single thread per block (simple version)
        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            cache,
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
        params_buf: DeviceBuffer[dtype],  # Unused for Softmax
    ) raises:
        """Launch forward pass on GPU without caching (for inference)."""
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl_no_cache[BATCH](output, input)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
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
        params_buf: DeviceBuffer[dtype],  # Unused for Softmax
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],  # Unused for Softmax
    ) raises:
        """Launch backward pass on GPU."""
        var grad_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](grad_input_buf.unsafe_ptr())
        var grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache_buf.unsafe_ptr())

        @always_inline
        fn kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output, cache)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_input,
            grad_output,
            cache,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    # =========================================================================
    # GPU Workspace Methods (for Sequential compatibility)
    # Softmax is a leaf layer, so workspace is unused - just delegate.
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Softmax
    ) raises:
        """GPU forward with workspace (workspace unused for Softmax)."""
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Softmax
    ) raises:
        """GPU forward without cache, with workspace (workspace unused)."""
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Softmax
    ) raises:
        """GPU backward with workspace (workspace unused for Softmax)."""
        Self.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )
