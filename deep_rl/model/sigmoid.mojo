from ..constants import dtype, TPB
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from math import exp


struct Sigmoid[dim: Int](Model):
    """Sigmoid activation: y = 1 / (1 + exp(-x)).

    CACHE_SIZE = dim (caches sigmoid output for backward pass)
    WORKSPACE_SIZE_PER_SAMPLE = 0 (leaf layer, no intermediate buffers needed)

    Derivative: dy/dx = y * (1 - y), which is why we cache the output.
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim  # Cache sigmoid output for backward
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
        batch_size: Int,
    ):
        """Forward: y = 1 / (1 + exp(-x)).

        Caches sigmoid output for backward pass (derivative uses output).
        Note: params is unused (Sigmoid has no parameters).
        """
        for batch in range(batch_size):
            for i in range(Self.dim):
                var val = rebind[Scalar[dtype]](input[batch, i])
                var val_f64 = Float64(val)
                var sigmoid_f64 = 1.0 / (1.0 + exp(-val_f64))
                var sigmoid_val = Scalar[dtype](sigmoid_f64)
                output[batch, i] = sigmoid_val
                cache[batch, i] = sigmoid_val  # Cache output for backward

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
        batch_size: Int,
    ):
        """Forward pass without caching (for inference).

        Note: params is unused (Sigmoid has no parameters).
        """
        for batch in range(batch_size):
            for i in range(Self.dim):
                var val = rebind[Scalar[dtype]](input[batch, i])
                var val_f64 = Float64(val)
                var sigmoid_f64 = 1.0 / (1.0 + exp(-val_f64))
                var sigmoid_val = Scalar[dtype](sigmoid_f64)
                output[batch, i] = sigmoid_val

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
        batch_size: Int,
    ):
        """Backward: dx = dy * y * (1 - y).

        Uses cached sigmoid output from forward pass.
        Note: params and grads are unused (Sigmoid has no parameters).
        """
        var one = Scalar[dtype](1.0)
        for batch in range(batch_size):
            for i in range(Self.dim):
                var y = rebind[Scalar[dtype]](cache[batch, i])  # Cached sigmoid output
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                # Derivative: sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y)
                grad_input[batch, i] = dy * y * (one - y)

    # =========================================================================
    # GPU Kernel Implementations (@always_inline for fusion)
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
        batch_size: Int,
    ):
        """Forward pass kernel: y = 1 / (1 + exp(-x)) with caching.

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= batch_size * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = rebind[Scalar[dtype]](input[row, col])
        var zero = Scalar[dtype](0.0)
        var one = Scalar[dtype](1.0)
        # Numerically stable sigmoid
        var sigmoid_val: Scalar[dtype]
        if val >= zero:
            sigmoid_val = one / (one + exp(-val))
        else:
            var exp_val = exp(val)
            sigmoid_val = exp_val / (one + exp_val)
        output[row, col] = sigmoid_val
        cache[row, col] = sigmoid_val  # Cache for backward

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
        batch_size: Int,
    ):
        """Forward pass kernel without caching (for inference).

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= batch_size * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = rebind[Scalar[dtype]](input[row, col])
        var zero = Scalar[dtype](0.0)
        var one = Scalar[dtype](1.0)
        # Numerically stable sigmoid
        var sigmoid_val: Scalar[dtype]
        if val >= zero:
            sigmoid_val = one / (one + exp(-val))
        else:
            var exp_val = exp(val)
            sigmoid_val = exp_val / (one + exp_val)
        output[row, col] = sigmoid_val

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
        batch_size: Int,
    ):
        """Backward pass kernel: dx = dy * y * (1 - y).

        Uses cached sigmoid output from forward pass.

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= batch_size * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var one = Scalar[dtype](1.0)
        var y = rebind[Scalar[dtype]](cache[row, col])  # Cached sigmoid output
        var dy = rebind[Scalar[dtype]](grad_output[row, col])
        grad_input[row, col] = dy * y * (one - y)

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
        params_buf: DeviceBuffer[dtype],  # Unused for Sigmoid
        cache_buf: DeviceBuffer[dtype],
        batch_size: Int,
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

        var total_elements = batch_size * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

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
            batch_size: Int,
        ):
            Self.forward_kernel_impl[BATCH](output, input, cache, batch_size)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            cache,
            batch_size,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],  # Unused for Sigmoid
        batch_size: Int,
    ) raises:
        """Launch forward pass on GPU without caching (for inference)."""
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())

        var total_elements = batch_size * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            batch_size: Int,
        ):
            Self.forward_kernel_impl_no_cache[BATCH](output, input, batch_size)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            batch_size,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],  # Unused for Sigmoid
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],  # Unused for Sigmoid
        batch_size: Int,
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

        var total_elements = batch_size * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

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
            batch_size: Int,
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output, cache, batch_size)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_input,
            grad_output,
            cache,
            batch_size,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Workspace Methods (for Sequential compatibility)
    # Sigmoid is a leaf layer, so workspace is unused - just delegate.
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Sigmoid
        batch_size: Int,
    ) raises:
        """GPU forward with workspace (workspace unused for Sigmoid)."""
        Self.forward_gpu[BATCH](
            ctx, output_buf, input_buf, params_buf, cache_buf, batch_size
        )

    @staticmethod
    fn forward_gpu_no_cache_ws[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],  # Unused for Sigmoid
        batch_size: Int,
    ) raises:
        """GPU forward without cache, with workspace (workspace unused)."""
        Self.forward_gpu_no_cache[BATCH](ctx, output_buf, input_buf, params_buf, batch_size)

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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Sigmoid
        batch_size: Int,
    ) raises:
        """GPU backward with workspace (workspace unused for Sigmoid)."""
        Self.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
            batch_size,
        )
