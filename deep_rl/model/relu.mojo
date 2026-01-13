from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

# GPU constant
comptime TPB = 256  # Threads per block for elementwise ops


struct ReLU[dim: Int](Model):
    """ReLU activation: y = max(0, x).

    CACHE_SIZE = dim (caches pre-activation values for backward pass)
    WORKSPACE_SIZE_PER_SAMPLE = 0 (leaf layer, no intermediate buffers needed)
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim  # Cache pre-activation for backward
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
        """Forward: y = max(0, x).

        Caches pre-activation values for backward pass.
        Note: params is unused (ReLU has no parameters).
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var val = input[batch, i]
                cache[batch, i] = val  # Cache for backward
                output[batch, i] = val if val > 0 else 0

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

        Note: params is unused (ReLU has no parameters).
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var val = input[batch, i]
                output[batch, i] = val if val > 0 else 0

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
        """Backward: dx = dy * (x > 0).

        Uses cached pre-activation values from forward pass.
        Note: params and grads are unused (ReLU has no parameters).
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var pre = cache[batch, i]
                grad_input[batch, i] = grad_output[batch, i] if pre > 0 else 0

    # =========================================================================
    # GPU Kernel Implementations (@always_inline for fusion)
    # =========================================================================
    #
    # These are the core GPU computations that can be inlined into fused kernels.
    # ReLU uses 1D elementwise parallelism.
    #
    # Grid: ((BATCH * dim + TPB - 1) // TPB,)
    # Block: (TPB,)
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
        """Forward pass kernel implementation: y = max(0, x) with caching.

        This is the core GPU computation that can be inlined into fused kernels.
        Uses 1D elementwise parallelism.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)

        Args:
            output: Output tensor [BATCH, dim] (written).
            input: Input tensor [BATCH, dim].
            cache: Cache buffer [BATCH, dim] for backward pass (written).
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = input[row, col]
        cache[row, col] = val  # Cache for backward
        output[row, col] = val if val > 0 else 0

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
        """Forward pass kernel implementation without caching (for inference).

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = input[row, col]
        output[row, col] = val if val > 0 else 0

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
        """Backward pass kernel implementation: dx = dy * (x > 0).

        Uses cached pre-activation values from forward pass.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var pre = cache[row, col]
        grad_input[row, col] = grad_output[row, col] if pre > 0 else 0

    # =========================================================================
    # GPU Launchers (with DeviceContext)
    # =========================================================================
    #
    # These functions handle buffer-to-tensor conversion, grid/block config,
    # and kernel launch. They call the _kernel_impl functions.
    #
    # Note: ReLU has no parameters, so params_buf and grads_buf are unused
    # but kept for API consistency with Linear.
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],  # Unused for ReLU
        cache_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU with caching.

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * dim].
            input_buf: Input buffer [BATCH * dim].
            params_buf: Parameters buffer (unused for ReLU, kept for API consistency).
            cache_buf: Cache buffer [BATCH * dim] for backward pass.
        """
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](cache_buf.unsafe_ptr())

        comptime total_elements = BATCH * Self.dim
        comptime grid_x = (total_elements + TPB - 1) // TPB

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

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input,
            cache,
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
        params_buf: DeviceBuffer[dtype],  # Unused for ReLU
    ) raises:
        """Launch forward pass on GPU without caching (for inference).

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * dim].
            input_buf: Input buffer [BATCH * dim].
            params_buf: Parameters buffer (unused for ReLU, kept for API consistency).
        """
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())

        comptime total_elements = BATCH * Self.dim
        comptime grid_x = (total_elements + TPB - 1) // TPB

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
        params_buf: DeviceBuffer[dtype],  # Unused for ReLU
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],  # Unused for ReLU
    ) raises:
        """Launch backward pass on GPU.

        ReLU has no parameters, so only grad_input is computed.

        Args:
            ctx: GPU device context.
            grad_input_buf: Gradient w.r.t. input [BATCH * dim] (written).
            grad_output_buf: Gradient w.r.t. output [BATCH * dim].
            params_buf: Parameters buffer (unused for ReLU).
            cache_buf: Cached pre-activation from forward pass [BATCH * dim].
            grads_buf: Parameter gradients (unused for ReLU).
        """
        var grad_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](grad_input_buf.unsafe_ptr())
        var grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache_buf.unsafe_ptr())

        comptime total_elements = BATCH * Self.dim
        comptime grid_x = (total_elements + TPB - 1) // TPB

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
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Workspace Methods (for Sequential compatibility)
    # ReLU is a leaf layer, so workspace is unused - just delegate to regular methods.
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for ReLU
    ) raises:
        """GPU forward with workspace (workspace unused for ReLU)."""
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for ReLU
    ) raises:
        """GPU forward without cache, with workspace (workspace unused for ReLU).
        """
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for ReLU
    ) raises:
        """GPU backward with workspace (workspace unused for ReLU)."""
        Self.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )
