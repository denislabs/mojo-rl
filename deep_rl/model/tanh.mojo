from ..constants import dtype, TPB
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from math import exp


struct Tanh[dim: Int](Model):
    """Tanh activation: y = tanh(x).

    CACHE_SIZE = dim (caches tanh output for backward pass: dx = dy * (1 - tanh(x)^2))
    WORKSPACE_SIZE_PER_SAMPLE = 0 (leaf layer, no intermediate buffers needed)
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim  # Cache tanh output for backward
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
        """Forward: y = tanh(x).

        Caches tanh output for backward pass (needed for derivative).
        Note: params is unused (Tanh has no parameters).
        """
        from math import exp

        for batch in range(BATCH):
            for i in range(Self.dim):
                var val_scalar: Scalar[dtype] = rebind[Scalar[dtype]](
                    input[batch, i]
                )
                var val = Float64(val_scalar)
                # Compute tanh manually: (e^x - e^-x) / (e^x + e^-x)
                var exp_val = exp(val)
                var exp_neg_val = exp(-val)
                var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
                var t = Scalar[dtype](tanh_val[0])  # Extract scalar from SIMD
                cache[batch, i] = t  # Cache tanh output for backward
                output[batch, i] = t

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

        Note: params is unused (Tanh has no parameters).
        """
        from math import exp

        for batch in range(BATCH):
            for i in range(Self.dim):
                var val_scalar: Scalar[dtype] = rebind[Scalar[dtype]](
                    input[batch, i]
                )
                var val = Float64(val_scalar)
                var exp_val = exp(val)
                var exp_neg_val = exp(-val)
                var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
                output[batch, i] = Scalar[dtype](tanh_val[0])

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
        """Backward: dx = dy * (1 - tanh(x)^2).

        Uses cached tanh output from forward pass.
        Note: params and grads are unused (Tanh has no parameters).
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var t = cache[batch, i]  # tanh(x) cached
                grad_input[batch, i] = grad_output[batch, i] * (1 - t * t)

    # =========================================================================
    # GPU Kernel Implementations (@always_inline for fusion)
    # =========================================================================
    #
    # These are the core GPU computations that can be inlined into fused kernels.
    # Tanh uses 1D elementwise parallelism.
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
        """Forward pass kernel implementation: y = tanh(x) with caching.

        This is the core GPU computation that can be inlined into fused kernels.
        Uses 1D elementwise parallelism.

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
        Block: (TPB,)

        Args:
            output: Output tensor [BATCH, dim] (written).
            input: Input tensor [BATCH, dim].
            cache: Cache buffer [BATCH, dim] for backward pass (written).
                   Stores tanh output for computing derivative.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = input[row, col]
        var val_f32 = rebind[Scalar[DType.float32]](val)
        var exp_val = exp(val_f32)
        var exp_neg_val = exp(-val_f32)
        var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
        var result = rebind[output.element_type](tanh_val)
        cache[row, col] = result  # Cache tanh output for backward
        output[row, col] = result

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

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = input[row, col]
        var val_f32 = rebind[Scalar[DType.float32]](val)
        var exp_val = exp(val_f32)
        var exp_neg_val = exp(-val_f32)
        var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
        output[row, col] = rebind[output.element_type](tanh_val)

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
        """Backward pass kernel implementation: dx = dy * (1 - tanh(x)^2).

        Uses cached tanh output from forward pass.

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var t = cache[row, col]  # tanh(x) was cached
        grad_input[row, col] = grad_output[row, col] * (1 - t * t)

    # =========================================================================
    # GPU Launchers (with DeviceContext)
    # =========================================================================
    #
    # These functions handle buffer-to-tensor conversion, grid/block config,
    # and kernel launch. They call the _kernel_impl functions.
    #
    # Note: Tanh has no parameters, so params_buf and grads_buf are unused
    # but kept for API consistency with Linear.
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],  # Unused for Tanh
        cache_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU with caching.

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * dim].
            input_buf: Input buffer [BATCH * dim].
            params_buf: Parameters buffer (unused for Tanh, kept for API consistency).
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
        params_buf: DeviceBuffer[dtype],  # Unused for Tanh
    ) raises:
        """Launch forward pass on GPU without caching (for inference).

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * dim].
            input_buf: Input buffer [BATCH * dim].
            params_buf: Parameters buffer (unused for Tanh, kept for API consistency).
        """
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())

        comptime total_elements = BATCH * Self.dim
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
        params_buf: DeviceBuffer[dtype],  # Unused for Tanh
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],  # Unused for Tanh
    ) raises:
        """Launch backward pass on GPU.

        Tanh has no parameters, so only grad_input is computed.

        Args:
            ctx: GPU device context.
            grad_input_buf: Gradient w.r.t. input [BATCH * dim] (written).
            grad_output_buf: Gradient w.r.t. output [BATCH * dim].
            params_buf: Parameters buffer (unused for Tanh).
            cache_buf: Cached tanh output from forward pass [BATCH * dim].
            grads_buf: Parameter gradients (unused for Tanh).
            batch_size: Runtime batch size for bounds checking.
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
    # Tanh is a leaf layer, so workspace is unused - just delegate to regular methods.
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Tanh
    ) raises:
        """GPU forward with workspace (workspace unused for Tanh)."""
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Tanh
    ) raises:
        """GPU forward without cache, with workspace (workspace unused for Tanh).
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
        workspace_buf: DeviceBuffer[dtype],  # Unused for Tanh
    ) raises:
        """GPU backward with workspace (workspace unused for Tanh)."""
        Self.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )
