from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import exp


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

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        """Move constructor for Sequential composition."""
        pass

    def __init__(out self, *, copy: Self):
        """Copy constructor for Copyable trait."""
        pass

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        pass

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
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
        """Forward: y = tanh(x).

        Caches tanh output for backward pass (needed for derivative).
        Note: params is unused (Tanh has no parameters).
        """
        from std.math import exp

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

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
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

        Note: params is unused (Tanh has no parameters).
        """
        from std.math import exp

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

    @staticmethod
    def backward[
        BATCH: Int, dtype: DType = DType.float32
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
    def forward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32,
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
    def forward_kernel_impl_no_cache[
        BATCH: Int, dtype: DType = DType.float32,
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
    def backward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32,
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
    def forward_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch forward pass on GPU with caching."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
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
            Self.forward_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int, dtype: DType = DType.float32
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
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch forward pass on GPU without caching (for inference)."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl_no_cache[BATCH, dtype](output, input)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
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
        """GPU forward on stream — delegates to default stream."""
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, workspace)

    @staticmethod
    def backward_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch backward pass on GPU.

        Tanh has no parameters, so only grad_input is computed.
        """
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache.ptr)

        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
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
            Self.backward_kernel_impl[BATCH, dtype](grad_input, grad_output, cache)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
