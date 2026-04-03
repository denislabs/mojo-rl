from ..constants import dtype
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.random.philox import Random as PhiloxRandom
from ..constants import TPB


struct Dropout[dim: Int, p: Float64, SEED: UInt64, training: Bool](Model):
    """Dropout regularization layer.

    During training: y = x * mask / (1 - p) where mask ~ Bernoulli(1-p)
    During inference: y = x (identity)

    Training mode = forward WITH cache (drops + caches mask for backward).
    Inference mode = forward WITHOUT cache (identity passthrough).

    Parameters:
        dim: Feature dimension.
        p: Dropout probability (fraction to drop).
        SEED: Base seed for PhiloxRandom.
        training: Whether in training mode (compile-time flag).

    PARAM_SIZE = 0 (no learnable parameters)
    CACHE_SIZE = dim if training else 0 (cache mask for backward)
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    # Only need cache during training (to store mask)
    comptime CACHE_SIZE: Int = Self.dim if Self.training else 0
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0  # Leaf layer

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
        """Training forward: apply dropout mask, cache for backward."""

        comptime if Self.training:
            var scale = Scalar[dtype](1.0 / (1.0 - Self.p))
            var zero = Scalar[dtype](0.0)
            var threshold = Scalar[dtype](Self.p)

            for batch in range(BATCH):
                for i in range(Self.dim):
                    var rng = PhiloxRandom(
                        seed=Self.SEED,
                        offset=UInt64(batch * Self.dim + i),
                    )
                    var rand = Scalar[dtype](rng.step_uniform()[0])
                    var mask: Scalar[dtype] = scale if rand >= threshold else zero
                    cache[batch, i] = mask
                    var in_val = rebind[Scalar[dtype]](input[batch, i])
                    output[batch, i] = in_val * mask
        else:
            for batch in range(BATCH):
                for i in range(Self.dim):
                    output[batch, i] = rebind[Scalar[dtype]](input[batch, i])

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
        """Inference forward: identity passthrough (no dropout)."""
        for batch in range(BATCH):
            for i in range(Self.dim):
                output[batch, i] = rebind[Scalar[dtype]](input[batch, i])

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
        """Backward pass: dx = dy * mask."""

        comptime if Self.training:
            for batch in range(BATCH):
                for i in range(Self.dim):
                    var mask = rebind[Scalar[dtype]](cache[batch, i])
                    var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                    grad_input[batch, i] = dy * mask
        else:
            for batch in range(BATCH):
                for i in range(Self.dim):
                    grad_input[batch, i] = rebind[Scalar[dtype]](
                        grad_output[batch, i]
                    )

    # =========================================================================
    # GPU Kernel Implementations
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
        """Training forward kernel: dropout with PhiloxRandom.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """

        comptime if Self.training:
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.dim:
                return

            var row = idx // Self.dim
            var col = idx % Self.dim

            # PhiloxRandom per element — no Float64, Metal-safe
            var rng = PhiloxRandom(seed=Self.SEED, offset=UInt64(idx))
            var rand = Scalar[dtype](rng.step_uniform()[0])

            var threshold = Scalar[dtype](Self.p)
            var scale = Scalar[dtype](1.0 / (1.0 - Self.p))
            var zero = Scalar[dtype](0.0)
            var mask: Scalar[dtype] = scale if rand >= threshold else zero

            cache[row, col] = mask
            var in_val = rebind[Scalar[dtype]](input[row, col])
            output[row, col] = in_val * mask
        else:
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.dim:
                return
            var row = idx // Self.dim
            var col = idx % Self.dim
            output[row, col] = rebind[Scalar[dtype]](input[row, col])

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
        """Inference forward kernel: identity passthrough.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        output[row, col] = rebind[Scalar[dtype]](input[row, col])

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
        """Backward kernel: dx = dy * mask.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """

        comptime if Self.training:
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.dim:
                return

            var row = idx // Self.dim
            var col = idx % Self.dim
            var dy = rebind[Scalar[dtype]](grad_output[row, col])
            var mask = rebind[Scalar[dtype]](cache[row, col])
            grad_input[row, col] = dy * mask
        else:
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.dim:
                return
            var row = idx // Self.dim
            var col = idx % Self.dim
            grad_input[row, col] = rebind[Scalar[dtype]](grad_output[row, col])

    # =========================================================================
    # GPU Launchers
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
        """GPU training forward with caching."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        comptime total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        comptime if Self.training:
            var cache_view = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ](cache.ptr)

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
                cache_view,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            def kernel_wrapper_infer(
                output: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
                ],
                input: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
                ],
            ):
                Self.forward_kernel_impl_no_cache[BATCH, dtype](output, input)

            ctx.enqueue_function[kernel_wrapper_infer, kernel_wrapper_infer](
                output,
                input_immut,
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
        """GPU inference forward: identity passthrough (no dropout)."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

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

        comptime total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

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
        """GPU backward pass."""
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)

        comptime total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        comptime if Self.training:
            var cache_immut = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ](cache.ptr)

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
        else:

            @always_inline
            def kernel_wrapper_infer(
                grad_input: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
                ],
                grad_output: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= BATCH * Self.dim:
                    return
                var row = idx // Self.dim
                var col = idx % Self.dim
                grad_input[row, col] = rebind[Scalar[dtype]](
                    grad_output[row, col]
                )

            ctx.enqueue_function[kernel_wrapper_infer, kernel_wrapper_infer](
                grad_input,
                grad_output_immut,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
            )
