from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random import seed, random_ui64
from ..constants import TPB


struct Dropout[dim: Int, p: Float64, SEED: UInt64, training: Bool](Model):
    """Dropout regularization layer.

    During training: y = x * mask / (1 - p) where mask ~ Bernoulli(1-p)
    During inference: y = x (identity)

    The training flag is compile-time for zero overhead when disabled.

    Parameters:
        dim: Feature dimension.
        p: Dropout probability (fraction to drop).
        SEED: Seed for random number generation.
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
    fn _random_from_seed(seed: UInt64) -> Float64:
        """Generate random float in [0, 1) using xorshift64."""
        var x = seed
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        # Convert to [0, 1)
        return Float64(x & 0xFFFFFFFFFFFF) / Float64(0xFFFFFFFFFFFF)

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
        """Forward pass with caching."""

        comptime if Self.training:
            # Training mode: apply dropout mask
            var scale = Scalar[dtype](1.0 / (1.0 - Self.p))
            var zero = Scalar[dtype](0.0)

            for batch in range(BATCH):
                for i in range(Self.dim):
                    # Generate deterministic random from element index
                    var elem_seed = Self.SEED ^ UInt64(
                        (batch * Self.dim + i) * 2654435761
                    )
                    var rand = Self._random_from_seed(elem_seed)
                    var keep = rand >= Self.p
                    var mask: Scalar[dtype] = scale if keep else zero
                    cache[batch, i] = mask  # Cache mask for backward
                    var in_val = rebind[Scalar[dtype]](input[batch, i])
                    output[batch, i] = in_val * mask
        else:
            # Inference mode: identity pass-through
            for batch in range(BATCH):
                for i in range(Self.dim):
                    output[batch, i] = rebind[Scalar[dtype]](input[batch, i])

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
        """Forward pass with caching."""

        comptime if Self.training:
            # Training mode: apply dropout mask
            var scale = Scalar[dtype](1.0 / (1.0 - Self.p))
            var zero = Scalar[dtype](0.0)

            for batch in range(BATCH):
                for i in range(Self.dim):
                    # Generate deterministic random from element index
                    var elem_seed = Self.SEED ^ UInt64(
                        (batch * Self.dim + i) * 2654435761
                    )
                    var rand = Self._random_from_seed(elem_seed)
                    var keep = rand >= Self.p
                    var mask: Scalar[dtype] = scale if keep else zero
                    var in_val = rebind[Scalar[dtype]](input[batch, i])
                    output[batch, i] = in_val * mask
        else:
            # Inference mode: identity pass-through
            for batch in range(BATCH):
                for i in range(Self.dim):
                    output[batch, i] = rebind[Scalar[dtype]](input[batch, i])

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
        """Backward pass: dx = dy * mask."""

        comptime if Self.training:
            # Apply same mask as forward
            for batch in range(BATCH):
                for i in range(Self.dim):
                    var mask = rebind[Scalar[dtype]](cache[batch, i])
                    var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                    grad_input[batch, i] = dy * mask
        else:
            # Inference mode: identity gradient
            for batch in range(BATCH):
                for i in range(Self.dim):
                    grad_input[batch, i] = rebind[Scalar[dtype]](
                        grad_output[batch, i]
                    )

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================
    #
    # GPU dropout uses thread index as part of random seed to generate
    # independent random numbers per element.
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
        seed_base: UInt64,
    ):
        """Forward pass kernel with caching (training mode).

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """

        comptime if Self.training:
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.dim:
                return

            var row = idx // Self.dim
            var col = idx % Self.dim

            # Generate random number using xorshift with unique seed per element
            var x = seed_base ^ UInt64(idx * 2654435761)
            x ^= x << 13
            x ^= x >> 7
            x ^= x << 17
            var rand = Float64(x & 0xFFFFFFFFFFFF) / Float64(0xFFFFFFFFFFFF)

            var keep = rand >= Self.p
            var scale = Scalar[dtype](1.0 / (1.0 - Self.p))
            var zero = Scalar[dtype](0.0)
            var mask: Scalar[dtype] = scale if keep else zero

            cache[row, col] = mask
            var in_val = rebind[Scalar[dtype]](input[row, col])
            output[row, col] = in_val * mask
        else:
            # Inference mode
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH * Self.dim:
                return
            var row = idx // Self.dim
            var col = idx % Self.dim
            output[row, col] = rebind[Scalar[dtype]](input[row, col])

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
        """Forward pass kernel without caching (identity).

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
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
        """Backward pass kernel: dx = dy * mask.

        Grid: ((batch_size * dim + TPB - 1) // TPB,)
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
        """Launch forward pass on GPU with caching."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        comptime total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        comptime if Self.training:
            # CACHE_SIZE == Self.dim when training=True, so cache has the right layout
            var cache_view = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ](cache.ptr)

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
                seed: UInt64,
            ):
                Self.forward_kernel_impl[BATCH](output, input, cache, seed)

            ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
                output,
                input_immut,
                cache_view,
                Self.SEED,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            fn kernel_wrapper_infer(
                output: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
                ],
                input: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
                ],
            ):
                Self.forward_kernel_impl_no_cache[BATCH](output, input)

            ctx.enqueue_function[kernel_wrapper_infer, kernel_wrapper_infer](
                output,
                input_immut,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
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
        """Launch forward pass on GPU without caching (identity)."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

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

        comptime total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
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
        """Launch backward pass on GPU."""
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)

        comptime total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        comptime if Self.training:
            # CACHE_SIZE == Self.dim when training=True
            var cache_immut = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ](cache.ptr)

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
                grad_output_immut,
                cache_immut,
                grid_dim=(grid_x,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            fn kernel_wrapper_infer(
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
