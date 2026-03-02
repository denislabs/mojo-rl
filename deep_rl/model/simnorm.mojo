from ..constants import dtype, TPB
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from math import exp


struct SimNorm[dim: Int, simplex_dim: Int = 8](Model):
    """Simplicial Normalization: softmax applied independently over groups.

    SimNorm(simplex_dim)(x):
      1. Reshape x: [B, D] -> [B, D / simplex_dim, simplex_dim]
      2. Apply Softmax over the last (group) dimension
      3. Reshape back -> [B, D]

    Used on the dynamics model output to stabilize the latent space.
    Replaces LayerNorm on the dynamics head's final projection.

    Parameters:
        dim: Feature dimension. Must be divisible by simplex_dim.
        simplex_dim: Group size for softmax (default: 8).

    No learned parameters (pure normalization).

    PARAM_SIZE = 0
    CACHE_SIZE = dim (caches softmax output per group for backward)
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim   # softmax outputs for backward
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0
    comptime N_GROUPS: Int = Self.dim // Self.simplex_dim

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn __copyinit__(out self, other: Self):
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
        """Forward: apply softmax independently to each group of simplex_dim elements.

        Caches softmax output for backward pass.
        """
        for batch in range(BATCH):
            for g in range(Self.N_GROUPS):
                var base = g * Self.simplex_dim
                # Find max for numerical stability
                var max_val = Float64(
                    rebind[Scalar[dtype]](input[batch, base])
                )
                for k in range(1, Self.simplex_dim):
                    var v = Float64(
                        rebind[Scalar[dtype]](input[batch, base + k])
                    )
                    if v > max_val:
                        max_val = v
                # Compute sum of exp
                var sum_exp: Float64 = 0.0
                for k in range(Self.simplex_dim):
                    var v = Float64(
                        rebind[Scalar[dtype]](input[batch, base + k])
                    )
                    sum_exp += exp(v - max_val)
                # Compute softmax and cache
                for k in range(Self.simplex_dim):
                    var v = Float64(
                        rebind[Scalar[dtype]](input[batch, base + k])
                    )
                    var s = Scalar[dtype](exp(v - max_val) / sum_exp)
                    output[batch, base + k] = s
                    cache[batch, base + k] = s

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
        """Forward pass without caching (inference)."""
        for batch in range(BATCH):
            for g in range(Self.N_GROUPS):
                var base = g * Self.simplex_dim
                var max_val = Float64(
                    rebind[Scalar[dtype]](input[batch, base])
                )
                for k in range(1, Self.simplex_dim):
                    var v = Float64(
                        rebind[Scalar[dtype]](input[batch, base + k])
                    )
                    if v > max_val:
                        max_val = v
                var sum_exp: Float64 = 0.0
                for k in range(Self.simplex_dim):
                    var v = Float64(
                        rebind[Scalar[dtype]](input[batch, base + k])
                    )
                    sum_exp += exp(v - max_val)
                for k in range(Self.simplex_dim):
                    var v = Float64(
                        rebind[Scalar[dtype]](input[batch, base + k])
                    )
                    output[batch, base + k] = Scalar[dtype](
                        exp(v - max_val) / sum_exp
                    )

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
        """Backward for softmax groups: dx_i = y_i * (dy_i - sum_j(dy_j * y_j)).

        Standard softmax backward applied to each group independently.
        """
        for batch in range(BATCH):
            for g in range(Self.N_GROUPS):
                var base = g * Self.simplex_dim
                # Compute dot product: sum_j(dy_j * y_j) within group
                var dot: Float64 = 0.0
                for k in range(Self.simplex_dim):
                    var dy = Float64(
                        rebind[Scalar[dtype]](grad_output[batch, base + k])
                    )
                    var y = Float64(
                        rebind[Scalar[dtype]](cache[batch, base + k])
                    )
                    dot += dy * y
                # dx_i = y_i * (dy_i - dot)
                for k in range(Self.simplex_dim):
                    var dy = Float64(
                        rebind[Scalar[dtype]](grad_output[batch, base + k])
                    )
                    var y = Float64(
                        rebind[Scalar[dtype]](cache[batch, base + k])
                    )
                    grad_input[batch, base + k] = Scalar[dtype](
                        y * (dy - dot)
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
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        """GPU forward: one thread per (batch, group).

        Grid: ((BATCH * N_GROUPS + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.N_GROUPS:
            return

        var b = idx // Self.N_GROUPS
        var g = idx % Self.N_GROUPS
        var base = g * Self.simplex_dim

        # Find max
        var max_val = rebind[Scalar[DType.float32]](input[b, base])
        for k in range(1, Self.simplex_dim):
            var v = rebind[Scalar[DType.float32]](input[b, base + k])
            if v > max_val:
                max_val = v

        # Sum exp
        var sum_exp: Scalar[DType.float32] = 0.0
        for k in range(Self.simplex_dim):
            var v = rebind[Scalar[DType.float32]](input[b, base + k])
            sum_exp = sum_exp + exp(v - max_val)

        # Write softmax
        for k in range(Self.simplex_dim):
            var v = rebind[Scalar[DType.float32]](input[b, base + k])
            var s = exp(v - max_val) / sum_exp
            var result = rebind[output.element_type](s)
            output[b, base + k] = result
            cache[b, base + k] = result

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
        """GPU forward without caching."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.N_GROUPS:
            return

        var b = idx // Self.N_GROUPS
        var g = idx % Self.N_GROUPS
        var base = g * Self.simplex_dim

        var max_val = rebind[Scalar[DType.float32]](input[b, base])
        for k in range(1, Self.simplex_dim):
            var v = rebind[Scalar[DType.float32]](input[b, base + k])
            if v > max_val:
                max_val = v

        var sum_exp: Scalar[DType.float32] = 0.0
        for k in range(Self.simplex_dim):
            var v = rebind[Scalar[DType.float32]](input[b, base + k])
            sum_exp = sum_exp + exp(v - max_val)

        for k in range(Self.simplex_dim):
            var v = rebind[Scalar[DType.float32]](input[b, base + k])
            output[b, base + k] = rebind[output.element_type](
                exp(v - max_val) / sum_exp
            )

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
        """GPU backward: one thread per (batch, group)."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.N_GROUPS:
            return

        var b = idx // Self.N_GROUPS
        var g = idx % Self.N_GROUPS
        var base = g * Self.simplex_dim

        # dot = sum_j(dy_j * y_j)
        var dot: Scalar[DType.float32] = 0.0
        for k in range(Self.simplex_dim):
            var dy = rebind[Scalar[DType.float32]](grad_output[b, base + k])
            var y = rebind[Scalar[DType.float32]](cache[b, base + k])
            dot = dot + dy * y

        for k in range(Self.simplex_dim):
            var dy = rebind[Scalar[DType.float32]](grad_output[b, base + k])
            var y = rebind[Scalar[DType.float32]](cache[b, base + k])
            grad_input[b, base + k] = rebind[grad_input.element_type](
                y * (dy - dot)
            )

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
        workspace_buf: DeviceBuffer[dtype],
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

        comptime total = BATCH * Self.N_GROUPS
        var grid_x = (total + TPB - 1) // TPB

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
            output, input, cache,
            grid_dim=(grid_x,), block_dim=(TPB,),
        )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch forward pass on GPU without caching."""
        var output = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](output_buf.unsafe_ptr())
        var input = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input_buf.unsafe_ptr())

        comptime total = BATCH * Self.N_GROUPS
        var grid_x = (total + TPB - 1) // TPB

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
            output, input,
            grid_dim=(grid_x,), block_dim=(TPB,),
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
        workspace_buf: DeviceBuffer[dtype],
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

        comptime total = BATCH * Self.N_GROUPS
        var grid_x = (total + TPB - 1) // TPB

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
            grad_input, grad_output, cache,
            grid_dim=(grid_x,), block_dim=(TPB,),
        )
