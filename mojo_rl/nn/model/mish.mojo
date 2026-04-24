from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import exp, log


struct Mish[dim: Int](Model):
    """Mish activation: y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x))).

    Default activation for all NormedLinear blocks in TDMPC2.

    Gradient:
        sp = softplus(x) = log(1 + exp(x))
        tanh_sp = tanh(sp)
        dy/dx = tanh_sp + x * sigmoid(x) * (1 - tanh_sp^2)

    CACHE_SIZE = 2 * dim: stores [tanh_sp (dim) | x (dim)] for backward.
    WORKSPACE_SIZE_PER_SAMPLE = 0 (leaf layer).
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 2 * Self.dim  # [tanh_sp | x] per sample
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward: y = x * tanh(softplus(x)).

        Caches tanh(softplus(x)) and x for backward pass.
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var x_val = Float64(rebind[Scalar[dtype]](input[batch, i]))
                # softplus(x) = log(1 + exp(x)), numerically stable:
                # for large x, softplus(x) ≈ x
                var sp: Float64
                if x_val > 20.0:
                    sp = x_val
                else:
                    sp = log(1.0 + exp(x_val))
                var exp_sp = exp(sp)
                var exp_neg_sp = exp(-sp)
                var tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
                var y = x_val * tanh_sp
                cache[batch, i] = Scalar[dtype](tanh_sp)  # tanh(sp)
                cache[batch, Self.dim + i] = Scalar[dtype](x_val)  # x
                output[batch, i] = Scalar[dtype](y)

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass without caching (for inference)."""
        for batch in range(BATCH):
            for i in range(Self.dim):
                var x_val = Float64(rebind[Scalar[dtype]](input[batch, i]))
                var sp: Float64
                if x_val > 20.0:
                    sp = x_val
                else:
                    sp = log(1.0 + exp(x_val))
                var exp_sp = exp(sp)
                var exp_neg_sp = exp(-sp)
                var tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
                output[batch, i] = Scalar[dtype](x_val * tanh_sp)

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward: dx = dy * (tanh_sp + x * sigmoid(x) * (1 - tanh_sp^2)).

        Uses cached tanh_sp and x from forward pass.
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var tanh_sp = Float64(rebind[Scalar[dtype]](cache[batch, i]))
                var x_val = Float64(
                    rebind[Scalar[dtype]](cache[batch, Self.dim + i])
                )
                var sigmoid_x = 1.0 / (1.0 + exp(-x_val))
                var d_mish = tanh_sp + x_val * sigmoid_x * (
                    1.0 - tanh_sp * tanh_sp
                )
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                grad_input[batch, i] = Scalar[dtype](Float64(dy) * d_mish)

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
            dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
        ],
    ):
        """Forward kernel: y = x * tanh(softplus(x)) with caching.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var x_val = rebind[Scalar[DType.float32]](input[row, col])

        # Mish: x * tanh(softplus(x))
        # For large x (>15), tanh(softplus(x)) ≈ 1, so Mish(x) ≈ x
        # For very negative x (<-15), tanh(softplus(x)) ≈ 0, so Mish(x) ≈ 0
        var tanh_sp: Scalar[DType.float32]
        if x_val > Scalar[DType.float32](15.0):
            tanh_sp = Scalar[DType.float32](1.0)
        elif x_val < Scalar[DType.float32](-15.0):
            tanh_sp = Scalar[DType.float32](0.0)
        else:
            var sp: Scalar[DType.float32]
            if x_val > 20.0:
                sp = x_val
            else:
                sp = log(1.0 + exp(x_val))
            var exp_sp = exp(sp)
            var exp_neg_sp = exp(-sp)
            tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
        var result = rebind[output.element_type](x_val * tanh_sp)

        cache[row, col] = rebind[cache.element_type](tanh_sp)
        cache[row, Self.dim + col] = rebind[cache.element_type](x_val)
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
        """Forward kernel without caching (inference).

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var x_val = rebind[Scalar[DType.float32]](input[row, col])

        var tanh_sp: Scalar[DType.float32]
        if x_val > Scalar[DType.float32](15.0):
            tanh_sp = Scalar[DType.float32](1.0)
        elif x_val < Scalar[DType.float32](-15.0):
            tanh_sp = Scalar[DType.float32](0.0)
        else:
            var sp: Scalar[DType.float32]
            if x_val > 20.0:
                sp = x_val
            else:
                sp = log(1.0 + exp(x_val))
            var exp_sp = exp(sp)
            var exp_neg_sp = exp(-sp)
            tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
        output[row, col] = rebind[output.element_type](x_val * tanh_sp)

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
            dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
        ],
    ):
        """Backward kernel.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return

        var row = idx // Self.dim
        var col = idx % Self.dim
        var tanh_sp = rebind[Scalar[DType.float32]](cache[row, col])
        var x_val = rebind[Scalar[DType.float32]](cache[row, Self.dim + col])
        var sigmoid_x: Scalar[DType.float32] = 1.0 / (1.0 + exp(-x_val))
        var d_mish = tanh_sp + x_val * sigmoid_x * (1.0 - tanh_sp * tanh_sp)
        var dy = rebind[Scalar[DType.float32]](grad_output[row, col])
        grad_input[row, col] = rebind[grad_input.element_type](dy * d_mish)

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        var cache_2d = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
        ](cache.ptr)

        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
        @always_inline
        def kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            cache_2d,
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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch forward pass on GPU without caching (inference)."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward on stream — delegates to default stream."""
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, state, workspace)

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        """Launch backward pass on GPU."""
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
        ](cache.ptr)

        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
        @always_inline
        def kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
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
