from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block


struct ReduceSum[dim: Int](DiffOp):
    """ReduceSum: output[b, 0] = sum_i(input[b, i]).

    Reduces the feature dimension to a scalar per batch element.

    PARAM_SIZE = 0
    CACHE_SIZE = 0
    """

    comptime OP_ID: Int = OpID.REDUCE_SUM._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 0

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    fn eval[
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
        for b in range(BATCH):
            var s = Scalar[dtype](0)
            for i in range(Self.dim):
                s += rebind[Scalar[dtype]](input[b, i])
            output[b, 0] = s

    @staticmethod
    fn vjp[
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            var g = grad_output[b, 0]
            for i in range(Self.dim):
                grad_input[b, i] = g

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        """One block per batch element. Threads cooperate to sum."""
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var my_sum: output.element_type = 0
        var idx = local_i
        while idx < Self.dim:
            my_sum += rebind[Scalar[dtype]](input[b, idx])
            idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_sum)
        if local_i == 0:
            output[b, 0] = total[0]

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ],
    ):
        """Broadcast grad_output[b, 0] to all grad_input[b, i]."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        grad_input[row, col] = rebind[Scalar[dtype]](grad_output[row, 0])

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
        BATCH: Int
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
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        @always_inline
        fn wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn vjp_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ) raises:
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](grad_output.ptr)
        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )


struct ReduceMean[dim: Int](DiffOp):
    """ReduceMean: output[b, 0] = mean_i(input[b, i]).

    Same as ReduceSum but divides by dim.

    PARAM_SIZE = 0
    CACHE_SIZE = 0
    """

    comptime OP_ID: Int = OpID.REDUCE_MEAN._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 0

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    fn eval[
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
        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))
        for b in range(BATCH):
            var s = Scalar[dtype](0)
            for i in range(Self.dim):
                s += rebind[Scalar[dtype]](input[b, i])
            output[b, 0] = s * inv_dim

    @staticmethod
    fn vjp[
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))
        for b in range(BATCH):
            var g = rebind[Scalar[dtype]](grad_output[b, 0]) * inv_dim
            for i in range(Self.dim):
                grad_input[b, i] = g

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var my_sum: output.element_type = 0
        var idx = local_i
        while idx < Self.dim:
            my_sum += rebind[Scalar[dtype]](input[b, idx])
            idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_sum)
        if local_i == 0:
            output[b, 0] = total[0] * Scalar[dtype](1.0 / Float64(Self.dim))

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        grad_input[row, col] = rebind[Scalar[dtype]](grad_output[row, 0]) * Scalar[dtype](
            1.0 / Float64(Self.dim)
        )

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
        BATCH: Int
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
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        @always_inline
        fn wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn vjp_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ) raises:
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](grad_output.ptr)
        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
