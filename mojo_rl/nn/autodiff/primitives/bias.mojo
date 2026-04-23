from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block


struct BiasAdd[dim: Int](DiffOp):
    """BiasAdd: y = x + b  where x:(B, dim), b:(dim,), y:(B, dim).

    Broadcast addition of a bias vector.

    PARAM_SIZE = dim (the bias vector)
    CACHE_SIZE = 0 (no cache needed — bias backward is identity + sum)
    """

    comptime OP_ID: Int = OpID.BIAS_ADD._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = Self.dim
    comptime CACHE_SIZE: Int = 0
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    def eval[
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
        """Forward: output = input + bias."""
        for b in range(BATCH):
            for i in range(Self.dim):
                output[b, i] = input[b, i] + params[i]

    @staticmethod
    def vjp[
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward: grad_input = grad_out, db += sum(grad_out, axis=0)."""
        for b in range(BATCH):
            # grad_input = grad_output (identity for addition)
            for i in range(Self.dim):
                grad_input[b, i] = grad_output[b, i]

            # db += sum(grad_output, axis=0) (ACCUMULATE)
            for i in range(Self.dim):
                grad_params[i] = grad_params[i] + grad_output[b, i]

    # =========================================================================
    # GPU kernel implementations
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        bias: LayoutTensor[dtype, Layout.row_major(Self.dim), ImmutAnyOrigin],
    ):
        """Forward kernel: y = x + b.

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        output[row, col] = rebind[Scalar[dtype]](input[row, col]) + rebind[
            Scalar[dtype]
        ](bias[col])

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        """Backward kernel for grad_input: dx = dy (identity).

        Grid: ((BATCH * dim + TPB - 1) // TPB,)
        Block: (TPB,)
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        grad_input[row, col] = rebind[Scalar[dtype]](grad_output[row, col])

    @always_inline
    @staticmethod
    def backward_db_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        db: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        """Backward kernel for bias gradient: db = sum(dy, axis=0).

        Grid: (dim,)
        Block: (TPB,)
        """
        var col = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if col >= Self.dim:
            return

        var my_sum: db.element_type = 0
        var batch_idx = local_i
        while batch_idx < BATCH:
            my_sum += rebind[Scalar[dtype]](grad_output[batch_idx, col])
            batch_idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_sum)
        if local_i == 0:
            db[col] = total[0]

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    def eval_gpu[
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
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)
        var bias = LayoutTensor[
            dtype, Layout.row_major(Self.dim), ImmutAnyOrigin
        ](params.ptr)

        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            bias: LayoutTensor[
                dtype, Layout.row_major(Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, bias)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            bias,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    def vjp_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var db = LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin](
            grad_params.ptr
        )

        # Kernel 1: dx = dy (identity copy)
        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
        @always_inline
        def dx_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH, dtype](grad_input, grad_output)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

        # Kernel 2: db = sum(dy, axis=0)
        @parameter
        @always_inline
        def db_wrapper(
            db: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_db_kernel_impl[BATCH, dtype](db, grad_output)

        ctx.enqueue_function[db_wrapper, db_wrapper](
            db,
            grad_output_immut,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
        )
