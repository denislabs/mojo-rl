from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block


struct ElemMul[dim: Int](DiffOp):
    """ElemMul: y[i] = x[i] * gamma[i] (learned elementwise scaling).

    PARAM_SIZE = dim (gamma vector)
    CACHE_SIZE = dim (caches input x for backward)
    """

    comptime OP_ID: Int = OpID.ELEM_MUL._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = Self.dim
    comptime CACHE_SIZE: Int = Self.dim
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
        for b in range(BATCH):
            for i in range(Self.dim):
                var x = input[b, i]
                cache[b, i] = x
                output[b, i] = x * params[i]

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
        for b in range(BATCH):
            for i in range(Self.dim):
                # dx = grad * gamma
                grad_input[b, i] = grad_output[b, i] * params[i]
                # dgamma += grad * x (accumulate over batch)
                grad_params[i] = (
                    grad_params[i] + grad_output[b, i] * cache[b, i]
                )

    # =========================================================================
    # GPU kernels
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
        gamma: LayoutTensor[dtype, Layout.row_major(Self.dim), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var x = rebind[Scalar[dtype]](input[row, col])
        cache[row, col] = x
        output[row, col] = x * rebind[Scalar[dtype]](gamma[col])

    @always_inline
    @staticmethod
    def backward_dx_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        gamma: LayoutTensor[dtype, Layout.row_major(Self.dim), ImmutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        grad_input[row, col] = rebind[Scalar[dtype]](
            grad_output[row, col]
        ) * rebind[Scalar[dtype]](gamma[col])

    @always_inline
    @staticmethod
    def backward_dgamma_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        dgamma: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        """Dgamma[col] = sum_b(grad_output[b, col] * x[b, col]).

        Grid: (dim,)  Block: (TPB,)
        """
        var col = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if col >= Self.dim:
            return

        var my_sum: dgamma.element_type = 0
        var batch_idx = local_i
        while batch_idx < BATCH:
            my_sum += rebind[Scalar[dtype]](
                grad_output[batch_idx, col]
            ) * rebind[Scalar[dtype]](cache[batch_idx, col])
            batch_idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_sum)
        if local_i == 0:
            # Accumulate into dgamma (pre-zeroed via zero_grads) so multi-call
            # backward sequences sum scale gradients instead of overwriting.
            dgamma[col] = dgamma[col] + total[0]

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
        var gamma = LayoutTensor[
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
            gamma: LayoutTensor[
                dtype, Layout.row_major(Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, gamma, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            gamma,
            cache,
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
        var gamma = LayoutTensor[
            dtype, Layout.row_major(Self.dim), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache.ptr)
        var dgamma = LayoutTensor[
            dtype, Layout.row_major(Self.dim), MutAnyOrigin
        ](grad_params.ptr)

        # Kernel 1: dx = grad * gamma
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
            gamma: LayoutTensor[
                dtype, Layout.row_major(Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_dx_kernel_impl[BATCH, dtype](grad_input, grad_output, gamma)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            gamma,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

        # Kernel 2: dgamma = sum(grad * x, axis=0)
        @parameter
        @always_inline
        def dgamma_wrapper(
            dgamma: LayoutTensor[
                dtype, Layout.row_major(Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_dgamma_kernel_impl[BATCH, dtype](dgamma, grad_output, cache)

        ctx.enqueue_function[dgamma_wrapper, dgamma_wrapper](
            dgamma,
            grad_output_immut,
            cache_immut,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
        )
