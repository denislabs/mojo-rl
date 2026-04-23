from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import exp


struct SoftmaxOp[dim: Int](DiffOp):
    """SoftmaxOp: y = exp(x - max(x)) / sum(exp(x - max(x))).

    Numerically stable softmax with cached output for backward.

    PARAM_SIZE = 0
    CACHE_SIZE = dim (caches softmax output y)
    """

    comptime OP_ID: Int = OpID.SOFTMAX._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
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
            # Find max for numerical stability
            var max_val = Float64(rebind[Scalar[dtype]](input[b, 0]))
            for i in range(1, Self.dim):
                var v = Float64(rebind[Scalar[dtype]](input[b, i]))
                if v > max_val:
                    max_val = v

            # Compute exp(x - max) and sum
            var sum_exp: Float64 = 0.0
            for i in range(Self.dim):
                var v = Float64(rebind[Scalar[dtype]](input[b, i]))
                var e = exp(v - max_val)
                output[b, i] = Scalar[dtype](e)
                sum_exp += e

            # Normalize
            var inv_sum = 1.0 / sum_exp
            for i in range(Self.dim):
                var y = Scalar[dtype](
                    Float64(rebind[Scalar[dtype]](output[b, i])) * inv_sum
                )
                output[b, i] = y
                cache[b, i] = y

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
        # dx[b,i] = y[b,i] * (grad[b,i] - sum_j(grad[b,j] * y[b,j]))
        for b in range(BATCH):
            # Compute dot = sum_j(grad[b,j] * y[b,j])
            var dot: Float64 = 0.0
            for j in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, j]))
                var y = Float64(rebind[Scalar[dtype]](cache[b, j]))
                dot += g * y

            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var y = Float64(rebind[Scalar[dtype]](cache[b, i]))
                grad_input[b, i] = Scalar[dtype](y * (g - dot))

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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        """Per-sample softmax. Grid: (BATCH,), Block: (TPB,)."""
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        # Phase 1: find max
        var my_max = Scalar[dtype](-1e30)
        var idx = local_i
        while idx < Self.dim:
            var v = rebind[Scalar[dtype]](input[b, idx])
            if v > my_max:
                my_max = v
            idx += TPB

        var global_max = block.max[block_size=TPB, broadcast=True](val=my_max)

        # Phase 2: compute exp(x - max) and sum
        var my_sum = Scalar[dtype](0)
        idx = local_i
        while idx < Self.dim:
            var e = exp(rebind[Scalar[dtype]](input[b, idx]) - global_max)
            output[b, idx] = e
            my_sum += e
            idx += TPB

        var total_sum = block.sum[block_size=TPB, broadcast=True](val=my_sum)

        # Phase 3: normalize
        var inv_sum = Scalar[dtype](1.0) / total_sum
        idx = local_i
        while idx < Self.dim:
            var y = rebind[Scalar[dtype]](output[b, idx]) * inv_sum
            output[b, idx] = y
            cache[b, idx] = y
            idx += TPB

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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        """Per-sample softmax backward. Grid: (BATCH,), Block: (TPB,)."""
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        # Phase 1: compute dot = sum_j(grad[b,j] * y[b,j])
        var my_dot = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            my_dot += rebind[Scalar[dtype]](grad_output[b, idx]) * rebind[
                Scalar[dtype]
            ](cache[b, idx])
            idx += TPB

        var dot = block.sum[block_size=TPB, broadcast=True](val=my_dot)

        # Phase 2: dx[b,i] = y[b,i] * (grad[b,i] - dot)
        idx = local_i
        while idx < Self.dim:
            var y = rebind[Scalar[dtype]](cache[b, idx])
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            grad_input[b, idx] = y * (g - dot)
            idx += TPB

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

        @parameter
        @always_inline
        def wrapper(
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
            Self.eval_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(BATCH,),
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
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache.ptr)

        @parameter
        @always_inline
        def wrapper(
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

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )
