from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import exp
from std.math import max as math_max
from std.sys import simd_width_of


comptime _CPU_SIMD_W = simd_width_of[dtype]()


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
        comptime W = _CPU_SIMD_W
        var in_p = input.ptr
        var out_p = output.ptr
        var c_p = cache.ptr
        for b in range(BATCH):
            var off = b * Self.dim

            # Pass 1: row max (SIMD reduce + scalar tail)
            var max_vec = SIMD[dtype, W](in_p[off])
            var max_scalar = in_p[off]
            var j = 0
            while j + W <= Self.dim:
                max_vec = math_max(max_vec, in_p.load[width=W](off + j))
                j += W
            while j < Self.dim:
                var v = in_p[off + j]
                if v > max_scalar:
                    max_scalar = v
                j += 1
            var row_max = max_scalar
            var vmax = max_vec.reduce_max()
            if vmax > row_max:
                row_max = vmax
            var rm_v = SIMD[dtype, W](row_max)

            # Pass 2: exp(x - max), accumulate sum
            var sum_vec = SIMD[dtype, W](0)
            var sum_tail = Scalar[dtype](0)
            j = 0
            while j + W <= Self.dim:
                var e = exp(in_p.load[width=W](off + j) - rm_v)
                out_p.store(off + j, e)
                sum_vec += e
                j += W
            while j < Self.dim:
                var e = exp(in_p[off + j] - row_max)
                out_p[off + j] = e
                sum_tail += e
                j += 1
            var inv_sum = Scalar[dtype](1) / (sum_vec.reduce_add() + sum_tail)
            var inv_v = SIMD[dtype, W](inv_sum)

            # Pass 3: normalize + store cache
            j = 0
            while j + W <= Self.dim:
                var y = out_p.load[width=W](off + j) * inv_v
                out_p.store(off + j, y)
                c_p.store(off + j, y)
                j += W
            while j < Self.dim:
                var y = out_p[off + j] * inv_sum
                out_p[off + j] = y
                c_p[off + j] = y
                j += 1

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
        comptime W = _CPU_SIMD_W
        var go_p = grad_output.ptr
        var gi_p = grad_input.ptr
        var c_p = cache.ptr
        for b in range(BATCH):
            var off = b * Self.dim

            # Pass 1: dot = sum_j grad[b,j] * y[b,j]
            var dot_vec = SIMD[dtype, W](0)
            var dot_tail = Scalar[dtype](0)
            var j = 0
            while j + W <= Self.dim:
                dot_vec += go_p.load[width=W](off + j) * c_p.load[width=W](
                    off + j
                )
                j += W
            while j < Self.dim:
                dot_tail += go_p[off + j] * c_p[off + j]
                j += 1
            var dot = dot_vec.reduce_add() + dot_tail
            var dot_v = SIMD[dtype, W](dot)

            # Pass 2: dx = y * (g - dot)
            j = 0
            while j + W <= Self.dim:
                var g = go_p.load[width=W](off + j)
                var y = c_p.load[width=W](off + j)
                gi_p.store(off + j, y * (g - dot_v))
                j += W
            while j < Self.dim:
                gi_p[off + j] = c_p[off + j] * (go_p[off + j] - dot)
                j += 1

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

        ctx.enqueue_function[wrapper](
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

        ctx.enqueue_function[wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )
