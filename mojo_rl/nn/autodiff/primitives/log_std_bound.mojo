"""LogStdBoundOp: smooth tanh-based clamp for policy log_std.

y = log_std_min + 0.5 * (log_std_max - log_std_min) * (tanh(x) + 1)

Reference: TD-MPC2 `math.log_std` (common/math.py:13). Used to bound the raw
log_std output of a Gaussian policy head into [log_std_min, log_std_max].

PARAM_SIZE = 0
CACHE_SIZE = dim   (caches tanh(x) for backward; dy/dx = 0.5*dif*(1-tanh²(x)))
"""

from std.math import exp

from ...constants import dtype, TPB
from ..op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct LogStdBoundOp[
    dim: Int,
    log_std_min_num: Int,
    log_std_min_den: Int,
    log_std_max_num: Int,
    log_std_max_den: Int,
](DiffOp):
    """LogStdBoundOp: y = lo + 0.5*dif*(tanh(x) + 1) where dif = hi - lo.

    Bounds are passed as comptime fractions (num/den) so they survive the
    DiffOp generic instantiation rules. For the standard TD-MPC2 setting
    `log_std_min=-10, log_std_max=2` use `LogStdBoundOp[dim, -10, 1, 2, 1]`.

    Parameters:
        dim: Per-sample feature dimension.
        log_std_min_num/den: Numerator/denominator of the lower bound.
        log_std_max_num/den: Numerator/denominator of the upper bound.
    """

    comptime OP_ID: Int = OpID.LOG_STD_BOUND._value
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

    @staticmethod
    def _lo[dtype: DType = DType.float32]() -> Scalar[dtype]:
        return Scalar[dtype](
            Float64(Self.log_std_min_num) / Float64(Self.log_std_min_den)
        )

    @staticmethod
    def _hi[dtype: DType = DType.float32]() -> Scalar[dtype]:
        return Scalar[dtype](
            Float64(Self.log_std_max_num) / Float64(Self.log_std_max_den)
        )

    @staticmethod
    def _half_dif[dtype: DType = DType.float32]() -> Scalar[dtype]:
        return Scalar[dtype](
            0.5
            * (
                Float64(Self.log_std_max_num) / Float64(Self.log_std_max_den)
                - Float64(Self.log_std_min_num) / Float64(Self.log_std_min_den)
            )
        )

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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var lo = Self._lo[dtype]()
        var half_dif = Self._half_dif[dtype]()
        for b in range(BATCH):
            for i in range(Self.dim):
                var x = rebind[Scalar[dtype]](input[b, i])
                var ex = exp(x)
                var enx = exp(-x)
                var t = (ex - enx) / (ex + enx)
                cache[b, i] = t
                # y = lo + half_dif * (tanh(x) + 1)
                output[b, i] = lo + half_dif * (t + Scalar[dtype](1.0))

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
        var half_dif = Self._half_dif[dtype]()
        for b in range(BATCH):
            for i in range(Self.dim):
                var t = rebind[Scalar[dtype]](cache[b, i])
                var deriv = half_dif * (Scalar[dtype](1.0) - t * t)
                grad_input[b, i] = (
                    rebind[Scalar[dtype]](grad_output[b, i]) * deriv
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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var x = rebind[Scalar[dtype]](input[row, col])
        var ex = exp(x)
        var enx = exp(-x)
        var t = (ex - enx) / (ex + enx)
        var lo = Scalar[dtype](
            Float64(Self.log_std_min_num) / Float64(Self.log_std_min_den)
        )
        var half_dif = Scalar[dtype](
            0.5
            * (
                Float64(Self.log_std_max_num) / Float64(Self.log_std_max_den)
                - Float64(Self.log_std_min_num) / Float64(Self.log_std_min_den)
            )
        )
        cache[row, col] = t
        output[row, col] = lo + half_dif * (t + Scalar[dtype](1.0))

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
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var t = rebind[Scalar[dtype]](cache[row, col])
        var half_dif = Scalar[dtype](
            0.5
            * (
                Float64(Self.log_std_max_num) / Float64(Self.log_std_max_den)
                - Float64(Self.log_std_min_num) / Float64(Self.log_std_min_den)
            )
        )
        var deriv = half_dif * (Scalar[dtype](1.0) - t * t)
        grad_input[row, col] = (
            rebind[Scalar[dtype]](grad_output[row, col]) * deriv
        )

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
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
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
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache.ptr)
        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

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
            Self.backward_kernel_impl[BATCH, dtype](
                grad_input, grad_output, cache
            )

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
