"""LayerNormNoAffineOp: LayerNorm without learned scale/shift.

Forward:  y = (x - mean) / sqrt(var + eps)
Backward: dx = inv_std * (g - mean(g) - x_hat * mean(g * x_hat))

Used in AdaLN-zero conditional transformer blocks where the affine
modulation (scale, shift) is provided externally by ModulateOp. Matches
`nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` in PyTorch.

PARAM_SIZE = 0  (no learned gamma/beta)
CACHE_SIZE = dim + 1  (x_hat per element + inv_std)
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import sqrt


struct LayerNormNoAffineOp[dim: Int](DiffOp):
    """LayerNorm without learned affine parameters."""

    comptime OP_ID: Int = OpID.LAYER_NORM._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim + 1
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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var eps: Float64 = 1e-6
        var inv_dim = 1.0 / Float64(Self.dim)

        for b in range(BATCH):
            var mean: Float64 = 0.0
            for i in range(Self.dim):
                mean += Float64(rebind[Scalar[dtype]](input[b, i]))
            mean *= inv_dim

            var var_val: Float64 = 0.0
            for i in range(Self.dim):
                var diff = Float64(rebind[Scalar[dtype]](input[b, i])) - mean
                var_val += diff * diff
            var_val *= inv_dim

            var inv_std = 1.0 / sqrt(var_val + eps)
            cache[b, Self.dim] = Scalar[dtype](inv_std)

            for i in range(Self.dim):
                var x_hat = (
                    Float64(rebind[Scalar[dtype]](input[b, i])) - mean
                ) * inv_std
                cache[b, i] = Scalar[dtype](x_hat)
                output[b, i] = Scalar[dtype](x_hat)

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
        var inv_dim = 1.0 / Float64(Self.dim)

        for b in range(BATCH):
            var inv_std = Float64(rebind[Scalar[dtype]](cache[b, Self.dim]))

            # No gamma: dx_hat = g directly.
            var mean_g: Float64 = 0.0
            var mean_g_xhat: Float64 = 0.0
            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var x_hat = Float64(rebind[Scalar[dtype]](cache[b, i]))
                mean_g += g
                mean_g_xhat += g * x_hat
            mean_g *= inv_dim
            mean_g_xhat *= inv_dim

            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var x_hat = Float64(rebind[Scalar[dtype]](cache[b, i]))
                var dx = inv_std * (g - mean_g - x_hat * mean_g_xhat)
                grad_input[b, i] = Scalar[dtype](dx)

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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)
        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))

        var my_sum = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            my_sum += rebind[Scalar[dtype]](input[b, idx])
            idx += TPB
        var mean_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
        )

        var my_var = Scalar[dtype](0)
        idx = local_i
        while idx < Self.dim:
            var diff = rebind[Scalar[dtype]](input[b, idx]) - mean_val
            my_var += diff * diff
            idx += TPB
        var var_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim
        )

        var inv_std = Scalar[dtype](1.0) / sqrt(var_val + Scalar[dtype](1e-6))
        if local_i == 0:
            cache[b, Self.dim] = inv_std

        idx = local_i
        while idx < Self.dim:
            var x_hat = (
                rebind[Scalar[dtype]](input[b, idx]) - mean_val
            ) * inv_std
            cache[b, idx] = x_hat
            output[b, idx] = x_hat
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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)
        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))
        var inv_std = rebind[Scalar[dtype]](cache[b, Self.dim])

        var my_g = Scalar[dtype](0)
        var my_g_xhat = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            my_g += g
            my_g_xhat += g * x_hat
            idx += TPB

        var mean_g = (
            block.sum[block_size=TPB, broadcast=True](val=my_g) * inv_dim
        )
        var mean_g_xhat = (
            block.sum[block_size=TPB, broadcast=True](val=my_g_xhat) * inv_dim
        )

        idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            grad_input[b, idx] = inv_std * (g - mean_g - x_hat * mean_g_xhat)
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
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
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
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH, dtype](
                grad_input, grad_output, cache
            )

        ctx.enqueue_function[wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )
