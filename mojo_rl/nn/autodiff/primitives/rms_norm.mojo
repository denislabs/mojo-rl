from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import sqrt


struct RMSNormOp[dim: Int](DiffOp):
    """RMSNormOp: y = gamma * x / rms(x), where rms(x) = sqrt(mean(x^2) + eps).

    Epsilon = 1e-5 (hardcoded).

    PARAM_SIZE = dim (gamma only, no beta)
    CACHE_SIZE = dim + 1 (x_hat[0..dim-1] + rms_inv)

    Cache layout per sample: [x_hat_0..x_hat_{dim-1}, rms_inv]
    """

    comptime OP_ID: Int = OpID.RMS_NORM._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = Self.dim
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
        var eps: Float64 = 1e-5
        var inv_dim = 1.0 / Float64(Self.dim)

        for b in range(BATCH):
            # Compute mean(x^2)
            var mean_sq: Float64 = 0.0
            for i in range(Self.dim):
                var x = Float64(rebind[Scalar[dtype]](input[b, i]))
                mean_sq += x * x
            mean_sq *= inv_dim

            var rms_inv = 1.0 / sqrt(mean_sq + eps)
            cache[b, Self.dim] = Scalar[dtype](rms_inv)

            for i in range(Self.dim):
                var x = Float64(rebind[Scalar[dtype]](input[b, i]))
                var x_hat = x * rms_inv
                cache[b, i] = Scalar[dtype](x_hat)
                var gamma = Float64(rebind[Scalar[dtype]](params[i]))
                output[b, i] = Scalar[dtype](gamma * x_hat)

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
            var rms_inv = Float64(rebind[Scalar[dtype]](cache[b, Self.dim]))

            # Compute mean(grad * gamma * x_hat)
            var mean_gx: Float64 = 0.0
            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var gamma = Float64(rebind[Scalar[dtype]](params[i]))
                var x_hat = Float64(rebind[Scalar[dtype]](cache[b, i]))

                # Accumulate dgamma
                grad_params[i] = rebind[Scalar[dtype]](grad_params[i]) + Scalar[
                    dtype
                ](g * x_hat)

                mean_gx += g * gamma * x_hat

            mean_gx *= inv_dim

            # dx = rms_inv * (grad * gamma - x_hat * mean(grad * gamma * x_hat))
            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var gamma = Float64(rebind[Scalar[dtype]](params[i]))
                var x_hat = Float64(rebind[Scalar[dtype]](cache[b, i]))
                var dx = rms_inv * (g * gamma - x_hat * mean_gx)
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
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Per-sample RMSNorm. Grid: (BATCH,), Block: (TPB,)."""
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))

        # Phase 1: compute mean(x^2)
        var my_sq = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            var x = rebind[Scalar[dtype]](input[b, idx])
            my_sq += x * x
            idx += TPB
        var mean_sq = (
            block.sum[block_size=TPB, broadcast=True](val=my_sq) * inv_dim
        )

        var rms_inv = Scalar[dtype](1.0) / sqrt(mean_sq + Scalar[dtype](1e-5))

        if local_i == 0:
            cache[b, Self.dim] = rms_inv

        # Phase 2: normalize and scale
        idx = local_i
        while idx < Self.dim:
            var x_hat = rebind[Scalar[dtype]](input[b, idx]) * rms_inv
            cache[b, idx] = x_hat
            output[b, idx] = rebind[Scalar[dtype]](params[idx]) * x_hat
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
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Per-sample RMSNorm backward (dx only). Grid: (BATCH,), Block: (TPB,).
        """
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))
        var rms_inv = rebind[Scalar[dtype]](cache[b, Self.dim])

        # Phase 1: mean(grad * gamma * x_hat)
        var my_gx = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var gamma = rebind[Scalar[dtype]](params[idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            my_gx += g * gamma * x_hat
            idx += TPB

        var mean_gx = (
            block.sum[block_size=TPB, broadcast=True](val=my_gx) * inv_dim
        )

        # Phase 2: dx
        idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var gamma = rebind[Scalar[dtype]](params[idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            grad_input[b, idx] = rms_inv * (g * gamma - x_hat * mean_gx)
            idx += TPB

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
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Reduce over batch for dgamma. Grid: (dim,), Block: (TPB,)."""
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
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        @parameter
        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, params, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            params_immut,
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
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        # Kernel 1: dx (per-sample reduction)
        @parameter
        @always_inline
        def dx_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_kernel_impl[BATCH, dtype](
                grad_input, grad_output, params, cache
            )

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            params_immut,
            cache_immut,
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

        # Kernel 2: dgamma (reduce over batch)
        var dgamma = LayoutTensor[
            dtype, Layout.row_major(Self.dim), MutAnyOrigin
        ](grad_params.ptr)

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
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
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
