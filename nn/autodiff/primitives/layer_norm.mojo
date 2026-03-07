from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import sqrt


struct LayerNormOp[dim: Int](DiffOp):
    """LayerNormOp: y = gamma * (x - mean) / sqrt(var + eps) + beta.

    Epsilon = 1e-5 (hardcoded).

    PARAM_SIZE = 2 * dim (gamma + beta)
    CACHE_SIZE = dim + 1 (x_hat[0..dim-1] + inv_std)

    Params layout: [gamma_0..gamma_{dim-1}, beta_0..beta_{dim-1}]
    Cache layout per sample: [x_hat_0..x_hat_{dim-1}, inv_std]
    """

    comptime OP_ID: Int = OpID.LAYER_NORM._value
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 2 * Self.dim
    comptime CACHE_SIZE: Int = Self.dim + 1

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
        var eps: Float64 = 1e-5
        var inv_dim = 1.0 / Float64(Self.dim)

        for b in range(BATCH):
            # Compute mean
            var mean: Float64 = 0.0
            for i in range(Self.dim):
                mean += Float64(rebind[Scalar[dtype]](input[b, i]))
            mean *= inv_dim

            # Compute variance
            var var_val: Float64 = 0.0
            for i in range(Self.dim):
                var diff = Float64(rebind[Scalar[dtype]](input[b, i])) - mean
                var_val += diff * diff
            var_val *= inv_dim

            # inv_std
            var inv_std = 1.0 / sqrt(var_val + eps)
            cache[b, Self.dim] = Scalar[dtype](inv_std)

            # Normalize, scale and shift
            for i in range(Self.dim):
                var x_hat = (
                    Float64(rebind[Scalar[dtype]](input[b, i])) - mean
                ) * inv_std
                cache[b, i] = Scalar[dtype](x_hat)
                var gamma = Float64(rebind[Scalar[dtype]](params[i]))
                var beta = Float64(
                    rebind[Scalar[dtype]](params[Self.dim + i])
                )
                output[b, i] = Scalar[dtype](gamma * x_hat + beta)

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
        var inv_dim = 1.0 / Float64(Self.dim)

        for b in range(BATCH):
            var inv_std = Float64(
                rebind[Scalar[dtype]](cache[b, Self.dim])
            )

            # Compute dx_hat = grad * gamma, and accumulate dgamma/dbeta
            # Also compute mean(dx_hat) and mean(dx_hat * x_hat)
            var mean_dxhat: Float64 = 0.0
            var mean_dxhat_xhat: Float64 = 0.0

            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var x_hat = Float64(rebind[Scalar[dtype]](cache[b, i]))
                var gamma = Float64(rebind[Scalar[dtype]](params[i]))

                # Accumulate dgamma and dbeta
                grad_params[i] = rebind[Scalar[dtype]](grad_params[i]) + Scalar[
                    dtype
                ](g * x_hat)
                grad_params[Self.dim + i] = rebind[Scalar[dtype]](
                    grad_params[Self.dim + i]
                ) + Scalar[dtype](g)

                var dx_hat = g * gamma
                mean_dxhat += dx_hat
                mean_dxhat_xhat += dx_hat * x_hat

            mean_dxhat *= inv_dim
            mean_dxhat_xhat *= inv_dim

            # dx = inv_std * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var x_hat = Float64(rebind[Scalar[dtype]](cache[b, i]))
                var gamma = Float64(rebind[Scalar[dtype]](params[i]))
                var dx_hat = g * gamma
                var dx = inv_std * (
                    dx_hat - mean_dxhat - x_hat * mean_dxhat_xhat
                )
                grad_input[b, i] = Scalar[dtype](dx)

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
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
        """Per-sample LayerNorm. Grid: (BATCH,), Block: (TPB,)."""
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))

        # Phase 1: compute mean
        var my_sum = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            my_sum += rebind[Scalar[dtype]](input[b, idx])
            idx += TPB
        var mean_val = block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim

        # Phase 2: compute variance
        var my_var = Scalar[dtype](0)
        idx = local_i
        while idx < Self.dim:
            var diff = rebind[Scalar[dtype]](input[b, idx]) - mean_val
            my_var += diff * diff
            idx += TPB
        var var_val = block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim

        var inv_std = Scalar[dtype](1.0) / sqrt(var_val + Scalar[dtype](1e-5))

        # Store inv_std in cache (thread 0 only)
        if local_i == 0:
            cache[b, Self.dim] = inv_std

        # Phase 3: normalize, scale, shift
        idx = local_i
        while idx < Self.dim:
            var x_hat = (rebind[Scalar[dtype]](input[b, idx]) - mean_val) * inv_std
            cache[b, idx] = x_hat
            output[b, idx] = rebind[Scalar[dtype]](params[idx]) * x_hat + rebind[Scalar[dtype]](params[Self.dim + idx])
            idx += TPB

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
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
        """Per-sample LayerNorm backward (dx only). Grid: (BATCH,), Block: (TPB,)."""
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))
        var inv_std = rebind[Scalar[dtype]](cache[b, Self.dim])

        # Phase 1: compute mean(dx_hat) and mean(dx_hat * x_hat)
        var my_dxhat = Scalar[dtype](0)
        var my_dxhat_xhat = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var gamma = rebind[Scalar[dtype]](params[idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            var dx_hat = g * gamma
            my_dxhat += dx_hat
            my_dxhat_xhat += dx_hat * x_hat
            idx += TPB

        var mean_dxhat = block.sum[block_size=TPB, broadcast=True](
            val=my_dxhat
        ) * inv_dim
        var mean_dxhat_xhat = block.sum[block_size=TPB, broadcast=True](
            val=my_dxhat_xhat
        ) * inv_dim

        # Phase 2: compute dx
        idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var gamma = rebind[Scalar[dtype]](params[idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            var dx_hat = g * gamma
            grad_input[b, idx] = inv_std * (
                dx_hat - mean_dxhat - x_hat * mean_dxhat_xhat
            )
            idx += TPB

    @always_inline
    @staticmethod
    fn backward_dparams_kernel_impl[
        BATCH: Int
    ](
        dgamma: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
        dbeta: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Reduce over batch for dgamma and dbeta. Grid: (dim,), Block: (TPB,)."""
        var col = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if col >= Self.dim:
            return

        var my_dgamma: dgamma.element_type = 0
        var my_dbeta: dbeta.element_type = 0
        var batch_idx = local_i
        while batch_idx < BATCH:
            var g = rebind[Scalar[dtype]](grad_output[batch_idx, col])
            var x_hat = rebind[Scalar[dtype]](cache[batch_idx, col])
            my_dgamma += g * x_hat
            my_dbeta += g
            batch_idx += TPB

        var total_dgamma = block.sum[block_size=TPB, broadcast=False](
            val=my_dgamma
        )
        var total_dbeta = block.sum[block_size=TPB, broadcast=False](
            val=my_dbeta
        )
        if local_i == 0:
            dgamma[col] = total_dgamma[0]
            dbeta[col] = total_dbeta[0]

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
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        @always_inline
        fn wrapper(
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
            Self.eval_kernel_impl[BATCH](output, input, params, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            params_immut,
            cache,
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
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        # Kernel 1: dx (per-sample reduction)
        @always_inline
        fn dx_wrapper(
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
            Self.backward_kernel_impl[BATCH](
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

        # Kernel 2: dgamma and dbeta (reduce over batch)
        var dgamma = LayoutTensor[
            dtype, Layout.row_major(Self.dim), MutAnyOrigin
        ](grad_params.ptr)
        var dbeta = LayoutTensor[
            dtype, Layout.row_major(Self.dim), MutAnyOrigin
        ](grad_params.ptr + Self.dim)

        @always_inline
        fn dp_wrapper(
            dgamma: LayoutTensor[
                dtype, Layout.row_major(Self.dim), MutAnyOrigin
            ],
            dbeta: LayoutTensor[
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
            Self.backward_dparams_kernel_impl[BATCH](
                dgamma, dbeta, grad_output, cache
            )

        ctx.enqueue_function[dp_wrapper, dp_wrapper](
            dgamma,
            dbeta,
            grad_output_immut,
            cache_immut,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
        )
