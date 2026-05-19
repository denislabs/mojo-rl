from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import sqrt
from std.sys import simd_width_of


comptime _CPU_SIMD_W = simd_width_of[dtype]()


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
        # SIMD path: reductions accumulate in Float64 (matches original
        # precision); affine pass stays in dtype SIMD for full speedup.
        # Without the Float64 promotion, NormedLinear[16,8] / [12,24] gradchecks
        # regress (max_rel jumps from <0.02 to ~0.1-0.2 — see commit log).
        comptime W = _CPU_SIMD_W
        var in_p = input.ptr
        var out_p = output.ptr
        var c_p = cache.ptr
        var gamma_p = params.ptr
        var beta_p = params.ptr + Self.dim
        var eps64: Float64 = 1e-5
        var inv_dim64 = 1.0 / Float64(Self.dim)

        for b in range(BATCH):
            var off = b * Self.dim

            # Pass 1: mean — Float64 accumulator via SIMD-cast.
            var s_vec64 = SIMD[DType.float64, W](0)
            var s_tail64 = Float64(0)
            var j = 0
            while j + W <= Self.dim:
                s_vec64 += in_p.load[width=W](off + j).cast[DType.float64]()
                j += W
            while j < Self.dim:
                s_tail64 += Float64(in_p[off + j])
                j += 1
            var mean64 = (s_vec64.reduce_add() + s_tail64) * inv_dim64
            var mean = Scalar[dtype](mean64)
            var mean_v = SIMD[dtype, W](mean)

            # Pass 2: variance — Float64 accumulator.
            var v_vec64 = SIMD[DType.float64, W](0)
            var v_tail64 = Float64(0)
            j = 0
            while j + W <= Self.dim:
                var d = in_p.load[width=W](off + j).cast[DType.float64]() - SIMD[
                    DType.float64, W
                ](mean64)
                v_vec64 += d * d
                j += W
            while j < Self.dim:
                var d = Float64(in_p[off + j]) - mean64
                v_tail64 += d * d
                j += 1
            var var_val64 = (v_vec64.reduce_add() + v_tail64) * inv_dim64
            var inv_std64 = 1.0 / sqrt(var_val64 + eps64)
            var inv_std = Scalar[dtype](inv_std64)
            var inv_std_v = SIMD[dtype, W](inv_std)
            c_p[off + Self.dim] = inv_std

            # Pass 3: normalize + affine, cache x_hat (Float32 SIMD).
            j = 0
            while j + W <= Self.dim:
                var x_hat = (in_p.load[width=W](off + j) - mean_v) * inv_std_v
                c_p.store(off + j, x_hat)
                out_p.store(
                    off + j,
                    gamma_p.load[width=W](j) * x_hat + beta_p.load[width=W](j),
                )
                j += W
            while j < Self.dim:
                var x_hat = (in_p[off + j] - mean) * inv_std
                c_p[off + j] = x_hat
                out_p[off + j] = gamma_p[j] * x_hat + beta_p[j]
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
        # Same precision discipline as eval: reductions in Float64.
        comptime W = _CPU_SIMD_W
        var go_p = grad_output.ptr
        var gi_p = grad_input.ptr
        var c_p = cache.ptr
        var gamma_p = params.ptr
        var dgamma_p = grad_params.ptr
        var dbeta_p = grad_params.ptr + Self.dim
        var inv_dim64 = 1.0 / Float64(Self.dim)

        for b in range(BATCH):
            var off = b * Self.dim
            var inv_std = c_p[off + Self.dim]
            var inv_std_v = SIMD[dtype, W](inv_std)

            # Pass 1: row reductions in Float64 + per-element dgamma/dbeta.
            var mdx_vec64 = SIMD[DType.float64, W](0)
            var mdxxh_vec64 = SIMD[DType.float64, W](0)
            var mdx_tail64 = Float64(0)
            var mdxxh_tail64 = Float64(0)
            var j = 0
            while j + W <= Self.dim:
                var g = go_p.load[width=W](off + j)
                var x_hat = c_p.load[width=W](off + j)
                var gamma = gamma_p.load[width=W](j)
                var dx_hat = g * gamma
                mdx_vec64 += dx_hat.cast[DType.float64]()
                mdxxh_vec64 += (dx_hat * x_hat).cast[DType.float64]()
                dgamma_p.store(j, dgamma_p.load[width=W](j) + g * x_hat)
                dbeta_p.store(j, dbeta_p.load[width=W](j) + g)
                j += W
            while j < Self.dim:
                var g = go_p[off + j]
                var x_hat = c_p[off + j]
                var gamma = gamma_p[j]
                var dx_hat = g * gamma
                mdx_tail64 += Float64(dx_hat)
                mdxxh_tail64 += Float64(dx_hat) * Float64(x_hat)
                dgamma_p[j] = dgamma_p[j] + g * x_hat
                dbeta_p[j] = dbeta_p[j] + g
                j += 1
            var mean_dxhat64 = (mdx_vec64.reduce_add() + mdx_tail64) * inv_dim64
            var mean_dxhat_xhat64 = (
                mdxxh_vec64.reduce_add() + mdxxh_tail64
            ) * inv_dim64
            var mean_dxhat = Scalar[dtype](mean_dxhat64)
            var mean_dxhat_xhat = Scalar[dtype](mean_dxhat_xhat64)
            var mdx_v = SIMD[dtype, W](mean_dxhat)
            var mdxxh_v = SIMD[dtype, W](mean_dxhat_xhat)

            # Pass 2: dx = inv_std * (dx_hat - mean - x_hat * mean_xhat)
            j = 0
            while j + W <= Self.dim:
                var g = go_p.load[width=W](off + j)
                var x_hat = c_p.load[width=W](off + j)
                var gamma = gamma_p.load[width=W](j)
                var dx_hat = g * gamma
                gi_p.store(
                    off + j,
                    inv_std_v * (dx_hat - mdx_v - x_hat * mdxxh_v),
                )
                j += W
            while j < Self.dim:
                var g = go_p[off + j]
                var x_hat = c_p[off + j]
                var gamma = gamma_p[j]
                var dx_hat = g * gamma
                gi_p[off + j] = inv_std * (
                    dx_hat - mean_dxhat - x_hat * mean_dxhat_xhat
                )
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
        var mean_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
        )

        # Phase 2: compute variance
        var my_var = Scalar[dtype](0)
        idx = local_i
        while idx < Self.dim:
            var diff = rebind[Scalar[dtype]](input[b, idx]) - mean_val
            my_var += diff * diff
            idx += TPB
        var var_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim
        )

        var inv_std = Scalar[dtype](1.0) / sqrt(var_val + Scalar[dtype](1e-5))

        # Store inv_std in cache (thread 0 only)
        if local_i == 0:
            cache[b, Self.dim] = inv_std

        # Phase 3: normalize, scale, shift
        idx = local_i
        while idx < Self.dim:
            var x_hat = (
                rebind[Scalar[dtype]](input[b, idx]) - mean_val
            ) * inv_std
            cache[b, idx] = x_hat
            output[b, idx] = rebind[Scalar[dtype]](
                params[idx]
            ) * x_hat + rebind[Scalar[dtype]](params[Self.dim + idx])
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
        """Per-sample LayerNorm backward (dx only). Grid: (BATCH,), Block: (TPB,).
        """
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

        var mean_dxhat = (
            block.sum[block_size=TPB, broadcast=True](val=my_dxhat) * inv_dim
        )
        var mean_dxhat_xhat = (
            block.sum[block_size=TPB, broadcast=True](val=my_dxhat_xhat)
            * inv_dim
        )

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
    def backward_dparams_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
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
        """Reduce over batch for dgamma and dbeta. Grid: (dim,), Block: (TPB,).
        """
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
            # Accumulate into dgamma/dbeta (pre-zeroed via zero_grads) so
            # multi-call backward sequences (MuZero K-step unroll,
            # DreamerV3/TD-MPC2 BPTT) sum gradients instead of overwriting.
            dgamma[col] = dgamma[col] + total_dgamma[0]
            dbeta[col] = dbeta[col] + total_dbeta[0]

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

        ctx.enqueue_function[wrapper](
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
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH, dtype](
                grad_input, grad_output, params, cache
            )

        ctx.enqueue_function[dx_wrapper](
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

        @parameter
        @always_inline
        def dp_wrapper(
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
            Self.backward_dparams_kernel_impl[BATCH, dtype](
                dgamma, dbeta, grad_output, cache
            )

        ctx.enqueue_function[dp_wrapper](
            dgamma,
            dbeta,
            grad_output_immut,
            cache_immut,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
        )
