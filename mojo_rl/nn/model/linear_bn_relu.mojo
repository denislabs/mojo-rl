"""Fused Linear + BatchNorm + ReLU as a single Model layer.

Eliminates 2 extra layers in Sequential (BN + ReLU), reducing kernel
launches from 6 per block (3 fwd + 3 bwd) to 4 (2 fwd + 2 bwd).

Forward:
  1. Linear (matmul + bias) → pre_bn
  2. BatchNorm + ReLU per feature → output

Backward:
  1. ReLU + BN backward per feature → grad_pre_bn
  2. Linear backward (dW, db, dx)

Params: [W | bias | bn_gamma | bn_beta | bn_running_mean | bn_running_var]
Cache per sample: [input | x_hat | inv_std_per_feature]

BatchNorm is 1D (spatial=1): normalizes each output feature across the batch.
"""

from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import sqrt


struct LinearBatchNormReLU[
    in_dim: Int,
    out_dim: Int,
    BN_MOMENTUM: Float64 = 0.1,
    BN_EPSILON: Float64 = 1e-5,
](Model):
    """Fused Linear → BatchNorm1D → ReLU in a single Model.

    Parameters match Linear: in_dim, out_dim.
    Adds BN_MOMENTUM and BN_EPSILON for BatchNorm configuration.
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim

    # Params: W (in*out) + bias (out) + BN gamma/beta/rmean/rvar (4*out)
    comptime W_SIZE: Int = Self.in_dim * Self.out_dim
    comptime LINEAR_PARAM_SIZE: Int = Self.W_SIZE + Self.out_dim
    comptime BN_PARAM_SIZE: Int = 4 * Self.out_dim
    comptime PARAM_SIZE: Int = Self.LINEAR_PARAM_SIZE + Self.BN_PARAM_SIZE

    # Param offsets
    comptime W_OFF: Int = 0
    comptime BIAS_OFF: Int = Self.W_SIZE
    comptime GAMMA_OFF: Int = Self.LINEAR_PARAM_SIZE
    comptime BETA_OFF: Int = Self.LINEAR_PARAM_SIZE + Self.out_dim
    comptime RMEAN_OFF: Int = Self.LINEAR_PARAM_SIZE + 2 * Self.out_dim
    comptime RVAR_OFF: Int = Self.LINEAR_PARAM_SIZE + 3 * Self.out_dim

    # Cache: input (for matmul backward) + x_hat (for BN backward) + inv_std
    comptime LINEAR_CACHE: Int = Self.in_dim
    comptime CACHE_SIZE: Int = Self.LINEAR_CACHE + Self.out_dim + Self.out_dim

    # Cache offsets (per sample)
    comptime XHAT_OFF: Int = Self.LINEAR_CACHE
    comptime INVSTD_OFF: Int = Self.LINEAR_CACHE + Self.out_dim

    # Workspace for GPU matmul backward (grad_pre_bn buffer)
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.out_dim

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Init W with INIT, bias=0, BN gamma=1, beta=0, rmean=0, rvar=1."""
        var w_params = LayoutTensor[
            dtype, Layout.row_major(Self.W_SIZE), MutAnyOrigin
        ](params.ptr)
        INIT.init[Self.W_SIZE, Self.in_dim, Self.out_dim](w_params)
        # Bias = 0
        for i in range(Self.out_dim):
            params.ptr[Self.BIAS_OFF + i] = Scalar[dtype](0.0)
        # BN params
        for i in range(Self.out_dim):
            params.ptr[Self.GAMMA_OFF + i] = Scalar[dtype](1.0)
            params.ptr[Self.BETA_OFF + i] = Scalar[dtype](0.0)
            params.ptr[Self.RMEAN_OFF + i] = Scalar[dtype](0.0)
            params.ptr[Self.RVAR_OFF + i] = Scalar[dtype](1.0)

    # =========================================================================
    # CPU Forward (training — with cache)
    # =========================================================================

    @staticmethod
    def forward[
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
        """Training forward: Linear → BN (batch stats) → ReLU."""
        # Step 1: Matmul + bias → output (pre-BN), cache input
        for b in range(BATCH):
            for i in range(Self.in_dim):
                cache[b, i] = input[b, i]
            for j in range(Self.out_dim):
                var acc = rebind[Scalar[dtype]](params[Self.BIAS_OFF + j])
                for i in range(Self.in_dim):
                    acc += rebind[Scalar[dtype]](input[b, i]) * rebind[Scalar[dtype]](params[i * Self.out_dim + j])
                output[b, j] = acc

        # Step 2: BN + ReLU (per feature, across batch)
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var mom = Scalar[dtype](Self.BN_MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n = Scalar[dtype](BATCH)

        for f in range(Self.out_dim):
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])

            # Batch mean
            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                mean += rebind[Scalar[dtype]](output[b, f])
            mean = mean / n

            # Batch variance
            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                var diff = rebind[Scalar[dtype]](output[b, f]) - mean
                var_ += diff * diff
            var_ = var_ / n

            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](sqrt(Float64(var_ + eps)))

            # Normalize + scale + shift + ReLU
            for b in range(BATCH):
                var x = rebind[Scalar[dtype]](output[b, f])
                var x_hat = (x - mean) * inv_std
                cache[b, Self.XHAT_OFF + f] = x_hat
                var pre_relu = gamma * x_hat + beta
                output[b, f] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
                cache[b, Self.INVSTD_OFF + f] = inv_std

            # Update running stats
            params.ptr[Self.RMEAN_OFF + f] = one_m * rebind[Scalar[dtype]](params[Self.RMEAN_OFF + f]) + mom * mean
            params.ptr[Self.RVAR_OFF + f] = one_m * rebind[Scalar[dtype]](params[Self.RVAR_OFF + f]) + mom * var_

    # =========================================================================
    # CPU Forward (inference — no cache, batch stats)
    # =========================================================================

    @staticmethod
    def forward[
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
    ):
        """Inference forward: Linear → BN (batch stats) → ReLU, no caching."""
        # Linear into output
        for b in range(BATCH):
            for j in range(Self.out_dim):
                var acc = rebind[Scalar[dtype]](params[Self.BIAS_OFF + j])
                for i in range(Self.in_dim):
                    acc += rebind[Scalar[dtype]](input[b, i]) * rebind[Scalar[dtype]](params[i * Self.out_dim + j])
                output[b, j] = acc

        # BN + ReLU using batch stats
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var n = Scalar[dtype](BATCH)

        for f in range(Self.out_dim):
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])

            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                mean += rebind[Scalar[dtype]](output[b, f])
            mean = mean / n

            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                var diff = rebind[Scalar[dtype]](output[b, f]) - mean
                var_ += diff * diff
            var_ = var_ / n

            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](sqrt(Float64(var_ + eps)))

            for b in range(BATCH):
                var x = rebind[Scalar[dtype]](output[b, f])
                var pre_relu = gamma * (x - mean) * inv_std + beta
                output[b, f] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    def backward[
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
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward: ReLU+BN grad → Linear grad."""
        var n = Scalar[dtype](BATCH)

        # Step 1: ReLU + BN backward per feature → grad w.r.t. linear output
        var grad_pre_bn = List[Scalar[dtype]](capacity=BATCH * Self.out_dim)
        for _ in range(BATCH * Self.out_dim):
            grad_pre_bn.append(Scalar[dtype](0.0))

        for f in range(Self.out_dim):
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])
            var inv_std = rebind[Scalar[dtype]](cache[0, Self.INVSTD_OFF + f])

            var d_gamma = Scalar[dtype](0.0)
            var d_beta = Scalar[dtype](0.0)
            var sum_dy_g = Scalar[dtype](0.0)
            var sum_dy_g_xh = Scalar[dtype](0.0)

            for b in range(BATCH):
                var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + f])
                var pre_relu = gamma * x_hat + beta
                var dy = rebind[Scalar[dtype]](grad_output[b, f])
                if pre_relu <= Scalar[dtype](0.0):
                    dy = Scalar[dtype](0.0)
                d_gamma += dy * x_hat
                d_beta += dy
                sum_dy_g += dy * gamma
                sum_dy_g_xh += dy * gamma * x_hat

            grads.ptr[Self.GAMMA_OFF + f] = rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + f]) + d_gamma
            grads.ptr[Self.BETA_OFF + f] = rebind[Scalar[dtype]](grads[Self.BETA_OFF + f]) + d_beta

            for b in range(BATCH):
                var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + f])
                var pre_relu = gamma * x_hat + beta
                var dy = rebind[Scalar[dtype]](grad_output[b, f])
                if pre_relu <= Scalar[dtype](0.0):
                    dy = Scalar[dtype](0.0)
                grad_pre_bn[b * Self.out_dim + f] = inv_std * (
                    dy * gamma - sum_dy_g / n - x_hat * sum_dy_g_xh / n
                )

        # Step 2: Linear backward (dW, db, dx)
        for b in range(BATCH):
            # dW += input^T @ grad_pre_bn
            for i in range(Self.in_dim):
                var cached_input = rebind[Scalar[dtype]](cache[b, i])
                for j in range(Self.out_dim):
                    grads.ptr[i * Self.out_dim + j] = rebind[Scalar[dtype]](grads[i * Self.out_dim + j]) + cached_input * grad_pre_bn[b * Self.out_dim + j]

            # db += grad_pre_bn
            for j in range(Self.out_dim):
                grads.ptr[Self.BIAS_OFF + j] = rebind[Scalar[dtype]](grads[Self.BIAS_OFF + j]) + grad_pre_bn[b * Self.out_dim + j]

            # dx = grad_pre_bn @ W^T
            for i in range(Self.in_dim):
                var acc = Scalar[dtype](0.0)
                for j in range(Self.out_dim):
                    acc += grad_pre_bn[b * Self.out_dim + j] * rebind[Scalar[dtype]](params[i * Self.out_dim + j])
                grad_input[b, i] = acc

    # =========================================================================
    # GPU Kernels — Linear matmul then fused BN+ReLU
    # =========================================================================

    @always_inline
    @staticmethod
    def bn_relu_kernel_impl[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Fused BN+ReLU kernel. Reads pre-BN from output, writes final output.

        Grid: (out_dim,), Block: (1,)
        One thread per feature — computes stats across BATCH.
        """
        var f = Int(block_idx.x)
        if f >= Self.out_dim or thread_idx.x != 0:
            return

        var eps = Scalar[dtype](Self.BN_EPSILON)
        var mom = Scalar[dtype](Self.BN_MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n = Scalar[dtype](BATCH)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])

        # Compute batch mean
        var mean = Scalar[dtype](0.0)
        for b in range(BATCH):
            mean += rebind[Scalar[dtype]](output[b, f])
        mean = mean / n

        # Compute batch variance
        var var_ = Scalar[dtype](0.0)
        for b in range(BATCH):
            var diff = rebind[Scalar[dtype]](output[b, f]) - mean
            var_ += diff * diff
        var_ = var_ / n

        var inv_std: Scalar[dtype] = 1.0 / sqrt(var_ + eps)

        # Normalize + scale + shift + ReLU
        for b in range(BATCH):
            var x = rebind[Scalar[dtype]](output[b, f])
            var x_hat = (x - mean) * inv_std
            cache[b, Self.XHAT_OFF + f] = x_hat
            var pre_relu = gamma * x_hat + beta
            output[b, f] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
            cache[b, Self.INVSTD_OFF + f] = inv_std

        # Update running stats
        var rm = rebind[Scalar[dtype]](params[Self.RMEAN_OFF + f])
        var rv = rebind[Scalar[dtype]](params[Self.RVAR_OFF + f])
        params.ptr[Self.RMEAN_OFF + f] = one_m * rm + mom * mean
        params.ptr[Self.RVAR_OFF + f] = one_m * rv + mom * var_

    @always_inline
    @staticmethod
    def bn_relu_kernel_impl_no_cache[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
    ):
        """Fused BN+ReLU inference kernel (batch stats, no cache).

        Grid: (out_dim,), Block: (1,)
        """
        var f = Int(block_idx.x)
        if f >= Self.out_dim or thread_idx.x != 0:
            return

        var eps = Scalar[dtype](Self.BN_EPSILON)
        var n = Scalar[dtype](BATCH)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])

        var mean = Scalar[dtype](0.0)
        for b in range(BATCH):
            mean += rebind[Scalar[dtype]](output[b, f])
        mean = mean / n

        var var_ = Scalar[dtype](0.0)
        for b in range(BATCH):
            var diff = rebind[Scalar[dtype]](output[b, f]) - mean
            var_ += diff * diff
        var_ = var_ / n

        var inv_std: Scalar[dtype] = 1.0 / sqrt(var_ + eps)

        for b in range(BATCH):
            var x = rebind[Scalar[dtype]](output[b, f])
            var pre_relu = gamma * (x - mean) * inv_std + beta
            output[b, f] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)

    @always_inline
    @staticmethod
    def relu_bn_backward_kernel_impl[
        BATCH: Int,
    ](
        grad_pre_bn: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Fused ReLU+BN backward kernel. Produces grad w.r.t. linear output.

        Grid: (out_dim,), Block: (1,)
        """
        var f = Int(block_idx.x)
        if f >= Self.out_dim or thread_idx.x != 0:
            return

        var n = Scalar[dtype](BATCH)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])
        var inv_std = rebind[Scalar[dtype]](cache[0, Self.INVSTD_OFF + f])

        var d_gamma = Scalar[dtype](0.0)
        var d_beta = Scalar[dtype](0.0)
        var sum_dy_g = Scalar[dtype](0.0)
        var sum_dy_g_xh = Scalar[dtype](0.0)

        for b in range(BATCH):
            var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + f])
            var pre_relu = gamma * x_hat + beta
            var dy = rebind[Scalar[dtype]](grad_output[b, f])
            if pre_relu <= Scalar[dtype](0.0):
                dy = Scalar[dtype](0.0)
            d_gamma += dy * x_hat
            d_beta += dy
            sum_dy_g += dy * gamma
            sum_dy_g_xh += dy * gamma * x_hat

        grads.ptr[Self.GAMMA_OFF + f] = rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + f]) + d_gamma
        grads.ptr[Self.BETA_OFF + f] = rebind[Scalar[dtype]](grads[Self.BETA_OFF + f]) + d_beta

        for b in range(BATCH):
            var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + f])
            var pre_relu = gamma * x_hat + beta
            var dy = rebind[Scalar[dtype]](grad_output[b, f])
            if pre_relu <= Scalar[dtype](0.0):
                dy = Scalar[dtype](0.0)
            grad_pre_bn[b, f] = inv_std * (
                dy * gamma - sum_dy_g / n - x_hat * sum_dy_g_xh / n
            )

    # =========================================================================
    # GPU Launchers
    # =========================================================================

    @staticmethod
    def forward_gpu[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU training forward: Linear matmul → fused BN+ReLU."""
        from ..autodiff.primitives.matmul import MatMul

        comptime MM = MatMul[Self.in_dim, Self.out_dim]

        # Linear params
        var mm_params = LayoutTensor[
            dtype, Layout.row_major(MM.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)

        # Allocate separate matmul cache (stride mismatch: MM.CACHE_SIZE < Self.CACHE_SIZE)
        comptime MM_CS = MM.CACHE_SIZE
        var mm_cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * MM_CS if MM_CS > 0 else 1)
        var mm_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, MM_CS), MutAnyOrigin
        ](mm_cache_buf.unsafe_ptr())

        # Step 1: MatMul → output (pre-bias)
        MM.eval_gpu[BATCH](ctx, output, input, mm_params, mm_cache, workspace.unsafe_ptr())

        # Copy cached input from mm_cache into our cache (needed for backward)
        comptime COPY_SIZE = BATCH * MM_CS
        @always_inline
        def copy_input_cache(
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
            src: LayoutTensor[dtype, Layout.row_major(BATCH, MM_CS), MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= COPY_SIZE:
                return
            var b = tid // MM_CS
            var i = tid % MM_CS
            dst.ptr[b * Self.CACHE_SIZE + i] = src.ptr[tid]

        comptime COPY_BLOCKS = (COPY_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[copy_input_cache, copy_input_cache](
            cache, mm_cache,
            grid_dim=(COPY_BLOCKS,), block_dim=(TPB,),
        )

        # Step 2: BiasAdd in-place via simple kernel
        var bias = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
        ](params.ptr + Self.BIAS_OFF)
        comptime TOTAL = BATCH * Self.out_dim
        comptime BLOCKS = (TOTAL + TPB - 1) // TPB

        @always_inline
        def bias_add_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin],
            bias: LayoutTensor[dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < TOTAL:
                output[idx] = rebind[Scalar[dtype]](output[idx]) + rebind[Scalar[dtype]](bias[idx % Self.out_dim])

        var out_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](output.ptr)
        ctx.enqueue_function[bias_add_wrapper, bias_add_wrapper](
            out_flat, bias, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )

        # Step 3: Fused BN+ReLU kernel
        @always_inline
        def bn_relu_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        ):
            Self.bn_relu_kernel_impl[BATCH](output, cache, params)

        ctx.enqueue_function[bn_relu_wrapper, bn_relu_wrapper](
            output, cache, params,
            grid_dim=(Self.out_dim,),
            block_dim=(1,),
        )

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU inference forward: Linear matmul → fused BN+ReLU (batch stats)."""
        from ..autodiff.primitives.matmul import MatMul

        comptime MM = MatMul[Self.in_dim, Self.out_dim]

        var mm_params = LayoutTensor[
            dtype, Layout.row_major(MM.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)

        # Dummy cache for matmul (not needed for inference, separate buffer)
        comptime MM_CS = MM.CACHE_SIZE
        var dummy_buf = ctx.enqueue_create_buffer[dtype](BATCH * MM_CS if MM_CS > 0 else 1)
        var dummy_mm_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, MM_CS), MutAnyOrigin
        ](dummy_buf.unsafe_ptr())

        MM.eval_gpu[BATCH](ctx, output, input, mm_params, dummy_mm_cache, workspace.unsafe_ptr())

        # BiasAdd in-place
        var bias = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
        ](params.ptr + Self.BIAS_OFF)
        comptime TOTAL = BATCH * Self.out_dim
        comptime BLOCKS = (TOTAL + TPB - 1) // TPB

        @always_inline
        def bias_add_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin],
            bias: LayoutTensor[dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < TOTAL:
                output[idx] = rebind[Scalar[dtype]](output[idx]) + rebind[Scalar[dtype]](bias[idx % Self.out_dim])

        var out_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](output.ptr)
        ctx.enqueue_function[bias_add_wrapper, bias_add_wrapper](
            out_flat, bias, grid_dim=(BLOCKS,), block_dim=(TPB,),
        )

        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        @always_inline
        def bn_relu_nc_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin],
        ):
            Self.bn_relu_kernel_impl_no_cache[BATCH](output, params)

        ctx.enqueue_function[bn_relu_nc_wrapper, bn_relu_nc_wrapper](
            output, params_immut,
            grid_dim=(Self.out_dim,),
            block_dim=(1,),
        )

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH](ctx, output, input, params, workspace)

    @staticmethod
    def backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU backward: fused ReLU+BN backward → Linear backward."""
        from ..autodiff.primitives.matmul import MatMul

        comptime MM = MatMul[Self.in_dim, Self.out_dim]

        # Allocate temp buffer for grad_pre_bn (same size as output)
        var grad_pre_bn_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.out_dim)
        var grad_pre_bn = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](grad_pre_bn_buf.unsafe_ptr())

        # Step 1: Fused ReLU+BN backward
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        @always_inline
        def relu_bn_bwd_wrapper(
            grad_pre_bn: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin],
            grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        ):
            Self.relu_bn_backward_kernel_impl[BATCH](grad_pre_bn, grad_output, params, cache, grads)

        ctx.enqueue_function[relu_bn_bwd_wrapper, relu_bn_bwd_wrapper](
            grad_pre_bn, grad_output_immut, params_immut, cache_immut, grads,
            grid_dim=(Self.out_dim,),
            block_dim=(1,),
        )

        # Step 2: Bias grad accumulation: db[j] += sum_b(grad_pre_bn[b, j])
        var bias_grads = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grads.ptr + Self.BIAS_OFF)

        @always_inline
        def bias_grad_wrapper(
            db: LayoutTensor[dtype, Layout.row_major(Self.out_dim), MutAnyOrigin],
            gpb: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        ):
            var j = Int(block_dim.x * block_idx.x + thread_idx.x)
            if j < Self.out_dim:
                var acc = Scalar[dtype](0.0)
                for b in range(BATCH):
                    acc += rebind[Scalar[dtype]](gpb[b, j])
                db[j] = rebind[Scalar[dtype]](db[j]) + acc

        comptime BG_BLOCKS = (Self.out_dim + TPB - 1) // TPB
        ctx.enqueue_function[bias_grad_wrapper, bias_grad_wrapper](
            bias_grads, grad_pre_bn,
            grid_dim=(BG_BLOCKS,), block_dim=(TPB,),
        )

        # Step 3: MatMul backward (dW, dx)
        # Extract cached input from our cache into a separate buffer (stride mismatch)
        comptime MM_CS = MM.CACHE_SIZE
        var mm_cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * MM_CS if MM_CS > 0 else 1)
        var mm_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, MM_CS), MutAnyOrigin
        ](mm_cache_buf.unsafe_ptr())

        comptime COPY_SIZE = BATCH * MM_CS
        @always_inline
        def copy_input_bwd(
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, MM_CS), MutAnyOrigin],
            src: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= COPY_SIZE:
                return
            var b = tid // MM_CS
            var i = tid % MM_CS
            dst.ptr[tid] = src.ptr[b * Self.CACHE_SIZE + i]

        comptime COPY_BLOCKS = (COPY_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[copy_input_bwd, copy_input_bwd](
            mm_cache, cache,
            grid_dim=(COPY_BLOCKS,), block_dim=(TPB,),
        )

        var mm_params = LayoutTensor[
            dtype, Layout.row_major(MM.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var mm_grads = LayoutTensor[
            dtype, Layout.row_major(MM.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        MM.vjp_gpu[BATCH](ctx, grad_pre_bn, grad_input, mm_params, mm_cache, mm_grads, workspace.unsafe_ptr())
