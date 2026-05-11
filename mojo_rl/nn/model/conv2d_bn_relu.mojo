"""Fused Conv2D + BatchNorm + ReLU as a single Model layer.

Eliminates 2 extra layers in Sequential (BN + ReLU), reducing kernel
launches from 6 per block (3 fwd + 3 bwd) to 4 (2 fwd + 2 bwd).

Forward:
  1. Conv2D (im2col + matmul + bias) → pre_bn
  2. BatchNorm + ReLU per channel → output

Backward:
  1. ReLU + BN backward per channel → grad_pre_bn
  2. Conv2D backward (dW, db, dx)

Params: [conv_W | conv_bias | bn_gamma | bn_beta | bn_running_mean | bn_running_var]
Cache per sample: [im2col | x_hat | batch_inv_std_per_channel]
"""

from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import sqrt
from std.random.philox import Random as PhiloxRandom


struct Conv2DBatchNormReLU[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
    BN_MOMENTUM: Float64 = 0.1,
    BN_EPSILON: Float64 = 1e-5,
](Model):
    """Fused Conv2D → BatchNorm → ReLU in a single Model.

    Parameters match Conv2DLayer: ic, oc, k, s, p, h, w.
    Adds BN_MOMENTUM and BN_EPSILON for BatchNorm configuration.
    """

    comptime out_h: Int = (
        Self.in_h + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime out_w: Int = (
        Self.in_w + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime col_size: Int = Self.in_channels * Self.kernel_size * Self.kernel_size
    comptime spatial_out: Int = Self.out_h * Self.out_w

    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.spatial_out

    # Params: conv_W (oc * col_size) + conv_bias (oc) + BN gamma/beta (2*oc, gradient-tracked)
    comptime CONV_W_SIZE: Int = Self.out_channels * Self.col_size
    comptime CONV_PARAM_SIZE: Int = Self.CONV_W_SIZE + Self.out_channels
    comptime BN_PARAM_SIZE: Int = 2 * Self.out_channels  # gamma, beta only
    comptime PARAM_SIZE: Int = Self.CONV_PARAM_SIZE + Self.BN_PARAM_SIZE

    # Param offsets (within PARAM_SIZE)
    comptime W_OFF: Int = 0
    comptime BIAS_OFF: Int = Self.CONV_W_SIZE
    comptime GAMMA_OFF: Int = Self.CONV_PARAM_SIZE
    comptime BETA_OFF: Int = Self.CONV_PARAM_SIZE + Self.out_channels
    # State offsets (within STATE_SIZE) — running stats live here post-Phase-3.
    comptime RMEAN_OFF: Int = 0
    comptime RVAR_OFF: Int = Self.out_channels

    # Cache: im2col + x_hat + batch_inv_std
    comptime CONV_CACHE: Int = Self.col_size * Self.spatial_out
    comptime CACHE_SIZE: Int = Self.CONV_CACHE + Self.OUT_DIM + Self.out_channels

    # Cache offsets
    comptime XHAT_OFF: Int = Self.CONV_CACHE
    comptime INVSTD_OFF: Int = Self.CONV_CACHE + Self.OUT_DIM

    # Workspace for GPU conv matmul + temp buffers (conv cache, grad_pre_bn)
    # Conv2D internal workspace: CONV_CACHE + col_size*OC (out_temp/dcol + w_t)
    # Temp conv cache: CONV_CACHE (for stride-mismatch copy to/from our cache)
    # grad_pre_bn (backward): OUT_DIM (reuses temp conv cache region in forward)
    comptime CONV2D_WS: Int = Self.CONV_CACHE + Self.col_size * Self.out_channels
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.CONV2D_WS + Self.CONV_CACHE + Self.OUT_DIM
    comptime STATE_SIZE: Int = 2 * Self.out_channels  # running_mean, running_var

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer,
        dtype: DType = DType.float32,
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Init conv weights with INIT, bias=0, BN gamma=1, beta=0. Running stats are owned by `state`."""
        # Conv weights: use the provided initializer
        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_W_SIZE), MutAnyOrigin
        ](params.ptr)
        INIT.init[Self.CONV_W_SIZE, Self.in_channels, Self.out_channels](
            conv_params
        )
        # Conv bias = 0
        for i in range(Self.out_channels):
            params.ptr[Self.BIAS_OFF + i] = Scalar[dtype](0.0)
        # BN params (gradient-tracked)
        for i in range(Self.out_channels):
            params.ptr[Self.GAMMA_OFF + i] = Scalar[dtype](1.0)
            params.ptr[Self.BETA_OFF + i] = Scalar[dtype](0.0)

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize BN running_mean=0, running_var=1."""
        for i in range(Self.out_channels):
            state.ptr[Self.RMEAN_OFF + i] = Scalar[dtype](0.0)
            state.ptr[Self.RVAR_OFF + i] = Scalar[dtype](1.0)

    # =========================================================================
    # CPU Forward (training — with cache)
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Training forward: Conv → BN (batch stats) → ReLU."""
        # Step 1: Im2col + Conv matmul → store pre-BN output in `output` temporarily
        for b in range(BATCH):
            # Im2col
            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for c in range(Self.in_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var ih = oh * Self.stride - Self.padding + kh
                                var iw = ow * Self.stride - Self.padding + kw
                                var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                                var col_idx = s * Self.col_size + c_k
                                if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                                    cache[b, col_idx] = input[b, c * Self.in_h * Self.in_w + ih * Self.in_w + iw]
                                else:
                                    cache[b, col_idx] = 0

            # Matmul: output[oc, s] = W[oc] @ col[s] + bias[oc]
            for oc in range(Self.out_channels):
                for s in range(Self.spatial_out):
                    var acc = rebind[Scalar[dtype]](params[Self.BIAS_OFF + oc])
                    for k in range(Self.col_size):
                        acc += rebind[Scalar[dtype]](params[oc * Self.col_size + k]) * rebind[Scalar[dtype]](cache[b, s * Self.col_size + k])
                    output[b, oc * Self.spatial_out + s] = acc

        # Step 2: BN + ReLU (per channel, across batch × spatial)
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var mom = Scalar[dtype](Self.BN_MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n = Scalar[dtype](BATCH * Self.spatial_out)

        for c in range(Self.out_channels):
            var c_off = c * Self.spatial_out
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

            # Batch mean
            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    mean += rebind[Scalar[dtype]](output[b, c_off + s])
            mean = mean / n

            # Batch variance
            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var diff = rebind[Scalar[dtype]](output[b, c_off + s]) - mean
                    var_ += diff * diff
            var_ = var_ / n

            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](sqrt(Float64(var_ + eps)))

            # Normalize + scale + shift + ReLU
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var x = rebind[Scalar[dtype]](output[b, c_off + s])
                    var x_hat = (x - mean) * inv_std
                    cache[b, Self.XHAT_OFF + c_off + s] = x_hat
                    var pre_relu = gamma * x_hat + beta
                    output[b, c_off + s] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
                cache[b, Self.INVSTD_OFF + c] = inv_std

            # Update running stats (EMA) in the persistent state buffer.
            state.ptr[Self.RMEAN_OFF + c] = one_m * rebind[Scalar[dtype]](state[Self.RMEAN_OFF + c]) + mom * mean
            state.ptr[Self.RVAR_OFF + c] = one_m * rebind[Scalar[dtype]](state[Self.RVAR_OFF + c]) + mom * var_

    # =========================================================================
    # CPU Forward (inference — no cache, batch stats)
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Inference forward: Conv → BN (running stats) → ReLU, no caching."""
        # Conv matmul into output
        for b in range(BATCH):
            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for oc in range(Self.out_channels):
                        var acc = rebind[Scalar[dtype]](params[Self.BIAS_OFF + oc])
                        for c in range(Self.in_channels):
                            for kh in range(Self.kernel_size):
                                for kw in range(Self.kernel_size):
                                    var ih = oh * Self.stride - Self.padding + kh
                                    var iw = ow * Self.stride - Self.padding + kw
                                    if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                                        var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                                        acc += rebind[Scalar[dtype]](params[oc * Self.col_size + c_k]) * rebind[Scalar[dtype]](input[b, c * Self.in_h * Self.in_w + ih * Self.in_w + iw])
                        output[b, oc * Self.spatial_out + s] = acc

        # BN + ReLU using running stats (inference)
        var eps = Scalar[dtype](Self.BN_EPSILON)

        for c in range(Self.out_channels):
            var c_off = c * Self.spatial_out
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
            var rmean = rebind[Scalar[dtype]](state[Self.RMEAN_OFF + c])
            var rvar = rebind[Scalar[dtype]](state[Self.RVAR_OFF + c])

            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](sqrt(Float64(rvar + eps)))

            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var x = rebind[Scalar[dtype]](output[b, c_off + s])
                    var pre_relu = gamma * (x - rmean) * inv_std + beta
                    output[b, c_off + s] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    def backward[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward: ReLU+BN grad → Conv grad."""
        var n = Scalar[dtype](BATCH * Self.spatial_out)

        # Step 1: ReLU + BN backward per channel → produces grad w.r.t. conv output
        # Allocate temp for grad_pre_bn (reuse a list)
        var grad_pre_bn = List[Scalar[dtype]](capacity=BATCH * Self.OUT_DIM)
        for _ in range(BATCH * Self.OUT_DIM):
            grad_pre_bn.append(Scalar[dtype](0.0))

        for c in range(Self.out_channels):
            var c_off = c * Self.spatial_out
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
            var inv_std = rebind[Scalar[dtype]](cache[0, Self.INVSTD_OFF + c])

            var d_gamma = Scalar[dtype](0.0)
            var d_beta = Scalar[dtype](0.0)
            var sum_dy_g = Scalar[dtype](0.0)
            var sum_dy_g_xh = Scalar[dtype](0.0)

            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
                    # ReLU mask: recompute pre_relu from x_hat
                    var pre_relu = gamma * x_hat + rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
                    var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
                    # Apply ReLU mask
                    if pre_relu <= Scalar[dtype](0.0):
                        dy = Scalar[dtype](0.0)
                    d_gamma += dy * x_hat
                    d_beta += dy
                    sum_dy_g += dy * gamma
                    sum_dy_g_xh += dy * gamma * x_hat

            # Accumulate BN param grads
            grads.ptr[Self.GAMMA_OFF + c] = rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + c]) + d_gamma
            grads.ptr[Self.BETA_OFF + c] = rebind[Scalar[dtype]](grads[Self.BETA_OFF + c]) + d_beta

            # Compute grad w.r.t. conv output (pre-BN)
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
                    var pre_relu = gamma * x_hat + rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
                    var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
                    if pre_relu <= Scalar[dtype](0.0):
                        dy = Scalar[dtype](0.0)
                    grad_pre_bn[b * Self.OUT_DIM + c_off + s] = inv_std * (
                        dy * gamma - sum_dy_g / n - x_hat * sum_dy_g_xh / n
                    )

        # Step 2: Conv backward (dW, db, dx) using grad_pre_bn
        for b in range(BATCH):
            # dW += grad_pre_bn @ col.T
            for oc in range(Self.out_channels):
                for k in range(Self.col_size):
                    var acc: Scalar[dtype] = 0
                    for s in range(Self.spatial_out):
                        acc += grad_pre_bn[b * Self.OUT_DIM + oc * Self.spatial_out + s] * rebind[Scalar[dtype]](cache[b, s * Self.col_size + k])
                    grads.ptr[oc * Self.col_size + k] = rebind[Scalar[dtype]](grads[oc * Self.col_size + k]) + acc

            # db += sum(grad_pre_bn, spatial)
            for oc in range(Self.out_channels):
                var acc: Scalar[dtype] = 0
                for s in range(Self.spatial_out):
                    acc += grad_pre_bn[b * Self.OUT_DIM + oc * Self.spatial_out + s]
                grads.ptr[Self.BIAS_OFF + oc] = rebind[Scalar[dtype]](grads[Self.BIAS_OFF + oc]) + acc

            # grad_input via col2im
            for i in range(Self.IN_DIM):
                grad_input[b, i] = 0
            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for c in range(Self.in_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var ih = oh * Self.stride - Self.padding + kh
                                var iw = ow * Self.stride - Self.padding + kw
                                if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                                    var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                    var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                                    var dcol: Scalar[dtype] = 0
                                    for oc in range(Self.out_channels):
                                        dcol += rebind[Scalar[dtype]](params[oc * Self.col_size + c_k]) * grad_pre_bn[b * Self.OUT_DIM + oc * Self.spatial_out + s]
                                    grad_input[b, in_idx] = rebind[Scalar[dtype]](grad_input[b, in_idx]) + dcol

    # =========================================================================
    # GPU Kernels — Conv matmul then fused BN+ReLU
    # =========================================================================

    @always_inline
    @staticmethod
    def bn_relu_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Fused BN+ReLU kernel. Reads pre-BN from output, writes final output.

        Grid: (out_channels,), Block: (TPB,)
        Optimized: 2-pass Welford (sum+sumsq in pass 1, normalize+relu in pass 2).
        """
        var c = Int(block_idx.x)
        if c >= Self.out_channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.spatial_out
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var mom = Scalar[dtype](Self.BN_MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        comptime N = BATCH * Self.spatial_out
        var n_f = Scalar[dtype](N)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
        comptime OUT_STRIDE = Self.OUT_DIM

        var smem = LayoutTensor[
            dtype, Layout.row_major(TPB), MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Pass 1: Mean
        var local_sum = Scalar[dtype](0.0)
        var base = output.ptr + c_off
        var idx = tid
        while idx < N:
            var b = idx // Self.spatial_out
            var s = idx - b * Self.spatial_out
            local_sum += (base + b * OUT_STRIDE + s)[]
            idx += TPB

        smem[tid] = local_sum
        barrier()
        var st = TPB // 2
        while st > 0:
            if tid < st:
                smem[tid] = smem[tid] + smem[tid + st]
            barrier()
            st = st // 2
        var mean = rebind[Scalar[dtype]](smem[0]) / n_f
        barrier()

        # Pass 1b: Variance using (x - mean)² — numerically stable
        var local_var = Scalar[dtype](0.0)
        idx = tid
        while idx < N:
            var b = idx // Self.spatial_out
            var s = idx - b * Self.spatial_out
            var diff = (base + b * OUT_STRIDE + s)[] - mean
            local_var += diff * diff
            idx += TPB

        smem[tid] = local_var
        barrier()
        st = TPB // 2
        while st > 0:
            if tid < st:
                smem[tid] = smem[tid] + smem[tid + st]
            barrier()
            st = st // 2
        var var_ = rebind[Scalar[dtype]](smem[0]) / n_f
        var inv_std: Scalar[dtype] = 1.0 / sqrt(var_ + eps)
        barrier()

        # Pass 2: Normalize + scale + shift + ReLU + cache
        idx = tid
        while idx < N:
            var b = idx // Self.spatial_out
            var s = idx - b * Self.spatial_out
            var out_off = b * OUT_STRIDE + c_off + s
            var x = output.ptr[out_off]
            var x_hat = (x - mean) * inv_std
            (cache.ptr + b * Self.CACHE_SIZE + Self.XHAT_OFF + c_off + s)[] = x_hat
            var pre_relu = gamma * x_hat + beta
            output.ptr[out_off] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
            idx += TPB

        # Parallel inv_std cache write + running stats
        if tid < BATCH:
            (cache.ptr + tid * Self.CACHE_SIZE + Self.INVSTD_OFF + c)[] = inv_std
        if tid == 0:
            var rm = state.ptr[Self.RMEAN_OFF + c]
            var rv = state.ptr[Self.RVAR_OFF + c]
            state.ptr[Self.RMEAN_OFF + c] = one_m * rm + mom * mean
            state.ptr[Self.RVAR_OFF + c] = one_m * rv + mom * var_

    @always_inline
    @staticmethod
    def bn_relu_kernel_impl_no_cache[
        BATCH: Int, dtype: DType = DType.float32,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Fused BN+ReLU inference kernel using running stats (no cache, no update).

        Grid: (out_channels,), Block: (TPB,)
        """
        var c = Int(block_idx.x)
        if c >= Self.out_channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.spatial_out
        var eps = Scalar[dtype](Self.BN_EPSILON)
        comptime N = BATCH * Self.spatial_out
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
        var rmean = rebind[Scalar[dtype]](state[Self.RMEAN_OFF + c])
        var rvar = rebind[Scalar[dtype]](state[Self.RVAR_OFF + c])
        var inv_std: Scalar[dtype] = 1.0 / sqrt(rvar + eps)
        comptime OUT_STRIDE = Self.OUT_DIM

        # Parallel scatter: normalize + ReLU
        var idx = tid
        while idx < N:
            var b = idx // Self.spatial_out
            var s = idx - b * Self.spatial_out
            var out_off = b * OUT_STRIDE + c_off + s
            var x = output.ptr[out_off]
            var pre_relu = gamma * (x - rmean) * inv_std + beta
            output.ptr[out_off] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
            idx += TPB

    @always_inline
    @staticmethod
    def relu_bn_backward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32,
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
        """Fused ReLU+BN backward kernel. Produces grad w.r.t. conv output.

        Grid: (out_channels,), Block: (TPB,)
        Block-parallel reduction for gradient accumulation.
        """
        var c = Int(block_idx.x)
        if c >= Self.out_channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.spatial_out
        var n_f = Scalar[dtype](BATCH * Self.spatial_out)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
        var inv_std = rebind[Scalar[dtype]](cache[0, Self.INVSTD_OFF + c])

        var smem = LayoutTensor[
            dtype, Layout.row_major(TPB), MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Pass 1: Accumulate 4 partial sums per thread
        var local_d_gamma = Scalar[dtype](0.0)
        var local_d_beta = Scalar[dtype](0.0)
        var local_sum_dy_g = Scalar[dtype](0.0)
        var local_sum_dy_g_xh = Scalar[dtype](0.0)

        var idx = tid
        while idx < BATCH * Self.spatial_out:
            var b = idx // Self.spatial_out
            var s = idx % Self.spatial_out
            var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
            var pre_relu = gamma * x_hat + beta
            var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
            if pre_relu <= Scalar[dtype](0.0):
                dy = Scalar[dtype](0.0)
            local_d_gamma += dy * x_hat
            local_d_beta += dy
            local_sum_dy_g += dy * gamma
            local_sum_dy_g_xh += dy * gamma * x_hat
            idx += TPB

        # Reduce d_gamma
        smem[tid] = local_d_gamma
        barrier()
        var st = TPB // 2
        while st > 0:
            if tid < st:
                smem[tid] = smem[tid] + smem[tid + st]
            barrier()
            st = st // 2
        var d_gamma = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Reduce d_beta
        smem[tid] = local_d_beta
        barrier()
        st = TPB // 2
        while st > 0:
            if tid < st:
                smem[tid] = smem[tid] + smem[tid + st]
            barrier()
            st = st // 2
        var d_beta = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Reduce sum_dy_g
        smem[tid] = local_sum_dy_g
        barrier()
        st = TPB // 2
        while st > 0:
            if tid < st:
                smem[tid] = smem[tid] + smem[tid + st]
            barrier()
            st = st // 2
        var sum_dy_g = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Reduce sum_dy_g_xh
        smem[tid] = local_sum_dy_g_xh
        barrier()
        st = TPB // 2
        while st > 0:
            if tid < st:
                smem[tid] = smem[tid] + smem[tid + st]
            barrier()
            st = st // 2
        var sum_dy_g_xh = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Write param grads (thread 0 only)
        if tid == 0:
            grads.ptr[Self.GAMMA_OFF + c] = rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + c]) + d_gamma
            grads.ptr[Self.BETA_OFF + c] = rebind[Scalar[dtype]](grads[Self.BETA_OFF + c]) + d_beta

        # Pass 2: Compute grad_input (parallel scatter)
        idx = tid
        while idx < BATCH * Self.spatial_out:
            var b = idx // Self.spatial_out
            var s = idx % Self.spatial_out
            var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
            var pre_relu = gamma * x_hat + beta
            var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
            if pre_relu <= Scalar[dtype](0.0):
                dy = Scalar[dtype](0.0)
            grad_pre_bn[b, c_off + s] = inv_std * (
                dy * gamma - sum_dy_g / n_f - x_hat * sum_dy_g_xh / n_f
            )
            idx += TPB

    # =========================================================================
    # GPU Launchers
    # =========================================================================

    @staticmethod
    def forward_gpu[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU training forward: Conv matmul → fused BN+ReLU."""
        # Import Conv2D for its GPU kernels
        from ..autodiff.fused import FusedConv2DActivation
        from ..autodiff.fused.activation import ReLUActivation

        # Use FusedConv2DActivation's GPU matmul but with identity activation
        # Actually, just run the conv as a plain matmul + bias into output,
        # then run BN+ReLU kernel.
        # For simplicity, delegate to Conv2DLayer's forward_gpu (no activation)
        from ..autodiff import Conv2D
        comptime ConvOp = Conv2D[
            Self.in_channels, Self.out_channels,
            Self.kernel_size, Self.stride, Self.padding,
            Self.in_h, Self.in_w,
        ]

        # Conv params are the first CONV_PARAM_SIZE elements
        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](params.ptr)

        # Temp conv cache in workspace (after Conv2D's internal region)
        # Conv2D uses workspace[0 : BATCH*CONV_CACHE + col_size*OC] internally.
        # We place the temp cache right after that.
        comptime CONV_CS = ConvOp.CACHE_SIZE
        comptime TEMP_CACHE_OFF = BATCH * Self.CONV_CACHE + Self.col_size * Self.out_channels
        var conv_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ](workspace.unsafe_ptr() + TEMP_CACHE_OFF)

        # Run Conv2D GPU forward (writes pre-BN output to `output`, im2col to conv_cache)
        ConvOp.eval_gpu[BATCH](ctx, output, input, conv_params, conv_cache, workspace.unsafe_ptr())

        # Copy im2col from conv_cache into our cache (needed for backward)
        # The BN+ReLU kernel reads x_hat/inv_std from our cache, and backward needs im2col
        comptime COPY_SIZE = BATCH * CONV_CS
        @parameter
        @always_inline
        def copy_im2col(
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
            src: LayoutTensor[dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin],
        ):
            # Copy im2col: for each sample, copy CONV_CS elements
            # Source stride = CONV_CS, dest stride = CACHE_SIZE
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= COPY_SIZE:
                return
            var b = tid // CONV_CS
            var i = tid % CONV_CS
            dst.ptr[b * Self.CACHE_SIZE + i] = src.ptr[tid]

        comptime COPY_BLOCKS = (COPY_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[copy_im2col](
            cache, conv_cache,
            grid_dim=(COPY_BLOCKS,),
            block_dim=(TPB,),
        )

        # Run fused BN+ReLU kernel
        @parameter
        @always_inline
        def bn_relu_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
            state: LayoutTensor[dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin],
        ):
            Self.bn_relu_kernel_impl[BATCH, dtype](output, cache, params, state)

        ctx.enqueue_function[bn_relu_wrapper](
            output, cache, params, state,
            grid_dim=(Self.out_channels,),
            block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU inference forward: Conv matmul → fused BN+ReLU (running stats)."""
        from ..autodiff import Conv2D
        comptime ConvOp = Conv2D[
            Self.in_channels, Self.out_channels,
            Self.kernel_size, Self.stride, Self.padding,
            Self.in_h, Self.in_w,
        ]

        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](params.ptr)

        # Dummy cache in workspace (after Conv2D's internal region — data discarded)
        comptime CONV_CS = ConvOp.CACHE_SIZE
        comptime TEMP_CACHE_OFF = BATCH * Self.CONV_CACHE + Self.col_size * Self.out_channels
        var dummy_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ](workspace.unsafe_ptr() + TEMP_CACHE_OFF)

        ConvOp.eval_gpu[BATCH](ctx, output, input, conv_params, dummy_cache, workspace.unsafe_ptr())

        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var state_immut = LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ](state.ptr)

        @parameter
        @always_inline
        def bn_relu_nc_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin],
            state: LayoutTensor[dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin],
        ):
            Self.bn_relu_kernel_impl_no_cache[BATCH, dtype](output, params, state)

        ctx.enqueue_function[bn_relu_nc_wrapper](
            output, params_immut, state_immut,
            grid_dim=(Self.out_channels,),
            block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, state, workspace)

    @staticmethod
    def backward_gpu[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        """GPU backward: fused ReLU+BN backward → Conv backward."""
        from ..autodiff import Conv2D
        comptime ConvOp = Conv2D[
            Self.in_channels, Self.out_channels,
            Self.kernel_size, Self.stride, Self.padding,
            Self.in_h, Self.in_w,
        ]

        # grad_pre_bn in workspace (after Conv2D's region + temp conv cache)
        # Layout: [Conv2D WS | temp_conv_cache: CONV_CACHE | grad_pre_bn: OUT_DIM]
        comptime GRAD_PRE_BN_OFF = BATCH * Self.CONV_CACHE + Self.col_size * Self.out_channels + BATCH * Self.CONV_CACHE
        var grad_pre_bn = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](workspace.unsafe_ptr() + GRAD_PRE_BN_OFF)

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

        @parameter
        @always_inline
        def relu_bn_bwd_wrapper(
            grad_pre_bn: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin],
            grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        ):
            Self.relu_bn_backward_kernel_impl[BATCH, dtype](grad_pre_bn, grad_output, params, cache, grads)

        ctx.enqueue_function[relu_bn_bwd_wrapper](
            grad_pre_bn, grad_output_immut, params_immut, cache_immut, grads,
            grid_dim=(Self.out_channels,),
            block_dim=(TPB,),
        )

        # Step 2: Conv backward (dW, db, dx)
        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var conv_grads = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)

        # Temp conv cache in workspace (after Conv2D's region, same offset as forward)
        comptime CONV_CS = ConvOp.CACHE_SIZE
        comptime TEMP_CACHE_OFF = BATCH * Self.CONV_CACHE + Self.col_size * Self.out_channels
        var conv_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ](workspace.unsafe_ptr() + TEMP_CACHE_OFF)

        # Copy im2col from our cache (stride=CACHE_SIZE) to conv_cache (stride=CONV_CS)
        comptime COPY_SIZE = BATCH * CONV_CS
        @parameter
        @always_inline
        def copy_im2col_bwd(
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin],
            src: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= COPY_SIZE:
                return
            var b = tid // CONV_CS
            var i = tid % CONV_CS
            dst.ptr[tid] = src.ptr[b * Self.CACHE_SIZE + i]

        comptime COPY_BLOCKS = (COPY_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[copy_im2col_bwd](
            conv_cache, cache,
            grid_dim=(COPY_BLOCKS,),
            block_dim=(TPB,),
        )

        ConvOp.vjp_gpu[BATCH](
            ctx, grad_pre_bn, grad_input, conv_params, conv_cache, conv_grads, workspace.unsafe_ptr()
        )

    # =========================================================================
    # Inference-mode forward + backward (Phase 3.5b)
    #
    # Forward uses BN running stats (no batch reductions, no EMA update on
    # state). Backward reads the same cache layout as training but uses
    # running-stat `inv_std` and the simpler `dx_BN = γ · inv_std_r · dy_pre_relu`
    # formula (no batch sums). BN param grads (γ/β) are NOT written, mirroring
    # `BatchNorm1D` / `BatchNorm2D` `backward_gpu_inference`. Conv params
    # (W, b) are still updated — diverges from the BN1D/BN2D "skip all
    # grad_params writes" contract because the composite owns non-BN
    # trainable params; callers wanting full freeze must zero conv grads
    # themselves (matches REDQ-OFE's `zero_grads` pattern).
    # =========================================================================

    @always_inline
    @staticmethod
    def bn_relu_kernel_impl_inference_with_cache[
        BATCH: Int, dtype: DType = DType.float32,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Inference fused BN+ReLU: running stats, populates `cache` for
        inference-mode backward. Single-pass (no batch reduction)."""
        var c = Int(block_idx.x)
        if c >= Self.out_channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.spatial_out
        var eps = Scalar[dtype](Self.BN_EPSILON)
        comptime N = BATCH * Self.spatial_out
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
        var rmean = rebind[Scalar[dtype]](state[Self.RMEAN_OFF + c])
        var rvar = rebind[Scalar[dtype]](state[Self.RVAR_OFF + c])
        var inv_std: Scalar[dtype] = 1.0 / sqrt(rvar + eps)
        comptime OUT_STRIDE = Self.OUT_DIM

        var idx = tid
        while idx < N:
            var b = idx // Self.spatial_out
            var s = idx - b * Self.spatial_out
            var out_off = b * OUT_STRIDE + c_off + s
            var x = output.ptr[out_off]
            var x_hat = (x - rmean) * inv_std
            (cache.ptr + b * Self.CACHE_SIZE + Self.XHAT_OFF + c_off + s)[] = x_hat
            var pre_relu = gamma * x_hat + beta
            output.ptr[out_off] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
            idx += TPB

        # Cache inv_std for inference backward
        if tid < BATCH:
            (cache.ptr + tid * Self.CACHE_SIZE + Self.INVSTD_OFF + c)[] = inv_std

    @always_inline
    @staticmethod
    def relu_bn_backward_kernel_impl_inference[
        BATCH: Int, dtype: DType = DType.float32,
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
    ):
        """Inference ReLU+BN backward: `dx = γ·inv_std_r·dy_pre_relu`, no reductions, no BN param grads."""
        var c = Int(block_idx.x)
        if c >= Self.out_channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.spatial_out
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
        var inv_std = rebind[Scalar[dtype]](cache[0, Self.INVSTD_OFF + c])
        var scale = gamma * inv_std

        var idx = tid
        while idx < BATCH * Self.spatial_out:
            var b = idx // Self.spatial_out
            var s = idx % Self.spatial_out
            var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
            var pre_relu = gamma * x_hat + beta
            var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
            if pre_relu <= Scalar[dtype](0.0):
                dy = Scalar[dtype](0.0)
            grad_pre_bn[b, c_off + s] = scale * dy
            idx += TPB

    @staticmethod
    def forward_gpu_inference_with_cache[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Inference forward: Conv → BN(running stats) → ReLU. Populates cache for inference backward."""
        from ..autodiff import Conv2D
        comptime ConvOp = Conv2D[
            Self.in_channels, Self.out_channels,
            Self.kernel_size, Self.stride, Self.padding,
            Self.in_h, Self.in_w,
        ]

        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](params.ptr)

        comptime CONV_CS = ConvOp.CACHE_SIZE
        comptime TEMP_CACHE_OFF = BATCH * Self.CONV_CACHE + Self.col_size * Self.out_channels
        var conv_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ](workspace.unsafe_ptr() + TEMP_CACHE_OFF)

        ConvOp.eval_gpu[BATCH](ctx, output, input, conv_params, conv_cache, workspace.unsafe_ptr())

        # Copy im2col into our cache (Conv backward needs it)
        comptime COPY_SIZE = BATCH * CONV_CS
        @parameter
        @always_inline
        def copy_im2col_inf(
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
            src: LayoutTensor[dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= COPY_SIZE:
                return
            var b = tid // CONV_CS
            var i = tid % CONV_CS
            dst.ptr[b * Self.CACHE_SIZE + i] = src.ptr[tid]

        comptime COPY_BLOCKS = (COPY_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[copy_im2col_inf](
            cache, conv_cache,
            grid_dim=(COPY_BLOCKS,),
            block_dim=(TPB,),
        )

        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var state_immut = LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ](state.ptr)

        @parameter
        @always_inline
        def bn_relu_inf_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin],
            state: LayoutTensor[dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin],
        ):
            Self.bn_relu_kernel_impl_inference_with_cache[BATCH, dtype](output, cache, params, state)

        ctx.enqueue_function[bn_relu_inf_wrapper](
            output, cache, params_immut, state_immut,
            grid_dim=(Self.out_channels,),
            block_dim=(TPB,),
        )

    @staticmethod
    def backward_gpu_inference[
        BATCH: Int,
        dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        """Inference backward: dx_BN = γ·inv_std_r·dy_pre_relu → Conv backward.

        Skips BN param-grad writes (γ/β). Conv param grads (W, b) are still
        accumulated via `Conv2D.vjp_gpu` — see header comment.
        """
        from ..autodiff import Conv2D
        comptime ConvOp = Conv2D[
            Self.in_channels, Self.out_channels,
            Self.kernel_size, Self.stride, Self.padding,
            Self.in_h, Self.in_w,
        ]

        comptime GRAD_PRE_BN_OFF = BATCH * Self.CONV_CACHE + Self.col_size * Self.out_channels + BATCH * Self.CONV_CACHE
        var grad_pre_bn = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](workspace.unsafe_ptr() + GRAD_PRE_BN_OFF)

        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        @parameter
        @always_inline
        def relu_bn_bwd_inf_wrapper(
            grad_pre_bn: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin],
        ):
            Self.relu_bn_backward_kernel_impl_inference[BATCH, dtype](
                grad_pre_bn, grad_output, params, cache
            )

        ctx.enqueue_function[relu_bn_bwd_inf_wrapper](
            grad_pre_bn, grad_output_immut, params_immut, cache_immut,
            grid_dim=(Self.out_channels,),
            block_dim=(TPB,),
        )

        # Conv backward — same as training-mode (writes conv grads, grad_input).
        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var conv_grads = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)

        comptime CONV_CS = ConvOp.CACHE_SIZE
        comptime TEMP_CACHE_OFF = BATCH * Self.CONV_CACHE + Self.col_size * Self.out_channels
        var conv_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ](workspace.unsafe_ptr() + TEMP_CACHE_OFF)

        comptime COPY_SIZE = BATCH * CONV_CS
        @parameter
        @always_inline
        def copy_im2col_bwd_inf(
            dst: LayoutTensor[dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin],
            src: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= COPY_SIZE:
                return
            var b = tid // CONV_CS
            var i = tid % CONV_CS
            dst.ptr[tid] = src.ptr[b * Self.CACHE_SIZE + i]

        comptime COPY_BLOCKS = (COPY_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[copy_im2col_bwd_inf](
            conv_cache, cache,
            grid_dim=(COPY_BLOCKS,),
            block_dim=(TPB,),
        )

        ConvOp.vjp_gpu[BATCH](
            ctx, grad_pre_bn, grad_input, conv_params, conv_cache, conv_grads, workspace.unsafe_ptr()
        )
