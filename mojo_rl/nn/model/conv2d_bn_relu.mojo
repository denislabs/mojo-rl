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
from std.gpu import thread_idx, block_idx, block_dim
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

    # Params: conv_W (oc * col_size) + conv_bias (oc) + BN gamma/beta/rmean/rvar (4*oc)
    comptime CONV_W_SIZE: Int = Self.out_channels * Self.col_size
    comptime CONV_PARAM_SIZE: Int = Self.CONV_W_SIZE + Self.out_channels
    comptime BN_PARAM_SIZE: Int = 4 * Self.out_channels
    comptime PARAM_SIZE: Int = Self.CONV_PARAM_SIZE + Self.BN_PARAM_SIZE

    # Param offsets
    comptime W_OFF: Int = 0
    comptime BIAS_OFF: Int = Self.CONV_W_SIZE
    comptime GAMMA_OFF: Int = Self.CONV_PARAM_SIZE
    comptime BETA_OFF: Int = Self.CONV_PARAM_SIZE + Self.out_channels
    comptime RMEAN_OFF: Int = Self.CONV_PARAM_SIZE + 2 * Self.out_channels
    comptime RVAR_OFF: Int = Self.CONV_PARAM_SIZE + 3 * Self.out_channels

    # Cache: im2col + x_hat + batch_inv_std
    comptime CONV_CACHE: Int = Self.col_size * Self.spatial_out
    comptime CACHE_SIZE: Int = Self.CONV_CACHE + Self.OUT_DIM + Self.out_channels

    # Cache offsets
    comptime XHAT_OFF: Int = Self.CONV_CACHE
    comptime INVSTD_OFF: Int = Self.CONV_CACHE + Self.OUT_DIM

    # Workspace for GPU conv matmul
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.CONV_CACHE + Self.OUT_DIM + Self.col_size * Self.out_channels

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
        """Init conv weights with INIT, bias=0, BN gamma=1, beta=0, rmean=0, rvar=1."""
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
        # BN params
        for i in range(Self.out_channels):
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

            # Update running stats
            params.ptr[Self.RMEAN_OFF + c] = one_m * rebind[Scalar[dtype]](params[Self.RMEAN_OFF + c]) + mom * mean
            params.ptr[Self.RVAR_OFF + c] = one_m * rebind[Scalar[dtype]](params[Self.RVAR_OFF + c]) + mom * var_

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
        """Inference forward: Conv → BN (batch stats) → ReLU, no caching."""
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

        # BN + ReLU using batch stats
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var n = Scalar[dtype](BATCH * Self.spatial_out)

        for c in range(Self.out_channels):
            var c_off = c * Self.spatial_out
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    mean += rebind[Scalar[dtype]](output[b, c_off + s])
            mean = mean / n

            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var diff = rebind[Scalar[dtype]](output[b, c_off + s]) - mean
                    var_ += diff * diff
            var_ = var_ / n

            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](sqrt(Float64(var_ + eps)))

            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var x = rebind[Scalar[dtype]](output[b, c_off + s])
                    var pre_relu = gamma * (x - mean) * inv_std + beta
                    output[b, c_off + s] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)

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

        Grid: (out_channels,), Block: (1,)
        One thread per channel — computes stats across BATCH * spatial.
        """
        var c = Int(block_idx.x)
        if c >= Self.out_channels or thread_idx.x != 0:
            return

        var c_off = c * Self.spatial_out
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var mom = Scalar[dtype](Self.BN_MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n = Scalar[dtype](BATCH * Self.spatial_out)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

        # Compute batch mean
        var mean = Scalar[dtype](0.0)
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                mean += rebind[Scalar[dtype]](output[b, c_off + s])
        mean = mean / n

        # Compute batch variance
        var var_ = Scalar[dtype](0.0)
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var diff = rebind[Scalar[dtype]](output[b, c_off + s]) - mean
                var_ += diff * diff
        var_ = var_ / n

        var inv_std: Scalar[dtype] = 1.0 / sqrt(var_ + eps)

        # Normalize + scale + shift + ReLU
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var x = rebind[Scalar[dtype]](output[b, c_off + s])
                var x_hat = (x - mean) * inv_std
                cache[b, Self.XHAT_OFF + c_off + s] = x_hat
                var pre_relu = gamma * x_hat + beta
                output[b, c_off + s] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
            cache[b, Self.INVSTD_OFF + c] = inv_std

        # Update running stats
        var rm = rebind[Scalar[dtype]](params[Self.RMEAN_OFF + c])
        var rv = rebind[Scalar[dtype]](params[Self.RVAR_OFF + c])
        params.ptr[Self.RMEAN_OFF + c] = one_m * rm + mom * mean
        params.ptr[Self.RVAR_OFF + c] = one_m * rv + mom * var_

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

        Grid: (out_channels,), Block: (1,)
        """
        var c = Int(block_idx.x)
        if c >= Self.out_channels or thread_idx.x != 0:
            return

        var c_off = c * Self.spatial_out
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var n = Scalar[dtype](BATCH * Self.spatial_out)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

        var mean = Scalar[dtype](0.0)
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                mean += rebind[Scalar[dtype]](output[b, c_off + s])
        mean = mean / n

        var var_ = Scalar[dtype](0.0)
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var diff = rebind[Scalar[dtype]](output[b, c_off + s]) - mean
                var_ += diff * diff
        var_ = var_ / n

        var inv_std: Scalar[dtype] = 1.0 / sqrt(var_ + eps)

        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var x = rebind[Scalar[dtype]](output[b, c_off + s])
                var pre_relu = gamma * (x - mean) * inv_std + beta
                output[b, c_off + s] = pre_relu if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)

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
        """Fused ReLU+BN backward kernel. Produces grad w.r.t. conv output.

        Grid: (out_channels,), Block: (1,)
        """
        var c = Int(block_idx.x)
        if c >= Self.out_channels or thread_idx.x != 0:
            return

        var c_off = c * Self.spatial_out
        var n = Scalar[dtype](BATCH * Self.spatial_out)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])
        var inv_std = rebind[Scalar[dtype]](cache[0, Self.INVSTD_OFF + c])

        var d_gamma = Scalar[dtype](0.0)
        var d_beta = Scalar[dtype](0.0)
        var sum_dy_g = Scalar[dtype](0.0)
        var sum_dy_g_xh = Scalar[dtype](0.0)

        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
                var pre_relu = gamma * x_hat + beta
                var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
                if pre_relu <= Scalar[dtype](0.0):
                    dy = Scalar[dtype](0.0)
                d_gamma += dy * x_hat
                d_beta += dy
                sum_dy_g += dy * gamma
                sum_dy_g_xh += dy * gamma * x_hat

        grads.ptr[Self.GAMMA_OFF + c] = rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + c]) + d_gamma
        grads.ptr[Self.BETA_OFF + c] = rebind[Scalar[dtype]](grads[Self.BETA_OFF + c]) + d_beta

        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
                var pre_relu = gamma * x_hat + beta
                var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
                if pre_relu <= Scalar[dtype](0.0):
                    dy = Scalar[dtype](0.0)
                grad_pre_bn[b, c_off + s] = inv_std * (
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

        # Rebind cache to Conv2D's expected layout (im2col part)
        comptime CONV_CS = ConvOp.CACHE_SIZE
        var conv_cache = rebind[LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ]](cache)

        # Run Conv2D GPU forward (writes pre-BN output to `output`)
        ConvOp.eval_gpu[BATCH](ctx, output, input, conv_params, conv_cache, workspace.unsafe_ptr())

        # Run fused BN+ReLU kernel
        @always_inline
        def bn_relu_wrapper(
            output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
            params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        ):
            Self.bn_relu_kernel_impl[BATCH](output, cache, params)

        ctx.enqueue_function[bn_relu_wrapper, bn_relu_wrapper](
            output, cache, params,
            grid_dim=(Self.out_channels,),
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
        """GPU inference forward: Conv matmul → fused BN+ReLU (batch stats)."""
        from ..autodiff import Conv2D
        comptime ConvOp = Conv2D[
            Self.in_channels, Self.out_channels,
            Self.kernel_size, Self.stride, Self.padding,
            Self.in_h, Self.in_w,
        ]

        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](params.ptr)

        # Allocate dummy cache for Conv2D (needed by eval_gpu)
        comptime CONV_CS = ConvOp.CACHE_SIZE
        var dummy_cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * CONV_CS if CONV_CS > 0 else 1)
        var dummy_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ](dummy_cache_buf.unsafe_ptr())

        ConvOp.eval_gpu[BATCH](ctx, output, input, conv_params, dummy_cache, workspace.unsafe_ptr())

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
            grid_dim=(Self.out_channels,),
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
        """GPU backward: fused ReLU+BN backward → Conv backward."""
        from ..autodiff import Conv2D
        comptime ConvOp = Conv2D[
            Self.in_channels, Self.out_channels,
            Self.kernel_size, Self.stride, Self.padding,
            Self.in_h, Self.in_w,
        ]

        # Allocate temp buffer for grad_pre_bn (same size as output)
        var grad_pre_bn_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.OUT_DIM)
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
            grid_dim=(Self.out_channels,),
            block_dim=(1,),
        )

        # Step 2: Conv backward (dW, db, dx)
        var conv_params = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var conv_grads = LayoutTensor[
            dtype, Layout.row_major(Self.CONV_PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)

        # Rebind cache to Conv2D's expected layout (im2col portion = first CONV_CACHE per sample)
        comptime CONV_CS = ConvOp.CACHE_SIZE
        var conv_cache = rebind[LayoutTensor[
            dtype, Layout.row_major(BATCH, CONV_CS), MutAnyOrigin
        ]](cache)
        ConvOp.vjp_gpu[BATCH](
            ctx, grad_pre_bn, grad_input, conv_params, conv_cache, conv_grads, workspace.unsafe_ptr()
        )
