"""Fused ResBlock with BatchNorm: y = ReLU(BN2(Conv2(Conv1BNReLU(x))) + x).

Reduces kernel launches vs the decomposed Sequential[Residual[Sequential[
Conv2DBatchNormReLU, Conv2DLayer, BatchNorm2D]], ReLU] by:
- Fusing BN2 + skip-add + ReLU into a single GPU kernel (instead of 3 separate)
- Using Conv2DBatchNormReLU directly for Conv1 (already fused)
- Sharing workspace between the two conv layers

Kernel launch count:
  Forward:  Conv1(2) + Conv2(1) + BN2+skip+ReLU(1) + copy_im2col(1) = 5
  Backward: BN2+skip+ReLU_bwd(1) + Conv2_bwd(AutoFused) + copy(1) + Conv1_bwd(2) + add(1) = ~8
  Total: ~13 per ResBlock (vs ~14 for decomposed, plus less combinator overhead)

The real win is eliminating Sequential/Residual/ReLU Model wrappers that each
add their own workspace management and kernel launch overhead.

Requires in_channels == out_channels and same padding (spatial dims preserved).
"""

from ..constants import dtype, TPB
from ..model.model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from .conv2d_bn_relu import Conv2DBatchNormReLU
from ..autodiff import AutoFused, Conv2D
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import sqrt, ceildiv


# ─── GPU kernels ──────────────────────────────────────────────────────────


def _bn_skip_relu_fwd_kernel[
    BATCH: Int,
    channels: Int,
    spatial: Int,
    BN_PS: Int,
    CACHE_SIZE: Int,
    GAMMA_OFF: Int,
    BETA_OFF: Int,
    RMEAN_OFF: Int,
    RVAR_OFF: Int,
    XHAT_OFF: Int,
    INVSTD_OFF: Int,
    BN_EPSILON: Float64,
    BN_MOMENTUM: Float64,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    skip: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    params: LayoutTensor[dtype, Layout.row_major(BN_PS), MutAnyOrigin],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    """Fused BN + skip-add + ReLU + cache. One block per channel.

    Optimized: 2-pass (Welford mean+var in pass 1, normalize+skip+relu in pass 2).
    Uses flat pointer math to avoid divmod per element.
    """
    var c = Int(block_idx.x)
    if c >= channels:
        return
    var tid = Int(thread_idx.x)
    var c_off = c * spatial
    var eps = Scalar[dtype](BN_EPSILON)
    var mom = Scalar[dtype](BN_MOMENTUM)
    var one_m = Scalar[dtype](1.0) - mom
    comptime N = BATCH * spatial
    var n_f = Scalar[dtype](N)
    var gamma = rebind[Scalar[dtype]](params[GAMMA_OFF + c])
    var beta = rebind[Scalar[dtype]](params[BETA_OFF + c])

    # Stride for output/skip: elements between output[b, c_off] and output[b+1, c_off]
    comptime OUT_STRIDE = channels * spatial

    var smem = LayoutTensor[
        dtype, Layout.row_major(TPB), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # ── Pass 1: Mean ──
    var local_sum = Scalar[dtype](0.0)
    var base = output.ptr + c_off
    var idx = tid
    while idx < N:
        var b = idx // spatial
        var s = idx - b * spatial
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

    # ── Pass 1b: Variance using (x - mean)² (numerically stable) ──
    var local_var = Scalar[dtype](0.0)
    idx = tid
    while idx < N:
        var b = idx // spatial
        var s = idx - b * spatial
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

    # ── Pass 2: Normalize + skip + ReLU + cache ──
    var skip_base = skip.ptr + c_off
    var cache_xhat_base = cache.ptr  # cache layout: [BATCH, CACHE_SIZE]
    idx = tid
    while idx < N:
        var b = idx // spatial
        var s = idx - b * spatial
        var out_off = b * OUT_STRIDE + c_off + s
        var x = output.ptr[out_off]
        var x_hat = (x - mean) * inv_std
        # Cache x_hat
        (cache.ptr + b * CACHE_SIZE + XHAT_OFF + c_off + s)[] = x_hat
        var bn_out = gamma * x_hat + beta
        var val = bn_out + (skip_base + b * OUT_STRIDE + s)[]
        # Cache pre-relu
        (cache.ptr + b * CACHE_SIZE + INVSTD_OFF + channels + c_off + s)[] = val
        output.ptr[out_off] = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        idx += TPB

    # Running stats + inv_std cache (parallel across threads)
    if tid < BATCH:
        (cache.ptr + tid * CACHE_SIZE + INVSTD_OFF + c)[] = inv_std
    if tid == 0:
        var rm = params.ptr[RMEAN_OFF + c]
        var rv = params.ptr[RVAR_OFF + c]
        params.ptr[RMEAN_OFF + c] = one_m * rm + mom * mean
        params.ptr[RVAR_OFF + c] = one_m * rv + mom * var_


def _bn_skip_relu_fwd_kernel_no_cache[
    BATCH: Int,
    channels: Int,
    spatial: Int,
    BN_PS: Int,
    GAMMA_OFF: Int,
    BETA_OFF: Int,
    RMEAN_OFF: Int,
    RVAR_OFF: Int,
    BN_EPSILON: Float64,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    skip: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    params: LayoutTensor[dtype, Layout.row_major(BN_PS), ImmutAnyOrigin],
):
    """Fused BN (running stats) + skip-add + ReLU. Inference, no cache, no update.

    One block per channel; parallel scatter over BATCH * spatial.
    """
    var c = Int(block_idx.x)
    if c >= channels:
        return
    var tid = Int(thread_idx.x)
    var c_off = c * spatial
    var eps = Scalar[dtype](BN_EPSILON)
    var gamma = rebind[Scalar[dtype]](params[GAMMA_OFF + c])
    var beta = rebind[Scalar[dtype]](params[BETA_OFF + c])
    var rmean = rebind[Scalar[dtype]](params[RMEAN_OFF + c])
    var rvar = rebind[Scalar[dtype]](params[RVAR_OFF + c])
    var inv_std: Scalar[dtype] = 1.0 / sqrt(rvar + eps)

    comptime N = BATCH * spatial
    comptime OUT_STRIDE = channels * spatial
    var base = output.ptr + c_off
    var skip_base = skip.ptr + c_off

    var idx = tid
    while idx < N:
        var b = idx // spatial
        var s = idx - b * spatial
        var off = b * OUT_STRIDE + s
        var x = (base + off)[]
        var bn_out = gamma * (x - rmean) * inv_std + beta
        var val = bn_out + (skip_base + off)[]
        (base + off)[] = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        idx += TPB


def _bn_skip_relu_bwd_kernel[
    BATCH: Int,
    channels: Int,
    spatial: Int,
    BN_PS: Int,
    CACHE_SIZE: Int,
    GAMMA_OFF: Int,
    BETA_OFF: Int,
    XHAT_OFF: Int,
    INVSTD_OFF: Int,
    dtype: DType = DType.float32,
](
    grad_conv2: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    grad_skip: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    params: LayoutTensor[dtype, Layout.row_major(BN_PS), MutAnyOrigin],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
    grads: LayoutTensor[dtype, Layout.row_major(BN_PS), MutAnyOrigin],
):
    """Fused backward: ReLU mask → BN backward → grad_conv2, + skip grad accumulate.

    From grad_output (gradient of loss w.r.t. block output):
    1. Apply ReLU mask using cached pre-relu values
    2. Accumulate into grad_skip (skip connection gradient)
    3. BN backward: compute grad w.r.t. conv2 output + BN param grads
    """
    var c = Int(block_idx.x)
    if c >= channels:
        return
    var tid = Int(thread_idx.x)
    var c_off = c * spatial
    var n_f = Scalar[dtype](BATCH * spatial)
    var gamma = rebind[Scalar[dtype]](params[GAMMA_OFF + c])
    var inv_std = rebind[Scalar[dtype]](cache[0, INVSTD_OFF + c])

    var smem = LayoutTensor[
        dtype, Layout.row_major(TPB), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Pass 1: Accumulate partials with ReLU mask
    var local_d_gamma = Scalar[dtype](0.0)
    var local_d_beta = Scalar[dtype](0.0)
    var local_sum_dy_g = Scalar[dtype](0.0)
    var local_sum_dy_g_xh = Scalar[dtype](0.0)

    var idx = tid
    while idx < BATCH * spatial:
        var b = idx // spatial
        var s = idx % spatial
        var pre_relu = rebind[Scalar[dtype]](cache[b, INVSTD_OFF + channels + c_off + s])
        var go = rebind[Scalar[dtype]](grad_output[b, c_off + s])
        # ReLU mask
        var dy = go if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        # Skip gradient (write, not accumulate — buffer may contain stale data)
        grad_skip[b, c_off + s] = dy
        # BN backward partials
        var x_hat = rebind[Scalar[dtype]](cache[b, XHAT_OFF + c_off + s])
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

    # Write param grads
    if tid == 0:
        grads.ptr[GAMMA_OFF + c] = rebind[Scalar[dtype]](grads[GAMMA_OFF + c]) + d_gamma
        grads.ptr[BETA_OFF + c] = rebind[Scalar[dtype]](grads[BETA_OFF + c]) + d_beta

    # Pass 2: grad w.r.t. conv2 output
    idx = tid
    while idx < BATCH * spatial:
        var b = idx // spatial
        var s = idx % spatial
        var pre_relu = rebind[Scalar[dtype]](cache[b, INVSTD_OFF + channels + c_off + s])
        var go = rebind[Scalar[dtype]](grad_output[b, c_off + s])
        var dy = go if pre_relu > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        var x_hat = rebind[Scalar[dtype]](cache[b, XHAT_OFF + c_off + s])
        grad_conv2[b, c_off + s] = inv_std * (
            dy * gamma - sum_dy_g / n_f - x_hat * sum_dy_g_xh / n_f
        )
        idx += TPB


def _add_kernel_flat[
    SIZE: Int,
    dtype: DType = DType.float32,
](
    a: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    a[idx] = rebind[Scalar[dtype]](a[idx]) + rebind[Scalar[dtype]](b[idx])


# ─── Struct ───────────────────────────────────────────────────────────────


@fieldwise_init
struct ResBlockConv2DBN[
    channels: Int,
    kernel_size: Int,
    padding: Int,
    h: Int,
    w: Int,
    BN_MOMENTUM: Float64 = 0.1,
    BN_EPSILON: Float64 = 1e-5,
](Model):
    """Fused ResBlock with BatchNorm: y = ReLU(BN2(Conv2(Conv1BNReLU(x))) + x).

    Conv1 uses Conv2DBatchNormReLU (fused conv+bn+relu).
    Conv2 uses AutoFused[Conv2D] (plain conv, no activation).
    BN2 + skip-add + ReLU are fused into a single GPU kernel.
    """

    comptime Conv1 = Conv2DBatchNormReLU[
        Self.channels, Self.channels, Self.kernel_size, 1, Self.padding,
        Self.h, Self.w, Self.BN_MOMENTUM, Self.BN_EPSILON,
    ]
    comptime Conv2 = AutoFused[Conv2D[
        Self.channels, Self.channels, Self.kernel_size, 1, Self.padding,
        Self.h, Self.w,
    ]]

    comptime DIM: Int = Self.channels * Self.h * Self.w
    comptime spatial: Int = Self.h * Self.w
    comptime IN_DIM: Int = Self.DIM
    comptime OUT_DIM: Int = Self.DIM

    # Params: Conv1 params + Conv2 params + BN2 params (gamma, beta, rmean, rvar)
    comptime CONV1_PS: Int = Self.Conv1.PARAM_SIZE
    comptime CONV2_PS: Int = Self.Conv2.PARAM_SIZE
    comptime BN2_PS: Int = 4 * Self.channels
    comptime PARAM_SIZE: Int = Self.CONV1_PS + Self.CONV2_PS + Self.BN2_PS

    # BN2 param offsets (within params)
    comptime BN2_OFF: Int = Self.CONV1_PS + Self.CONV2_PS
    comptime BN2_GAMMA_OFF: Int = 0
    comptime BN2_BETA_OFF: Int = Self.channels
    comptime BN2_RMEAN_OFF: Int = 2 * Self.channels
    comptime BN2_RVAR_OFF: Int = 3 * Self.channels

    # Cache: Conv1 cache + Conv2 cache + BN2 cache
    # BN2 cache per sample: x_hat(DIM) + inv_std(channels) + pre_relu(DIM)
    comptime CONV1_CS: Int = Self.Conv1.CACHE_SIZE
    comptime CONV2_CS: Int = Self.Conv2.CACHE_SIZE
    comptime BN2_CS: Int = Self.DIM + Self.channels + Self.DIM  # x_hat + inv_std + pre_relu
    comptime CACHE_SIZE: Int = Self.CONV1_CS + Self.CONV2_CS + Self.BN2_CS

    # BN2 cache offsets (within per-sample cache)
    comptime BN2_CACHE_OFF: Int = Self.CONV1_CS + Self.CONV2_CS
    comptime BN2_XHAT_OFF: Int = 0
    comptime BN2_INVSTD_OFF: Int = Self.DIM  # inv_std(C) followed by pre_relu(DIM)

    # Workspace
    comptime CONV1_WS: Int = Self.Conv1.WORKSPACE_SIZE_PER_SAMPLE
    comptime CONV2_WS: Int = Self.Conv2.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_CONV_WS: Int = Self.CONV1_WS if Self.CONV1_WS > Self.CONV2_WS else Self.CONV2_WS
    # conv ws + grad_conv2 buffer + temp_gi buffer + grad_inter buffer
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.MAX_CONV_WS + Self.DIM + Self.DIM + Self.DIM
    # TODO(phase 3): once BatchNorm2D / Conv2DBatchNormReLU migrate their running
    # stats into state, this should become
    #   Self.Conv1.STATE_SIZE + Self.Conv2.STATE_SIZE + (BN2 running stats size).
    # For Phase 1 BN's STATE_SIZE is still 0 so a pure pass-through zero-size
    # state is valid; sub-model state slices are zero-length views.
    comptime STATE_SIZE: Int = 0

    # ── Initialization ─────────────────────────────────────────────

    @staticmethod
    def initialize_params[
        INIT: Initializer,
        dtype: DType = DType.float32,
    ](
        mut params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        Self.Conv1.initialize_params[INIT](p1)
        Self.Conv2.initialize_params[INIT](p2)
        # BN2: gamma=1, beta=0, rmean=0, rvar=1
        for c in range(Self.channels):
            params.ptr[Self.BN2_OFF + Self.BN2_GAMMA_OFF + c] = Scalar[dtype](1.0)
            params.ptr[Self.BN2_OFF + Self.BN2_BETA_OFF + c] = Scalar[dtype](0.0)
            params.ptr[Self.BN2_OFF + Self.BN2_RMEAN_OFF + c] = Scalar[dtype](0.0)
            params.ptr[Self.BN2_OFF + Self.BN2_RVAR_OFF + c] = Scalar[dtype](1.0)

    # ── CPU Forward ────────────────────────────────────────────────

    @staticmethod
    def forward[BATCH: Int, dtype: DType = DType.float32](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
    ):
        """CPU forward: Conv1BNReLU → Conv2 → BN2 + skip + ReLU."""
        from std.memory import alloc
        from std.math import sqrt as msqrt

        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV1_CS), MutAnyOrigin](cache.ptr)
        var c2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)
        var bn2c = cache.ptr + BATCH * Self.BN2_CACHE_OFF
        # Sub-model state slices. BN's STATE_SIZE is still 0 in Phase 1, so these
        # are zero-length views; Phase 3 will split a real state buffer here.
        var s1 = LayoutTensor[dtype, Layout.row_major(Self.Conv1.STATE_SIZE), MutAnyOrigin](state.ptr)
        var s2 = LayoutTensor[dtype, Layout.row_major(Self.Conv2.STATE_SIZE), MutAnyOrigin](state.ptr)

        # 1. Conv1BNReLU → inter
        var inter = alloc[Scalar[dtype]](BATCH * Self.DIM)
        var inter_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward[BATCH, dtype](in_rb, inter_t, p1, s1, c1)

        # 2. Conv2 → output (holds conv2 output pre-BN)
        var inter_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin]](inter_t)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward[BATCH, dtype](inter_rb, out_rb, p2, s2, c2)

        # 3. BN2 + skip + ReLU + cache
        var eps = Scalar[dtype](Self.BN_EPSILON)
        var mom = Scalar[dtype](Self.BN_MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        comptime C = Self.channels
        comptime S = Self.spatial
        comptime N_ELEM = BATCH * S

        for c in range(C):
            var gamma = params.ptr[Self.BN2_OFF + Self.BN2_GAMMA_OFF + c]
            var beta = params.ptr[Self.BN2_OFF + Self.BN2_BETA_OFF + c]

            # Mean
            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(S):
                    mean += output.ptr[b * Self.DIM + c * S + s]
            mean = mean / Scalar[dtype](N_ELEM)

            # Variance
            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(S):
                    var diff = output.ptr[b * Self.DIM + c * S + s] - mean
                    var_ += diff * diff
            var_ = var_ / Scalar[dtype](N_ELEM)
            var inv_std = Scalar[dtype](1.0 / msqrt(Float64(var_ + eps)))

            # Normalize + skip + ReLU + cache
            for b in range(BATCH):
                (bn2c + b * Self.BN2_CS + Self.BN2_INVSTD_OFF + c)[] = inv_std
                for s in range(S):
                    var idx = b * Self.DIM + c * S + s
                    var x = output.ptr[idx]
                    var x_hat = (x - mean) * inv_std
                    (bn2c + b * Self.BN2_CS + Self.BN2_XHAT_OFF + c * S + s)[] = x_hat
                    var bn_out = gamma * x_hat + beta
                    var val = bn_out + input.ptr[idx]
                    (bn2c + b * Self.BN2_CS + Self.BN2_INVSTD_OFF + C + c * S + s)[] = val
                    output.ptr[idx] = val if Float64(val) > 0.0 else Scalar[dtype](0.0)

            # Update running stats
            var rm = params.ptr[Self.BN2_OFF + Self.BN2_RMEAN_OFF + c]
            var rv = params.ptr[Self.BN2_OFF + Self.BN2_RVAR_OFF + c]
            params.ptr[Self.BN2_OFF + Self.BN2_RMEAN_OFF + c] = one_m * rm + mom * mean
            params.ptr[Self.BN2_OFF + Self.BN2_RVAR_OFF + c] = one_m * rv + mom * var_

        inter.free()

    @staticmethod
    def forward[BATCH: Int, dtype: DType = DType.float32](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """CPU inference forward: Conv1BNReLU (running stats) → Conv2 → BN2 (running stats) + skip + ReLU."""
        from std.memory import alloc
        from std.math import sqrt as msqrt

        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        # Sub-model state slices. BN's STATE_SIZE is still 0 in Phase 1, so these
        # are zero-length views; Phase 3 will split a real state buffer here.
        var s1 = LayoutTensor[dtype, Layout.row_major(Self.Conv1.STATE_SIZE), MutAnyOrigin](state.ptr)
        var s2 = LayoutTensor[dtype, Layout.row_major(Self.Conv2.STATE_SIZE), MutAnyOrigin](state.ptr)

        # 1. Conv1BNReLU inference → inter
        var inter = alloc[Scalar[dtype]](BATCH * Self.DIM)
        var inter_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward[BATCH, dtype](in_rb, inter_t, p1, s1)

        # 2. Conv2 inference → output (pre-BN2)
        var inter_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin]](inter_t)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward[BATCH, dtype](inter_rb, out_rb, p2, s2)

        # 3. BN2 (running stats) + skip + ReLU
        var eps = Scalar[dtype](Self.BN_EPSILON)
        comptime C = Self.channels
        comptime S = Self.spatial

        for c in range(C):
            var gamma = params.ptr[Self.BN2_OFF + Self.BN2_GAMMA_OFF + c]
            var beta = params.ptr[Self.BN2_OFF + Self.BN2_BETA_OFF + c]
            var rmean = params.ptr[Self.BN2_OFF + Self.BN2_RMEAN_OFF + c]
            var rvar = params.ptr[Self.BN2_OFF + Self.BN2_RVAR_OFF + c]
            var inv_std = Scalar[dtype](1.0 / msqrt(Float64(rvar + eps)))

            for b in range(BATCH):
                for s in range(S):
                    var idx = b * Self.DIM + c * S + s
                    var x = output.ptr[idx]
                    var bn_out = gamma * (x - rmean) * inv_std + beta
                    var val = bn_out + input.ptr[idx]
                    output.ptr[idx] = val if Float64(val) > 0.0 else Scalar[dtype](0.0)

        inter.free()

    @staticmethod
    def backward[BATCH: Int, dtype: DType = DType.float32](
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        """CPU backward: ReLU+BN2 backward → Conv2 backward → Conv1 backward.

        Mirrors the GPU backward_gpu logic step by step.
        """
        from std.memory import alloc, memset

        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV1_CS), MutAnyOrigin](cache.ptr)
        var c2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)
        var g1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](grads.ptr)
        var g2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](grads.ptr + Self.CONV1_PS)
        var bn2c = cache.ptr + BATCH * Self.BN2_CACHE_OFF
        # Sub-model state slices. BN's STATE_SIZE is still 0 in Phase 1, so these
        # are zero-length views; Phase 3 will split a real state buffer here.
        var s1 = LayoutTensor[dtype, Layout.row_major(Self.Conv1.STATE_SIZE), MutAnyOrigin](state.ptr)
        var s2 = LayoutTensor[dtype, Layout.row_major(Self.Conv2.STATE_SIZE), MutAnyOrigin](state.ptr)

        comptime C = Self.channels
        comptime S = Self.spatial
        comptime N_ELEM = BATCH * S

        # 1. Fused ReLU + BN2 backward per channel → grad_conv2 + skip grad
        var grad_conv2 = alloc[Scalar[dtype]](BATCH * Self.DIM)
        memset(grad_conv2, 0, BATCH * Self.DIM)

        for c in range(C):
            var gamma = params.ptr[Self.BN2_OFF + Self.BN2_GAMMA_OFF + c]
            var inv_std = (bn2c + 0 * Self.BN2_CS + Self.BN2_INVSTD_OFF + c)[]
            var n_f = Scalar[dtype](N_ELEM)

            # Pass 1: accumulate BN partials + skip grad
            var d_gamma = Scalar[dtype](0.0)
            var d_beta = Scalar[dtype](0.0)
            var sum_dy_g = Scalar[dtype](0.0)
            var sum_dy_g_xh = Scalar[dtype](0.0)

            for b in range(BATCH):
                for s in range(S):
                    var pre_relu = (bn2c + b * Self.BN2_CS + Self.BN2_INVSTD_OFF + C + c * S + s)[]
                    var go = grad_output.ptr[b * Self.DIM + c * S + s]
                    var dy = go if Float64(pre_relu) > 0.0 else Scalar[dtype](0.0)
                    grad_input.ptr[b * Self.DIM + c * S + s] = dy  # skip path

                    var x_hat = (bn2c + b * Self.BN2_CS + Self.BN2_XHAT_OFF + c * S + s)[]
                    d_gamma += dy * x_hat
                    d_beta += dy
                    sum_dy_g += dy * gamma
                    sum_dy_g_xh += dy * gamma * x_hat

            # Accumulate BN2 param grads
            var g_off = Self.CONV1_PS + Self.CONV2_PS
            grads.ptr[g_off + Self.BN2_GAMMA_OFF + c] = grads.ptr[g_off + Self.BN2_GAMMA_OFF + c] + d_gamma
            grads.ptr[g_off + Self.BN2_BETA_OFF + c] = grads.ptr[g_off + Self.BN2_BETA_OFF + c] + d_beta

            # Pass 2: grad w.r.t. conv2 output
            for b in range(BATCH):
                for s in range(S):
                    var pre_relu = (bn2c + b * Self.BN2_CS + Self.BN2_INVSTD_OFF + C + c * S + s)[]
                    var go = grad_output.ptr[b * Self.DIM + c * S + s]
                    var dy = go if Float64(pre_relu) > 0.0 else Scalar[dtype](0.0)
                    var x_hat = (bn2c + b * Self.BN2_CS + Self.BN2_XHAT_OFF + c * S + s)[]
                    (grad_conv2 + b * Self.DIM + c * S + s)[] = inv_std * (
                        dy * gamma - sum_dy_g / n_f - x_hat * sum_dy_g_xh / n_f
                    )

        # 2. Conv2 backward: grad_conv2 → grad_inter
        var grad_inter = alloc[Scalar[dtype]](BATCH * Self.DIM)
        memset(grad_inter, 0, BATCH * Self.DIM)
        var grad_conv2_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin](grad_conv2)
        var grad_inter_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](grad_inter)
        Self.Conv2.backward[BATCH, dtype](grad_conv2_t, grad_inter_t, p2, s2, c2, g2)

        # 3. Conv1BNReLU backward: grad_inter → temp_gi
        var temp_gi = alloc[Scalar[dtype]](BATCH * Self.DIM)
        memset(temp_gi, 0, BATCH * Self.DIM)
        var go_c1 = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin]](grad_inter_t)
        var temp_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin](temp_gi)
        Self.Conv1.backward[BATCH, dtype](go_c1, temp_gi_t, p1, s1, c1, g1)

        # 4. grad_input += conv1's grad_input
        for i in range(BATCH * Self.DIM):
            grad_input.ptr[i] = grad_input.ptr[i] + (temp_gi + i)[]

        grad_conv2.free()
        grad_inter.free()
        temp_gi.free()

    # ── GPU Forward (with cache) ─────────────────────────────────

    @staticmethod
    def forward_gpu[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        var c1_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV1_CS), MutAnyOrigin](cache.ptr)
        var c2_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)
        # Sub-model state slices. BN's STATE_SIZE is still 0 in Phase 1, so these
        # are zero-length views; Phase 3 will split a real state buffer here.
        var s1 = LayoutTensor[dtype, Layout.row_major(Self.Conv1.STATE_SIZE), MutAnyOrigin](state.ptr)
        var s2 = LayoutTensor[dtype, Layout.row_major(Self.Conv2.STATE_SIZE), MutAnyOrigin](state.ptr)

        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        # Conv1+BN1+ReLU → inter
        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var inter_out = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter_ptr)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward_gpu[BATCH](ctx, inter_out, in_rb, p1, s1, c1_v, conv_ws)

        # Conv2 (no activation) → output (temporarily holds conv2 output pre-BN)
        var inter_in = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](inter_ptr)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward_gpu[BATCH](ctx, out_rb, inter_in, p2, s2, c2_v, conv_ws)

        # Fused BN2 + skip-add + ReLU (one kernel, one block per channel)
        var bn2_params = LayoutTensor[dtype, Layout.row_major(Self.BN2_PS), MutAnyOrigin](params.ptr + Self.BN2_OFF)
        var bn2_cache = LayoutTensor[dtype, Layout.row_major(BATCH, Self.BN2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.BN2_CACHE_OFF)
        var skip_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](input.ptr)

        comptime fwd_k = _bn_skip_relu_fwd_kernel[
            BATCH, Self.channels, Self.spatial, Self.BN2_PS, Self.BN2_CS,
            Self.BN2_GAMMA_OFF, Self.BN2_BETA_OFF, Self.BN2_RMEAN_OFF, Self.BN2_RVAR_OFF,
            Self.BN2_XHAT_OFF, Self.BN2_INVSTD_OFF,
            Self.BN_EPSILON, Self.BN_MOMENTUM, dtype,
        ]
        ctx.enqueue_function[fwd_k, fwd_k](
            output, skip_t, bn2_params, bn2_cache,
            grid_dim=(Self.channels,), block_dim=(TPB,),
        )

    # ── GPU Forward (inference, no cache) ────────────────────────

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        # Sub-model state slices. BN's STATE_SIZE is still 0 in Phase 1, so these
        # are zero-length views; Phase 3 will split a real state buffer here.
        var s1 = LayoutTensor[dtype, Layout.row_major(Self.Conv1.STATE_SIZE), MutAnyOrigin](state.ptr)
        var s2 = LayoutTensor[dtype, Layout.row_major(Self.Conv2.STATE_SIZE), MutAnyOrigin](state.ptr)

        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var inter_out = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter_ptr)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward_gpu_no_cache[BATCH](ctx, inter_out, in_rb, p1, s1, conv_ws)

        var inter_in = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](inter_ptr)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward_gpu_no_cache[BATCH](ctx, out_rb, inter_in, p2, s2, conv_ws)

        # BN2 (running stats) + skip + ReLU — inference, no cache, no stat update
        var bn2_params = LayoutTensor[dtype, Layout.row_major(Self.BN2_PS), ImmutAnyOrigin](params.ptr + Self.BN2_OFF)
        var skip_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](input.ptr)

        comptime fwd_k = _bn_skip_relu_fwd_kernel_no_cache[
            BATCH, Self.channels, Self.spatial, Self.BN2_PS,
            Self.BN2_GAMMA_OFF, Self.BN2_BETA_OFF, Self.BN2_RMEAN_OFF, Self.BN2_RVAR_OFF,
            Self.BN_EPSILON, dtype,
        ]
        ctx.enqueue_function[fwd_k, fwd_k](
            output, skip_t, bn2_params,
            grid_dim=(Self.channels,), block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, state, workspace)

    # ── GPU Backward ─────────────────────────────────────────────

    @staticmethod
    def backward_gpu[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        # Sub-model state slices. BN's STATE_SIZE is still 0 in Phase 1, so these
        # are zero-length views; Phase 3 will split a real state buffer here.
        var s1 = LayoutTensor[dtype, Layout.row_major(Self.Conv1.STATE_SIZE), MutAnyOrigin](state.ptr)
        var s2 = LayoutTensor[dtype, Layout.row_major(Self.Conv2.STATE_SIZE), MutAnyOrigin](state.ptr)

        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        comptime TOTAL = BATCH * Self.DIM
        comptime BLOCKS = ceildiv(TOTAL, TPB)

        # Workspace layout:
        #   [0, MAX_CONV_WS)         — conv workspace (shared by Conv1/Conv2 backward)
        #   [MAX_CONV_WS, +DIM)      — grad_conv2 (BN backward output)
        #   [MAX_CONV_WS+DIM, +DIM)  — temp_gi (Conv1 backward output)
        #   [MAX_CONV_WS+2*DIM, +DIM) — grad_inter (Conv2 backward output)
        var grad_conv2_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var temp_gi_ptr = grad_conv2_ptr + BATCH * Self.DIM
        var grad_inter_ptr = temp_gi_ptr + BATCH * Self.DIM

        # 1. Fused BN2+skip+ReLU backward: grad_output → grad_conv2 + grad_skip
        var bn2_params = LayoutTensor[dtype, Layout.row_major(Self.BN2_PS), MutAnyOrigin](params.ptr + Self.BN2_OFF)
        var bn2_cache = LayoutTensor[dtype, Layout.row_major(BATCH, Self.BN2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.BN2_CACHE_OFF)
        var bn2_grads = LayoutTensor[dtype, Layout.row_major(Self.BN2_PS), MutAnyOrigin](grads.ptr + Self.BN2_OFF)
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](grad_output.ptr)
        var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](grad_input.ptr)
        var grad_conv2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](grad_conv2_ptr)

        comptime bwd_k = _bn_skip_relu_bwd_kernel[
            BATCH, Self.channels, Self.spatial, Self.BN2_PS, Self.BN2_CS,
            Self.BN2_GAMMA_OFF, Self.BN2_BETA_OFF, Self.BN2_XHAT_OFF, Self.BN2_INVSTD_OFF,
            dtype,
        ]
        ctx.enqueue_function[bwd_k, bwd_k](
            grad_conv2, go_t, gi_t, bn2_params, bn2_cache, bn2_grads,
            grid_dim=(Self.channels,), block_dim=(TPB,),
        )

        # 2. Conv2 backward: grad_conv2 → grad_inter (separate workspace region)
        var grad_inter = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](grad_inter_ptr)
        var go_c2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin](grad_conv2_ptr)
        var c2_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)
        var g2_v = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](grads.ptr + Self.CONV1_PS)
        Self.Conv2.backward_gpu[BATCH](ctx, grad_inter, go_c2, p2, s2, c2_v, g2_v, conv_ws)

        # 3. Conv1 backward: grad_inter → temp_gi (separate workspace region)
        var go_c1 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](grad_inter_ptr)
        var temp_gi = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin](temp_gi_ptr)
        var c1_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV1_CS), MutAnyOrigin](cache.ptr)
        var g1_v = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](grads.ptr)
        Self.Conv1.backward_gpu[BATCH](ctx, temp_gi, go_c1, p1, s1, c1_v, g1_v, conv_ws)

        # 4. Add conv1's grad_input to skip grad
        @parameter
        @always_inline
        def add_wrapper(
            a: LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin],
            b: LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

        var temp_flat = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](temp_gi_ptr)
        ctx.enqueue_function[add_wrapper, add_wrapper](
            grad_input, temp_flat,
            grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
