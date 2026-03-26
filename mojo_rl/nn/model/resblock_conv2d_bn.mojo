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
    dtype: DType where dtype.is_floating_point(),
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    skip: LayoutTensor[dtype, Layout.row_major(BATCH, channels * spatial), MutAnyOrigin],
    params: LayoutTensor[dtype, Layout.row_major(BN_PS), MutAnyOrigin],
    cache: LayoutTensor[dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    """Fused BN + skip-add + ReLU + cache. One block per channel.

    output = ReLU(gamma * (conv2_out - mean) / sqrt(var + eps) + beta + skip)
    Caches x_hat and inv_std for backward.
    """
    var c = Int(block_idx.x)
    if c >= channels:
        return
    var tid = Int(thread_idx.x)
    var c_off = c * spatial
    var eps = Scalar[dtype](BN_EPSILON)
    var mom = Scalar[dtype](BN_MOMENTUM)
    var one_m = Scalar[dtype](1.0) - mom
    var n_f = Scalar[dtype](BATCH * spatial)
    var gamma = rebind[Scalar[dtype]](params[GAMMA_OFF + c])
    var beta = rebind[Scalar[dtype]](params[BETA_OFF + c])

    var smem = LayoutTensor[
        dtype, Layout.row_major(TPB), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Pass 1: mean
    var local_sum = Scalar[dtype](0.0)
    var idx = tid
    while idx < BATCH * spatial:
        var b = idx // spatial
        var s = idx % spatial
        local_sum += rebind[Scalar[dtype]](output[b, c_off + s])
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

    # Pass 2: variance
    var local_var = Scalar[dtype](0.0)
    idx = tid
    while idx < BATCH * spatial:
        var b = idx // spatial
        var s = idx % spatial
        local_var += (rebind[Scalar[dtype]](output[b, c_off + s]) - mean) ** 2
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

    # Pass 3: normalize + skip + relu + cache
    idx = tid
    while idx < BATCH * spatial:
        var b = idx // spatial
        var s = idx % spatial
        var x = rebind[Scalar[dtype]](output[b, c_off + s])
        var x_hat = (x - mean) * inv_std
        cache[b, XHAT_OFF + c_off + s] = x_hat
        var bn_out = gamma * x_hat + beta
        var val = bn_out + rebind[Scalar[dtype]](skip[b, c_off + s])
        # Cache pre-relu for backward
        cache[b, INVSTD_OFF + channels + c_off + s] = val
        output[b, c_off + s] = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        idx += TPB

    # Running stats + inv_std cache
    if tid == 0:
        for b in range(BATCH):
            cache[b, INVSTD_OFF + c] = inv_std
        var rm = rebind[Scalar[dtype]](params[RMEAN_OFF + c])
        var rv = rebind[Scalar[dtype]](params[RVAR_OFF + c])
        params.ptr[RMEAN_OFF + c] = one_m * rm + mom * mean
        params.ptr[RVAR_OFF + c] = one_m * rv + mom * var_


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
    dtype: DType where dtype.is_floating_point(),
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
        # Skip gradient accumulation
        grad_skip[b, c_off + s] = rebind[Scalar[dtype]](grad_skip[b, c_off + s]) + dy
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
    dtype: DType where dtype.is_floating_point(),
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
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.MAX_CONV_WS + Self.DIM  # conv ws + inter buffer

    # ── Initialization ─────────────────────────────────────────────

    @staticmethod
    def initialize_params[
        INIT: Initializer,
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
    def forward[BATCH: Int](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
    ):
        pass  # CPU forward not needed — use GPU

    @staticmethod
    def forward[BATCH: Int](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        pass  # CPU inference not needed — use GPU

    @staticmethod
    def backward[BATCH: Int](
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        pass  # CPU backward not needed — use GPU

    # ── GPU Forward (with cache) ─────────────────────────────────

    @staticmethod
    def forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        var c1_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV1_CS), MutAnyOrigin](cache.ptr)
        var c2_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)

        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        # Conv1+BN1+ReLU → inter
        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var inter_out = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter_ptr)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward_gpu[BATCH](ctx, inter_out, in_rb, p1, c1_v, conv_ws)

        # Conv2 (no activation) → output (temporarily holds conv2 output pre-BN)
        var inter_in = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](inter_ptr)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward_gpu[BATCH](ctx, out_rb, inter_in, p2, c2_v, conv_ws)

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
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)

        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var inter_out = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter_ptr)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward_gpu_no_cache[BATCH](ctx, inter_out, in_rb, p1, conv_ws)

        var inter_in = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](inter_ptr)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward_gpu_no_cache[BATCH](ctx, out_rb, inter_in, p2, conv_ws)

        # BN2 + skip + ReLU (inference: use batch stats like training for MCTS consistency)
        var bn2_params = LayoutTensor[dtype, Layout.row_major(Self.BN2_PS), MutAnyOrigin](params.ptr + Self.BN2_OFF)
        # Allocate temp BN cache in workspace inter region (we're done with it)
        var bn2_cache = LayoutTensor[dtype, Layout.row_major(BATCH, Self.BN2_CS), MutAnyOrigin](inter_ptr)
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

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH](ctx, output, input, params, workspace)

    # ── GPU Backward ─────────────────────────────────────────────

    @staticmethod
    def backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)

        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        comptime TOTAL = BATCH * Self.DIM
        comptime BLOCKS = ceildiv(TOTAL, TPB)

        # 1. Fused BN2+skip+ReLU backward: grad_output → grad_conv2 + grad_skip
        #    grad_conv2 goes to grad_output buffer (reused), grad_skip accumulates into grad_input
        var bn2_params = LayoutTensor[dtype, Layout.row_major(Self.BN2_PS), MutAnyOrigin](params.ptr + Self.BN2_OFF)
        var bn2_cache = LayoutTensor[dtype, Layout.row_major(BATCH, Self.BN2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.BN2_CACHE_OFF)
        var bn2_grads = LayoutTensor[dtype, Layout.row_major(Self.BN2_PS), MutAnyOrigin](grads.ptr + Self.BN2_OFF)
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](grad_output.ptr)
        var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](grad_input.ptr)

        # grad_conv2 output goes into workspace inter region
        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var grad_conv2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin](inter_ptr)

        comptime bwd_k = _bn_skip_relu_bwd_kernel[
            BATCH, Self.channels, Self.spatial, Self.BN2_PS, Self.BN2_CS,
            Self.BN2_GAMMA_OFF, Self.BN2_BETA_OFF, Self.BN2_XHAT_OFF, Self.BN2_INVSTD_OFF,
            dtype,
        ]
        ctx.enqueue_function[bwd_k, bwd_k](
            grad_conv2, go_t, gi_t, bn2_params, bn2_cache, bn2_grads,
            grid_dim=(Self.channels,), block_dim=(TPB,),
        )

        # 2. Conv2 backward: grad_conv2 → grad_inter (reuse grad_output buffer)
        var grad_inter = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](grad_output.ptr)
        var go_c2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin](inter_ptr)
        var c2_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)
        var g2_v = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](grads.ptr + Self.CONV1_PS)
        Self.Conv2.backward_gpu[BATCH](ctx, grad_inter, go_c2, p2, c2_v, g2_v, conv_ws)

        # 3. Conv1 backward: grad_inter → temp_gi (reuse inter region in workspace)
        var go_c1 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](grad_output.ptr)
        var temp_gi = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin](inter_ptr)
        var c1_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV1_CS), MutAnyOrigin](cache.ptr)
        var g1_v = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](grads.ptr)
        Self.Conv1.backward_gpu[BATCH](ctx, temp_gi, go_c1, p1, c1_v, g1_v, conv_ws)

        # 4. Add conv1's grad_input to skip grad
        comptime add_k = _add_kernel_flat[TOTAL, dtype]
        ctx.enqueue_function[add_k, add_k](
            gi_t,
            LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](inter_ptr),
            grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
