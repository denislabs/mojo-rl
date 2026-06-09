"""NormConvPCBlock — conv PC level with FUSED input-side per-channel RMSNorm.

The PC-compatible way to normalize a conv stack WITHOUT paying the
inference-depth cost of a separate norm level (which adds a latent and hurt
accuracy — see ChannelNormPCBlock). Here normalization is folded into the conv
block's prediction:

    a       = RMSNorm_ch( ACT(x_below) )      # parameter-free, per INPUT channel
    μ       = Conv(a; W, b)                    # standard conv

`RMSNorm_ch` normalizes each input channel over its in_h·in_w positions, per
sample. Because it is parameter-free, the Jacobian split fits PCBlockTrait with
NO interface change:

    pull_back:           z = Convᵀ(ε; W)                     (unchanged — needs only W)
    act_derivative_mul:  z ← ACT'(x) ⊙ RMSNorm_ch_Jᵀ(z; ACT(x))   (has x_below; no params)

Normalizing the conv INPUT normalizes the previous layer's output (the same
inter-layer activations BatchNorm targets), but adds no latent — so it does not
slow PC inference like a separate norm level.

Implemented by delegating the conv ops (predict-conv / pull_back / weight_grad /
init) to an inner `ConvPCBlock[..., PCIdentity]` (reusing its Accelerate
kernels); only the input-norm (predict) and the norm-Jacobian (act_deriv) are
new. CPU; GPU deferred (raise-stubs) — the CIFAR experiment runs on CPU.
"""

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext
from std.memory import alloc
from std.math import sqrt

from mojo_rl.nn.initializer import Initializer

from .predictive_model import PCActivation, PCReLU, PCBlockTrait
from .pc_conv_block import ConvPCBlock

comptime _RMS_EPS: Float64 = 1e-6


struct NormConvPCBlock[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
    ACT: PCActivation = PCReLU,
](PCBlockTrait):
    # Inner conv (Identity act): does the conv of the already-normalized input.
    comptime INNER = ConvPCBlock[
        Self.in_channels,
        Self.out_channels,
        Self.kernel_size,
        Self.stride,
        Self.padding,
        Self.in_h,
        Self.in_w,
        PCReLU,  # unused activation path; we call its conv ops with pre-act input
    ]
    comptime spatial_in: Int = Self.in_h * Self.in_w

    comptime IN_DIM: Int = Self.INNER.IN_DIM
    comptime OUT_DIM: Int = Self.INNER.OUT_DIM
    comptime PARAM_SIZE: Int = Self.INNER.PARAM_SIZE

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        Self.INNER.initialize_params[INIT, dtype](params)

    # ── parameter-free per-input-channel RMSNorm:  out = u / rms_c  ───────────
    @staticmethod
    def _channel_rmsnorm[
        BATCH: Int, dtype: DType
    ](
        u: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        for b in range(BATCH):
            for c in range(Self.in_channels):
                var off = b * Self.IN_DIM + c * Self.spatial_in
                var ss: Float64 = 0.0
                for s in range(Self.spatial_in):
                    var v = Float64(u[off + s])
                    ss += v * v
                var inv_r = Scalar[dtype](
                    1.0 / sqrt(ss / Float64(Self.spatial_in) + _RMS_EPS)
                )
                for s in range(Self.spatial_in):
                    dst[off + s] = u[off + s] * inv_r

    # =========================================================================
    # predict:  a = RMSNorm_ch(ACT(x_below));  μ = Conv(a)   (a cached for wgrad)
    # =========================================================================

    @staticmethod
    def predict[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        # u = ACT(x_below)  → scratch;  a_below = RMSNorm_ch(u)
        var u_buf = alloc[Scalar[dtype]](BATCH * Self.IN_DIM)
        var u = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](u_buf)
        Self.ACT.apply[BATCH, Self.IN_DIM, dtype](x_below, u)
        Self._channel_rmsnorm[BATCH, dtype](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](u.ptr),
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](a_below.ptr),
        )
        # μ = Conv(a_below).  INNER (Identity-style use): pass a_below as the
        # conv input; its a_below_out scratch (=u) is discarded.
        Self.INNER._conv_forward[BATCH, dtype](a_below, params, mu)
        u_buf.free()

    @staticmethod
    def eps_compute[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        Self.INNER.eps_compute[BATCH, dtype](x_above, mu, eps)

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut z_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        # Convᵀ(ε) — independent of the input-side norm.
        Self.INNER.pull_back[BATCH, dtype](eps_above, params, z_below)

    # =========================================================================
    # act_derivative_mul:  z ← ACT'(x) ⊙ RMSNorm_ch_Jᵀ(z; u),  u = ACT(x_below)
    # =========================================================================

    @staticmethod
    def act_derivative_mul[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        z_in: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut z_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        var xp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](x_below.ptr)
        var zi = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](z_in.ptr)

        # u = ACT(x_below)
        var u_buf = alloc[Scalar[dtype]](BATCH * Self.IN_DIM)
        var u = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](u_buf)
        Self.ACT.apply[BATCH, Self.IN_DIM, dtype](x_below, u)
        var up = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](u.ptr)

        # g = RMSNorm_ch_Jᵀ(z_in; u)  (per input channel over spatial_in)
        var g_buf = alloc[Scalar[dtype]](BATCH * Self.IN_DIM)
        for b in range(BATCH):
            for c in range(Self.in_channels):
                var off = b * Self.IN_DIM + c * Self.spatial_in
                var ss: Float64 = 0.0
                for s in range(Self.spatial_in):
                    var v = Float64(up[off + s])
                    ss += v * v
                var inv_r = 1.0 / sqrt(ss / Float64(Self.spatial_in) + _RMS_EPS)
                var dot: Float64 = 0.0
                for s in range(Self.spatial_in):
                    dot += Float64(zi[off + s]) * Float64(up[off + s]) * inv_r
                var dot_over = dot / Float64(Self.spatial_in)
                for s in range(Self.spatial_in):
                    var n_s = Float64(up[off + s]) * inv_r
                    g_buf[off + s] = Scalar[dtype](
                        inv_r * (Float64(zi[off + s]) - n_s * dot_over)
                    )
        # z_out = ACT'(x_below) ⊙ g
        var g = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](g_buf)
        Self.ACT.apply_derivative_mul[BATCH, Self.IN_DIM, dtype](
            x_below, g, z_out
        )
        u_buf.free()
        g_buf.free()

    @staticmethod
    def weight_grad[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        # a_below is the cached NORMALIZED conv input → standard conv weight grad.
        Self.INNER.weight_grad[BATCH, dtype](eps_above, a_below, grads)

    # =========================================================================
    # GPU dispatchers — deferred (CPU experiment). Raise until ported.
    # =========================================================================

    @staticmethod
    def predict_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ) raises:
        raise Error("NormConvPCBlock.predict_gpu: GPU port deferred")

    @staticmethod
    def eps_compute_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        raise Error("NormConvPCBlock.eps_compute_gpu: GPU port deferred")

    @staticmethod
    def pull_back_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut z_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ) raises:
        raise Error("NormConvPCBlock.pull_back_gpu: GPU port deferred")

    @staticmethod
    def act_derivative_mul_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        z_in: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut z_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ) raises:
        raise Error("NormConvPCBlock.act_derivative_mul_gpu: GPU port deferred")

    @staticmethod
    def weight_grad_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ) raises:
        raise Error("NormConvPCBlock.weight_grad_gpu: GPU port deferred")
