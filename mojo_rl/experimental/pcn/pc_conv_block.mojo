"""ConvPCBlock — one convolutional PCN level (Bogacz canonical, bottom-up).

SPIKE (P0/P1): CPU naive implementation. See docs/PCN_CONV_DESIGN.md.
Conforms to `PCBlockTrait` so it composes into `PCSequential` / `PCTrainer`,
but the five `*_gpu` methods are **P2 stubs that raise** — the CPU
`train_one_batch` path never calls them. Real GPU dispatchers (reusing the
Conv2D GPU kernels) land in P2. The CPU surface mirrors `PCBlock` exactly.

A ConvPCBlock predicts the *above* feature map from the *below* feature map via
a convolution (cross-correlation), in the same flat layout the `Conv2D`
autodiff primitive uses:

    a_below = ACT(x_below)                                 # [B, C_in·H·W]
    μ       = Conv(a_below; W, b)                          # [B, C_out·H'·W']
    ε       = x_above − μ
    z_below = Convᵀ(ε; W)            (col2im, w.r.t a_below)# pull_back
    z_gated = z_below ⊙ ACT'(x_below)                      # act_derivative_mul
    dW      = −corr(im2col(a_below), ε) ; db = −Σ ε        # weight_grad (−sign)

Layout (identical to nn Conv2D so a params buffer is interchangeable):
  - x_below idx:  c·in_h·in_w + ih·in_w + iw           (C_in·in_h·in_w = IN_DIM)
  - μ idx:        oc·spatial_out + s,  s = oh·out_w+ow (C_out·out_h·out_w = OUT_DIM)
  - W:            params[oc·col_size + c_k],  c_k = c·k² + kh·k + kw
  - bias:         params[C_out·col_size + oc]
  - PARAM_SIZE =  C_out·col_size + C_out,  col_size = C_in·k²

The −sign on weight_grad is baked in (as in PCBlock) so callers can do
`params -= lr·grads` directly. pull_back writes (does not accumulate) z_below.
"""

from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.memory import alloc
from std.sys import CompilationTarget

from mojo_rl.nn.constants import TPB
from mojo_rl.nn.initializer import Initializer
from mojo_rl.nn.autodiff.apple_cblas import apple_sgemm_accum

from .predictive_model import PCActivation, PCReLU, PCBlockTrait


struct ConvPCBlock[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
    ACT: PCActivation = PCReLU,
](PCBlockTrait):
    comptime out_h: Int = (
        Self.in_h + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime out_w: Int = (
        Self.in_w + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime col_size: Int = (
        Self.in_channels * Self.kernel_size * Self.kernel_size
    )
    comptime spatial_out: Int = Self.out_h * Self.out_w
    comptime CACHE: Int = Self.spatial_out * Self.col_size  # im2col per sample

    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.out_h * Self.out_w
    comptime PARAM_SIZE: Int = (
        Self.out_channels * Self.col_size + Self.out_channels
    )

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    # =========================================================================
    # Initialization — conv fans (fan_in = C_in·k², fan_out = C_out·k²)
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        comptime W_SIZE = Self.out_channels * Self.col_size
        var W_view = LayoutTensor[
            dtype, Layout.row_major(W_SIZE), MutAnyOrigin
        ](params.ptr)
        INIT.init[
            W_SIZE,
            Self.in_channels * Self.kernel_size * Self.kernel_size,
            Self.out_channels * Self.kernel_size * Self.kernel_size,
            dtype,
        ](W_view)
        for j in range(Self.out_channels):
            params.ptr[W_SIZE + j] = Scalar[dtype](0)

    # =========================================================================
    # im2col helper (CPU, Accelerate fast path): a_below[B, IN] → cache row per
    # sample [spatial_out, col_size] at cache[b*CACHE + s*col_size + c_k].
    # =========================================================================

    @staticmethod
    def _im2col_cpu[
        BATCH: Int, dtype: DType
    ](
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        cache: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        var ap = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](a_below.ptr)
        for b in range(BATCH):
            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for c in range(Self.in_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var ih = oh * Self.stride - Self.padding + kh
                                var iw = ow * Self.stride - Self.padding + kw
                                var c_k = (
                                    c * Self.kernel_size * Self.kernel_size
                                    + kh * Self.kernel_size
                                    + kw
                                )
                                var col_idx = (
                                    b * Self.CACHE + s * Self.col_size + c_k
                                )
                                if (
                                    ih >= 0
                                    and ih < Self.in_h
                                    and iw >= 0
                                    and iw < Self.in_w
                                ):
                                    cache[col_idx] = ap[
                                        b * Self.IN_DIM
                                        + c * Self.in_h * Self.in_w
                                        + ih * Self.in_w
                                        + iw
                                    ]
                                else:
                                    cache[col_idx] = Scalar[dtype](0)

    # =========================================================================
    # predict:  a_below = ACT(x_below);  μ = Conv(a_below; W) + b
    #   Apple/fp32: im2col + per-batch Accelerate sgemm (W @ colᵀ) + bias.
    #   else: naive scalar loops (oracle/fallback).
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
        # a_below = ACT(x_below) (cached for weight_grad)
        Self.ACT.apply[BATCH, Self.IN_DIM, dtype](x_below, a_below)
        Self._conv_forward[BATCH, dtype](a_below, params, mu)

    @staticmethod
    def _conv_forward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        """Conv-only forward (no activation): μ = Conv(a_below) + bias. Caller
        supplies the already-activated/normalized conv input. Reused by
        NormConvPCBlock to convolve a pre-normalized input."""
        comptime W_SIZE = Self.out_channels * Self.col_size
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )

        comptime if use_apple:
            # im2col → per-batch  out_b[oc,s] = Σ_k W[oc,k]·col_b[s,k]  (W @ colᵀ)
            var cache = alloc[Scalar[dtype]](BATCH * Self.CACHE)
            Self._im2col_cpu[BATCH, dtype](a_below, cache)
            var Wp = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](params.ptr)
            for b in range(BATCH):
                var col_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    cache + b * Self.CACHE
                )
                var out_b = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                    mu.ptr + b * Self.OUT_DIM
                )
                try:
                    apple_sgemm_accum[transpose_a=False, transpose_b=True](
                        Self.out_channels,
                        Self.spatial_out,
                        Self.col_size,
                        Float32(1.0),
                        Wp,
                        Self.col_size,
                        col_b,
                        Self.col_size,
                        Float32(0.0),
                        out_b,
                        Self.spatial_out,
                    )
                except:
                    pass
            # + bias
            var mp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](mu.ptr)
            var wp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                params.ptr
            )
            for b in range(BATCH):
                for oc in range(Self.out_channels):
                    var bias_val = wp[W_SIZE + oc]
                    var off = b * Self.OUT_DIM + oc * Self.spatial_out
                    for s in range(Self.spatial_out):
                        mp[off + s] = mp[off + s] + bias_val
            cache.free()
        else:
            var ap = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                a_below.ptr
            )
            var wp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                params.ptr
            )
            var mp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](mu.ptr)
            for b in range(BATCH):
                for oc in range(Self.out_channels):
                    var bias_val = wp[W_SIZE + oc]
                    for oh in range(Self.out_h):
                        for ow in range(Self.out_w):
                            var s = oh * Self.out_w + ow
                            var acc: Scalar[dtype] = bias_val
                            for c in range(Self.in_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var ih = (
                                            oh * Self.stride - Self.padding + kh
                                        )
                                        var iw = (
                                            ow * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            ih >= 0
                                            and ih < Self.in_h
                                            and iw >= 0
                                            and iw < Self.in_w
                                        ):
                                            var c_k = (
                                                c
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            var in_idx = (
                                                c * Self.in_h * Self.in_w
                                                + ih * Self.in_w
                                                + iw
                                            )
                                            acc += (
                                                wp[oc * Self.col_size + c_k]
                                                * ap[b * Self.IN_DIM + in_idx]
                                            )
                            mp[
                                b * Self.OUT_DIM + oc * Self.spatial_out + s
                            ] = acc

    # =========================================================================
    # eps_compute:  ε = x_above − μ   (elementwise, identical to PCBlock)
    # =========================================================================

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
        var xp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](x_above.ptr)
        var mp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](mu.ptr)
        var ep = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](eps.ptr)
        for i in range(BATCH * Self.OUT_DIM):
            ep[i] = xp[i] - mp[i]

    # =========================================================================
    # pull_back:  z_below = Convᵀ(ε; W)  (col2im scatter; w.r.t a_below, no act)
    # =========================================================================

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
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )
        var zp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](z_below.ptr)
        for i in range(BATCH * Self.IN_DIM):
            zp[i] = Scalar[dtype](0)

        comptime if use_apple:
            # per-batch  dcol_b[s,k] = Σ_oc ε_b[oc,s]·W[oc,k]  (εᵀ @ W), then
            # col2im scatter dcol → z_below (accumulate over overlaps).
            var dcol = alloc[Scalar[dtype]](BATCH * Self.CACHE)
            var Wp = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](params.ptr)
            for b in range(BATCH):
                var eps_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    eps_above.ptr + b * Self.OUT_DIM
                )
                var dcol_b = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                    dcol + b * Self.CACHE
                )
                try:
                    apple_sgemm_accum[transpose_a=True, transpose_b=False](
                        Self.spatial_out,
                        Self.col_size,
                        Self.out_channels,
                        Float32(1.0),
                        eps_b,
                        Self.spatial_out,
                        Wp,
                        Self.col_size,
                        Float32(0.0),
                        dcol_b,
                        Self.col_size,
                    )
                except:
                    pass
            var dcp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](dcol)
            for b in range(BATCH):
                for oh in range(Self.out_h):
                    for ow in range(Self.out_w):
                        var s = oh * Self.out_w + ow
                        for c in range(Self.in_channels):
                            for kh in range(Self.kernel_size):
                                for kw in range(Self.kernel_size):
                                    var ih = oh * Self.stride - Self.padding + kh
                                    var iw = ow * Self.stride - Self.padding + kw
                                    if (
                                        ih >= 0
                                        and ih < Self.in_h
                                        and iw >= 0
                                        and iw < Self.in_w
                                    ):
                                        var c_k = (
                                            c
                                            * Self.kernel_size
                                            * Self.kernel_size
                                            + kh * Self.kernel_size
                                            + kw
                                        )
                                        var in_idx = (
                                            c * Self.in_h * Self.in_w
                                            + ih * Self.in_w
                                            + iw
                                        )
                                        zp[b * Self.IN_DIM + in_idx] += dcp[
                                            b * Self.CACHE + s * Self.col_size
                                            + c_k
                                        ]
            dcol.free()
        else:
            var wp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                params.ptr
            )
            var ep = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                eps_above.ptr
            )
            for b in range(BATCH):
                for oc in range(Self.out_channels):
                    for oh in range(Self.out_h):
                        for ow in range(Self.out_w):
                            var s = oh * Self.out_w + ow
                            var g = ep[
                                b * Self.OUT_DIM + oc * Self.spatial_out + s
                            ]
                            for c in range(Self.in_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var ih = (
                                            oh * Self.stride - Self.padding + kh
                                        )
                                        var iw = (
                                            ow * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            ih >= 0
                                            and ih < Self.in_h
                                            and iw >= 0
                                            and iw < Self.in_w
                                        ):
                                            var c_k = (
                                                c
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            var in_idx = (
                                                c * Self.in_h * Self.in_w
                                                + ih * Self.in_w
                                                + iw
                                            )
                                            zp[b * Self.IN_DIM + in_idx] += (
                                                wp[oc * Self.col_size + c_k] * g
                                            )

    # =========================================================================
    # act_derivative_mul:  z_out = z_in ⊙ ACT'(x_below)  (reused verbatim)
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
        Self.ACT.apply_derivative_mul[BATCH, Self.IN_DIM, dtype](
            x_below, z_in, z_out
        )

    # =========================================================================
    # weight_grad:  dW = −corr(im2col(a_below), ε) ;  db = −Σ ε   (−sign baked)
    # =========================================================================

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
        comptime W_SIZE = Self.out_channels * Self.col_size
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )
        var gp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](grads.ptr)
        var ep = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            eps_above.ptr
        )
        for i in range(Self.PARAM_SIZE):
            gp[i] = Scalar[dtype](0)

        comptime if use_apple:
            # im2col, then accumulate  dW += (−1)·ε_b @ col_b  over the batch
            # (alpha=−1, beta=1) → dW = −Σ_b ε_b @ col_b. db = −Σ ε (scalar).
            var cache = alloc[Scalar[dtype]](BATCH * Self.CACHE)
            Self._im2col_cpu[BATCH, dtype](a_below, cache)
            var Cw = rebind[UnsafePointer[Float32, MutAnyOrigin]](grads.ptr)
            for b in range(BATCH):
                var eps_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    eps_above.ptr + b * Self.OUT_DIM
                )
                var col_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    cache + b * Self.CACHE
                )
                try:
                    apple_sgemm_accum[transpose_a=False, transpose_b=False](
                        Self.out_channels,
                        Self.col_size,
                        Self.spatial_out,
                        Float32(-1.0),
                        eps_b,
                        Self.spatial_out,
                        col_b,
                        Self.col_size,
                        Float32(1.0),
                        Cw,
                        Self.col_size,
                    )
                except:
                    pass
            for oc in range(Self.out_channels):
                var db_acc: Scalar[dtype] = 0
                for b in range(BATCH):
                    var off = b * Self.OUT_DIM + oc * Self.spatial_out
                    for s in range(Self.spatial_out):
                        db_acc += ep[off + s]
                gp[W_SIZE + oc] = -db_acc
            cache.free()
        else:
            var ap = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                a_below.ptr
            )
            # dW[oc, c_k] = −Σ_{b,s} ε[b, oc·so+s] · (im2col(a_below) patch)
            # db[oc]      = −Σ_{b,s} ε[b, oc·so+s]
            for b in range(BATCH):
                for oc in range(Self.out_channels):
                    var db_acc: Scalar[dtype] = 0
                    for oh in range(Self.out_h):
                        for ow in range(Self.out_w):
                            var s = oh * Self.out_w + ow
                            var g = ep[
                                b * Self.OUT_DIM + oc * Self.spatial_out + s
                            ]
                            db_acc += g
                            for c in range(Self.in_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var ih = (
                                            oh * Self.stride - Self.padding + kh
                                        )
                                        var iw = (
                                            ow * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            ih >= 0
                                            and ih < Self.in_h
                                            and iw >= 0
                                            and iw < Self.in_w
                                        ):
                                            var c_k = (
                                                c
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            var in_idx = (
                                                c * Self.in_h * Self.in_w
                                                + ih * Self.in_w
                                                + iw
                                            )
                                            gp[oc * Self.col_size + c_k] += (
                                                -g * ap[b * Self.IN_DIM + in_idx]
                                            )
                    gp[W_SIZE + oc] += -db_acc

    # =========================================================================
    # GPU kernels (P2) — naive, atomic-free GATHER forms so each output thread
    # accumulates locally. The gather indexing makes GPU bit-comparable to the
    # CPU loops (no scatter/atomics, deterministic). See PCN_CONV_DESIGN.md.
    # =========================================================================

    @staticmethod
    def _predict_conv_kernel[
        BATCH: Int, dtype: DType
    ](
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.OUT_DIM:
            return
        comptime W_SIZE = Self.out_channels * Self.col_size
        var b = idx // Self.OUT_DIM
        var rem = idx % Self.OUT_DIM
        var oc = rem // Self.spatial_out
        var s = rem % Self.spatial_out
        var oh = s // Self.out_w
        var ow = s % Self.out_w
        var acc: Scalar[dtype] = params.ptr[W_SIZE + oc]  # bias
        for c in range(Self.in_channels):
            for kh in range(Self.kernel_size):
                for kw in range(Self.kernel_size):
                    var ih = oh * Self.stride - Self.padding + kh
                    var iw = ow * Self.stride - Self.padding + kw
                    if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                        var c_k = (
                            c * Self.kernel_size * Self.kernel_size
                            + kh * Self.kernel_size
                            + kw
                        )
                        var in_idx = (
                            c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                        )
                        acc += (
                            params.ptr[oc * Self.col_size + c_k]
                            * a_below.ptr[b * Self.IN_DIM + in_idx]
                        )
        mu.ptr[idx] = acc

    @staticmethod
    def _eps_kernel[
        BATCH: Int, dtype: DType
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.OUT_DIM:
            return
        eps.ptr[idx] = x_above.ptr[idx] - mu.ptr[idx]

    @staticmethod
    def _pull_back_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        z_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        # One thread per input element (gather over all outputs whose
        # receptive field covers it): z[b,in_idx] = Σ_{oc,kh,kw} W·ε.
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.IN_DIM:
            return
        var b = idx // Self.IN_DIM
        var in_idx = idx % Self.IN_DIM
        var c = in_idx // (Self.in_h * Self.in_w)
        var rem = in_idx % (Self.in_h * Self.in_w)
        var ih = rem // Self.in_w
        var iw = rem % Self.in_w
        var acc: Scalar[dtype] = 0
        for oc in range(Self.out_channels):
            for kh in range(Self.kernel_size):
                for kw in range(Self.kernel_size):
                    var oh_num = ih + Self.padding - kh
                    var ow_num = iw + Self.padding - kw
                    if (
                        oh_num >= 0
                        and ow_num >= 0
                        and oh_num % Self.stride == 0
                        and ow_num % Self.stride == 0
                    ):
                        var oh = oh_num // Self.stride
                        var ow = ow_num // Self.stride
                        if oh < Self.out_h and ow < Self.out_w:
                            var c_k = (
                                c * Self.kernel_size * Self.kernel_size
                                + kh * Self.kernel_size
                                + kw
                            )
                            var s = oh * Self.out_w + ow
                            acc += (
                                params.ptr[oc * Self.col_size + c_k]
                                * eps_above.ptr[
                                    b * Self.OUT_DIM + oc * Self.spatial_out + s
                                ]
                            )
        z_below.ptr[idx] = acc

    @staticmethod
    def _weight_grad_W_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        # One thread per weight (oc, c_k): dW = −Σ_{b,oh,ow} ε·a_below_patch.
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= Self.out_channels * Self.col_size:
            return
        var oc = idx // Self.col_size
        var c_k = idx % Self.col_size
        var c = c_k // (Self.kernel_size * Self.kernel_size)
        var kk = c_k % (Self.kernel_size * Self.kernel_size)
        var kh = kk // Self.kernel_size
        var kw = kk % Self.kernel_size
        var acc: Scalar[dtype] = 0
        for b in range(BATCH):
            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var ih = oh * Self.stride - Self.padding + kh
                    var iw = ow * Self.stride - Self.padding + kw
                    if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                        var s = oh * Self.out_w + ow
                        var in_idx = (
                            c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                        )
                        acc += (
                            eps_above.ptr[
                                b * Self.OUT_DIM + oc * Self.spatial_out + s
                            ]
                            * a_below.ptr[b * Self.IN_DIM + in_idx]
                        )
        grads.ptr[idx] = -acc

    @staticmethod
    def _weight_grad_b_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        # One thread per output channel: db = −Σ_{b,oh,ow} ε.
        var oc = Int(block_dim.x * block_idx.x + thread_idx.x)
        if oc >= Self.out_channels:
            return
        comptime W_SIZE = Self.out_channels * Self.col_size
        var acc: Scalar[dtype] = 0
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                acc += eps_above.ptr[
                    b * Self.OUT_DIM + oc * Self.spatial_out + s
                ]
        grads.ptr[W_SIZE + oc] = -acc

    # ── GPU dispatchers ──────────────────────────────────────────────────────

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
        Self.ACT.apply_gpu[BATCH, Self.IN_DIM, dtype](ctx, x_below, a_below)
        comptime k = Self._predict_conv_kernel[BATCH, dtype]
        var threads = BATCH * Self.OUT_DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            a_below, params, mu, grid_dim=(blocks,), block_dim=(TPB,)
        )

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
        comptime k = Self._eps_kernel[BATCH, dtype]
        var threads = BATCH * Self.OUT_DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x_above, mu, eps, grid_dim=(blocks,), block_dim=(TPB,)
        )

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
        comptime k = Self._pull_back_kernel[BATCH, dtype]
        var threads = BATCH * Self.IN_DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            eps_above, params, z_below, grid_dim=(blocks,), block_dim=(TPB,)
        )

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
        Self.ACT.apply_derivative_mul_gpu[BATCH, Self.IN_DIM, dtype](
            ctx, x_below, z_in, z_out
        )

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
        comptime kw = Self._weight_grad_W_kernel[BATCH, dtype]
        var w_threads = Self.out_channels * Self.col_size
        var w_blocks = (w_threads + TPB - 1) // TPB
        ctx.enqueue_function[kw](
            eps_above, a_below, grads, grid_dim=(w_blocks,), block_dim=(TPB,)
        )

        comptime kb = Self._weight_grad_b_kernel[BATCH, dtype]
        var b_blocks = (Self.out_channels + TPB - 1) // TPB
        ctx.enqueue_function[kb](
            eps_above, grads, grid_dim=(b_blocks,), block_dim=(TPB,)
        )
