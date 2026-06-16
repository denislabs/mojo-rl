"""ConvTransposePCBlock — one transposed-convolution PCN level (Bogacz, bottom-up).

The generative dual of `ConvPCBlock`. Where ConvPCBlock.predict is a forward
(downsampling) conv, this block's predict is a TRANSPOSED (upsampling) conv:
a small below latent predicts a larger above latent. A transposed conv is the
adjoint of a forward conv, so the three ops mirror ConvPCBlock:

    a_below = ACT(x_below)                                  # [B, C_in·H·W]   (small)
    μ       = ConvT(a_below; W, b)                          # [B, C_out·H'·W'](big)   ← predict (scatter)
    ε       = x_above − μ
    z_below = Conv(ε; W)                                    # [B, C_in·H·W]          ← pull_back (im2col+GEMM)
    z_gated = z_below ⊙ ACT'(x_below)
    dW      = −corr(a_below, im2col(ε)) ; db = −Σ ε         #                        ← weight_grad

Output spatial (PyTorch ConvTranspose2d convention):
    out_h = (in_h − 1)·stride − 2·padding + kernel_size

Layout:
  - x_below idx:  ic·in_h·in_w + ih·in_w + iw            (C_in·in_h·in_w = IN_DIM, small)
  - μ idx:        oc·big_spatial + oh·out_w + ow         (C_out·out_h·out_w = OUT_DIM, big)
  - small↔big:    oh = ih·stride − padding + kh,  ow = iw·stride − padding + kw
  - W:            params[ic·col_size + (oc·k² + kh·k + kw)],  col_size = C_out·k²
  - bias:         params[W_SIZE + oc]   (per BIG output channel)
  - PARAM_SIZE =  C_in·col_size + C_out

CPU-only (naive scalar fallback + Apple Accelerate fast path). No GPU / no
PCBlockTrait conformance — callers invoke the static methods directly (the
bidirectional generative path does not go through PCSequential).
"""

from layout import Layout, LayoutTensor
from std.memory import alloc
from std.sys import CompilationTarget

from .pc_initializer import PCInitializer
from .pc_apple_cblas import apple_sgemm_accum

from .predictive_model import PCActivation, PCReLU


struct ConvTransposePCBlock[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
    ACT: PCActivation = PCReLU,
](Movable & ImplicitlyCopyable):
    comptime out_h: Int = (
        (Self.in_h - 1) * Self.stride - 2 * Self.padding + Self.kernel_size
    )
    comptime out_w: Int = (
        (Self.in_w - 1) * Self.stride - 2 * Self.padding + Self.kernel_size
    )
    comptime small_spatial: Int = Self.in_h * Self.in_w
    comptime big_spatial: Int = Self.out_h * Self.out_w
    comptime col_size: Int = (
        Self.out_channels * Self.kernel_size * Self.kernel_size
    )
    comptime CACHE: Int = Self.small_spatial * Self.col_size

    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.out_h * Self.out_w
    comptime W_SIZE: Int = Self.in_channels * Self.col_size
    comptime PARAM_SIZE: Int = Self.W_SIZE + Self.out_channels

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    # =========================================================================
    # Init — transposed-conv fans (fan_in = C_in·k², fan_out = C_out·k²)
    # =========================================================================

    @staticmethod
    def pc_init_params[
        INIT: PCInitializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ) raises:
        """nn init: conv-transpose W via INIT.fill(fan_in/out); zero bias."""
        var W_view = LayoutTensor[
            dtype, Layout.row_major(Self.W_SIZE), MutAnyOrigin
        ](params.ptr)
        INIT.fill[
            Self.W_SIZE,
            Self.in_channels * Self.kernel_size * Self.kernel_size,
            Self.out_channels * Self.kernel_size * Self.kernel_size,
            dtype,
        ](W_view)
        for j in range(Self.out_channels):
            params.ptr[Self.W_SIZE + j] = Scalar[dtype](0)

    # =========================================================================
    # im2col of the BIG tensor (ε or x_above) → ecol[b, p_small·col_size + ocol]
    #   ecol[p_small, oc·k²+kh·k+kw] = big[oc, oh, ow]  (oh = ih·s − p + kh) or 0
    # =========================================================================

    @staticmethod
    def _im2col_big[
        BATCH: Int, dtype: DType
    ](
        big: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        ecol: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        var bp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](big.ptr)
        for b in range(BATCH):
            for ih in range(Self.in_h):
                for iw in range(Self.in_w):
                    var p_small = ih * Self.in_w + iw
                    for oc in range(Self.out_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var oh = ih * Self.stride - Self.padding + kh
                                var ow = iw * Self.stride - Self.padding + kw
                                var ocol = (
                                    oc * Self.kernel_size * Self.kernel_size
                                    + kh * Self.kernel_size
                                    + kw
                                )
                                var ci = (
                                    b * Self.CACHE
                                    + p_small * Self.col_size
                                    + ocol
                                )
                                if (
                                    oh >= 0
                                    and oh < Self.out_h
                                    and ow >= 0
                                    and ow < Self.out_w
                                ):
                                    ecol[ci] = bp[
                                        b * Self.OUT_DIM
                                        + oc * Self.big_spatial
                                        + oh * Self.out_w
                                        + ow
                                    ]
                                else:
                                    ecol[ci] = Scalar[dtype](0)

    # =========================================================================
    # predict:  a_below = ACT(x_below);  μ = ConvT(a_below; W) + b   (upsample)
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
        Self.ACT.apply[BATCH, Self.IN_DIM, dtype](x_below, a_below)
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )
        var mp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](mu.ptr)
        var wp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](params.ptr)
        var ap = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](a_below.ptr)

        for i in range(BATCH * Self.OUT_DIM):
            mp[i] = Scalar[dtype](0)

        comptime if use_apple:
            # Step 1: dcol[b, p_small·col_size + ocol] = Σ_ic a_below[ic,p_small]·W[ic,ocol]
            #         (transpose_a GEMM: a_belowᵀ @ W per batch)
            var dcol = alloc[Scalar[dtype]](BATCH * Self.CACHE)
            var Wp = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](params.ptr)
            for b in range(BATCH):
                var a_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    a_below.ptr + b * Self.IN_DIM
                )
                var dcol_b = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                    dcol + b * Self.CACHE
                )
                try:
                    apple_sgemm_accum[transpose_a=True, transpose_b=False](
                        Self.small_spatial,
                        Self.col_size,
                        Self.in_channels,
                        Float32(1.0),
                        a_b,
                        Self.small_spatial,
                        Wp,
                        Self.col_size,
                        Float32(0.0),
                        dcol_b,
                        Self.col_size,
                    )
                except:
                    pass
            # Step 2: col2im scatter dcol → μ (accumulate over overlapping taps)
            var dcp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](dcol)
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        var p_small = ih * Self.in_w + iw
                        for oc in range(Self.out_channels):
                            for kh in range(Self.kernel_size):
                                for kw in range(Self.kernel_size):
                                    var oh = (
                                        ih * Self.stride - Self.padding + kh
                                    )
                                    var ow = (
                                        iw * Self.stride - Self.padding + kw
                                    )
                                    if (
                                        oh >= 0
                                        and oh < Self.out_h
                                        and ow >= 0
                                        and ow < Self.out_w
                                    ):
                                        var ocol = (
                                            oc
                                            * Self.kernel_size
                                            * Self.kernel_size
                                            + kh * Self.kernel_size
                                            + kw
                                        )
                                        mp[
                                            b * Self.OUT_DIM
                                            + oc * Self.big_spatial
                                            + oh * Self.out_w
                                            + ow
                                        ] += dcp[
                                            b * Self.CACHE
                                            + p_small * Self.col_size
                                            + ocol
                                        ]
            dcol.free()
        else:
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        for ic in range(Self.in_channels):
                            var av = ap[
                                b * Self.IN_DIM
                                + ic * Self.small_spatial
                                + ih * Self.in_w
                                + iw
                            ]
                            for oc in range(Self.out_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var oh = (
                                            ih * Self.stride - Self.padding + kh
                                        )
                                        var ow = (
                                            iw * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            oh >= 0
                                            and oh < Self.out_h
                                            and ow >= 0
                                            and ow < Self.out_w
                                        ):
                                            var widx = (
                                                ic * Self.col_size
                                                + oc
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            mp[
                                                b * Self.OUT_DIM
                                                + oc * Self.big_spatial
                                                + oh * Self.out_w
                                                + ow
                                            ] += (wp[widx] * av)

        # + bias per BIG output channel
        for b in range(BATCH):
            for oc in range(Self.out_channels):
                var bias_val = wp[Self.W_SIZE + oc]
                var off = b * Self.OUT_DIM + oc * Self.big_spatial
                for s in range(Self.big_spatial):
                    mp[off + s] = mp[off + s] + bias_val

    # =========================================================================
    # eps_compute:  ε = x_above − μ   (elementwise)
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
    # pull_back:  z_below = Conv(ε; W)   (forward conv = adjoint of predict)
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
        var wp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](params.ptr)
        var ep = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            eps_above.ptr
        )

        comptime if use_apple:
            # im2col(ε_big) → ecol, then z_small[ic,p] = Σ_ocol W[ic,ocol]·ecol[p,ocol]
            # (W @ ecolᵀ per batch).
            var ecol = alloc[Scalar[dtype]](BATCH * Self.CACHE)
            Self._im2col_big[BATCH, dtype](eps_above, ecol)
            var Wp = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](params.ptr)
            for b in range(BATCH):
                var ecol_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    ecol + b * Self.CACHE
                )
                var z_b = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                    z_below.ptr + b * Self.IN_DIM
                )
                try:
                    apple_sgemm_accum[transpose_a=False, transpose_b=True](
                        Self.in_channels,
                        Self.small_spatial,
                        Self.col_size,
                        Float32(1.0),
                        Wp,
                        Self.col_size,
                        ecol_b,
                        Self.col_size,
                        Float32(0.0),
                        z_b,
                        Self.small_spatial,
                    )
                except:
                    pass
            ecol.free()
        else:
            for i in range(BATCH * Self.IN_DIM):
                zp[i] = Scalar[dtype](0)
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        for ic in range(Self.in_channels):
                            var acc: Scalar[dtype] = 0
                            for oc in range(Self.out_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var oh = (
                                            ih * Self.stride - Self.padding + kh
                                        )
                                        var ow = (
                                            iw * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            oh >= 0
                                            and oh < Self.out_h
                                            and ow >= 0
                                            and ow < Self.out_w
                                        ):
                                            var widx = (
                                                ic * Self.col_size
                                                + oc
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            acc += (
                                                wp[widx]
                                                * ep[
                                                    b * Self.OUT_DIM
                                                    + oc * Self.big_spatial
                                                    + oh * Self.out_w
                                                    + ow
                                                ]
                                            )
                            zp[
                                b * Self.IN_DIM
                                + ic * Self.small_spatial
                                + ih * Self.in_w
                                + iw
                            ] = acc

    # =========================================================================
    # act_derivative_mul:  z_out = z_in ⊙ ACT'(x_below)
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
    # weight_grad:  dW = −Σ_b a_below @ im2col(ε) ;  db = −Σ ε   (−sign baked)
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
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )
        var gp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](grads.ptr)
        var ep = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            eps_above.ptr
        )
        var ap = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](a_below.ptr)
        for i in range(Self.PARAM_SIZE):
            gp[i] = Scalar[dtype](0)

        comptime if use_apple:
            # ecol = im2col(ε_big); dW[ic,ocol] += (−1)·Σ_p a_below[ic,p]·ecol[p,ocol]
            # (alpha=−1, beta=1 accumulate over batch).
            var ecol = alloc[Scalar[dtype]](BATCH * Self.CACHE)
            Self._im2col_big[BATCH, dtype](eps_above, ecol)
            var Cw = rebind[UnsafePointer[Float32, MutAnyOrigin]](grads.ptr)
            for b in range(BATCH):
                var a_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    a_below.ptr + b * Self.IN_DIM
                )
                var ecol_b = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                    ecol + b * Self.CACHE
                )
                try:
                    apple_sgemm_accum[transpose_a=False, transpose_b=False](
                        Self.in_channels,
                        Self.col_size,
                        Self.small_spatial,
                        Float32(-1.0),
                        a_b,
                        Self.small_spatial,
                        ecol_b,
                        Self.col_size,
                        Float32(1.0),
                        Cw,
                        Self.col_size,
                    )
                except:
                    pass
            ecol.free()
        else:
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        var p_small = ih * Self.in_w + iw
                        _ = p_small
                        for ic in range(Self.in_channels):
                            var av = ap[
                                b * Self.IN_DIM
                                + ic * Self.small_spatial
                                + ih * Self.in_w
                                + iw
                            ]
                            for oc in range(Self.out_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var oh = (
                                            ih * Self.stride - Self.padding + kh
                                        )
                                        var ow = (
                                            iw * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            oh >= 0
                                            and oh < Self.out_h
                                            and ow >= 0
                                            and ow < Self.out_w
                                        ):
                                            var widx = (
                                                ic * Self.col_size
                                                + oc
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            gp[widx] += (
                                                -av
                                                * ep[
                                                    b * Self.OUT_DIM
                                                    + oc * Self.big_spatial
                                                    + oh * Self.out_w
                                                    + ow
                                                ]
                                            )

        # db[oc] = −Σ_{b, big spatial} ε
        for oc in range(Self.out_channels):
            var acc: Scalar[dtype] = 0
            for b in range(BATCH):
                var off = b * Self.OUT_DIM + oc * Self.big_spatial
                for s in range(Self.big_spatial):
                    acc += ep[off + s]
            gp[Self.W_SIZE + oc] = -acc
