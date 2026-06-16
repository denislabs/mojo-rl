"""NormPCBlock — an RMSNorm PCN level (Bogacz canonical, bottom-up).

A normalization level: predicts the normalized version of the below latent,

    μ = γ ⊙ RMSNorm(x_below),   RMSNorm(x)_i = x_i / r,  r = sqrt(mean_i(x_i²) + ε)

with IN_DIM == OUT_DIM == DIM and a learnable per-feature scale γ (DIM params).
Inserted between conv levels it yields the standard Conv → Norm → ReLU → Conv
ordering (the ReLU is the next ConvPCBlock's activation), bounding latent
magnitude so the deep conv stack trains stably at higher LR.

RMSNorm's Jacobian depends on the INPUT, but `pull_back` only sees `params` and
`act_derivative_mul` only sees the input. The split (the framework calls
`pull_back` then `act_derivative_mul(x_below, ·)`):

    pull_back:           z = ε ⊙ γ                            (has ε, γ)
    act_derivative_mul:  z ← (1/r)[z − n·(Σ_k z_k n_k)/D]     (has x_below → n, r)

This composes to exactly J^T ε for μ = γ⊙(x/r). Per-feature reduction is over
the FULL flattened feature vector (global RMSNorm), per sample.

CPU naive + GPU (per-row reduction kernels). Conforms to PCBlockTrait.
"""

from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.math import sqrt

from .pc_constants import TPB
from .pc_initializer import PCInitializer

from .predictive_model import PCBlockTrait

comptime _RMS_EPS: Float64 = 1e-6


struct NormPCBlock[dim: Int](PCBlockTrait):
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = Self.dim  # γ (per-feature scale)

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def pc_init_params[
        INIT: PCInitializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ) raises:
        """nn init: γ = 1 (INIT unused — RMSNorm scale, not a weight)."""
        for i in range(Self.dim):
            params.ptr[i] = Scalar[dtype](1)

    # =========================================================================
    # predict:  μ = γ ⊙ (x_below / r);  a_below cached = x_below (raw)
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
        var xp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](x_below.ptr)
        var gp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](params.ptr)
        var mp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](mu.ptr)
        var ap = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](a_below.ptr)
        for i in range(BATCH * Self.dim):
            ap[i] = xp[i]
        for b in range(BATCH):
            var off = b * Self.dim
            var ss: Float64 = 0.0
            for i in range(Self.dim):
                var v = Float64(xp[off + i])
                ss += v * v
            var r = sqrt(ss / Float64(Self.dim) + _RMS_EPS)
            var inv_r = Scalar[dtype](1.0 / r)
            for i in range(Self.dim):
                mp[off + i] = gp[i] * xp[off + i] * inv_r

    # =========================================================================
    # eps_compute:  ε = x_above − μ
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
        for i in range(BATCH * Self.dim):
            ep[i] = xp[i] - mp[i]

    # =========================================================================
    # pull_back:  z = ε ⊙ γ   (per-feature scale; RMSNorm Jacobian deferred to
    #             act_derivative_mul, which sees x_below)
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
        var ep = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            eps_above.ptr
        )
        var gp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](params.ptr)
        var zp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](z_below.ptr)
        for b in range(BATCH):
            var off = b * Self.dim
            for i in range(Self.dim):
                zp[off + i] = ep[off + i] * gp[i]

    # =========================================================================
    # act_derivative_mul:  apply the RMSNorm Jacobian to z (=ε⊙γ) using x_below:
    #   z_out_i = (1/r)[ z_in_i − n_i · (Σ_k z_in_k n_k) / D ],  n = x_below/r
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
        var zo = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](z_out.ptr)
        for b in range(BATCH):
            var off = b * Self.dim
            var ss: Float64 = 0.0
            for i in range(Self.dim):
                var v = Float64(xp[off + i])
                ss += v * v
            var r = sqrt(ss / Float64(Self.dim) + _RMS_EPS)
            var inv_r = 1.0 / r
            # s = Σ_k z_in_k · n_k  with n_k = x_k / r
            var s: Float64 = 0.0
            for i in range(Self.dim):
                s += Float64(zi[off + i]) * Float64(xp[off + i]) * inv_r
            var s_over_d = s / Float64(Self.dim)
            for i in range(Self.dim):
                var n_i = Float64(xp[off + i]) * inv_r
                zo[off + i] = Scalar[dtype](
                    inv_r * (Float64(zi[off + i]) - n_i * s_over_d)
                )

    # =========================================================================
    # weight_grad:  dγ_i = −Σ_b ε[b,i] · n[b,i]   (n = a_below/r, −sign baked)
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
        var ep = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            eps_above.ptr
        )
        var ap = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](a_below.ptr)
        var gp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](grads.ptr)
        for i in range(Self.dim):
            gp[i] = Scalar[dtype](0)
        for b in range(BATCH):
            var off = b * Self.dim
            var ss: Float64 = 0.0
            for i in range(Self.dim):
                var v = Float64(ap[off + i])
                ss += v * v
            var inv_r = 1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS)
            for i in range(Self.dim):
                var n_i = Float64(ap[off + i]) * inv_r
                gp[i] = gp[i] - Scalar[dtype](Float64(ep[off + i]) * n_i)

    # =========================================================================
    # GPU kernels (per-row reductions; one thread per sample row)
    # =========================================================================

    @staticmethod
    def _predict_kernel[
        BATCH: Int, dtype: DType
    ](
        x_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var off = b * Self.dim
        var ss: Float64 = 0.0
        for i in range(Self.dim):
            var v = Float64(x_below.ptr[off + i])
            ss += v * v
            a_below.ptr[off + i] = x_below.ptr[off + i]
        var inv_r = Scalar[dtype](1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS))
        for i in range(Self.dim):
            mu.ptr[off + i] = params.ptr[i] * x_below.ptr[off + i] * inv_r

    @staticmethod
    def _eps_kernel[
        BATCH: Int, dtype: DType
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        eps.ptr[idx] = x_above.ptr[idx] - mu.ptr[idx]

    @staticmethod
    def _pull_back_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
        z_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var col = idx % Self.dim
        z_below.ptr[idx] = eps_above.ptr[idx] * params.ptr[col]

    @staticmethod
    def _act_deriv_kernel[
        BATCH: Int, dtype: DType
    ](
        x_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        z_in: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        z_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var off = b * Self.dim
        var ss: Float64 = 0.0
        for i in range(Self.dim):
            var v = Float64(x_below.ptr[off + i])
            ss += v * v
        var inv_r = 1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS)
        var s: Float64 = 0.0
        for i in range(Self.dim):
            s += (
                Float64(z_in.ptr[off + i])
                * Float64(x_below.ptr[off + i])
                * inv_r
            )
        var s_over_d = s / Float64(Self.dim)
        for i in range(Self.dim):
            var n_i = Float64(x_below.ptr[off + i]) * inv_r
            z_out.ptr[off + i] = Scalar[dtype](
                inv_r * (Float64(z_in.ptr[off + i]) - n_i * s_over_d)
            )

    @staticmethod
    def _rms_per_row_kernel[
        BATCH: Int, dtype: DType
    ](
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        inv_r_buf: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var off = b * Self.dim
        var ss: Float64 = 0.0
        for i in range(Self.dim):
            var v = Float64(a_below.ptr[off + i])
            ss += v * v
        inv_r_buf.ptr[b] = Scalar[dtype](
            1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS)
        )

    @staticmethod
    def _weight_grad_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        inv_r_buf: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        grads: LayoutTensor[dtype, Layout.row_major(Self.dim), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= Self.dim:
            return
        var acc: Float64 = 0.0
        for b in range(BATCH):
            var off = b * Self.dim
            var n_i = Float64(a_below.ptr[off + i]) * Float64(inv_r_buf.ptr[b])
            acc += Float64(eps_above.ptr[off + i]) * n_i
        grads.ptr[i] = Scalar[dtype](-acc)

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
        comptime k = Self._predict_kernel[BATCH, dtype]
        var blocks = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x_below, params, mu, a_below, grid_dim=(blocks,), block_dim=(TPB,)
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
        var threads = BATCH * Self.dim
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
        var threads = BATCH * Self.dim
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
        comptime k = Self._act_deriv_kernel[BATCH, dtype]
        var blocks = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x_below, z_in, z_out, grid_dim=(blocks,), block_dim=(TPB,)
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
        var inv_r_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var inv_r = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            inv_r_buf.unsafe_ptr()
        )
        comptime kr = Self._rms_per_row_kernel[BATCH, dtype]
        var rblocks = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[kr](
            a_below, inv_r, grid_dim=(rblocks,), block_dim=(TPB,)
        )
        comptime kg = Self._weight_grad_kernel[BATCH, dtype]
        var gblocks = (Self.dim + TPB - 1) // TPB
        ctx.enqueue_function[kg](
            eps_above,
            a_below,
            inv_r,
            grads,
            grid_dim=(gblocks,),
            block_dim=(TPB,),
        )
        _ = inv_r_buf
