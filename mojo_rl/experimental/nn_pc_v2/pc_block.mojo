"""PCBlock — one PCN level (Bogacz canonical, bottom-up).

W stored as `[in_dim, out_dim]`, bias as `[out_dim]`. Bottom-up prediction:

    a_below = ACT(x_below)                                 # cached
    μ_above = a_below @ W + b                              # [B, OUT_DIM]
    ε_above = x_above − μ_above                            # local error
    z_below = ε_above @ W^T                                # pull-back to below
    f'-mod  = z_below ⊙ act'(x_below)                      # gain-modulated
    dE/dW   = −a_below^T @ ε_above                         # weight gradient
    dE/db   = −Σ_b ε_above[b, :]                           # bias gradient

Sign convention: ε = x − μ (latent above minus prediction), uniform across
interior and readout. The block does NOT add the "−" sign for ε itself —
that's handled by the caller. But weight_grad / bias_grad bake in the −sign
so that callers can pass the result directly to an Optimizer.step().

For the readout, ε_L = target − output (target plays the role of x_above).
"""

from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.runtime.asyncrt import DeviceContextPtr
from std.sys import has_nvidia_gpu_accelerator
from linalg.matmul import matmul as max_matmul
from layout.tile_tensor import lt_to_tt

from mojo_rl.nn.constants import TPB, MMA_BLOCK_M, MMA_BLOCK_N, MMA_BLOCK_THREADS
from mojo_rl.nn.initializer import Initializer

from .predictive_model import PCActivation, PCReLU, PCBlockTrait


struct PCBlock[
    in_dim: Int,
    out_dim: Int,
    ACT: PCActivation = PCReLU,
    USE_MAX_KERNELS: Bool = True,
](PCBlockTrait):
    """One PCN level: W [in_dim, out_dim] + b [out_dim] + bundled `ACT`.

    Naming matches `nn.Linear[in_dim, out_dim]`:
      - in_dim  is the below side (predicting block reads x_below of this dim)
      - out_dim is the above side (latent x_above of this dim is predicted)

    For the readout, pass `ACT=PCIdentity` and target plays role of x_above.

    USE_MAX_KERNELS (NVIDIA only): when True, route predict / pull_back /
    weight_grad GPU matmuls through `linalg.matmul` (the optimized max_matmul
    GEMM). When False, fall through to the 2×2 register-tiled fallback
    kernels. Apple always uses the tiled fallback regardless of this flag.
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim + Self.out_dim

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Init W via INIT(fan_in=in_dim, fan_out=out_dim); zero biases."""
        var W_view = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim * Self.out_dim),
            MutAnyOrigin,
        ](params.ptr)
        INIT.init[
            Self.in_dim * Self.out_dim, Self.in_dim, Self.out_dim, dtype
        ](W_view)
        # Zero biases
        for j in range(Self.out_dim):
            params.ptr[Self.in_dim * Self.out_dim + j] = Scalar[dtype](0)

    # =========================================================================
    # Bottom-up prediction:  a_below = ACT(x_below);  μ = a_below @ W + b
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
        """μ[b, j] = sum_i ACT(x_below[b, i]) * W[i, j] + b[j].

        a_below is cached for use in weight_grad later in the same training
        step; caller owns the buffer.
        """
        # 1. a_below = ACT(x_below)
        Self.ACT.apply[BATCH, Self.in_dim, dtype](x_below, a_below)

        # 2. μ = a_below @ W + b
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.out_dim),
            MutAnyOrigin,
        ](params.ptr)
        var b_view = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](params.ptr + Self.in_dim * Self.out_dim)

        for sb in range(BATCH):
            for j in range(Self.out_dim):
                var s: Scalar[dtype] = rebind[Scalar[dtype]](b_view[j])
                for i in range(Self.in_dim):
                    s += rebind[Scalar[dtype]](a_below[sb, i]) * rebind[
                        Scalar[dtype]
                    ](W[i, j])
                mu[sb, j] = s

    # =========================================================================
    # Local prediction error:  ε = x_above − μ
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
        for b in range(BATCH):
            for j in range(Self.out_dim):
                eps[b, j] = (
                    rebind[Scalar[dtype]](x_above[b, j])
                    - rebind[Scalar[dtype]](mu[b, j])
                )

    # =========================================================================
    # Pull-back to latent below:  z_below = ε_above @ W^T
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
        """z_below[b, i] = sum_j ε_above[b, j] * W[i, j]."""
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.out_dim),
            MutAnyOrigin,
        ](params.ptr)
        for b in range(BATCH):
            for i in range(Self.in_dim):
                var s: Scalar[dtype] = 0
                for j in range(Self.out_dim):
                    s += rebind[Scalar[dtype]](eps_above[b, j]) * rebind[
                        Scalar[dtype]
                    ](W[i, j])
                z_below[b, i] = s

    # =========================================================================
    # Activation derivative gating:  z_out = z_in ⊙ act'(x_below)
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
        Self.ACT.apply_derivative_mul[BATCH, Self.in_dim, dtype](
            x_below, z_in, z_out
        )

    # =========================================================================
    # Weight + bias gradients (signed for direct Optimizer.step consumption)
    #   dE/dW[i, j] = −Σ_b ε_above[b, j] * a_below[b, i]
    #   dE/db[j]    = −Σ_b ε_above[b, j]
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
        """Writes gradients (W, b) into `grads` in [W flat | b] layout.

        Sign baked in so that `params -= lr * grads` performs gradient descent
        on the local energy (or output loss for the readout block).
        """
        var W_grad = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.out_dim),
            MutAnyOrigin,
        ](grads.ptr)
        var b_grad = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grads.ptr + Self.in_dim * Self.out_dim)

        # dE/dW[i, j] = −Σ_b ε_above[b, j] * a_below[b, i]
        for i in range(Self.in_dim):
            for j in range(Self.out_dim):
                var s: Scalar[dtype] = 0
                for sb in range(BATCH):
                    s += rebind[Scalar[dtype]](eps_above[sb, j]) * rebind[
                        Scalar[dtype]
                    ](a_below[sb, i])
                W_grad[i, j] = -s

        # dE/db[j] = −Σ_b ε_above[b, j]
        for j in range(Self.out_dim):
            var s: Scalar[dtype] = 0
            for sb in range(BATCH):
                s += rebind[Scalar[dtype]](eps_above[sb, j])
            b_grad[j] = -s

    # =========================================================================
    # GPU kernels (elementwise: ε computation, bias gradient)
    # =========================================================================

    @staticmethod
    def _eps_kernel[
        BATCH: Int, OUT: Int, dtype: DType,
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * OUT:
            return
        var b = idx // OUT
        var j = idx % OUT
        eps[b, j] = (
            rebind[Scalar[dtype]](x_above[b, j])
            - rebind[Scalar[dtype]](mu[b, j])
        )

    @staticmethod
    def _bias_grad_kernel[
        BATCH: Int, OUT: Int, dtype: DType,
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
        b_grad: LayoutTensor[
            dtype, Layout.row_major(OUT), MutAnyOrigin
        ],
    ):
        var j = Int(block_dim.x * block_idx.x + thread_idx.x)
        if j >= OUT:
            return
        var s: Scalar[dtype] = 0
        for sb in range(BATCH):
            s += rebind[Scalar[dtype]](eps_above[sb, j])
        b_grad[j] = -s

    # ── Helpers used by the max_matmul fast path ─────────────────────────────

    @staticmethod
    def _bias_add_kernel[
        BATCH: Int, OUT: Int, dtype: DType,
    ](
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(OUT), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * OUT:
            return
        var col = idx % OUT
        mu.ptr[idx] = mu.ptr[idx] + rebind[Scalar[dtype]](b[col])

    @staticmethod
    def _negate_kernel[
        N: Int, dtype: DType,
    ](
        buf: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= N:
            return
        buf.ptr[idx] = -buf.ptr[idx]

    @staticmethod
    def _transpose_2d_kernel[
        ROWS: Int, COLS: Int, dtype: DType,
    ](
        dst: LayoutTensor[
            dtype, Layout.row_major(COLS, ROWS), MutAnyOrigin
        ],
        src: LayoutTensor[
            dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= ROWS * COLS:
            return
        var row = idx // COLS
        var col = idx % COLS
        dst[col, row] = src[row, col]

    # =========================================================================
    # Register-tiled 2×2 GPU kernels (Apple / non-USE_MAX_KERNELS path)
    # =========================================================================
    # Each thread computes a 2×2 sub-block of the output. Block tile is BT×BT
    # (32×32), shared-memory reduction tile SK = 16. 256 threads per block
    # cover the 32×32 output tile (16 sub-rows × 16 sub-cols × 2×2 = 1024 elems).
    # Mirrors `eval_kernel_2x2` / `backward_dx_kernel_2x2` /
    # `backward_dW_kernel_2x2` in `nn/autodiff/primitives/matmul.mojo`.

    @always_inline
    @staticmethod
    def _predict_kernel_2x2[
        BATCH: Int, dtype: DType,
    ](
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
    ):
        """Tiled mu = a_below @ W (no bias; bias added by post-kernel)."""
        comptime BT = 32
        comptime SK = 16

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_row = Int(block_idx.y) * BT
        var block_col = Int(block_idx.x) * BT

        var a_smem = LayoutTensor[
            dtype,
            Layout.row_major(BT, SK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var b_smem = LayoutTensor[
            dtype,
            Layout.row_major(SK, BT),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var acc00: Scalar[dtype] = 0
        var acc01: Scalar[dtype] = 0
        var acc10: Scalar[dtype] = 0
        var acc11: Scalar[dtype] = 0

        comptime num_k_tiles = (Self.in_dim + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK

            if block_row + a_r0 < BATCH and k_off + a_c0 < Self.in_dim:
                a_smem[a_r0, a_c0] = a_below[
                    block_row + a_r0, k_off + a_c0
                ]
            else:
                a_smem[a_r0, a_c0] = 0
            if block_row + a_r1 < BATCH and k_off + a_c1 < Self.in_dim:
                a_smem[a_r1, a_c1] = a_below[
                    block_row + a_r1, k_off + a_c1
                ]
            else:
                a_smem[a_r1, a_c1] = 0

            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT

            if k_off + b_r0 < Self.in_dim and block_col + b_c0 < Self.out_dim:
                b_smem[b_r0, b_c0] = W[k_off + b_r0, block_col + b_c0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < Self.in_dim and block_col + b_c1 < Self.out_dim:
                b_smem[b_r1, b_c1] = W[k_off + b_r1, block_col + b_c1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < Self.in_dim:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](
                        a_smem[sub_r * 2 + 1, k]
                    )
                    var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1 = rebind[Scalar[dtype]](
                        b_smem[k, sub_c * 2 + 1]
                    )
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    acc10 += a1 * b0
                    acc11 += a1 * b1

            barrier()

        var gr0 = block_row + sub_r * 2
        var gc0 = block_col + sub_c * 2
        if gr0 < BATCH and gc0 < Self.out_dim:
            mu[gr0, gc0] = acc00
        if gr0 < BATCH and gc0 + 1 < Self.out_dim:
            mu[gr0, gc0 + 1] = acc01
        if gr0 + 1 < BATCH and gc0 < Self.out_dim:
            mu[gr0 + 1, gc0] = acc10
        if gr0 + 1 < BATCH and gc0 + 1 < Self.out_dim:
            mu[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    def _pull_back_kernel_2x2[
        BATCH: Int, dtype: DType,
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        z_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """Tiled z_below = eps_above @ W^T (W^T loaded transposed from W)."""
        comptime BT = 32
        comptime SK = 16

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_row = Int(block_idx.y) * BT  # BATCH axis
        var block_col = Int(block_idx.x) * BT  # in_dim axis

        var a_smem = LayoutTensor[
            dtype,
            Layout.row_major(BT, SK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var b_smem = LayoutTensor[
            dtype,
            Layout.row_major(SK, BT),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var acc00: Scalar[dtype] = 0
        var acc01: Scalar[dtype] = 0
        var acc10: Scalar[dtype] = 0
        var acc11: Scalar[dtype] = 0

        comptime num_k_tiles = (Self.out_dim + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            # A = eps_above
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if block_row + a_r0 < BATCH and k_off + a_c0 < Self.out_dim:
                a_smem[a_r0, a_c0] = eps_above[
                    block_row + a_r0, k_off + a_c0
                ]
            else:
                a_smem[a_r0, a_c0] = 0
            if block_row + a_r1 < BATCH and k_off + a_c1 < Self.out_dim:
                a_smem[a_r1, a_c1] = eps_above[
                    block_row + a_r1, k_off + a_c1
                ]
            else:
                a_smem[a_r1, a_c1] = 0

            # B = W^T: b_smem[k, c] = W[block_col + c, k_off + k]
            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < Self.out_dim and block_col + b_c0 < Self.in_dim:
                b_smem[b_r0, b_c0] = W[block_col + b_c0, k_off + b_r0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < Self.out_dim and block_col + b_c1 < Self.in_dim:
                b_smem[b_r1, b_c1] = W[block_col + b_c1, k_off + b_r1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < Self.out_dim:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](
                        a_smem[sub_r * 2 + 1, k]
                    )
                    var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1 = rebind[Scalar[dtype]](
                        b_smem[k, sub_c * 2 + 1]
                    )
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    acc10 += a1 * b0
                    acc11 += a1 * b1

            barrier()

        var gr0 = block_row + sub_r * 2
        var gc0 = block_col + sub_c * 2
        if gr0 < BATCH and gc0 < Self.in_dim:
            z_below[gr0, gc0] = acc00
        if gr0 < BATCH and gc0 + 1 < Self.in_dim:
            z_below[gr0, gc0 + 1] = acc01
        if gr0 + 1 < BATCH and gc0 < Self.in_dim:
            z_below[gr0 + 1, gc0] = acc10
        if gr0 + 1 < BATCH and gc0 + 1 < Self.in_dim:
            z_below[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    def _weight_grad_kernel_2x2[
        BATCH: Int, dtype: DType,
    ](
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
        W_grad: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
    ):
        """Tiled W_grad = -a_below^T @ eps_above (sign baked in)."""
        comptime BT = 32
        comptime SK = 16

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_row = Int(block_idx.y) * BT  # in_dim axis
        var block_col = Int(block_idx.x) * BT  # out_dim axis

        var a_smem = LayoutTensor[
            dtype,
            Layout.row_major(BT, SK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var b_smem = LayoutTensor[
            dtype,
            Layout.row_major(SK, BT),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var acc00: Scalar[dtype] = 0
        var acc01: Scalar[dtype] = 0
        var acc10: Scalar[dtype] = 0
        var acc11: Scalar[dtype] = 0

        comptime num_k_tiles = (BATCH + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            # A = a_below^T: a_smem[r, k] = a_below[k_off + k, block_row + r]
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if k_off + a_c0 < BATCH and block_row + a_r0 < Self.in_dim:
                a_smem[a_r0, a_c0] = a_below[
                    k_off + a_c0, block_row + a_r0
                ]
            else:
                a_smem[a_r0, a_c0] = 0
            if k_off + a_c1 < BATCH and block_row + a_r1 < Self.in_dim:
                a_smem[a_r1, a_c1] = a_below[
                    k_off + a_c1, block_row + a_r1
                ]
            else:
                a_smem[a_r1, a_c1] = 0

            # B = eps_above
            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < BATCH and block_col + b_c0 < Self.out_dim:
                b_smem[b_r0, b_c0] = eps_above[
                    k_off + b_r0, block_col + b_c0
                ]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < BATCH and block_col + b_c1 < Self.out_dim:
                b_smem[b_r1, b_c1] = eps_above[
                    k_off + b_r1, block_col + b_c1
                ]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < BATCH:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](
                        a_smem[sub_r * 2 + 1, k]
                    )
                    var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1 = rebind[Scalar[dtype]](
                        b_smem[k, sub_c * 2 + 1]
                    )
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    acc10 += a1 * b0
                    acc11 += a1 * b1

            barrier()

        # Bake in the −sign expected by Optimizer.step (params -= lr·grads).
        var gr0 = block_row + sub_r * 2
        var gc0 = block_col + sub_c * 2
        if gr0 < Self.in_dim and gc0 < Self.out_dim:
            W_grad[gr0, gc0] = -acc00
        if gr0 < Self.in_dim and gc0 + 1 < Self.out_dim:
            W_grad[gr0, gc0 + 1] = -acc01
        if gr0 + 1 < Self.in_dim and gc0 < Self.out_dim:
            W_grad[gr0 + 1, gc0] = -acc10
        if gr0 + 1 < Self.in_dim and gc0 + 1 < Self.out_dim:
            W_grad[gr0 + 1, gc0 + 1] = -acc11

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
        # 1. a_below = ACT(x_below)
        Self.ACT.apply_gpu[BATCH, Self.in_dim, dtype](ctx, x_below, a_below)

        # 2. μ = a_below @ W + b
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.out_dim),
            MutAnyOrigin,
        ](params.ptr)
        var b_view = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](params.ptr + Self.in_dim * Self.out_dim)

        comptime if Self.USE_MAX_KERNELS and has_nvidia_gpu_accelerator():
            # Fast path: max_matmul writes mu = a_below @ W (no bias),
            # then a separate bias-add kernel folds in b.
            max_matmul[target="gpu"](
                lt_to_tt(mu),
                lt_to_tt(a_below),
                lt_to_tt(W),
                DeviceContextPtr(ctx),
            )

            comptime kb = Self._bias_add_kernel[
                BATCH, Self.out_dim, dtype
            ]
            var ba_threads = BATCH * Self.out_dim
            var ba_blocks = (ba_threads + TPB - 1) // TPB
            ctx.enqueue_function[kb, kb](
                mu, b_view,
                grid_dim=(ba_blocks,), block_dim=(TPB,),
            )
        else:
            # Fallback (Apple / non-NVIDIA): 2×2 register-tiled GEMM, then
            # post-add bias.
            comptime kt = Self._predict_kernel_2x2[BATCH, dtype]
            comptime grid_x = (
                Self.out_dim + MMA_BLOCK_N - 1
            ) // MMA_BLOCK_N
            comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M
            ctx.enqueue_function[kt, kt](
                a_below, W, mu,
                grid_dim=(grid_x, grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

            comptime kb = Self._bias_add_kernel[
                BATCH, Self.out_dim, dtype
            ]
            var ba_threads = BATCH * Self.out_dim
            var ba_blocks = (ba_threads + TPB - 1) // TPB
            ctx.enqueue_function[kb, kb](
                mu, b_view,
                grid_dim=(ba_blocks,), block_dim=(TPB,),
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
        comptime k = Self._eps_kernel[BATCH, Self.out_dim, dtype]
        var threads = BATCH * Self.out_dim
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k, k](
            x_above, mu, eps,
            grid_dim=(blocks,), block_dim=(TPB,),
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
        var W = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.out_dim),
            MutAnyOrigin,
        ](params.ptr)
        comptime if Self.USE_MAX_KERNELS and has_nvidia_gpu_accelerator():
            # z_below[B, IN] = eps_above[B, OUT] @ W^T[OUT, IN]
            # max_matmul with transpose_b=True treats W (stored [IN, OUT])
            # as if it were W^T, giving exactly this contraction.
            max_matmul[transpose_b=True, target="gpu"](
                lt_to_tt(z_below),
                lt_to_tt(eps_above),
                lt_to_tt(W),
                DeviceContextPtr(ctx),
            )
        else:
            # Fallback (Apple / non-NVIDIA): 2×2 register-tiled GEMM with
            # transposed W loads.
            comptime kt = Self._pull_back_kernel_2x2[BATCH, dtype]
            comptime grid_x = (
                Self.in_dim + MMA_BLOCK_N - 1
            ) // MMA_BLOCK_N
            comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M
            ctx.enqueue_function[kt, kt](
                eps_above, W, z_below,
                grid_dim=(grid_x, grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
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
        Self.ACT.apply_derivative_mul_gpu[BATCH, Self.in_dim, dtype](
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
        var W_grad = LayoutTensor[
            dtype,
            Layout.row_major(Self.in_dim, Self.out_dim),
            MutAnyOrigin,
        ](grads.ptr)
        var b_grad = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grads.ptr + Self.in_dim * Self.out_dim)

        comptime if Self.USE_MAX_KERNELS and has_nvidia_gpu_accelerator():
            # dW[IN, OUT] = -a_below^T[IN, BATCH] @ eps_above[BATCH, OUT]
            # max_matmul has no transpose_a, so materialize a_below^T into
            # a scratch buffer first, then compute the GEMM and negate dW.
            var a_T_buf = ctx.enqueue_create_buffer[dtype](
                BATCH * Self.in_dim
            )
            var a_T = LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, BATCH),
                MutAnyOrigin,
            ](a_T_buf.unsafe_ptr())

            comptime kt = Self._transpose_2d_kernel[
                BATCH, Self.in_dim, dtype
            ]
            comptime t_blocks = (
                BATCH * Self.in_dim + TPB - 1
            ) // TPB
            ctx.enqueue_function[kt, kt](
                a_T, a_below,
                grid_dim=(t_blocks,), block_dim=(TPB,),
            )

            max_matmul[target="gpu"](
                lt_to_tt(W_grad),
                lt_to_tt(a_T),
                lt_to_tt(eps_above),
                DeviceContextPtr(ctx),
            )

            # Bake in the −sign expected by Optimizer.step (params -= lr·grads).
            var W_grad_flat = LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim * Self.out_dim),
                MutAnyOrigin,
            ](grads.ptr)
            comptime kn = Self._negate_kernel[
                Self.in_dim * Self.out_dim, dtype
            ]
            comptime n_blocks = (
                Self.in_dim * Self.out_dim + TPB - 1
            ) // TPB
            ctx.enqueue_function[kn, kn](
                W_grad_flat,
                grid_dim=(n_blocks,), block_dim=(TPB,),
            )
        else:
            # Fallback (Apple / non-NVIDIA): 2×2 register-tiled GEMM with
            # transposed a_below loads; sign baked into the store.
            comptime kw = Self._weight_grad_kernel_2x2[BATCH, dtype]
            comptime w_grid_x = (
                Self.out_dim + MMA_BLOCK_N - 1
            ) // MMA_BLOCK_N
            comptime w_grid_y = (
                Self.in_dim + MMA_BLOCK_M - 1
            ) // MMA_BLOCK_M
            ctx.enqueue_function[kw, kw](
                a_below, eps_above, W_grad,
                grid_dim=(w_grid_x, w_grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

        comptime kb = Self._bias_grad_kernel[BATCH, Self.out_dim, dtype]
        var b_threads = Self.out_dim
        var b_blocks = (b_threads + TPB - 1) // TPB
        ctx.enqueue_function[kb, kb](
            eps_above, b_grad,
            grid_dim=(b_blocks,), block_dim=(TPB,),
        )
