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
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from std.sys import has_nvidia_gpu_accelerator
from linalg.matmul import matmul as max_matmul
from layout.tile_tensor import lt_to_tt

from mojo_rl.nn.constants import TPB
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
    GEMM). When False, fall through to the naive elementwise kernels. Apple
    always uses the naive fallback regardless of this flag.
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
    # GPU kernels (naive: one thread per output element)
    # =========================================================================

    @staticmethod
    fn _predict_matmul_kernel[
        BATCH: Int, IN: Int, OUT: Int, dtype: DType,
    ](
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
        ],
        W: LayoutTensor[dtype, Layout.row_major(IN, OUT), MutAnyOrigin],
        b: LayoutTensor[dtype, Layout.row_major(OUT), MutAnyOrigin],
        mu: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * OUT:
            return
        var sb = idx // OUT
        var j = idx % OUT
        var s: Scalar[dtype] = rebind[Scalar[dtype]](b[j])
        for i in range(IN):
            s += rebind[Scalar[dtype]](a_below[sb, i]) * rebind[
                Scalar[dtype]
            ](W[i, j])
        mu[sb, j] = s

    @staticmethod
    fn _eps_kernel[
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
    fn _pull_back_kernel[
        BATCH: Int, IN: Int, OUT: Int, dtype: DType,
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
        W: LayoutTensor[dtype, Layout.row_major(IN, OUT), MutAnyOrigin],
        z_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * IN:
            return
        var b = idx // IN
        var i = idx % IN
        var s: Scalar[dtype] = 0
        for j in range(OUT):
            s += rebind[Scalar[dtype]](eps_above[b, j]) * rebind[
                Scalar[dtype]
            ](W[i, j])
        z_below[b, i] = s

    @staticmethod
    fn _weight_grad_kernel[
        BATCH: Int, IN: Int, OUT: Int, dtype: DType,
    ](
        eps_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ],
        a_below: LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
        ],
        W_grad: LayoutTensor[
            dtype, Layout.row_major(IN, OUT), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= IN * OUT:
            return
        var i = idx // OUT
        var j = idx % OUT
        var s: Scalar[dtype] = 0
        for sb in range(BATCH):
            s += rebind[Scalar[dtype]](eps_above[sb, j]) * rebind[
                Scalar[dtype]
            ](a_below[sb, i])
        W_grad[i, j] = -s

    @staticmethod
    fn _bias_grad_kernel[
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
    fn _bias_add_kernel[
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
    fn _negate_kernel[
        N: Int, dtype: DType,
    ](
        buf: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= N:
            return
        buf.ptr[idx] = -buf.ptr[idx]

    @staticmethod
    fn _transpose_2d_kernel[
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
            # Fallback: naive one-thread-per-output-element kernel.
            comptime k = Self._predict_matmul_kernel[
                BATCH, Self.in_dim, Self.out_dim, dtype
            ]
            var threads = BATCH * Self.out_dim
            var blocks = (threads + TPB - 1) // TPB
            ctx.enqueue_function[k, k](
                a_below, W, b_view, mu,
                grid_dim=(blocks,), block_dim=(TPB,),
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
            comptime k = Self._pull_back_kernel[
                BATCH, Self.in_dim, Self.out_dim, dtype
            ]
            var threads = BATCH * Self.in_dim
            var blocks = (threads + TPB - 1) // TPB
            ctx.enqueue_function[k, k](
                eps_above, W, z_below,
                grid_dim=(blocks,), block_dim=(TPB,),
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
            comptime kw = Self._weight_grad_kernel[
                BATCH, Self.in_dim, Self.out_dim, dtype
            ]
            var w_threads = Self.in_dim * Self.out_dim
            var w_blocks = (w_threads + TPB - 1) // TPB
            ctx.enqueue_function[kw, kw](
                eps_above, a_below, W_grad,
                grid_dim=(w_blocks,), block_dim=(TPB,),
            )

        comptime kb = Self._bias_grad_kernel[BATCH, Self.out_dim, dtype]
        var b_threads = Self.out_dim
        var b_blocks = (b_threads + TPB - 1) // TPB
        ctx.enqueue_function[kb, kb](
            eps_above, b_grad,
            grid_dim=(b_blocks,), block_dim=(TPB,),
        )
