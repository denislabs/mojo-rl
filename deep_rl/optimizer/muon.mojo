# =============================================================================
# Muon Optimizer (Momentum + Orthogonalization via Newton-Schulz)
# =============================================================================
#
# Reference: https://kellerjordan.github.io/posts/muon/
#
# Muon applies orthogonalization to momentum-based updates using Newton-Schulz
# iteration. This approximates the nearest semi-orthogonal matrix to the update,
# which can improve optimization dynamics for weight matrices.
#
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from math import sqrt
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer


struct Muon(Optimizer):
    """Muon optimizer: Momentum + Newton-Schulz orthogonalization.

    Update rule:
        1. Compute Nesterov momentum: m = beta * m + grad
        2. Look-ahead: g_nesterov = grad + beta * m
        3. Reshape to matrix (rows x cols) if dimensions known
        4. Apply Newton-Schulz iteration to approximate orthogonalization:
           - Normalize: X = G / ||G||
           - Iterate: A = X @ X.T, B = b*A + c*A@A, X = a*X + B@X
           - Result approaches UV^T (nearest semi-orthogonal matrix)
        5. Update: param -= lr * orthogonalized_update

    Newton-Schulz coefficients (tuned): a=3.4445, b=-4.7750, c=2.0315

    STATE_PER_PARAM = 1:
        - state[i, 0] = momentum buffer

    Note: For optimal performance, this optimizer is designed for 2D weight
    matrices. For 1D parameters (biases), it falls back to normalized momentum.
    """

    comptime STATE_PER_PARAM: Int = 1

    var lr: Float64
    var beta: Float64  # Momentum coefficient
    var ns_steps: Int  # Newton-Schulz iterations
    var eps: Float64
    # Newton-Schulz coefficients (tuned values from paper)
    var ns_a: Float64
    var ns_b: Float64
    var ns_c: Float64

    fn __init__(
        out self,
        lr: Float64 = 0.02,
        beta: Float64 = 0.95,
        ns_steps: Int = 5,
        eps: Float64 = 1e-7,
    ):
        """Initialize Muon optimizer.

        Args:
            lr: Learning rate (default 0.02, higher than Adam due to normalization).
            beta: Momentum coefficient (default 0.95).
            ns_steps: Number of Newton-Schulz iterations (default 5).
            eps: Small epsilon for numerical stability.
        """
        self.lr = lr
        self.beta = beta
        self.ns_steps = ns_steps
        self.eps = eps
        # Tuned Newton-Schulz coefficients from the paper
        self.ns_a = 3.4445
        self.ns_b = -4.7750
        self.ns_c = 2.0315

    fn step[
        PARAM_SIZE: Int
    ](
        mut self,
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM),
            MutAnyOrigin,
        ],
    ):
        """Muon update step (1D simplified version).

        For 1D parameters, Newton-Schulz degenerates to normalized momentum.
        This is still effective but doesn't capture full orthogonalization benefits.

        Args:
            params: Parameters to update.
            grads: Gradients.
            state: Optimizer state with layout `(PARAM_SIZE, 1)` for momentum.
        """
        var beta = Scalar[dtype](self.beta)
        var lr = Scalar[dtype](self.lr)
        var eps = Scalar[dtype](self.eps)
        var one = Scalar[dtype](1.0)

        # Step 1: Update momentum (Nesterov style)
        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var m_new = beta * m + g
            state[i, 0] = m_new

        # Step 2: Compute Nesterov look-ahead gradient
        # g_nesterov = grad + beta * m_new
        # For 1D case, we normalize the update vector (simplified Newton-Schulz)

        # Compute norm of Nesterov gradient
        var norm_sq = Scalar[dtype](0.0)
        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var g_nesterov = g + beta * m
            norm_sq += g_nesterov * g_nesterov

        var norm = sqrt(norm_sq) + eps
        var scale = lr / norm * Scalar[dtype](sqrt(Float64(PARAM_SIZE)))

        # Step 3: Apply normalized update
        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var g_nesterov = g + beta * m
            params[i] = rebind[Scalar[dtype]](params[i]) - scale * g_nesterov

    # =========================================================================
    # Newton-Schulz helper for matrix orthogonalization
    # =========================================================================

    @staticmethod
    fn _frobenius_norm[
        ROWS: Int, COLS: Int
    ](
        mat: LayoutTensor[dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    ) -> Scalar[dtype]:
        """Compute Frobenius norm of a matrix."""
        var norm_sq = Scalar[dtype](0.0)
        for i in range(ROWS):
            for j in range(COLS):
                var val = rebind[Scalar[dtype]](mat[i, j])
                norm_sq += val * val
        return sqrt(norm_sq)

    @staticmethod
    fn _matmul_AAT[
        ROWS: Int, COLS: Int
    ](
        A: LayoutTensor[dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin],
        out_AAT: LayoutTensor[
            dtype, Layout.row_major(ROWS, ROWS), MutAnyOrigin
        ],
    ):
        """Compute A @ A.T -> out_AAT (ROWS x ROWS)."""
        for i in range(ROWS):
            for j in range(ROWS):
                var sum = Scalar[dtype](0.0)
                for k in range(COLS):
                    sum += rebind[Scalar[dtype]](A[i, k]) * rebind[
                        Scalar[dtype]
                    ](A[j, k])
                out_AAT[i, j] = sum

    @staticmethod
    fn _matmul_square[
        DIM: Int
    ](
        A: LayoutTensor[dtype, Layout.row_major(DIM, DIM), MutAnyOrigin],
        B: LayoutTensor[dtype, Layout.row_major(DIM, DIM), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(DIM, DIM), MutAnyOrigin],
    ):
        """Compute A @ B -> out for square matrices."""
        for i in range(DIM):
            for j in range(DIM):
                var sum = Scalar[dtype](0.0)
                for k in range(DIM):
                    sum += rebind[Scalar[dtype]](A[i, k]) * rebind[
                        Scalar[dtype]
                    ](B[k, j])
                output[i, j] = sum

    @staticmethod
    fn _matmul_rect[
        M: Int, N: Int, K: Int
    ](
        A: LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin],
        B: LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    ):
        """Compute A @ B -> out for rectangular matrices."""
        for i in range(M):
            for j in range(N):
                var sum = Scalar[dtype](0.0)
                for k in range(K):
                    sum += rebind[Scalar[dtype]](A[i, k]) * rebind[
                        Scalar[dtype]
                    ](B[k, j])
                output[i, j] = sum

    # =========================================================================
    # 2D Matrix step (for weight matrices with known dimensions)
    # =========================================================================

    fn step_2d[
        ROWS: Int, COLS: Int
    ](
        mut self,
        mut params: LayoutTensor[
            dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin],
        mut momentum: LayoutTensor[
            dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin
        ],
    ):
        """Muon update step for 2D weight matrices with full Newton-Schulz.

        This is the recommended method for weight matrices where orthogonalization
        provides the most benefit.

        Args:
            params: Weight matrix to update (ROWS x COLS).
            grads: Gradient matrix (ROWS x COLS).
            momentum: Momentum buffer (ROWS x COLS).
        """
        var beta = Scalar[dtype](self.beta)
        var lr = Scalar[dtype](self.lr)
        var eps = Scalar[dtype](self.eps)
        var ns_a = Scalar[dtype](self.ns_a)
        var ns_b = Scalar[dtype](self.ns_b)
        var ns_c = Scalar[dtype](self.ns_c)

        # Step 1: Update momentum (Nesterov style)
        for i in range(ROWS):
            for j in range(COLS):
                var g = rebind[Scalar[dtype]](grads[i, j])
                var m = rebind[Scalar[dtype]](momentum[i, j])
                var m_new = beta * m + g
                momentum[i, j] = m_new

        # Step 2: Compute Nesterov look-ahead gradient into a temporary buffer
        # We'll reuse the gradient interpretation: G = grad + beta * momentum
        # For Newton-Schulz, we work with the smaller dimension

        # Determine if we should transpose (work with smaller dimension as rows)
        comptime should_transpose = ROWS > COLS

        @parameter
        if should_transpose:
            # Work with transposed version: COLS x ROWS
            # X starts as normalized G.T
            var norm = Self._frobenius_norm[ROWS, COLS](grads)
            norm = norm + eps

            # Allocate working buffers on stack (small matrices assumed)
            # For large matrices, consider heap allocation
            var X = InlineArray[Scalar[dtype], COLS * ROWS](Scalar[dtype](0.0))
            var X_new = InlineArray[Scalar[dtype], COLS * ROWS](
                Scalar[dtype](0.0)
            )
            var A = InlineArray[Scalar[dtype], COLS * COLS](Scalar[dtype](0.0))
            var B = InlineArray[Scalar[dtype], COLS * COLS](Scalar[dtype](0.0))
            var AA = InlineArray[Scalar[dtype], COLS * COLS](Scalar[dtype](0.0))
            var BX = InlineArray[Scalar[dtype], COLS * ROWS](Scalar[dtype](0.0))

            # Initialize X = (G + beta*M).T / norm
            for i in range(ROWS):
                for j in range(COLS):
                    var g = rebind[Scalar[dtype]](grads[i, j])
                    var m = rebind[Scalar[dtype]](momentum[i, j])
                    var g_nesterov = g + beta * m
                    # Transposed: X[j, i] = g_nesterov / norm
                    X[j * ROWS + i] = g_nesterov / norm

            # Newton-Schulz iterations
            for _ in range(self.ns_steps):
                # A = X @ X.T (COLS x COLS)
                for i in range(COLS):
                    for j in range(COLS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(ROWS):
                            sum += X[i * ROWS + k] * X[j * ROWS + k]
                        A[i * COLS + j] = sum

                # AA = A @ A (COLS x COLS)
                for i in range(COLS):
                    for j in range(COLS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(COLS):
                            sum += A[i * COLS + k] * A[k * COLS + j]
                        AA[i * COLS + j] = sum

                # B = b*A + c*AA (COLS x COLS)
                for i in range(COLS):
                    for j in range(COLS):
                        B[i * COLS + j] = (
                            ns_b * A[i * COLS + j] + ns_c * AA[i * COLS + j]
                        )

                # BX = B @ X (COLS x ROWS)
                for i in range(COLS):
                    for j in range(ROWS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(COLS):
                            sum += B[i * COLS + k] * X[k * ROWS + j]
                        BX[i * ROWS + j] = sum

                # X_new = a*X + BX
                for i in range(COLS):
                    for j in range(ROWS):
                        X_new[i * ROWS + j] = (
                            ns_a * X[i * ROWS + j] + BX[i * ROWS + j]
                        )

                # Copy X_new to X
                for i in range(COLS * ROWS):
                    X[i] = X_new[i]

            # Apply update: params -= lr * X.T (transpose back)
            for i in range(ROWS):
                for j in range(COLS):
                    params[i, j] = (
                        rebind[Scalar[dtype]](params[i, j])
                        - lr * X[j * ROWS + i]
                    )
        else:
            # No transpose needed: ROWS <= COLS
            # X starts as normalized G
            var norm = Self._frobenius_norm[ROWS, COLS](grads)
            norm = norm + eps

            var X = InlineArray[Scalar[dtype], ROWS * COLS](Scalar[dtype](0.0))
            var X_new = InlineArray[Scalar[dtype], ROWS * COLS](
                Scalar[dtype](0.0)
            )
            var A = InlineArray[Scalar[dtype], ROWS * ROWS](Scalar[dtype](0.0))
            var B = InlineArray[Scalar[dtype], ROWS * ROWS](Scalar[dtype](0.0))
            var AA = InlineArray[Scalar[dtype], ROWS * ROWS](Scalar[dtype](0.0))
            var BX = InlineArray[Scalar[dtype], ROWS * COLS](Scalar[dtype](0.0))

            # Initialize X = (G + beta*M) / norm
            for i in range(ROWS):
                for j in range(COLS):
                    var g = rebind[Scalar[dtype]](grads[i, j])
                    var m = rebind[Scalar[dtype]](momentum[i, j])
                    var g_nesterov = g + beta * m
                    X[i * COLS + j] = g_nesterov / norm

            # Newton-Schulz iterations
            for _ in range(self.ns_steps):
                # A = X @ X.T (ROWS x ROWS)
                for i in range(ROWS):
                    for j in range(ROWS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(COLS):
                            sum += X[i * COLS + k] * X[j * COLS + k]
                        A[i * ROWS + j] = sum

                # AA = A @ A (ROWS x ROWS)
                for i in range(ROWS):
                    for j in range(ROWS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(ROWS):
                            sum += A[i * ROWS + k] * A[k * ROWS + j]
                        AA[i * ROWS + j] = sum

                # B = b*A + c*AA
                for i in range(ROWS):
                    for j in range(ROWS):
                        B[i * ROWS + j] = (
                            ns_b * A[i * ROWS + j] + ns_c * AA[i * ROWS + j]
                        )

                # BX = B @ X (ROWS x COLS)
                for i in range(ROWS):
                    for j in range(COLS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(ROWS):
                            sum += B[i * ROWS + k] * X[k * COLS + j]
                        BX[i * COLS + j] = sum

                # X_new = a*X + BX
                for i in range(ROWS):
                    for j in range(COLS):
                        X_new[i * COLS + j] = (
                            ns_a * X[i * COLS + j] + BX[i * COLS + j]
                        )

                # Copy X_new to X
                for i in range(ROWS * COLS):
                    X[i] = X_new[i]

            # Apply update: params -= lr * X
            for i in range(ROWS):
                for j in range(COLS):
                    params[i, j] = (
                        rebind[Scalar[dtype]](params[i, j])
                        - lr * X[i * COLS + j]
                    )

    # =========================================================================
    # GPU kernel implementation (simplified 1D version)
    # =========================================================================

    @always_inline
    @staticmethod
    fn step_kernel_impl[
        PARAM_SIZE: Int
    ](
        params: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        state: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE, 1), MutAnyOrigin
        ],
        lr: Scalar[dtype],
        beta: Scalar[dtype],
        inv_norm: Scalar[dtype],
        scale_factor: Scalar[dtype],
    ):
        """Muon optimizer kernel (simplified 1D version).

        The norm is pre-computed on CPU; this kernel applies the normalized update.

        state layout: (PARAM_SIZE, 1) where state[i, 0] = momentum.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var g = rebind[Scalar[dtype]](grads[idx])
        var m_val = rebind[Scalar[dtype]](state[idx, 0])

        # Update momentum
        var m_new = beta * m_val + g
        state[idx, 0] = m_new

        # Nesterov look-ahead
        var g_nesterov = g + beta * m_new

        # Apply normalized update
        params[idx] = (
            rebind[Scalar[dtype]](params[idx])
            - lr * scale_factor * inv_norm * g_nesterov
        )

    # =========================================================================
    # GPU launcher
    # =========================================================================

    fn step_gpu[
        PARAM_SIZE: Int
    ](
        mut self,
        ctx: DeviceContext,
        params_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
        state_buf: DeviceBuffer[dtype],
    ) raises:
        """Launch Muon optimization step on GPU (simplified 1D version).

        Note: This is the simplified version for 1D parameters. For 2D weight
        matrices, use step_2d which provides full Newton-Schulz orthogonalization.

        Args:
            ctx: GPU device context.
            params_buf: Parameters buffer [PARAM_SIZE] (modified in place).
            grads_buf: Gradients buffer [PARAM_SIZE].
            state_buf: State buffer [PARAM_SIZE] (momentum).
        """
        # Pre-compute norm on CPU (requires reading gradients and momentum)
        # This is a simplification; production code would compute on GPU
        var beta = Scalar[dtype](self.beta)
        var lr = Scalar[dtype](self.lr)
        var eps = Scalar[dtype](self.eps)
        var scale_factor = Scalar[dtype](sqrt(Float64(PARAM_SIZE)))

        # Create LayoutTensor views
        var params = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ](params_buf.unsafe_ptr())
        var grads = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ](grads_buf.unsafe_ptr())
        var state = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE, 1), MutAnyOrigin
        ](state_buf.unsafe_ptr())

        # Compute Nesterov gradient norm on CPU (could be optimized with GPU reduction)
        var norm_sq = Scalar[dtype](0.0)
        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var g_nesterov = g + beta * (beta * m + g)
            norm_sq += g_nesterov * g_nesterov
        var inv_norm = Scalar[dtype](1.0) / (sqrt(norm_sq) + eps)

        # Kernel wrapper
        @always_inline
        fn kernel_wrapper(
            params: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            state: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE, 1), MutAnyOrigin
            ],
            lr: Scalar[dtype],
            beta: Scalar[dtype],
            inv_norm: Scalar[dtype],
            scale_factor: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE](
                params,
                grads,
                state,
                lr,
                beta,
                inv_norm,
                scale_factor,
            )

        # Launch
        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            state,
            lr,
            beta,
            inv_norm,
            scale_factor,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
