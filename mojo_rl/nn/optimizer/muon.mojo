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
from std.math import sqrt
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


struct Muon[
    LR: Float64 = 0.02,
    BETA: Float64 = 0.95,
    NS_STEPS: Int = 5,
    EPS: Float64 = 1e-7,
](Optimizer):
    """Muon optimizer: Momentum + Newton-Schulz orthogonalization.

    Update rule:
        1. Compute Nesterov momentum: m = beta * m + grad
        2. Look-ahead: g_nesterov = grad + beta * m
        3. Apply Newton-Schulz iteration to approximate orthogonalization
        4. Update: param -= lr * orthogonalized_update

    Newton-Schulz coefficients (tuned): a=3.4445, b=-4.7750, c=2.0315

    STATE_PER_PARAM = 1:
        - state[i, 0] = momentum buffer

    All hyperparameters are compile-time struct parameters.
    For 2D weight matrices, use step_2d() for full Newton-Schulz orthogonalization.
    """

    comptime STATE_PER_PARAM: Int = 1

    # Newton-Schulz coefficients (tuned values from paper)
    comptime NS_A: Float64 = 3.4445
    comptime NS_B: Float64 = -4.7750
    comptime NS_C: Float64 = 2.0315

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    fn step[
        PARAM_SIZE: Int
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM),
            MutAnyOrigin,
        ],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ):
        """Muon update step (1D simplified version). step_num is unused.

        For 1D parameters, Newton-Schulz degenerates to normalized momentum.
        """
        var beta = Scalar[dtype](Self.BETA)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)

        # Update momentum (Nesterov style)
        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            state[i, 0] = beta * m + g

        # Compute Nesterov look-ahead gradient norm
        var norm_sq = Scalar[dtype](0.0)
        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var g_nesterov = g + beta * m
            norm_sq += g_nesterov * g_nesterov

        var norm = sqrt(norm_sq) + eps
        var scale = lr / norm * Scalar[dtype](sqrt(Float64(PARAM_SIZE)))

        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var g_nesterov = g + beta * m
            params[i] = rebind[Scalar[dtype]](params[i]) - scale * g_nesterov

    # =========================================================================
    # Newton-Schulz helpers for matrix orthogonalization
    # =========================================================================

    @staticmethod
    fn _frobenius_norm[
        ROWS: Int, COLS: Int
    ](
        mat: LayoutTensor[dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    ) -> Scalar[dtype]:
        var norm_sq = Scalar[dtype](0.0)
        for i in range(ROWS):
            for j in range(COLS):
                var val = rebind[Scalar[dtype]](mat[i, j])
                norm_sq += val * val
        return sqrt(norm_sq)

    # =========================================================================
    # 2D Matrix step (for weight matrices with known dimensions)
    # =========================================================================

    fn step_2d[
        ROWS: Int, COLS: Int
    ](
        self,
        mut params: LayoutTensor[
            dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin],
        mut momentum: LayoutTensor[
            dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin
        ],
    ):
        """Muon update for 2D weight matrices with full Newton-Schulz.

        Args:
            params: Weight matrix to update (ROWS x COLS).
            grads: Gradient matrix (ROWS x COLS).
            momentum: Momentum buffer (ROWS x COLS).
        """
        var beta = Scalar[dtype](Self.BETA)
        var lr = Scalar[dtype](Self.LR)
        var eps = Scalar[dtype](Self.EPS)
        var ns_a = Scalar[dtype](Self.NS_A)
        var ns_b = Scalar[dtype](Self.NS_B)
        var ns_c = Scalar[dtype](Self.NS_C)

        # Update momentum (Nesterov style)
        for i in range(ROWS):
            for j in range(COLS):
                var g = rebind[Scalar[dtype]](grads[i, j])
                var m = rebind[Scalar[dtype]](momentum[i, j])
                momentum[i, j] = beta * m + g

        comptime should_transpose = ROWS > COLS

        comptime if should_transpose:
            var norm = Self._frobenius_norm[ROWS, COLS](grads) + eps

            var X = InlineArray[Scalar[dtype], COLS * ROWS](uninitialized=True)
            var X_new = InlineArray[Scalar[dtype], COLS * ROWS](
                uninitialized=True
            )
            var A = InlineArray[Scalar[dtype], COLS * COLS](uninitialized=True)
            var B = InlineArray[Scalar[dtype], COLS * COLS](uninitialized=True)
            var AA = InlineArray[Scalar[dtype], COLS * COLS](uninitialized=True)
            var BX = InlineArray[Scalar[dtype], COLS * ROWS](uninitialized=True)

            for i in range(ROWS):
                for j in range(COLS):
                    var g = rebind[Scalar[dtype]](grads[i, j])
                    var m = rebind[Scalar[dtype]](momentum[i, j])
                    X[j * ROWS + i] = (g + beta * m) / norm

            for _ in range(Self.NS_STEPS):
                for i in range(COLS):
                    for j in range(COLS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(ROWS):
                            sum += X[i * ROWS + k] * X[j * ROWS + k]
                        A[i * COLS + j] = sum

                for i in range(COLS):
                    for j in range(COLS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(COLS):
                            sum += A[i * COLS + k] * A[k * COLS + j]
                        AA[i * COLS + j] = sum

                for i in range(COLS):
                    for j in range(COLS):
                        B[i * COLS + j] = (
                            ns_b * A[i * COLS + j] + ns_c * AA[i * COLS + j]
                        )

                for i in range(COLS):
                    for j in range(ROWS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(COLS):
                            sum += B[i * COLS + k] * X[k * ROWS + j]
                        BX[i * ROWS + j] = sum

                for i in range(COLS * ROWS):
                    X_new[i] = ns_a * X[i] + BX[i]

                for i in range(COLS * ROWS):
                    X[i] = X_new[i]

            for i in range(ROWS):
                for j in range(COLS):
                    params[i, j] = (
                        rebind[Scalar[dtype]](params[i, j])
                        - lr * X[j * ROWS + i]
                    )
        else:
            var norm = Self._frobenius_norm[ROWS, COLS](grads) + eps

            var X = InlineArray[Scalar[dtype], ROWS * COLS](uninitialized=True)
            var X_new = InlineArray[Scalar[dtype], ROWS * COLS](
                uninitialized=True
            )
            var A = InlineArray[Scalar[dtype], ROWS * ROWS](uninitialized=True)
            var B = InlineArray[Scalar[dtype], ROWS * ROWS](uninitialized=True)
            var AA = InlineArray[Scalar[dtype], ROWS * ROWS](uninitialized=True)
            var BX = InlineArray[Scalar[dtype], ROWS * COLS](uninitialized=True)

            for i in range(ROWS):
                for j in range(COLS):
                    var g = rebind[Scalar[dtype]](grads[i, j])
                    var m = rebind[Scalar[dtype]](momentum[i, j])
                    X[i * COLS + j] = (g + beta * m) / norm

            for _ in range(Self.NS_STEPS):
                for i in range(ROWS):
                    for j in range(ROWS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(COLS):
                            sum += X[i * COLS + k] * X[j * COLS + k]
                        A[i * ROWS + j] = sum

                for i in range(ROWS):
                    for j in range(ROWS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(ROWS):
                            sum += A[i * ROWS + k] * A[k * ROWS + j]
                        AA[i * ROWS + j] = sum

                for i in range(ROWS):
                    for j in range(ROWS):
                        B[i * ROWS + j] = (
                            ns_b * A[i * ROWS + j] + ns_c * AA[i * ROWS + j]
                        )

                for i in range(ROWS):
                    for j in range(COLS):
                        var sum = Scalar[dtype](0.0)
                        for k in range(ROWS):
                            sum += B[i * ROWS + k] * X[k * COLS + j]
                        BX[i * COLS + j] = sum

                for i in range(ROWS * COLS):
                    X_new[i] = ns_a * X[i] + BX[i]

                for i in range(ROWS * COLS):
                    X[i] = X_new[i]

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
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var g = rebind[Scalar[dtype]](grads[idx])
        var m_val = rebind[Scalar[dtype]](state[idx, 0])

        var m_new = beta * m_val + g
        state[idx, 0] = m_new

        var g_nesterov = g + beta * m_new

        params[idx] = (
            rebind[Scalar[dtype]](params[idx])
            - lr * scale_factor * inv_norm * g_nesterov
        )

    # =========================================================================
    # GPU launcher
    # =========================================================================

    @staticmethod
    fn step_gpu[
        PARAM_SIZE: Int
    ](
        ctx: DeviceContext,
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM),
            MutAnyOrigin,
        ],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ) raises:
        """Launch Muon optimization step on GPU (simplified 1D version).

        Note: Norm is computed on CPU from device tensors (works on unified memory).
        For non-unified memory, move norm computation to GPU reduction kernel.
        step_num is unused.
        """
        var beta = Scalar[dtype](Self.BETA)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)
        var scale_factor = Scalar[dtype](sqrt(Float64(PARAM_SIZE)))

        # Compute Nesterov gradient norm on CPU (Apple Silicon unified memory)
        var norm_sq = Scalar[dtype](0.0)
        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var g_nesterov = g + beta * (beta * m + g)
            norm_sq += g_nesterov * g_nesterov
        var inv_norm = Scalar[dtype](1.0) / (sqrt(norm_sq) + eps)

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
