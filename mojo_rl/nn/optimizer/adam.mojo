# =============================================================================
# Adam Optimizer
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from std.math import sqrt
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


struct Adam[
    LR: Float64 = 0.001,
    BETA1: Float64 = 0.9,
    BETA2: Float64 = 0.999,
    EPS: Float64 = 1e-8,
](Optimizer):
    """Adam optimizer with adaptive learning rates.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^step)
        v_hat = v / (1 - beta2^step)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    STATE_PER_PARAM = 2:
        - state[i, 0] = m (first moment)
        - state[i, 1] = v (second moment)

    Hyperparameters are compile-time struct parameters.
    step_num is passed by the caller (stored in NetworkState).
    """

    comptime STATE_PER_PARAM: Int = 2

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
        """Adam update step.

        Args:
            params: Parameters to update.
            grads: Gradients.
            state: Optimizer state layout `(PARAM_SIZE, 2)`: m at col 0, v at col 1.
            step_num: Current step (1-based), used for bias correction.
            lr_scale: Multiplicative LR scale (default 1.0). Set < 1.0 for LR annealing.
        """
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var one_minus_beta1 = Scalar[dtype](1.0 - Self.BETA1)
        var one_minus_beta2 = Scalar[dtype](1.0 - Self.BETA2)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)

        for i in range(PARAM_SIZE):
            var g = grads[i]
            var m = state[i, 0]
            var v = state[i, 1]

            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g

            state[i, 0] = m_new
            state[i, 1] = v_new

            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2

            params[i] -= lr * m_hat / (sqrt(v_hat) + eps)

    # =========================================================================
    # GPU kernel implementation
    # =========================================================================

    @always_inline
    @staticmethod
    fn step_kernel_impl[
        PARAM_SIZE: Int
    ](
        params: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        state: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
        ],
        lr: Scalar[dtype],
        beta1: Scalar[dtype],
        beta2: Scalar[dtype],
        eps: Scalar[dtype],
        bias_correction1: Scalar[dtype],
        bias_correction2: Scalar[dtype],
    ):
        """Adam optimizer kernel.

        state layout: (PARAM_SIZE, 2) where state[i, 0] = m, state[i, 1] = v.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var g = rebind[Scalar[dtype]](grads[idx])
        var m_val = rebind[Scalar[dtype]](state[idx, 0])
        var v_val = rebind[Scalar[dtype]](state[idx, 1])

        var one = Scalar[dtype](1.0)
        var m_new = beta1 * m_val + (one - beta1) * g
        var v_new = beta2 * v_val + (one - beta2) * g * g

        state[idx, 0] = m_new
        state[idx, 1] = v_new

        var m_hat = m_new / bias_correction1
        var v_hat = v_new / bias_correction2

        params[idx] = rebind[Scalar[dtype]](params[idx]) - lr * m_hat / (
            sqrt(v_hat) + eps
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
        """Launch Adam optimization step on GPU.

        Args:
            ctx: GPU device context.
            params: Parameters [PARAM_SIZE] (modified in place).
            grads: Gradients [PARAM_SIZE].
            state: State [PARAM_SIZE, 2] (m and v moments).
            step_num: Current step (1-based), used for bias correction.
            lr_scale: Multiplicative LR scale (default 1.0). Set < 1.0 for LR annealing.
        """
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var eps = Scalar[dtype](Self.EPS)

        @always_inline
        fn kernel_wrapper(
            params: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            state: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
            ],
            lr: Scalar[dtype],
            beta1: Scalar[dtype],
            beta2: Scalar[dtype],
            eps: Scalar[dtype],
            bias_correction1: Scalar[dtype],
            bias_correction2: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE](
                params,
                grads,
                state,
                lr,
                beta1,
                beta2,
                eps,
                bias_correction1,
                bias_correction2,
            )

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            state,
            lr,
            beta1,
            beta2,
            eps,
            bias_correction1,
            bias_correction2,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
