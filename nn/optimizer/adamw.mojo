# =============================================================================
# AdamW Optimizer (Adam with Decoupled Weight Decay)
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from math import sqrt
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer


struct AdamW[
    LR: Float64 = 0.001,
    BETA1: Float64 = 0.9,
    BETA2: Float64 = 0.999,
    EPS: Float64 = 1e-8,
    WEIGHT_DECAY: Float64 = 0.01,
](Optimizer):
    """AdamW optimizer - Adam with decoupled weight decay.

    The key difference from Adam: weight decay is applied directly to parameters,
    not through the gradient. This leads to better generalization.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^step)
        v_hat = v / (1 - beta2^step)
        param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

    STATE_PER_PARAM = 2:
        - state[i, 0] = m (first moment)
        - state[i, 1] = v (second moment)

    All hyperparameters are compile-time struct parameters.
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
    ):
        """AdamW update step."""
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var one_minus_beta1 = Scalar[dtype](1.0 - Self.BETA1)
        var one_minus_beta2 = Scalar[dtype](1.0 - Self.BETA2)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var lr = Scalar[dtype](Self.LR)
        var eps = Scalar[dtype](Self.EPS)
        var wd_factor = Scalar[dtype](1.0 - Self.LR * Self.WEIGHT_DECAY)

        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var m = rebind[Scalar[dtype]](state[i, 0])
            var v = rebind[Scalar[dtype]](state[i, 1])

            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g

            state[i, 0] = m_new
            state[i, 1] = v_new

            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2

            var p = rebind[Scalar[dtype]](params[i])
            params[i] = p * wd_factor - lr * m_hat / (sqrt(v_hat) + eps)

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
        wd_factor: Scalar[dtype],
    ):
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

        var p = rebind[Scalar[dtype]](params[idx])
        params[idx] = p * wd_factor - lr * m_hat / (sqrt(v_hat) + eps)

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
    ) raises:
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var lr = Scalar[dtype](Self.LR)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var eps = Scalar[dtype](Self.EPS)
        var wd_factor = Scalar[dtype](1.0 - Self.LR * Self.WEIGHT_DECAY)

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
            wd_factor: Scalar[dtype],
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
                wd_factor,
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
            wd_factor,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
