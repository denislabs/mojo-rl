# =============================================================================
# RMSprop Optimizer
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from std.math import sqrt
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


struct RMSprop[
    LR: Float64 = 0.01,
    ALPHA: Float64 = 0.99,
    EPS: Float64 = 1e-8,
](Optimizer):
    """RMSprop optimizer with adaptive learning rates.

    Update rule:
        v = alpha * v + (1 - alpha) * grad^2
        param = param - lr * grad / (sqrt(v) + eps)

    STATE_PER_PARAM = 1:
        - state[i, 0] = v (squared gradient moving average)

    GLOBAL_STATE_SIZE = 1: slot 0 holds `lr_scale` (Scalar[dtype]) — written
    by GPUNetworkState.set_lr_scale, read by the kernel each step so LR
    schedules survive CUDA-graph replay.

    All hyperparameters are compile-time struct parameters.
    """

    comptime STATE_PER_PARAM: Int = 1
    comptime GLOBAL_STATE_SIZE: Int = 1

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def step[
        PARAM_SIZE: Int, dtype: DType = DType.float32
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
        mut opt_global_state: LayoutTensor[
            dtype, Layout.row_major(Self.GLOBAL_STATE_SIZE), MutAnyOrigin
        ],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ):
        """RMSprop update step. step_num is unused."""
        var alpha = Scalar[dtype](Self.ALPHA)
        var one_minus_alpha = Scalar[dtype](1.0 - Self.ALPHA)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)

        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])
            var v = rebind[Scalar[dtype]](state[i, 0])

            var v_new = alpha * v + one_minus_alpha * g * g
            state[i, 0] = v_new

            var p = rebind[Scalar[dtype]](params[i])
            params[i] = p - lr * g / (sqrt(v_new) + eps)

    # =========================================================================
    # GPU kernel implementation
    # =========================================================================

    @always_inline
    @staticmethod
    def step_kernel_impl[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        params: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        state: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE, 1), MutAnyOrigin
        ],
        lr_scale_view: LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ],
        base_lr: Scalar[dtype],
        alpha: Scalar[dtype],
        eps: Scalar[dtype],
    ):
        """RMSprop kernel. lr_scale is read from a 1-element device view so
        LR schedules survive CUDA-graph replay.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var lr = base_lr * rebind[Scalar[dtype]](lr_scale_view[0])
        var g = rebind[Scalar[dtype]](grads[idx])
        var v_val = rebind[Scalar[dtype]](state[idx, 0])

        var one = Scalar[dtype](1.0)
        var v_new = alpha * v_val + (one - alpha) * g * g
        state[idx, 0] = v_new

        params[idx] = rebind[Scalar[dtype]](params[idx]) - lr * g / (
            sqrt(v_new) + eps
        )

    # =========================================================================
    # GPU launcher
    # =========================================================================

    @staticmethod
    def step_gpu[
        PARAM_SIZE: Int, dtype: DType = DType.float32
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
        mut opt_global_state: LayoutTensor[
            dtype, Layout.row_major(Self.GLOBAL_STATE_SIZE), MutAnyOrigin
        ],
        step_num: Int,
    ) raises:
        """Launch RMSprop optimization step on GPU. `lr_scale` lives in
        `opt_global_state[0]` (the only slot). step_num is unused.
        """
        var base_lr = Scalar[dtype](Self.LR)
        var alpha = Scalar[dtype](Self.ALPHA)
        var eps = Scalar[dtype](Self.EPS)
        var lr_scale_view = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](opt_global_state.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            params: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            state: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE, 1), MutAnyOrigin
            ],
            lr_scale_view: LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ],
            base_lr: Scalar[dtype],
            alpha: Scalar[dtype],
            eps: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE, dtype](
                params, grads, state, lr_scale_view, base_lr, alpha, eps
            )

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            state,
            lr_scale_view,
            base_lr,
            alpha,
            eps,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
