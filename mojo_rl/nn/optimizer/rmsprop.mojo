# =============================================================================
# RMSprop Optimizer
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from std.math import sqrt
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.sys import simd_width_of


comptime _CPU_SIMD_W = simd_width_of[dtype]()


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

        # STATE_PER_PARAM=1 → state buffer is contiguous Float32 [v0, v1, ...].
        comptime W = _CPU_SIMD_W
        var p_p = params.ptr
        var g_p = grads.ptr
        var s_p = state.ptr
        var alpha_v = SIMD[dtype, W](alpha)
        var oma_v = SIMD[dtype, W](one_minus_alpha)
        var lr_v = SIMD[dtype, W](lr)
        var eps_v = SIMD[dtype, W](eps)
        var i = 0
        while i + W <= PARAM_SIZE:
            var g = g_p.load[width=W](i)
            var v = s_p.load[width=W](i)
            var v_new = alpha_v * v + oma_v * g * g
            s_p.store(i, v_new)
            var p = p_p.load[width=W](i)
            p_p.store(i, p - lr_v * g / (sqrt(v_new) + eps_v))
            i += W
        while i < PARAM_SIZE:
            var g = grads[i]
            var v = state[i, 0]
            var v_new = alpha * v + one_minus_alpha * g * g
            state[i, 0] = v_new
            params[i] = params[i] - lr * g / (sqrt(v_new) + eps)
            i += 1

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

        ctx.enqueue_function[kernel_wrapper](
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
