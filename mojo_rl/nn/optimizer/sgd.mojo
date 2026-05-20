# =============================================================================
# SGD Optimizer
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


struct SGD[LR: Float64 = 0.01](Optimizer):
    """Stochastic Gradient Descent optimizer.

    Update rule: param -= lr * grad

    STATE_PER_PARAM = 1 (unused, but minimum for valid tensor dimensions).
    GLOBAL_STATE_SIZE = 1: slot 0 holds `lr_scale` (Scalar[dtype]) — written
    by GPUNetworkState.set_lr_scale, read by the kernel each step so LR
    schedules survive CUDA-graph replay.
    LR is a compile-time struct parameter.
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
        """SGD update: param -= lr * lr_scale * grad. State and step_num are unused.
        """
        var lr = Scalar[dtype](Self.LR * lr_scale)
        for i in range(PARAM_SIZE):
            params[i] -= lr * grads[i]

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
        lr_scale_view: LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ],
        base_lr: Scalar[dtype],
    ):
        """SGD update kernel: param -= (base_lr * lr_scale) * grad.

        Reads lr_scale from a 1-element device view so LR schedules survive
        CUDA-graph replay.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return
        var lr = base_lr * rebind[Scalar[dtype]](lr_scale_view[0])
        params[idx] = rebind[Scalar[dtype]](params[idx]) - lr * rebind[
            Scalar[dtype]
        ](grads[idx])

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
        """Launch SGD optimization step on GPU. `lr_scale` is read from
        `opt_global_state[0]` (the only slot for SGD); per-param `state`
        and host `step_num` are unused.
        """
        var base_lr = Scalar[dtype](Self.LR)
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
            lr_scale_view: LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ],
            base_lr: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE, dtype](
                params, grads, lr_scale_view, base_lr
            )

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper](
            params,
            grads,
            lr_scale_view,
            base_lr,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
