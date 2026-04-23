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
    LR is a compile-time struct parameter.
    """

    comptime STATE_PER_PARAM: Int = 1

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
        lr: Scalar[dtype],
    ):
        """SGD update kernel: param -= lr * grad."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return
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
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ) raises:
        """Launch SGD optimization step on GPU. State and step_num are unused.
        """
        var lr = Scalar[dtype](Self.LR * lr_scale)

        @parameter
        @always_inline
        def kernel_wrapper(
            params: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
            ],
            lr: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE, dtype](params, grads, lr)

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            lr,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
