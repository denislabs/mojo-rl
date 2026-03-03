# =============================================================================
# SGD Optimizer
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer


struct SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Update rule: param -= lr * grad

    STATE_PER_PARAM = 1 (unused, but required for valid tensor dimensions).
    """

    comptime STATE_PER_PARAM: Int = 1

    var lr: Float64

    fn __init__(out self, lr: Float64 = 0.01):
        """Initialize SGD with learning rate."""
        self.lr = lr

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
        """SGD update: param -= lr * grad. State is unused."""
        for i in range(PARAM_SIZE):
            params[i] -= Scalar[dtype](self.lr) * grads[i]

    # =========================================================================
    # GPU kernel implementation (inlinable for fusion)
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
    ):
        """SGD update kernel: param -= lr * grad.

        State is unused for SGD but included for API consistency.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return
        params[idx] = rebind[Scalar[dtype]](params[idx]) - lr * rebind[
            Scalar[dtype]
        ](grads[idx])

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
        """Launch SGD optimization step on GPU.

        Args:
            ctx: GPU device context.
            params_buf: Parameters buffer [PARAM_SIZE] (modified in place).
            grads_buf: Gradients buffer [PARAM_SIZE].
            state_buf: State buffer [PARAM_SIZE] (unused for SGD).
        """
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

        var lr = Scalar[dtype](self.lr)

        # Kernel wrapper with explicit parameters
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
        ):
            Self.step_kernel_impl[PARAM_SIZE](params, grads, state, lr)

        # Launch
        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            state,
            lr,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
