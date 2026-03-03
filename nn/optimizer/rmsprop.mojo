# =============================================================================
# RMSprop Optimizer
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from math import sqrt
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer


struct RMSprop(Optimizer):
    """RMSprop optimizer with adaptive learning rates.

    Update rule:
        v = alpha * v + (1 - alpha) * grad^2
        param = param - lr * grad / (sqrt(v) + eps)

    STATE_PER_PARAM = 1:
        - state[i, 0] = v (squared gradient moving average)

    State is managed externally by the trainer and passed to step().
    """

    comptime STATE_PER_PARAM: Int = 1

    var lr: Float64
    var alpha: Float64  # Decay rate for squared gradient average
    var eps: Float64

    fn __init__(
        out self,
        lr: Float64 = 0.01,
        alpha: Float64 = 0.99,
        eps: Float64 = 1e-8,
    ):
        """Initialize RMSprop optimizer.

        Args:
            lr: Learning rate (default 0.01, typical for RMSprop).
            alpha: Decay rate for squared gradient average (default 0.99).
            eps: Small constant for numerical stability.
        """
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

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
        """RMSprop update step.

        Args:
            params: Parameters to update.
            grads: Gradients.
            state: Optimizer state with layout `(PARAM_SIZE, 1)`.
        """
        var alpha = Scalar[dtype](self.alpha)
        var one_minus_alpha = Scalar[dtype](1.0 - self.alpha)
        var lr = Scalar[dtype](self.lr)
        var eps = Scalar[dtype](self.eps)

        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])

            # Read current squared gradient average from state
            var v = rebind[Scalar[dtype]](state[i, 0])

            # Update squared gradient average
            var v_new = alpha * v + one_minus_alpha * g * g

            # Write updated state back
            state[i, 0] = v_new

            # Update parameters
            var p = rebind[Scalar[dtype]](params[i])
            params[i] = p - lr * g / (sqrt(v_new) + eps)

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
            dtype, Layout.row_major(PARAM_SIZE, 1), MutAnyOrigin
        ],
        lr: Scalar[dtype],
        alpha: Scalar[dtype],
        eps: Scalar[dtype],
    ):
        """RMSprop optimizer kernel.

        state layout: (PARAM_SIZE, 1) where state[i, 0] = v.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var g = rebind[Scalar[dtype]](grads[idx])
        var v_val = rebind[Scalar[dtype]](state[idx, 0])

        # Update squared gradient average
        var one = Scalar[dtype](1.0)
        var v_new = alpha * v_val + (one - alpha) * g * g

        # Write updated state back
        state[idx, 0] = v_new

        # Update parameters
        params[idx] = rebind[Scalar[dtype]](params[idx]) - lr * g / (
            sqrt(v_new) + eps
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
        """Launch RMSprop optimization step on GPU.

        Args:
            ctx: GPU device context.
            params_buf: Parameters buffer [PARAM_SIZE] (modified in place).
            grads_buf: Gradients buffer [PARAM_SIZE].
            state_buf: State buffer [PARAM_SIZE] (squared gradient average).
        """
        var lr = Scalar[dtype](self.lr)
        var alpha = Scalar[dtype](self.alpha)
        var eps = Scalar[dtype](self.eps)

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
            alpha: Scalar[dtype],
            eps: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE](
                params, grads, state, lr, alpha, eps
            )

        # Launch
        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            params,
            grads,
            state,
            lr,
            alpha,
            eps,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
