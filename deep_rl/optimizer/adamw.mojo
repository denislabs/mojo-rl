# =============================================================================
# AdamW Optimizer (Adam with Decoupled Weight Decay)
# =============================================================================

from ..constants import dtype, TPB
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from math import sqrt
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer


struct AdamW(Optimizer):
    """AdamW optimizer - Adam with decoupled weight decay.

    The key difference from Adam: weight decay is applied directly to parameters,
    not through the gradient. This leads to better generalization.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

    STATE_PER_PARAM = 2:
        - state[i, 0] = m (first moment)
        - state[i, 1] = v (second moment)

    State is managed externally by the trainer and passed to step().
    """

    comptime STATE_PER_PARAM: Int = 2

    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var weight_decay: Float64
    var t: Int  # Timestep

    fn __init__(
        out self,
        lr: Float64 = 0.001,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        eps: Float64 = 1e-8,
        weight_decay: Float64 = 0.01,
    ):
        """Initialize AdamW optimizer.

        Args:
            lr: Learning rate.
            beta1: Exponential decay rate for first moment (default 0.9).
            beta2: Exponential decay rate for second moment (default 0.999).
            eps: Small constant for numerical stability.
            weight_decay: Weight decay coefficient (default 0.01).
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

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
        """AdamW update step.

        Args:
            params: Parameters to update.
            grads: Gradients.
            state: Optimizer state with layout `(PARAM_SIZE, 2)`.
        """
        self.t += 1

        # Bias correction factors
        var bias_correction1 = Scalar[dtype](1.0 - (self.beta1**self.t))
        var bias_correction2 = Scalar[dtype](1.0 - (self.beta2**self.t))
        var one_minus_beta1 = Scalar[dtype](1.0 - self.beta1)
        var one_minus_beta2 = Scalar[dtype](1.0 - self.beta2)
        var beta1 = Scalar[dtype](self.beta1)
        var beta2 = Scalar[dtype](self.beta2)
        var lr = Scalar[dtype](self.lr)
        var eps = Scalar[dtype](self.eps)
        var wd_factor = Scalar[dtype](1.0 - self.lr * self.weight_decay)

        for i in range(PARAM_SIZE):
            var g = rebind[Scalar[dtype]](grads[i])

            # Read current moments from state
            var m = rebind[Scalar[dtype]](state[i, 0])
            var v = rebind[Scalar[dtype]](state[i, 1])

            # Update moments
            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g

            # Write updated moments back to state
            state[i, 0] = m_new
            state[i, 1] = v_new

            # Bias-corrected estimates
            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2

            # Decoupled weight decay + Adam update
            # param = param * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)
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
        """AdamW optimizer kernel.

        state layout: (PARAM_SIZE, 2) where state[i, 0] = m, state[i, 1] = v.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var g = rebind[Scalar[dtype]](grads[idx])
        var m_val = rebind[Scalar[dtype]](state[idx, 0])
        var v_val = rebind[Scalar[dtype]](state[idx, 1])

        # Update moments
        var one = Scalar[dtype](1.0)
        var m_new = beta1 * m_val + (one - beta1) * g
        var v_new = beta2 * v_val + (one - beta2) * g * g

        # Write updated moments back to state
        state[idx, 0] = m_new
        state[idx, 1] = v_new

        # Bias-corrected estimates
        var m_hat = m_new / bias_correction1
        var v_hat = v_new / bias_correction2

        # Decoupled weight decay + Adam update
        var p = rebind[Scalar[dtype]](params[idx])
        params[idx] = p * wd_factor - lr * m_hat / (sqrt(v_hat) + eps)

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
        """Launch AdamW optimization step on GPU.

        Args:
            ctx: GPU device context.
            params_buf: Parameters buffer [PARAM_SIZE] (modified in place).
            grads_buf: Gradients buffer [PARAM_SIZE].
            state_buf: State buffer [PARAM_SIZE * 2] (m and v moments).
        """
        # Increment timestep
        self.t += 1

        # Compute bias corrections on CPU
        var bias_correction1 = Scalar[dtype](1.0 - (self.beta1**self.t))
        var bias_correction2 = Scalar[dtype](1.0 - (self.beta2**self.t))
        var lr = Scalar[dtype](self.lr)
        var beta1 = Scalar[dtype](self.beta1)
        var beta2 = Scalar[dtype](self.beta2)
        var eps = Scalar[dtype](self.eps)
        var wd_factor = Scalar[dtype](1.0 - self.lr * self.weight_decay)

        # Create LayoutTensor views
        var params = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ](params_buf.unsafe_ptr())
        var grads = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ](grads_buf.unsafe_ptr())
        var state = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE, 2), MutAnyOrigin
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

        # Launch
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
