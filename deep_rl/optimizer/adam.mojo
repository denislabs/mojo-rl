# =============================================================================
# Adam Optimizer (Stateless)
# =============================================================================

from ..constants import dtype
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from math import sqrt
from gpu import thread_idx


struct Adam(Optimizer):
    """Adam optimizer with adaptive learning rates.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

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
    var t: Int  # Timestep (not per-parameter, stays in struct)

    fn __init__(
        out self,
        lr: Float64 = 0.001,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        eps: Float64 = 1e-8,
    ):
        """Initialize Adam optimizer."""
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
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
            dtype, Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM), MutAnyOrigin
        ],
    ):
        """Adam update step.

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

        for i in range(PARAM_SIZE):
            var g = grads[i]

            # Read current moments from state
            var m = state[i, 0]
            var v = state[i, 1]

            # Update moments
            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g

            # Write updated moments back to state
            state[i, 0] = m_new
            state[i, 1] = v_new

            # Bias-corrected estimates
            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2

            # Update parameters
            params[i] -= lr * m_hat / (sqrt(v_hat) + eps)

    # @always_inline
    # fn step_kernel[
    #     PARAM_SIZE: Int
    # ](
    #     self,
    #     mut params: LayoutTensor[
    #         dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    #     ],
    #     grads: LayoutTensor[
    #         dtype, Layout.row_major(PARAM_SIZE), ImmutAnyOrigin
    #     ],
    #     mut state: LayoutTensor[
    #         dtype, Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM), MutAnyOrigin
    #     ],
    # ):
    #     """Adam optimizer update kernel for GPU."""
    #     var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    #     if idx >= PARAM_SIZE:
    #         return
    #
    #     var g = grads[idx]
    #     var m = state[idx, 0]
    #     var v = state[idx, 1]
    #
    #     var m_new = Scalar[dtype](self.beta1) * m + (1 - Scalar[dtype](self.beta1)) * g
    #     var v_new = Scalar[dtype](self.beta2) * v + (1 - Scalar[dtype](self.beta2)) * g * g
    #
    #     state[idx, 0] = m_new
    #     state[idx, 1] = v_new
    #
    #     var m_hat = m_new / Scalar[dtype](1.0 - (self.beta1**self.t))
    #     var v_hat = v_new / Scalar[dtype](1.0 - (self.beta2**self.t))
    #
    #     params[idx] -= Scalar[dtype](self.lr) * m_hat / (sqrt(v_hat) + Scalar[dtype](self.eps))
