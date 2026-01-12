# =============================================================================
# Adam Optimizer (Trait-based)
# =============================================================================

from ..constants import dtype
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from math import sqrt
from gpu import thread_idx


struct Adam[param_size: Int](Optimizer):
    """Adam optimizer with adaptive learning rates.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    comptime PARAM_SIZE: Int = Self.param_size
    comptime GRAD_SIZE: Int = 3

    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var t: Int
    var m: InlineArray[Scalar[dtype], Self.param_size]  # First moment
    var v: InlineArray[Scalar[dtype], Self.param_size]  # Second moment

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
        self.m = InlineArray[Scalar[dtype], Self.param_size](uninitialized=True)
        self.v = InlineArray[Scalar[dtype], Self.param_size](uninitialized=True)

        # Zero initialize moments
        for i in range(Self.param_size):
            self.m[i] = 0
            self.v[i] = 0

    fn step(
        mut self,
        mut params: InlineArray[Scalar[dtype], Self.PARAM_SIZE],
        grads: InlineArray[Scalar[dtype], Self.PARAM_SIZE],
    ):
        """Adam update step."""
        self.t += 1

        # Bias correction factors
        var bias_correction1 = 1.0 - (self.beta1**self.t)
        var bias_correction2 = 1.0 - (self.beta2**self.t)
        var one_minus_beta1 = 1.0 - self.beta1
        var one_minus_beta2 = 1.0 - self.beta2

        for i in range(Self.PARAM_SIZE):
            var g = Float64(grads[i])

            # Update moments
            self.m[i] = Scalar[dtype](
                self.beta1 * Float64(self.m[i]) + one_minus_beta1 * g
            )
            self.v[i] = Scalar[dtype](
                self.beta2 * Float64(self.v[i]) + one_minus_beta2 * g * g
            )

            # Bias-corrected estimates
            var m_hat = Float64(self.m[i]) / bias_correction1
            var v_hat = Float64(self.v[i]) / bias_correction2

            # Update parameters
            params[i] = Scalar[dtype](
                Float64(params[i]) - self.lr * m_hat / (sqrt(v_hat) + self.eps)
            )

    # @always_inline
    # fn step_kernel(
    #     self,
    #     mut params: LayoutTensor[
    #         dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
    #     ],
    #     mut grads: LayoutTensor[
    #         dtype,
    #         Layout.row_major(Self.PARAM_SIZE, Self.GRAD_SIZE),
    #         ImmutAnyOrigin,
    #     ],
    # ):
    #     """Adam optimizer update kernel."""
    #     var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    #     if idx >= Self.PARAM_SIZE:
    #         return

    #     var g = grads[idx, 0]
    #     var m_val = grads[idx, 1]
    #     var v_val = grads[idx, 2]

    #     var m_new: grads.element_type = (
    #         Scalar[dtype](self.beta1) * m_val
    #         + (1 - Scalar[dtype](self.beta1)) * g
    #     )
    #     var v_new: grads.element_type = (
    #         Scalar[dtype](self.beta2) * v_val
    #         + (1 - Scalar[dtype](self.beta2)) * g * g
    #     )

    #     var m_hat = m_new / Scalar[dtype](1.0 - (self.beta1**self.t))
    #     var v_hat = v_new / Scalar[dtype](1.0 - (self.beta2**self.t))

    #     params[idx] = params[idx] - Scalar[dtype](self.lr) * m_hat / (
    #         sqrt(v_hat) + Scalar[dtype](self.eps)
    #     )

    #     grads[idx, 1] = m_new
    #     grads[idx, 2] = v_new
