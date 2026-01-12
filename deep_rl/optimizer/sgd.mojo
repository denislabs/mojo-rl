# =============================================================================
# SGD Optimizer
# =============================================================================

from ..constants import dtype
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from gpu import thread_idx


struct SGD[param_size: Int](Optimizer):
    """Stochastic Gradient Descent optimizer.

    Update rule: param -= lr * grad
    """

    comptime PARAM_SIZE: Int = Self.param_size
    comptime GRAD_SIZE: Int = 1

    var lr: Float64

    fn __init__(out self, lr: Float64 = 0.01):
        """Initialize SGD with learning rate."""
        self.lr = lr

    fn step(
        mut self,
        mut params: InlineArray[Scalar[dtype], Self.PARAM_SIZE],
        grads: InlineArray[Scalar[dtype], Self.PARAM_SIZE],
    ):
        """SGD update: param -= lr * grad."""
        for i in range(Self.PARAM_SIZE):
            params[i] = Scalar[dtype](
                Float64(params[i]) - self.lr * Float64(grads[i])
            )

    @always_inline
    fn step_kernel(
        self,
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE, 1), ImmutAnyOrigin
        ],
    ):
        """SGD update on GPU: param -= lr * grad."""
        var idx = thread_idx.x
        if idx < UInt(Self.PARAM_SIZE):
            params[idx] -= Scalar[dtype](self.lr) * grads[idx, 0]
