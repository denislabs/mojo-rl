# =============================================================================
# SGD Optimizer
# =============================================================================

from ..constants import dtype
from .optimizer import Optimizer
from layout import LayoutTensor, Layout
from gpu import thread_idx


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
            dtype, Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM), MutAnyOrigin
        ],
    ):
        """SGD update: param -= lr * grad. State is unused."""
        for i in range(PARAM_SIZE):
            params[i] -= Scalar[dtype](self.lr) * grads[i]

    @always_inline
    fn step_kernel[
        PARAM_SIZE: Int
    ](
        self,
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE, 1), ImmutAnyOrigin
        ],
    ):
        """SGD update on GPU: param -= lr * grad."""
        var idx = thread_idx.x
        if idx < UInt(PARAM_SIZE):
            params[idx] -= Scalar[dtype](self.lr) * grads[idx, 0]
