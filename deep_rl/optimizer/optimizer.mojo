# =============================================================================
# Optimizer Trait
# =============================================================================

from ..constants import dtype
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer


trait Optimizer(Movable & ImplicitlyCopyable):
    """Base trait for optimizers.

    Optimizers update parameters using gradients. State (e.g., moments for Adam)
    is passed externally by the trainer, making optimizers stateless with respect
    to parameter-sized buffers.

    STATE_PER_PARAM defines how many state values are needed per parameter:
    - SGD: 1 (unused, but minimum for valid tensor dimensions)
    - Adam: 2 (m = first moment, v = second moment)
    """

    comptime STATE_PER_PARAM: Int

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
        """Perform one optimization step.

        Args:
            params: Flattened parameters to update (modified in place).
            grads: Flattened gradients.
            state: Optimizer state (e.g., moments). Layout: (PARAM_SIZE, STATE_PER_PARAM).
        """
        ...

    # =========================================================================
    # GPU methods
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
        """Perform one optimization step on GPU.

        Args:
            ctx: GPU device context.
            params_buf: Parameters buffer [PARAM_SIZE] (modified in place).
            grads_buf: Gradients buffer [PARAM_SIZE].
            state_buf: Optimizer state buffer [PARAM_SIZE * STATE_PER_PARAM].
        """
        ...
