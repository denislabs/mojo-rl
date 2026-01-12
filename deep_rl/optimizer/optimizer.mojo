# =============================================================================
# Optimizer Trait
# =============================================================================

from ..constants import dtype
from layout import Layout, LayoutTensor


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
            dtype, Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM), MutAnyOrigin
        ],
    ):
        """Perform one optimization step.

        Args:
            params: Flattened parameters to update (modified in place).
            grads: Flattened gradients.
            state: Optimizer state (e.g., moments). Layout: (PARAM_SIZE, STATE_PER_PARAM).
        """
        ...

    # @always_inline
    # fn step_kernel(
    #     self,
    #     mut params: LayoutTensor[
    #         dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
    #     ],
    #     grads: LayoutTensor[
    #         dtype,
    #         Layout.row_major(Self.PARAM_SIZE, Self.GRAD_SIZE),
    #         ImmutAnyOrigin,
    #     ],
    # ):
    #     """Perform one optimization step on GPU.

    #     Note: SGD just needs params and grads.
    #     Adam needs additional moment buffers - use Adam.step_kernel directly.
    #     """
    #     ...
