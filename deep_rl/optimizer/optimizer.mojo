# =============================================================================
# Optimizer Trait
# =============================================================================

from ..constants import dtype


trait Optimizer(Movable & ImplicitlyCopyable):
    """Base trait for optimizers.

    Optimizers update parameters using gradients. They only need to know
    the total parameter size to operate on flattened parameter arrays.
    """

    comptime PARAM_SIZE: Int
    comptime GRAD_SIZE: Int

    fn step(
        mut self,
        mut params: InlineArray[Scalar[dtype], Self.PARAM_SIZE],
        grads: InlineArray[Scalar[dtype], Self.PARAM_SIZE],
    ):
        """Perform one optimization step.

        Args:
            params: Flattened parameters to update (modified in place).
            grads: Flattened gradients.
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
