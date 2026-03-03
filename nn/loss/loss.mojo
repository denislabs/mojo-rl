from ..constants import dtype
from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer


trait LossFunction(Movable & ImplicitlyCopyable):
    """Base trait for loss functions.

    Loss functions have:
    - forward() for computing loss
    - backward() for computing gradients
    """

    fn forward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
    ) -> Float64:
        """Forward pass for loss function."""
        ...

    fn backward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
        mut grad: InlineArray[Scalar[dtype], SIZE],
    ):
        """Backward pass for loss function."""
        ...

    # =========================================================================
    # GPU methods
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        ctx: DeviceContext,
        loss_buf: DeviceBuffer[dtype],
        predictions_buf: DeviceBuffer[dtype],
        targets_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass: compute loss value.

        Args:
            ctx: GPU device context.
            loss_buf: Output buffer [1] for scalar loss value.
            predictions_buf: Predictions buffer [BATCH * OUT_DIM].
            targets_buf: Targets buffer [BATCH * OUT_DIM].
        """
        ...

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        ctx: DeviceContext,
        grad_output_buf: DeviceBuffer[dtype],
        predictions_buf: DeviceBuffer[dtype],
        targets_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass: compute gradient of loss w.r.t. predictions.

        Args:
            ctx: GPU device context.
            grad_output_buf: Gradient buffer [BATCH * OUT_DIM] (written).
            predictions_buf: Predictions buffer [BATCH * OUT_DIM].
            targets_buf: Targets buffer [BATCH * OUT_DIM].
        """
        ...
