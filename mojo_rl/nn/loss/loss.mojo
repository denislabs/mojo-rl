from ..constants import dtype
from layout import LayoutTensor, Layout
from std.gpu.host import DeviceContext, DeviceBuffer


trait LossFunction(Movable & ImplicitlyCopyable):
    """Base trait for loss functions.

    Loss functions are stateless pure-computation types. Hyperparameters
    (e.g., Huber delta) are compile-time struct parameters.

    All methods are @staticmethod - no instance needed.

    CPU methods use LayoutTensor for both input and output.
    GPU methods use LayoutTensor for all tensors.
    """

    @staticmethod
    fn forward[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
    ) -> Float64:
        """Forward pass: compute scalar loss value.

        Args:
            output: Model predictions [BATCH, OUT_DIM].
            target: Ground truth targets [BATCH, OUT_DIM].

        Returns:
            Scalar loss value.
        """
        ...

    @staticmethod
    fn backward[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        mut grad: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
    ):
        """Backward pass: compute gradient of loss w.r.t. output.

        Args:
            output: Model predictions [BATCH, OUT_DIM].
            target: Ground truth targets [BATCH, OUT_DIM].
            grad: Gradient [BATCH, OUT_DIM] (written).
        """
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
        mut loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
        predictions: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        targets: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        """GPU forward pass: compute loss value.

        Args:
            ctx: GPU device context.
            loss: Output [1] for scalar loss value (written).
            predictions: Predictions [BATCH, OUT_DIM].
            targets: Targets [BATCH, OUT_DIM].
        """
        ...

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        ctx: DeviceContext,
        mut grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        predictions: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        targets: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        """GPU backward pass: compute gradient of loss w.r.t. predictions.

        Args:
            ctx: GPU device context.
            grad_output: Gradient [BATCH, OUT_DIM] (written).
            predictions: Predictions [BATCH, OUT_DIM].
            targets: Targets [BATCH, OUT_DIM].
        """
        ...
