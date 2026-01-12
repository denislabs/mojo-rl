from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim


@fieldwise_init
struct MSELoss(LossFunction):
    """Mean Squared Error loss: L = mean((output - target)^2)."""

    fn forward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
    ) -> Float64:
        """Mean Squared Error loss: L = mean((output - target)^2)."""
        var loss: Float64 = 0.0
        for i in range(SIZE):
            var diff = Float64(output[i]) - Float64(target[i])
            loss += diff * diff
        return loss / SIZE

    fn backward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
        mut grad: InlineArray[Scalar[dtype], SIZE],
    ):
        """Gradient of MSE loss: dL/dy = 2 * (output - target) / size."""
        for i in range(SIZE):
            var diff = Float64(output[i]) - Float64(target[i])
            grad[i] = Scalar[dtype](2.0 * diff / SIZE)


fn mse_loss_backward_kernel[
    BATCH: Int,
    OUT_DIM: Int,
](
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    predictions: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
    targets: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Compute gradient of MSE loss: dL/dy = 2 * (pred - target) / N."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * OUT_DIM:
        return

    var row = idx // OUT_DIM
    var col = idx % OUT_DIM
    var pred = predictions[row, col]
    var target = targets[row, col]
    grad_output[row, col] = 2.0 * (pred - target) / (BATCH * OUT_DIM)


fn mse_loss_kernel[
    BATCH: Int,
    OUT_DIM: Int,
](
    loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    predictions: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
    targets: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Compute MSE loss using block reduction."""
    from gpu import block

    var local_i = thread_idx.x

    var my_value: predictions.element_type = 0
    var idx = Int(local_i)
    while idx < BATCH * OUT_DIM:
        var row = idx // OUT_DIM
        var col = idx % OUT_DIM
        var diff = predictions[row, col] - targets[row, col]
        my_value += diff * diff
        idx += TPB

    var total = block.sum[block_size=TPB, broadcast=False](val=my_value)

    if local_i == 0:
        loss[0] = total[0] / (BATCH * OUT_DIM)
