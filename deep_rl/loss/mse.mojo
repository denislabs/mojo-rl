from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.primitives import block
from gpu.host import DeviceContext, DeviceBuffer


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

    # =========================================================================
    # GPU kernel implementations (inlinable for fusion)
    # =========================================================================

    @always_inline
    @staticmethod
    fn forward_kernel_impl[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
        predictions: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        targets: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
    ):
        """Compute MSE loss using block reduction.

        Must be launched with grid_dim=(1,), block_dim=(TPB,).
        """
        var local_i = thread_idx.x

        var my_value: Scalar[dtype] = 0
        var idx = Int(local_i)
        comptime SIZE = BATCH * OUT_DIM
        while idx < SIZE:
            var row = idx // OUT_DIM
            var col = idx % OUT_DIM
            var diff = rebind[Scalar[dtype]](predictions[row, col]) - rebind[
                Scalar[dtype]
            ](targets[row, col])
            my_value += diff * diff
            idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_value)

        if local_i == 0:
            loss[0] = total[0] / SIZE

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        predictions: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        targets: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
    ):
        """Compute gradient of MSE loss: dL/dy = 2 * (pred - target) / N."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        comptime SIZE = BATCH * OUT_DIM
        if idx >= SIZE:
            return

        var row = idx // OUT_DIM
        var col = idx % OUT_DIM
        var pred = predictions[row, col]
        var target = targets[row, col]
        grad_output[row, col] = 2.0 * (pred - target) / SIZE

    # =========================================================================
    # GPU launchers
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
        """Launch forward pass on GPU to compute MSE loss.

        Args:
            ctx: GPU device context.
            loss_buf: Output buffer [1] for scalar loss value.
            predictions_buf: Predictions buffer [BATCH * OUT_DIM].
            targets_buf: Targets buffer [BATCH * OUT_DIM].
        """
        # Create LayoutTensor views
        var loss = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            loss_buf.unsafe_ptr()
        )
        var predictions = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](predictions_buf.unsafe_ptr())
        var targets = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](targets_buf.unsafe_ptr())

        # Kernel wrapper with explicit parameters
        @always_inline
        fn kernel_wrapper(
            loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
            predictions: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            targets: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH, OUT_DIM](loss, predictions, targets)

        # Launch with single block for reduction
        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            loss,
            predictions,
            targets,
            grid_dim=(1,),
            block_dim=(TPB,),
        )

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
        """Launch backward pass on GPU to compute loss gradient.

        Args:
            ctx: GPU device context.
            grad_output_buf: Gradient buffer [BATCH * OUT_DIM] (written).
            predictions_buf: Predictions buffer [BATCH * OUT_DIM].
            targets_buf: Targets buffer [BATCH * OUT_DIM].
        """
        # Create LayoutTensor views
        var grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var predictions = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](predictions_buf.unsafe_ptr())
        var targets = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](targets_buf.unsafe_ptr())

        # Kernel wrapper with explicit parameters
        @always_inline
        fn kernel_wrapper(
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            predictions: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            targets: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH, OUT_DIM](
                grad_output, predictions, targets
            )

        # Launch with enough threads
        comptime total = BATCH * OUT_DIM
        comptime grid_size = (total + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_output,
            predictions,
            targets,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
