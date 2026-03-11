# =============================================================================
# MSE Loss
# =============================================================================

from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer


struct MSELoss(LossFunction):
    """Mean Squared Error loss: L = mean((output - target)^2)."""

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

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
        """Mean Squared Error loss: L = mean((output - target)^2)."""
        comptime SIZE = BATCH * OUT_DIM
        var loss: Float64 = 0.0
        for row in range(BATCH):
            for col in range(OUT_DIM):
                var diff = Float64(
                    rebind[Scalar[dtype]](output[row, col])
                ) - Float64(rebind[Scalar[dtype]](target[row, col]))
                loss += diff * diff
        return loss / Float64(SIZE)

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
        """Gradient of MSE loss: dL/dy = 2 * (output - target) / size."""
        comptime SIZE = BATCH * OUT_DIM
        for row in range(BATCH):
            for col in range(OUT_DIM):
                var diff = Float64(
                    rebind[Scalar[dtype]](output[row, col])
                ) - Float64(rebind[Scalar[dtype]](target[row, col]))
                grad[row, col] = Scalar[dtype](2.0 * diff / Float64(SIZE))

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
            loss[0] = total[0] / Scalar[dtype](SIZE)

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
        grad_output[row, col] = 2.0 * (pred - target) / Scalar[dtype](SIZE)

    # =========================================================================
    # GPU launchers
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
        """Launch forward pass on GPU to compute MSE loss."""

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
        """Launch backward pass on GPU to compute loss gradient."""

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

        comptime total = BATCH * OUT_DIM
        comptime grid_size = (total + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_output,
            predictions,
            targets,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
