# =============================================================================
# Huber Loss
# =============================================================================

from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs


struct HuberLoss[delta: Float64 = 1.0](LossFunction):
    """Huber Loss (Smooth L1): robust to outliers, useful for DQN.

    L = 0.5 * (y - t)^2                     if |y - t| <= delta
    L = delta * |y - t| - 0.5 * delta^2     otherwise

    Gradient:
    dL/dy = (y - t)                         if |y - t| <= delta
    dL/dy = delta * sign(y - t)             otherwise

    delta is a compile-time struct parameter.
    """

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
        """Huber Loss forward pass."""
        comptime SIZE = BATCH * OUT_DIM
        var loss: Float64 = 0.0
        var d = Self.delta
        var half_delta_sq = 0.5 * d * d
        for row in range(BATCH):
            for col in range(OUT_DIM):
                var diff = Float64(
                    rebind[Scalar[dtype]](output[row, col])
                ) - Float64(rebind[Scalar[dtype]](target[row, col]))
                var abs_diff = abs(diff)
                if abs_diff <= d:
                    loss += 0.5 * diff * diff
                else:
                    loss += d * abs_diff - half_delta_sq
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
        """Huber Loss backward pass: gradient dL/dy."""
        comptime SIZE = BATCH * OUT_DIM
        var d = Self.delta
        var inv_n = 1.0 / Float64(SIZE)
        for row in range(BATCH):
            for col in range(OUT_DIM):
                var diff = Float64(
                    rebind[Scalar[dtype]](output[row, col])
                ) - Float64(rebind[Scalar[dtype]](target[row, col]))
                var abs_diff = abs(diff)
                if abs_diff <= d:
                    grad[row, col] = Scalar[dtype](diff * inv_n)
                else:
                    var sign: Float64 = 1.0 if diff > 0 else -1.0
                    grad[row, col] = Scalar[dtype](d * sign * inv_n)

    # =========================================================================
    # GPU kernel implementations
    # Note: kernel params use 'd' instead of 'delta' to avoid shadowing
    # the struct's compile-time 'delta' parameter.
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
        d: Scalar[dtype],
    ):
        """Compute Huber loss using block reduction.

        Must be launched with grid_dim=(1,), block_dim=(TPB,).
        """
        var local_i = thread_idx.x
        var half_d_sq = Scalar[dtype](0.5) * d * d

        var my_value: Scalar[dtype] = 0
        var idx = Int(local_i)
        comptime SIZE = BATCH * OUT_DIM
        while idx < SIZE:
            var row = idx // OUT_DIM
            var col = idx % OUT_DIM
            var pred = rebind[Scalar[dtype]](predictions[row, col])
            var tgt = rebind[Scalar[dtype]](targets[row, col])
            var diff = pred - tgt
            var abs_diff = abs(diff)

            if abs_diff <= d:
                my_value = my_value + Scalar[dtype](0.5) * diff * diff
            else:
                my_value = my_value + d * abs_diff - half_d_sq

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
        d: Scalar[dtype],
    ):
        """Compute gradient of Huber loss."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        comptime SIZE = BATCH * OUT_DIM
        if idx >= SIZE:
            return

        var row = idx // OUT_DIM
        var col = idx % OUT_DIM
        var pred = predictions[row, col]
        var tgt = targets[row, col]
        var diff = pred - tgt
        var abs_diff = abs(diff)
        var n = Scalar[dtype](SIZE)
        var zero: predictions.element_type = 0.0

        if abs_diff <= d:
            grad_output[row, col] = diff / n
        else:
            var sign: predictions.element_type = 1.0 if diff > zero else Scalar[
                dtype
            ](-1.0)
            grad_output[row, col] = d * sign / n

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
        """Launch forward pass on GPU to compute Huber loss."""
        var d_scalar = Scalar[dtype](Self.delta)

        @always_inline
        fn kernel_wrapper(
            loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
            predictions: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            targets: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            d: Scalar[dtype],
        ):
            Self.forward_kernel_impl[BATCH, OUT_DIM](
                loss, predictions, targets, d
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            loss,
            predictions,
            targets,
            d_scalar,
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
        var d_scalar = Scalar[dtype](Self.delta)

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
            d: Scalar[dtype],
        ):
            Self.backward_kernel_impl[BATCH, OUT_DIM](
                grad_output, predictions, targets, d
            )

        comptime total = BATCH * OUT_DIM
        comptime grid_size = (total + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_output,
            predictions,
            targets,
            d_scalar,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
