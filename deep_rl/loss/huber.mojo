from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, block
from gpu.host import DeviceContext, DeviceBuffer


fn _abs(val: Scalar[dtype]) -> Scalar[dtype]:
    """Absolute value helper for Scalar[dtype]."""
    var zero = Scalar[dtype](0.0)
    return val if val >= zero else -val


fn _abs_f64(val: Float64) -> Float64:
    """Absolute value helper for Float64."""
    return val if val >= 0.0 else -val


struct HuberLoss(LossFunction):
    """Huber Loss (Smooth L1): robust to outliers, useful for DQN.

    L = 0.5 * (y - t)^2                     if |y - t| <= delta
    L = delta * |y - t| - 0.5 * delta^2     otherwise

    Gradient:
    dL/dy = (y - t)                         if |y - t| <= delta
    dL/dy = delta * sign(y - t)             otherwise
    """

    var delta: Float64

    fn __init__(out self, delta: Float64 = 1.0):
        """Initialize HuberLoss with delta threshold.

        Args:
            delta: Threshold for switching between quadratic and linear loss.
                   Default is 1.0 (standard Smooth L1).
        """
        self.delta = delta

    fn __moveinit__(out self, deinit other: Self):
        self.delta = other.delta

    fn __copyinit__(out self, other: Self):
        self.delta = other.delta

    fn forward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
    ) -> Float64:
        """Huber Loss forward pass."""
        var loss: Float64 = 0.0
        var delta = self.delta
        var half_delta_sq = 0.5 * delta * delta

        for i in range(SIZE):
            var diff = Float64(output[i]) - Float64(target[i])
            var abs_diff = _abs_f64(diff)
            if abs_diff <= delta:
                # Quadratic region
                loss += 0.5 * diff * diff
            else:
                # Linear region
                loss += delta * abs_diff - half_delta_sq

        return loss / SIZE

    fn backward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
        mut grad: InlineArray[Scalar[dtype], SIZE],
    ):
        """Huber Loss backward pass: gradient dL/dy."""
        var delta = self.delta

        for i in range(SIZE):
            var diff = Float64(output[i]) - Float64(target[i])
            var abs_diff = _abs_f64(diff)
            if abs_diff <= delta:
                # Quadratic region: gradient is (y - t)
                grad[i] = Scalar[dtype](diff / SIZE)
            else:
                # Linear region: gradient is delta * sign(diff)
                var sign: Float64 = 1.0 if diff > 0 else -1.0
                grad[i] = Scalar[dtype](delta * sign / SIZE)

    # =========================================================================
    # GPU kernel implementations
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
        delta: Scalar[dtype],
    ):
        """Compute Huber loss using block reduction.

        Must be launched with grid_dim=(1,), block_dim=(TPB,).
        """
        var local_i = thread_idx.x
        var half_delta_sq = Scalar[dtype](0.5) * delta * delta

        var my_value: Scalar[dtype] = 0
        var idx = Int(local_i)
        while idx < BATCH * OUT_DIM:
            var row = idx // OUT_DIM
            var col = idx % OUT_DIM
            var pred = rebind[Scalar[dtype]](predictions[row, col])
            var target = rebind[Scalar[dtype]](targets[row, col])
            var diff = pred - target
            var abs_diff = _abs(diff)

            if abs_diff <= delta:
                my_value = my_value + Scalar[dtype](0.5) * diff * diff
            else:
                my_value = my_value + delta * abs_diff - half_delta_sq

            idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_value)

        if local_i == 0:
            loss[0] = total[0] / Scalar[dtype](BATCH * OUT_DIM)

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
        delta: Scalar[dtype],
    ):
        """Compute gradient of Huber loss."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * OUT_DIM:
            return

        var row = idx // OUT_DIM
        var col = idx % OUT_DIM
        var pred = rebind[Scalar[dtype]](predictions[row, col])
        var target = rebind[Scalar[dtype]](targets[row, col])
        var diff = pred - target
        var abs_diff = _abs(diff)
        var n = Scalar[dtype](BATCH * OUT_DIM)
        var zero = Scalar[dtype](0.0)

        if abs_diff <= delta:
            # Quadratic region
            grad_output[row, col] = diff / n
        else:
            # Linear region
            var sign: Scalar[dtype] = Scalar[dtype](1.0) if diff > zero else Scalar[dtype](-1.0)
            grad_output[row, col] = delta * sign / n

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
        """Launch forward pass on GPU (trait-compatible, uses delta=1.0)."""
        Self._forward_gpu_impl[BATCH, OUT_DIM](ctx, loss_buf, predictions_buf, targets_buf, 1.0)

    @staticmethod
    fn _forward_gpu_impl[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        ctx: DeviceContext,
        loss_buf: DeviceBuffer[dtype],
        predictions_buf: DeviceBuffer[dtype],
        targets_buf: DeviceBuffer[dtype],
        delta: Float64,
    ) raises:
        """Launch forward pass on GPU to compute Huber loss.

        Args:
            ctx: GPU device context.
            loss_buf: Output buffer [1] for scalar loss value.
            predictions_buf: Predictions buffer [BATCH * OUT_DIM].
            targets_buf: Targets buffer [BATCH * OUT_DIM].
            delta: Huber loss delta threshold.
        """
        var loss = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            loss_buf.unsafe_ptr()
        )
        var predictions = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](predictions_buf.unsafe_ptr())
        var targets = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](targets_buf.unsafe_ptr())
        var delta_scalar = Scalar[dtype](delta)

        @always_inline
        fn kernel_wrapper(
            loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
            predictions: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            targets: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            delta: Scalar[dtype],
        ):
            Self.forward_kernel_impl[BATCH, OUT_DIM](
                loss, predictions, targets, delta
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            loss,
            predictions,
            targets,
            delta_scalar,
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
        """Launch backward pass on GPU (trait-compatible, uses delta=1.0)."""
        Self._backward_gpu_impl[BATCH, OUT_DIM](ctx, grad_output_buf, predictions_buf, targets_buf, 1.0)

    @staticmethod
    fn _backward_gpu_impl[
        BATCH: Int,
        OUT_DIM: Int,
    ](
        ctx: DeviceContext,
        grad_output_buf: DeviceBuffer[dtype],
        predictions_buf: DeviceBuffer[dtype],
        targets_buf: DeviceBuffer[dtype],
        delta: Float64,
    ) raises:
        """Launch backward pass on GPU to compute loss gradient.

        Args:
            ctx: GPU device context.
            grad_output_buf: Gradient buffer [BATCH * OUT_DIM] (written).
            predictions_buf: Predictions buffer [BATCH * OUT_DIM].
            targets_buf: Targets buffer [BATCH * OUT_DIM].
            delta: Huber loss delta threshold.
        """
        var grad_output = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var predictions = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](predictions_buf.unsafe_ptr())
        var targets = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](targets_buf.unsafe_ptr())
        var delta_scalar = Scalar[dtype](delta)

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
            delta: Scalar[dtype],
        ):
            Self.backward_kernel_impl[BATCH, OUT_DIM](
                grad_output, predictions, targets, delta
            )

        comptime total = BATCH * OUT_DIM
        comptime grid_size = (total + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_output,
            predictions,
            targets,
            delta_scalar,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
