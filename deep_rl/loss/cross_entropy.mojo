from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, block
from gpu.host import DeviceContext, DeviceBuffer
from math import exp, log


struct CrossEntropyLoss(LossFunction):
    """Cross-Entropy Loss for classification/policy gradients.

    Works with one-hot encoded targets or soft targets (probability distributions).

    For logits input:
        L = -sum(target * log_softmax(output))
        L = -sum(target * (output - log_sum_exp(output)))

    Uses log-sum-exp trick for numerical stability.

    Gradient:
        dL/dy = softmax(output) - target

    This is suitable for policy gradient methods where:
    - output: logits [num_actions]
    - target: one-hot encoded action or action probabilities [num_actions]
    """

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn forward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
    ) -> Float64:
        """Cross-Entropy Loss: L = -sum(target * log_softmax(output)).

        Uses log-sum-exp for numerical stability.
        """
        # Find max for numerical stability
        var max_val = Float64(output[0])
        for i in range(1, SIZE):
            var val = Float64(output[i])
            if val > max_val:
                max_val = val

        # Compute log_sum_exp
        var sum_exp: Float64 = 0.0
        for i in range(SIZE):
            sum_exp += exp(Float64(output[i]) - max_val)
        var log_sum_exp = max_val + log(sum_exp)

        # Compute cross-entropy: -sum(target * (output - log_sum_exp))
        var loss: Float64 = 0.0
        for i in range(SIZE):
            var log_softmax = Float64(output[i]) - log_sum_exp
            loss -= Float64(target[i]) * log_softmax

        return loss

    fn backward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
        mut grad: InlineArray[Scalar[dtype], SIZE],
    ):
        """Gradient of Cross-Entropy: dL/dy = softmax(output) - target."""
        # Find max for numerical stability
        var max_val = Float64(output[0])
        for i in range(1, SIZE):
            var val = Float64(output[i])
            if val > max_val:
                max_val = val

        # Compute softmax
        var sum_exp: Float64 = 0.0
        for i in range(SIZE):
            sum_exp += exp(Float64(output[i]) - max_val)

        for i in range(SIZE):
            var softmax_val = exp(Float64(output[i]) - max_val) / sum_exp
            grad[i] = Scalar[dtype](softmax_val - Float64(target[i]))

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
    ):
        """Compute Cross-Entropy loss using block reduction.

        Each sample's loss is computed, then summed across batch.
        Must be launched with grid_dim=(1,), block_dim=(TPB,).
        """
        var local_i = thread_idx.x

        var my_value: Scalar[dtype] = 0

        # Each thread processes multiple batch samples
        var batch_idx = Int(local_i)
        while batch_idx < BATCH:
            # Find max for this sample
            var max_val = rebind[Scalar[dtype]](predictions[batch_idx, 0])
            for j in range(1, OUT_DIM):
                var val = rebind[Scalar[dtype]](predictions[batch_idx, j])
                if val > max_val:
                    max_val = val

            # Compute log_sum_exp
            var sum_exp: Scalar[dtype] = 0.0
            for j in range(OUT_DIM):
                var pred = rebind[Scalar[dtype]](predictions[batch_idx, j])
                sum_exp = sum_exp + exp(pred - max_val)
            var log_sum_exp = max_val + log(sum_exp)

            # Compute cross-entropy for this sample
            var sample_loss: Scalar[dtype] = 0.0
            for j in range(OUT_DIM):
                var pred = rebind[Scalar[dtype]](predictions[batch_idx, j])
                var tgt = rebind[Scalar[dtype]](targets[batch_idx, j])
                var log_softmax = pred - log_sum_exp
                sample_loss = sample_loss - tgt * log_softmax

            my_value = my_value + sample_loss
            batch_idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_value)

        if local_i == 0:
            loss[0] = total[0] / Scalar[dtype](BATCH)

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
        """Compute gradient: dL/dy = (softmax(output) - target) / BATCH.

        Each block handles one sample (needs per-sample softmax computation).
        Grid: (BATCH,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        if thread_idx.x != 0:
            return

        # Find max for numerical stability
        var max_val = rebind[Scalar[dtype]](predictions[batch_idx, 0])
        for j in range(1, OUT_DIM):
            var val = rebind[Scalar[dtype]](predictions[batch_idx, j])
            if val > max_val:
                max_val = val

        # Compute softmax sum
        var sum_exp: Scalar[dtype] = 0.0
        for j in range(OUT_DIM):
            var pred = rebind[Scalar[dtype]](predictions[batch_idx, j])
            sum_exp = sum_exp + exp(pred - max_val)

        # Compute gradient
        var n = Scalar[dtype](BATCH)
        for j in range(OUT_DIM):
            var pred = rebind[Scalar[dtype]](predictions[batch_idx, j])
            var tgt = rebind[Scalar[dtype]](targets[batch_idx, j])
            var softmax_val = exp(pred - max_val) / sum_exp
            grad_output[batch_idx, j] = (softmax_val - tgt) / n

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
        """Launch forward pass on GPU to compute Cross-Entropy loss.

        Args:
            ctx: GPU device context.
            loss_buf: Output buffer [1] for scalar loss value.
            predictions_buf: Logits buffer [BATCH * OUT_DIM].
            targets_buf: One-hot or soft targets buffer [BATCH * OUT_DIM].
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

        # Single block for reduction
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
            predictions_buf: Logits buffer [BATCH * OUT_DIM].
            targets_buf: One-hot or soft targets buffer [BATCH * OUT_DIM].
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

        # One block per sample for softmax computation
        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_output,
            predictions,
            targets,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
