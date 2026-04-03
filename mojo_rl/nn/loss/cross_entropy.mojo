# =============================================================================
# Cross-Entropy Loss
# =============================================================================

from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, log


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

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def forward[
        BATCH: Int,
        OUT_DIM: Int,
        dtype: DType = DType.float32,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
    ) -> Float64:
        """Cross-Entropy Loss: per-sample log-softmax, averaged over batch.

        Uses log-sum-exp trick per sample for numerical stability.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var total_loss: Float64 = 0.0
        for row in range(BATCH):
            var max_val = Float64(rebind[Scalar[dtype]](output[row, 0]))
            for col in range(1, OUT_DIM):
                var val = Float64(rebind[Scalar[dtype]](output[row, col]))
                if val > max_val:
                    max_val = val
            var sum_exp: Float64 = 0.0
            for col in range(OUT_DIM):
                sum_exp += exp(
                    Float64(rebind[Scalar[dtype]](output[row, col])) - max_val
                )
            var log_sum_exp = max_val + log(sum_exp)
            var sample_loss: Float64 = 0.0
            for col in range(OUT_DIM):
                var log_sm = (
                    Float64(rebind[Scalar[dtype]](output[row, col]))
                    - log_sum_exp
                )
                sample_loss -= (
                    Float64(rebind[Scalar[dtype]](target[row, col])) * log_sm
                )
            total_loss += sample_loss
        return total_loss / Float64(BATCH)

    @staticmethod
    def backward[
        BATCH: Int,
        OUT_DIM: Int,
        dtype: DType = DType.float32,
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
        """Gradient of Cross-Entropy: dL/dy = (softmax(output) - target) / BATCH.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        for row in range(BATCH):
            var max_val = Float64(rebind[Scalar[dtype]](output[row, 0]))
            for col in range(1, OUT_DIM):
                var val = Float64(rebind[Scalar[dtype]](output[row, col]))
                if val > max_val:
                    max_val = val
            var sum_exp: Float64 = 0.0
            for col in range(OUT_DIM):
                sum_exp += exp(
                    Float64(rebind[Scalar[dtype]](output[row, col])) - max_val
                )
            for col in range(OUT_DIM):
                var sm = (
                    exp(
                        Float64(rebind[Scalar[dtype]](output[row, col]))
                        - max_val
                    )
                    / sum_exp
                )
                grad[row, col] = Scalar[dtype](
                    (sm - Float64(rebind[Scalar[dtype]](target[row, col])))
                    / Float64(BATCH)
                )

    # =========================================================================
    # GPU kernel implementations
    # =========================================================================

    @always_inline
    @staticmethod
    def forward_kernel_impl[
        BATCH: Int,
        OUT_DIM: Int,
        dtype: DType = DType.float32,
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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var local_i = thread_idx.x

        var my_value: predictions.element_type = 0.0

        comptime BATCH_SIZE = BATCH * OUT_DIM

        # Each thread processes multiple batch samples
        var batch_idx = Int(local_i)
        while batch_idx < BATCH_SIZE:
            # Find max for this sample
            var max_val = predictions[batch_idx, 0]
            for j in range(1, OUT_DIM):
                var val = predictions[batch_idx, j]
                if val > max_val:
                    max_val = val

            # Compute log_sum_exp
            var sum_exp: predictions.element_type = 0.0
            for j in range(OUT_DIM):
                var pred = predictions[batch_idx, j]
                sum_exp = sum_exp + exp(pred - max_val)
            var log_sum_exp = max_val + log(sum_exp)

            # Compute cross-entropy for this sample
            var sample_loss: predictions.element_type = 0.0
            for j in range(OUT_DIM):
                var pred = predictions[batch_idx, j]
                var tgt = targets[batch_idx, j]
                var log_softmax = pred - log_sum_exp
                sample_loss = sample_loss - tgt * log_softmax

            my_value = my_value + sample_loss
            batch_idx += TPB

        var total = block.sum[block_size=TPB, broadcast=False](val=my_value)

        if local_i == 0:
            loss[0] = total[0] / Scalar[dtype](BATCH_SIZE)

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int,
        OUT_DIM: Int,
        dtype: DType = DType.float32,
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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var batch_idx = Int(block_idx.x)
        comptime SIZE = BATCH * OUT_DIM
        if batch_idx >= SIZE:
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
        var n = Scalar[dtype](SIZE)
        for j in range(OUT_DIM):
            var pred = rebind[Scalar[dtype]](predictions[batch_idx, j])
            var tgt = rebind[Scalar[dtype]](targets[batch_idx, j])
            var softmax_val = exp(pred - max_val) / sum_exp
            grad_output[batch_idx, j] = (softmax_val - tgt) / n

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    def forward_gpu[
        BATCH: Int,
        OUT_DIM: Int,
        dtype: DType = DType.float32,
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
        """Launch forward pass on GPU to compute Cross-Entropy loss."""
        comptime assert dtype.is_floating_point(), "dtype must be floating point"

        @always_inline
        def kernel_wrapper(
            loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
            predictions: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
            targets: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH, OUT_DIM, dtype](loss, predictions, targets)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            loss,
            predictions,
            targets,
            grid_dim=(1,),
            block_dim=(TPB,),
        )

    @staticmethod
    def backward_gpu[
        BATCH: Int,
        OUT_DIM: Int,
        dtype: DType = DType.float32,
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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"

        @always_inline
        def kernel_wrapper(
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
            Self.backward_kernel_impl[BATCH, OUT_DIM, dtype](
                grad_output, predictions, targets
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_output,
            predictions,
            targets,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
