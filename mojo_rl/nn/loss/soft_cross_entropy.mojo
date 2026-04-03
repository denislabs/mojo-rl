# =============================================================================
# Soft Cross-Entropy Loss
# =============================================================================

from ..constants import dtype, TPB
from .loss import LossFunction
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, log


struct SoftCrossEntropyLoss(LossFunction):
    """Soft Cross-Entropy Loss for distributional RL (TDMPC2).

    Used for reward and Q-value heads where targets are two-hot vectors
    (soft probability distributions, not hard one-hot vectors).

    L = -sum_i target_i * log_softmax(logits)_i
      = -sum_i target_i * (logits_i - log_sum_exp(logits))

    Uses log-sum-exp trick for numerical stability.

    Gradient:
        dL/dy_i = softmax(logits)_i - target_i

    This is identical in form to CrossEntropyLoss but the docstring emphasizes
    that soft (non-integer) targets are expected, as in two-hot encoding.
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
        """Soft cross-entropy: per-sample L = -sum(target * log_softmax(output)), averaged over batch.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var total_loss: Float64 = 0.0
        for row in range(BATCH):
            var max_val = Float64(rebind[Scalar[dtype]](output[row, 0]))
            for col in range(1, OUT_DIM):
                var v = Float64(rebind[Scalar[dtype]](output[row, col]))
                if v > max_val:
                    max_val = v
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
        """Gradient: dL/dy = (softmax(output) - target) / BATCH."""
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        for row in range(BATCH):
            var max_val = Float64(rebind[Scalar[dtype]](output[row, 0]))
            for col in range(1, OUT_DIM):
                var v = Float64(rebind[Scalar[dtype]](output[row, col]))
                if v > max_val:
                    max_val = v
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
        """Compute soft cross-entropy using block reduction.

        Must be launched with grid_dim=(1,), block_dim=(TPB,).
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var local_i = thread_idx.x
        var my_value: Scalar[dtype] = 0

        var batch_idx = Int(local_i)
        while batch_idx < BATCH:
            # Find max for this sample
            var max_val = rebind[Scalar[dtype]](predictions[batch_idx, 0])
            for j in range(1, OUT_DIM):
                var v = rebind[Scalar[dtype]](predictions[batch_idx, j])
                if v > max_val:
                    max_val = v

            # Compute log_sum_exp
            var sum_exp: Scalar[dtype] = 0.0
            for j in range(OUT_DIM):
                var pred = rebind[Scalar[dtype]](predictions[batch_idx, j])
                sum_exp = sum_exp + exp(pred - max_val)
            var log_sum_exp = max_val + log(sum_exp)

            # Compute soft cross-entropy for this sample
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
        """Gradient: dL/dy = (softmax(output) - target) / BATCH.

        Grid: (BATCH,), Block: (1,)
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return
        if thread_idx.x != 0:
            return

        var max_val = rebind[Scalar[dtype]](predictions[batch_idx, 0])
        for j in range(1, OUT_DIM):
            var v = rebind[Scalar[dtype]](predictions[batch_idx, j])
            if v > max_val:
                max_val = v

        var sum_exp: Scalar[dtype] = 0.0
        for j in range(OUT_DIM):
            var pred = rebind[Scalar[dtype]](predictions[batch_idx, j])
            sum_exp = sum_exp + exp(pred - max_val)

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
        """Launch forward pass on GPU."""
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
        """Launch backward pass on GPU."""
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
