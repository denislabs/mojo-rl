"""GPU eval kernels: per-batch top-1 correctness + CE loss from int32 labels.

Both kernels take a `batch_idx` and write to slot `[batch_idx]` of a
device array sized `[N_VAL_BATCHES]`. No float atomics. The Trainer's
eval loop enqueues one `forward_gpu_no_cache` + `argmax_match_kernel`
+ `ce_loss_from_labels_kernel` per batch, then performs a single
`enqueue_copy` + `synchronize` after the loop and reduces both arrays
on the host.

Loss is computed directly from int32 labels (log-sum-exp formulation),
so callers need not upload one-hot validation targets.
"""

from ..constants import TPB
from layout import Layout, LayoutTensor
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives import block
from std.math import exp, log


@always_inline
def argmax_match_kernel[
    BATCH: Int,
    NUM_CLASSES: Int,
    N_VAL_BATCHES: Int,
    dtype: DType,
](
    correct_out: LayoutTensor[
        dtype, Layout.row_major(N_VAL_BATCHES), MutAnyOrigin
    ],
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, NUM_CLASSES), MutAnyOrigin
    ],
    labels_slice: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    batch_idx: Int,
):
    """Count rows where argmax(logits[r]) == labels_slice[r] for one batch.

    Launch with grid_dim=(1,), block_dim=(TPB,). Each thread handles
    ⌈BATCH/TPB⌉ rows in a strided loop; block-level sum reduction
    aggregates per-thread counts, thread 0 writes the total to
    correct_out[batch_idx] as a Scalar[dtype] (exact for BATCH up to 2^24).
    """
    var my_correct: Scalar[dtype] = 0
    var r = Int(thread_idx.x)
    while r < BATCH:
        var max_val = rebind[Scalar[dtype]](logits[r, 0])
        var best_idx: Int32 = 0
        for c in range(1, NUM_CLASSES):
            var v = rebind[Scalar[dtype]](logits[r, c])
            if v > max_val:
                max_val = v
                best_idx = Int32(c)
        var label = rebind[Scalar[DType.int32]](labels_slice[r])
        if best_idx == label:
            my_correct = my_correct + Scalar[dtype](1.0)
        r += TPB

    var total = block.sum[block_size=TPB, broadcast=False](val=my_correct)
    if thread_idx.x == 0:
        correct_out[batch_idx] = total[0]


@always_inline
def ce_loss_from_labels_kernel[
    BATCH: Int,
    NUM_CLASSES: Int,
    N_VAL_BATCHES: Int,
    dtype: DType,
](
    loss_out: LayoutTensor[
        dtype, Layout.row_major(N_VAL_BATCHES), MutAnyOrigin
    ],
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH, NUM_CLASSES), MutAnyOrigin
    ],
    labels_slice: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    batch_idx: Int,
):
    """Mean per-row cross-entropy loss from int32 labels.

    Per row: `loss_r = log_sum_exp(logits[r]) - logits[r, labels_slice[r]]`.
    Block-sum reduces across rows and thread 0 writes
    `sum_r loss_r / BATCH` to `loss_out[batch_idx]`.

    Launch with grid_dim=(1,), block_dim=(TPB,).
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    var my_loss: Scalar[dtype] = 0
    var r = Int(thread_idx.x)
    while r < BATCH:
        var max_val = rebind[Scalar[dtype]](logits[r, 0])
        for c in range(1, NUM_CLASSES):
            var v = rebind[Scalar[dtype]](logits[r, c])
            if v > max_val:
                max_val = v
        var sum_exp: Scalar[dtype] = 0
        for c in range(NUM_CLASSES):
            sum_exp = sum_exp + exp(
                rebind[Scalar[dtype]](logits[r, c]) - max_val
            )
        var lse = max_val + log(sum_exp)
        var label = Int(rebind[Scalar[DType.int32]](labels_slice[r]))
        var true_logit = rebind[Scalar[dtype]](logits[r, label])
        my_loss = my_loss + (lse - true_logit)
        r += TPB

    var total = block.sum[block_size=TPB, broadcast=False](val=my_loss)
    if thread_idx.x == 0:
        loss_out[batch_idx] = total[0] / Scalar[dtype](BATCH)
