"""REDQ-OFE kernels — auxiliary MSE loss + its gradient.

Ports the two scalar primitives from
`mojo_rl/deep_agents/redq_ofe/kernels.mojo` to the deep_agents surface.
The grad kernel is the load-bearing one — it seeds the backward chain
through `OFEPredictorHead → OFEActionBranch → OFEStateBranch`. The loss
kernel is a diagnostics-only host reduction.

GPU variants ship alongside the trainer GPU port (G.5). REDQ-OFE
doesn't capture under CUDA graphs (host control flow with subset
sampling + policy delay + aux interleave), so the per-step D2H of
the pred + target for the diagnostic loss is cheap.

Math:
    loss = (1 / (BATCH * OBS)) * Σ_b Σ_d (pred[b,d] - target[b,d])^2
    d loss / d pred[b,d] = 2 * (pred[b,d] - target[b,d]) / (BATCH * OBS)
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..training.trainer_block import TrainerState
from mojo_rl.nn.constants import DT, TPB


def aux_mse_grad_cpu[
    BATCH: Int, OBS: Int,
](
    pred_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_pred_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Writes `grad_pred[b,d] = 2 * (pred[b,d] - target[b,d]) / (BATCH * OBS)`
    over the full `BATCH * OBS` slab. Overwrites grad_pred."""
    var scale = Scalar[DT](2.0) / Scalar[DT](BATCH * OBS)
    for b in range(BATCH):
        for d in range(OBS):
            var off = b * OBS + d
            grad_pred_ptr[off] = scale * (pred_ptr[off] - target_ptr[off])


def aux_mse_loss_cpu[
    BATCH: Int, OBS: Int,
](
    pred_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    """Returns the scalar `(1 / (BATCH * OBS)) * Σ (pred - target)^2`.
    Pure host reduction — diagnostics only (the backward chain doesn't
    use this value, it uses the gradient kernel above)."""
    var s: Scalar[DT] = Scalar[DT](0.0)
    for b in range(BATCH):
        for d in range(OBS):
            var off = b * OBS + d
            var diff = pred_ptr[off] - target_ptr[off]
            s += diff * diff
    return s / Scalar[DT](BATCH * OBS)


# ────────────────────────────────────────────────────────────────────
# GPU MSE-grad kernel.
# ────────────────────────────────────────────────────────────────────


def _aux_mse_grad_kernel[
    BATCH: Int, OBS: Int,
](
    pred: LayoutTensor[
        DT, Layout.row_major(BATCH, OBS), MutAnyOrigin,
    ],
    target: LayoutTensor[
        DT, Layout.row_major(BATCH, OBS), MutAnyOrigin,
    ],
    grad_pred: LayoutTensor[
        DT, Layout.row_major(BATCH, OBS), MutAnyOrigin,
    ],
    scale: Scalar[DT],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OBS
    if idx >= total:
        return
    var b = idx // OBS
    var d = idx % OBS
    grad_pred[b, d] = scale * (
        rebind[Scalar[DT]](pred[b, d])
        - rebind[Scalar[DT]](target[b, d])
    )


def aux_mse_grad_gpu[
    BATCH: Int, OBS: Int,
](
    ctx: DeviceContext,
    pred_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_pred_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    """Device variant of `aux_mse_grad_cpu`. Writes
    `grad_pred[b,d] = 2·(pred[b,d] - target[b,d]) / (BATCH·OBS)`."""
    var scale = Scalar[DT](2.0) / Scalar[DT](BATCH * OBS)
    var pred_lt = LayoutTensor[
        DT, Layout.row_major(BATCH, OBS), MutAnyOrigin,
    ](pred_ptr)
    var target_lt = LayoutTensor[
        DT, Layout.row_major(BATCH, OBS), MutAnyOrigin,
    ](target_ptr)
    var grad_lt = LayoutTensor[
        DT, Layout.row_major(BATCH, OBS), MutAnyOrigin,
    ](grad_pred_ptr)
    comptime total = BATCH * OBS
    comptime n_blocks = (total + TPB - 1) // TPB
    comptime kernel = _aux_mse_grad_kernel[BATCH, OBS]
    ctx.enqueue_function[kernel](
        pred_lt, target_lt, grad_lt, scale,
        grid_dim=n_blocks, block_dim=TPB,
    )
