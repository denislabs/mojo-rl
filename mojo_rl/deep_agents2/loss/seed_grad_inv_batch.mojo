"""Seed gradient tile of shape [BATCH, 1] with 1/BATCH.

This is the standard "seed the backward pass for a mean-over-batch
loss" helper: when the forward output is `loss_per_b` of shape
[BATCH, 1] and the trainer wants `d(mean_b(loss_per_b))/d(loss_per_b)`,
that gradient is the constant `1/BATCH` in every slot.

Replaces the inline `_fill_constant_kernel` previously open-coded in
`sac_actor_loss.mojo`. Both CPU and GPU paths.

Phase 3 — Lives in `loss/` because mean-batch backward is the loss
contract. Other primitives that need a constant-fill should call this
helper rather than re-rolling.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT


def _seed_inv_batch_kernel[N: Int](
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    value: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = value


def seed_grad_inv_batch[
    target: StaticString,
    BATCH: Int,
](
    grad_out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fill `grad_out_ptr[0:BATCH]` with `1/BATCH`.

    Caller-provided `ctx` is required for `target='gpu'`; ignored for
    `target='cpu'`.
    """
    var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    comptime if target == "cpu":
        for b in range(BATCH):
            grad_out_ptr[b] = inv_b
    else:
        var c = ctx.value()
        var buf_lt = LayoutTensor[
            DT, Layout.row_major(BATCH), MutAnyOrigin,
        ](grad_out_ptr)
        comptime TPB = 128
        comptime n_blocks = (BATCH + TPB - 1) // TPB
        comptime kernel = _seed_inv_batch_kernel[BATCH]
        c.enqueue_function[kernel](
            buf_lt, inv_b, grid_dim=n_blocks, block_dim=TPB,
        )
