"""SAC math helpers — polyak soft-update + target-y (storage surface).

`polyak_tensor` is the per-Tensor soft-update used by `Module.polyak_from`
overrides (Linear params; combinators recurse). `sac_target_y` builds the
TD target from a batch (pure elementwise arithmetic). Both CPU + GPU.

  polyak:   dst = tau·src + (1-tau)·dst
  target_y: y[b] = r[b] + gamma·(1-done[b])·(min_q[b] - alpha·log_prob[b])
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor


def _polyak_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    tau: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = tau * rebind[Scalar[DT]](src[i]) + (Scalar[DT](1.0) - tau) * rebind[Scalar[DT]](dst[i])


def polyak_tensor[
    target: StaticString, N: Int
](mut dst: Tensor, mut src: Tensor, tau: Scalar[DT], ctx: Optional[DeviceContext]) raises:
    comptime if target == "cpu":
        for i in range(N):
            dst.data[i] = tau * src.data[i] + (Scalar[DT](1.0) - tau) * dst.data[i]
    else:
        var c = ctx.value()
        comptime nblk = (N + TPB - 1) // TPB
        c.enqueue_function[_polyak_kernel[N]](
            dst.lt_gpu[Layout.row_major(N)](), src.lt_gpu[Layout.row_major(N)](), tau,
            grid_dim=nblk, block_dim=TPB,
        )


def _target_y_kernel[B: Int](
    r: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    done: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    min_q: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    logp: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    y: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    gamma: Scalar[DT],
    alpha: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b < B:
        var soft = rebind[Scalar[DT]](min_q[b]) - alpha * rebind[Scalar[DT]](logp[b])
        y[b] = rebind[Scalar[DT]](r[b]) + gamma * (
            Scalar[DT](1.0) - rebind[Scalar[DT]](done[b])
        ) * soft


def sac_target_y[
    target: StaticString, B: Int
](
    mut r: Tensor, mut done: Tensor, mut min_q: Tensor, mut logp: Tensor,
    gamma: Scalar[DT], alpha: Scalar[DT], mut y: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """y[b] = r + gamma·(1-done)·(min_q - alpha·log_prob).  min_q/logp from the
    TARGET critics + actor on the NEXT state (caller detaches — no grad here)."""
    comptime if target == "cpu":
        y.ensure(B)
        for b in range(B):
            var soft = min_q.data[b] - alpha * logp.data[b]
            y.data[b] = r.data[b] + gamma * (Scalar[DT](1.0) - done.data[b]) * soft
    else:
        var c = ctx.value()
        y.ensure_gpu(c, B)
        comptime nblk = (B + TPB - 1) // TPB
        c.enqueue_function[_target_y_kernel[B]](
            r.lt_gpu[Layout.row_major(B)](), done.lt_gpu[Layout.row_major(B)](),
            min_q.lt_gpu[Layout.row_major(B)](), logp.lt_gpu[Layout.row_major(B)](),
            y.lt_gpu[Layout.row_major(B)](), gamma, alpha,
            grid_dim=nblk, block_dim=TPB,
        )
