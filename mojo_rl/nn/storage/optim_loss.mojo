"""SGDS + MSE — storage-native optimizer + loss (CPU + GPU).

All operate on `Tensor` storages with a `comptime target`. CPU reads `.data`;
GPU builds a device `LayoutTensor` via `lt_gpu[…](mut self)` and launches a
kernel (args `MutAnyOrigin` — the ABI boundary). `mse_forward` is CPU-only (a
scalar monitor; the GPU driver downloads `pred` first).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from .tensor import Tensor
from .param import ParamVisitorS


# ── kernels ────────────────────────────────────────────────────────────
def _sgd_kernel[
    N: Int
](
    param: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    lr: Scalar[DT],
    wd: Scalar[DT],
    apply_decay: Int,
):
    var i = Int(global_idx.x)
    if i < N:
        var d = grad[i]
        if apply_decay != 0:
            d += wd * param[i]
        param[i] -= lr * d


def _mse_back_kernel[
    M: Int
](
    pred: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    scale: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < M:
        grad[i] = scale * (pred[i] - tgt[i])


# ── optimizer ──────────────────────────────────────────────────────────
@fieldwise_init
struct SGDS(ParamVisitorS):
    var lr: Scalar[DT]
    var wd: Scalar[DT]

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        mut param: Tensor,
        mut grad: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            for i in range(N):
                var p = param.data[i]
                var d = grad.data[i]
                if apply_decay:
                    d += self.wd * p
                param.data[i] = p - self.lr * d
        else:
            var c = ctx.value()
            comptime layout = Layout.row_major(N)
            var pl = param.lt_gpu[layout]()
            var gl = grad.lt_gpu[layout]()
            comptime nblk = (N + 255) // 256
            c.enqueue_function[_sgd_kernel[N]](
                pl,
                gl,
                self.lr,
                self.wd,
                Int(apply_decay),
                grid_dim=nblk,
                block_dim=256,
            )


# ── loss ───────────────────────────────────────────────────────────────
def mse_forward[
    B: Int, DIM: Int
](ref pred: Tensor, ref tgt: Tensor) -> Scalar[DT]:
    """CPU monitor (reads `.data`). GPU driver downloads `pred` first."""
    var s: Scalar[DT] = 0
    for i in range(B * DIM):
        var d = pred.data[i] - tgt.data[i]
        s += d * d
    return s / Scalar[DT](B * DIM)


def mse_backward[
    target: StaticString, B: Int, DIM: Int
](
    mut pred: Tensor,
    mut tgt: Tensor,
    mut grad: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime M = B * DIM
    var scale = Scalar[DT](2) / Scalar[DT](M)
    comptime if target == "cpu":
        grad.ensure(M)
        for i in range(M):
            grad.data[i] = scale * (pred.data[i] - tgt.data[i])
    else:
        var c = ctx.value()
        grad.ensure_gpu(c, M)
        comptime layout = Layout.row_major(M)
        var pl = pred.lt_gpu[layout]()
        var tl = tgt.lt_gpu[layout]()
        var gl = grad.lt_gpu[layout]()
        comptime nblk = (M + 255) // 256
        c.enqueue_function[_mse_back_kernel[M]](
            pl, tl, gl, scale, grid_dim=nblk, block_dim=256
        )
