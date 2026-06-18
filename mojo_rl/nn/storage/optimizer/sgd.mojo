"""SGD + MSE — storage-native optimizer + loss (CPU + GPU).

All operate on `Tensor` storages with a `comptime target`. CPU reads `.data`;
GPU builds a device `LayoutTensor` via `lt_gpu[…](mut self)` and launches a
kernel (args `MutAnyOrigin` — the ABI boundary). `mse_forward` is CPU-only (a
scalar monitor; the GPU driver downloads `pred` first).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor
from ..core.param import ParamVisitor


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


# ── optimizer ──────────────────────────────────────────────────────────
@fieldwise_init
struct SGD(ParamVisitor):
    var lr: Scalar[DT]
    var wd: Scalar[DT]

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,   # unused (SGD is stateless)
        mut v: Tensor,   # unused
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


