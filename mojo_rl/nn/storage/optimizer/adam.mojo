"""Adam — storage-native Adam / AdamW optimizer (CPU + GPU).

A `ParamVisitor` like `SGD`, but stateful: the per-param 1st/2nd moments live on
the `Param` itself (`m`/`v` Tensors), lazily zero-allocated on the first step.
The optimizer carries only hyperparams + the step counter `t` and its bias-
correction running powers. Decoupled weight decay (`wd > 0`, gated by the param's
`APPLY_DECAY`) makes this AdamW; `wd == 0` is plain Adam.

Usage (the step counter must be bumped ONCE per optimizer step, before the walk):

    opt.begin_step()
    model.for_each_param[target](opt, ctx)

Update (matches legacy `nn.optimizer.Adam`):
    if decay: p -= lr·wd·p
    m = β1·m + (1-β1)·g ;  v = β2·v + (1-β2)·g²
    m̂ = m / (1-β1ᵗ) ;  v̂ = v / (1-β2ᵗ)
    p -= lr · m̂ / (√v̂ + eps)
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor
from ..core.module import Module


def _adam_update_kernel[N: Int](
    param: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    v: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
    wd: Scalar[DT],
    apply_decay: Int,
):
    var i = Int(global_idx.x)
    if i >= N:
        return
    var one = Scalar[DT](1.0)
    var p = rebind[Scalar[DT]](param[i])
    if apply_decay != 0:
        p -= lr * wd * p
    var g = rebind[Scalar[DT]](grad[i])
    var m_new = beta1 * rebind[Scalar[DT]](m[i]) + (one - beta1) * g
    var v_new = beta2 * rebind[Scalar[DT]](v[i]) + (one - beta2) * g * g
    m[i] = m_new
    v[i] = v_new
    var m_hat = m_new / bc1
    var v_hat = v_new / bc2
    param[i] = p - lr * m_hat / (sqrt(v_hat) + eps)


struct Adam(ParamVisitor, Movable):
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]   # β1ᵗ (running)
    var _b2_pow: Scalar[DT]   # β2ᵗ (running)
    var bc1: Scalar[DT]       # 1 - β1ᵗ
    var bc2: Scalar[DT]       # 1 - β2ᵗ

    def __init__(
        out self,
        lr: Scalar[DT] = 1e-3,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        wd: Scalar[DT] = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.t = 0
        self._b1_pow = Scalar[DT](1.0)
        self._b2_pow = Scalar[DT](1.0)
        self.bc1 = Scalar[DT](1.0)
        self.bc2 = Scalar[DT](1.0)

    def begin_step(mut self):
        """Bump the step counter + refresh bias corrections. Call ONCE per
        optimizer step, before `model.for_each_param(self, ...)`."""
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow

    def step[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Model-walking convenience (the legacy `opt.step(model)` call style):
        bump the step then apply Adam to every Param via `for_each_param`.
        Equivalent to `opt.begin_step(); model.for_each_param[target](opt, ctx)`."""
        self.begin_step()
        model.for_each_param[target](self, ctx)

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            m.ensure(N)  # lazy zero-alloc on first step
            v.ensure(N)
            var one = Scalar[DT](1.0)
            for i in range(N):
                var p = param.data[i]
                if apply_decay:
                    p -= self.lr * self.wd * p
                var g = grad.data[i]
                var m_new = self.beta1 * m.data[i] + (one - self.beta1) * g
                var v_new = self.beta2 * v.data[i] + (one - self.beta2) * g * g
                m.data[i] = m_new
                v.data[i] = v_new
                var m_hat = m_new / self.bc1
                var v_hat = v_new / self.bc2
                param.data[i] = p - self.lr * m_hat / (sqrt(v_hat) + self.eps)
        else:
            var c = ctx.value()
            if not m.dev:  # first step: allocate + zero the moments
                m.ensure_gpu(c, N)
                m.dev.value().enqueue_fill(Scalar[DT](0))
                v.ensure_gpu(c, N)
                v.dev.value().enqueue_fill(Scalar[DT](0))
            comptime layout = Layout.row_major(N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_adam_update_kernel[N]](
                param.lt_gpu[layout](), grad.lt_gpu[layout](),
                m.lt_gpu[layout](), v.lt_gpu[layout](),
                self.lr, self.beta1, self.beta2, self.eps,
                self.bc1, self.bc2, self.wd, Int(apply_decay),
                grid_dim=nblk, block_dim=TPB,
            )
