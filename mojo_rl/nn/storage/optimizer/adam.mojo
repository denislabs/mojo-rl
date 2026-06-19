"""Adam — storage-native Adam / AdamW optimizer (CPU + GPU).

A `ParamVisitor`, but stateful: the per-param 1st/2nd moments live on the `Param`
(`m`/`v` Tensors), lazily zero-allocated. Hyperparams + step counter `t` + bias-
correction running powers. Decoupled weight decay (`wd > 0`, gated by the param's
`APPLY_DECAY`) makes this AdamW; `wd == 0` is plain Adam.

Two GPU execution modes, ONE class, IDENTICAL math (bit-parity, gated):

  - per-param (default, CPU + GPU): `step` walks `for_each_param`, one kernel /
    CPU loop per Param. The universal correctness path; CPU↔GPU parity check.
  - arena (GPU, opt-in via `adopt`): `adopt[target](model)` packs every param into
    4 CONTIGUOUS device buffers (val/grd/m/v) and rebinds each Param's val/grd to
    `create_sub_buffer` slices — so forward reads / backward writes the arena
    transparently and grads land contiguous. `step` is then ONE flat kernel over
    `[0,total)`. Collapses N launches → 1, runs on Apple AND NVIDIA (the arenas
    are passed as their own buffers — no per-param address table). `adopt` is a
    NO-OP on CPU, so agent code is identical on both targets:

        opt.adopt[target](model, ctx)     # GPU: pack arena;  CPU: nothing
        opt.step[target](model, ctx)      # GPU+adopted: 1 kernel;  else per-param

Lifetime: the optimizer owns the arenas; the model's param slices reference them
(DeviceBuffer is refcounted → destruction order is safe).
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor
from ..core.module import Module
from ..core.named_params import named_params


def _adam_update_kernel[
    N: Int
](
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
    """Per-param update (one Param, comptime size N)."""
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


def _grouped_adam_kernel(
    val: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grd: UnsafePointer[Scalar[DT], MutAnyOrigin],
    m: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v: UnsafePointer[Scalar[DT], MutAnyOrigin],
    decay: UnsafePointer[Scalar[DT], MutAnyOrigin],
    total: Int,
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
    wd: Scalar[DT],
):
    """Arena update (all params at once over runtime-length flat buffers).
    `decay[i]` (0/1) gates AdamW's decoupled decay per element. UnsafePointer is
    required for runtime-length flat indexing — the accepted GPU-ABI boundary, NOT
    a per-param address table."""
    var i = Int(global_idx.x)
    if i >= total:
        return
    var one = Scalar[DT](1.0)
    var p = val[i]
    if decay[i] != Scalar[DT](0.0):
        p -= lr * wd * p
    var g = grd[i]
    var m_new = beta1 * m[i] + (one - beta1) * g
    var v_new = beta2 * v[i] + (one - beta2) * g * g
    m[i] = m_new
    v[i] = v_new
    var m_hat = m_new / bc1
    var v_hat = v_new / bc2
    val[i] = p - lr * m_hat / (sqrt(v_hat) + eps)


struct Adam(Movable, ParamVisitor):
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]  # β1ᵗ (running)
    var _b2_pow: Scalar[DT]  # β2ᵗ (running)
    var bc1: Scalar[DT]  # 1 - β1ᵗ
    var bc2: Scalar[DT]  # 1 - β2ᵗ
    # Arena mode (GPU, set by `adopt`). Empty / 0 / False when un-adopted.
    var val_arena: Tensor
    var grd_arena: Tensor
    var m_arena: Tensor
    var v_arena: Tensor
    var decay_mask: Tensor
    var total: Int
    var _adopt_off: Int  # running offset during the adopt walk
    var _placing: Bool  # True only during adopt's placement walk
    var _adopted: Bool

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
        self.val_arena = Tensor()
        self.grd_arena = Tensor()
        self.m_arena = Tensor()
        self.v_arena = Tensor()
        self.decay_mask = Tensor()
        self.total = 0
        self._adopt_off = 0
        self._placing = False
        self._adopted = False

    def begin_step(mut self):
        """Bump the step counter + refresh bias corrections. Once per step."""
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow

    def adopt[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Engage arena mode (GPU). Allocates the 4 arenas, copies param values
        in, rebinds each Param's val/grd to arena slices, builds the decay mask.
        NO-OP on CPU (per-param has no launch overhead to collapse). Call ONCE
        after the model is made + initialized, before the first step."""
        comptime if target == "gpu":
            var c = ctx.value()
            var nps = named_params["gpu"](model)
            var total = 0
            for i in range(len(nps)):
                total += nps[i].size
            self.total = total

            var dm = Tensor.alloc(total)  # host decay mask
            var off = 0
            for i in range(len(nps)):
                var d = Scalar[DT](1.0) if nps[i].decay else Scalar[DT](0.0)
                for k in range(nps[i].size):
                    dm.data[off + k] = d
                off += nps[i].size
            dm.upload(c)
            self.decay_mask = dm^

            self.val_arena = Tensor.alloc_gpu(c, total)  # zeroed
            self.grd_arena = Tensor.alloc_gpu(c, total)
            self.m_arena = Tensor.alloc_gpu(c, total)
            self.v_arena = Tensor.alloc_gpu(c, total)
            self._adopt_off = 0
            self._placing = True
            model.for_each_param["gpu"](self, Optional(c))
            self._placing = False
            self._adopted = True

    def step[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Bump the step then update every Param. GPU+adopted → one arena kernel;
        CPU or un-adopted GPU → per-param walk."""
        self.begin_step()
        comptime if target == "cpu":
            model.for_each_param["cpu"](self, ctx)
        else:
            if self._adopted:
                self._grouped_step(ctx.value())
            else:
                model.for_each_param["gpu"](self, ctx)

    def _grouped_step(mut self, c: DeviceContext) raises:
        if self.total == 0:
            return
        var nblk = (self.total + TPB - 1) // TPB
        c.enqueue_function[_grouped_adam_kernel](
            self.val_arena.dev.value(),
            self.grd_arena.dev.value(),
            self.m_arena.dev.value(),
            self.v_arena.dev.value(),
            self.decay_mask.dev.value(),
            self.total,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.bc1,
            self.bc2,
            self.wd,
            grid_dim=nblk,
            block_dim=TPB,
        )

    def zero_grad(mut self) raises:
        """Adopted GPU only: zero the whole grad arena in ONE fill (vs N per-param
        fills). For CPU / un-adopted, use `model.zero_grad[target]` as usual."""
        if self._adopted and self.total > 0:
            self.grd_arena.dev.value().enqueue_fill(Scalar[DT](0))

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        # Placement walk (adopt): GPU-only, gated comptime so the device ops never
        # compile into the CPU path. `param`/`grad` are the val/grd Tensors; the
        # mut refs chain back to the model's Param, so the rebinds persist.
        comptime if target == "gpu":
            if self._placing:
                var c = ctx.value()
                var vsub = self.val_arena.dev.value().create_sub_buffer[DT](
                    self._adopt_off, N
                )
                c.enqueue_copy(vsub, param.dev.value())  # preserve init values
                param.dev = Optional(vsub)
                param.n = N
                var gsub = self.grd_arena.dev.value().create_sub_buffer[DT](
                    self._adopt_off, N
                )
                grad.dev = Optional(gsub)
                grad.n = N
                self._adopt_off += N
                return

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
                param.lt["gpu", layout](),
                grad.lt["gpu", layout](),
                m.lt["gpu", layout](),
                v.lt["gpu", layout](),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.bc1,
                self.bc2,
                self.wd,
                Int(apply_decay),
                grid_dim=nblk,
                block_dim=TPB,
            )


# AdamW is just Adam with decoupled weight decay (`wd > 0`, gated per param by
# APPLY_DECAY) — both the per-param and arena paths apply `p -= lr·wd·p` before
# the moment update (AdamW's decoupled-decay rule). Construct as
# `AdamW(lr=..., wd=0.01)` (pass a nonzero `wd` — the Adam default is 0.0).
comptime AdamW = Adam
