"""Adam — storage-native Adam / AdamW optimizer (CPU + GPU).

A `ParamVisitor`, but stateful: per-param 1st/2nd moments. Two GPU modes, ONE
class, IDENTICAL math (bit-parity, gated):

  - per-param (default, CPU + GPU): `step` walks `for_each_param`, one kernel /
    CPU loop per Param. The universal correctness path + CPU↔GPU parity check.
    Moments live on the `Param` (`m`/`v` Tensors), lazily zero-allocated.
  - arena (GPU, opt-in via `adopt`): a shared `ParamArena` packs params into
    contiguous val/grd buffers (+ this optimizer's own contiguous m/v arenas);
    `step` is ONE flat kernel over `[0,total)`, collapsing N launches → 1. Runs on
    Apple AND NVIDIA. `adopt` is a NO-OP on CPU → agent code is target-agnostic:

        opt.adopt[target](model, ctx); opt.step[target](model, ctx)

Decoupled weight decay (`wd > 0`, gated by `APPLY_DECAY` / the arena `decay_mask`)
makes this AdamW. Lifetime: the arena/moments are optimizer-owned; param slices
reference them (DeviceBuffer is refcounted → destruction order is safe).
"""

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor
from ..core.module import Module
from .param_arena import ParamArena
from .grad_clip import clip_grad_norm, clip_arena_grads
from .optimizer import Optimizer


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
    """Per-param update (one Param, comptime size N).

    NOTE: `bc1/bc2` are host-baked here — fine for CPU and for the un-captured
    per-param GPU walk, but NOT CUDA-graph-safe (they'd freeze at capture-time).
    The capture path uses `adopt` → `_grouped_adam_kernel`, which reads β^t from
    a device buffer. Don't capture a non-adopted GPU optimizer."""
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


def _adam_advance_pow_kernel(
    powbuf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    """Advance the device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (1
    thread). β^t lives on-device so it advances on every CUDA-graph REPLAY —
    a host-baked `bc` scalar would freeze at the capture-time step, scaling
    every replayed update by the early (t≈1) correction. Mirrors ScalarAdam's
    on-device `β^t` state."""
    if Int(global_idx.x) != 0:
        return
    powbuf[0] = powbuf[0] * beta1
    powbuf[1] = powbuf[1] * beta2


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
    powbuf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    wd: Scalar[DT],
):
    """Arena update (all params at once over runtime-length flat buffers).
    `bc1/bc2` are read from the device `powbuf` (`[β₁ᵗ, β₂ᵗ]`, advanced by
    `_adam_advance_pow_kernel` just before) so they advance under graph replay.
    """
    var i = Int(global_idx.x)
    if i >= total:
        return
    var one = Scalar[DT](1.0)
    var bc1 = one - powbuf[0]
    var bc2 = one - powbuf[1]
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


struct Adam(Movable, ParamVisitor, Optimizer):
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]
    var _b2_pow: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]
    # Arena mode (GPU, set by `adopt`): shared val/grd packing + own m/v arenas.
    var arena: ParamArena
    var m_arena: Tensor
    var v_arena: Tensor
    # Device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (GPU arena path only,
    # allocated by `adopt`). Advanced on-device each step so the correction
    # advances under CUDA-graph replay instead of freezing at the host-baked
    # capture-time value. CPU + per-param paths use the host `bc1/bc2`.
    var _pow_dev: Tensor

    def __init__(out self):
        """No-arg default (satisfies Defaultable for the generic Trainer)."""
        self = Self(lr=Scalar[DT](1e-3))

    def __init__(
        out self,
        lr: Scalar[DT],  # required → disambiguates from the no-arg ctor
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
        self._pow_dev = Tensor()
        self.arena = ParamArena()
        self.m_arena = Tensor()
        self.v_arena = Tensor()

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
        """Engage arena mode (GPU); NO-OP on CPU. Packs val/grd via the shared
        ParamArena and allocates this optimizer's m/v arenas to match."""
        self.arena.adopt[target](model, ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            self.m_arena = Tensor.alloc_gpu(c, self.arena.total)
            self.v_arena = Tensor.alloc_gpu(c, self.arena.total)
            # `[β₁ᵗ, β₂ᵗ]` seeded to β^0 = 1; advanced on-device each step.
            self._pow_dev = Tensor.alloc_gpu(c, 2)
            self._pow_dev.dev.value().enqueue_fill(Scalar[DT](1.0))

    def step[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Bump the step then update every Param. GPU+adopted → one arena kernel;
        CPU or un-adopted GPU → per-param walk."""
        self.begin_step()
        comptime if target == "cpu":
            model.for_each_param["cpu"](self, ctx)
        else:
            if self.arena.adopted:
                self._grouped_step(ctx.value())
            else:
                model.for_each_param["gpu"](self, ctx)

    def _grouped_step(mut self, c: DeviceContext) raises:
        if self.arena.total == 0:
            return
        # Advance β^t on-device BEFORE the update reads it — captured into the
        # graph so it advances per replay (host `bc1/bc2` would freeze).
        c.enqueue_function[_adam_advance_pow_kernel](
            self._pow_dev.dev.value(),
            self.beta1,
            self.beta2,
            grid_dim=1,
            block_dim=1,
        )
        var nblk = (self.arena.total + TPB - 1) // TPB
        c.enqueue_function[_grouped_adam_kernel](
            self.arena.val.dev.value(),
            self.arena.grd.dev.value(),
            self.m_arena.dev.value(),
            self.v_arena.dev.value(),
            self.arena.decay_mask.dev.value(),
            self.arena.total,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self._pow_dev.dev.value(),
            self.wd,
            grid_dim=nblk,
            block_dim=TPB,
        )

    def zero_grad[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """GPU+adopted → zero the grad arena in ONE fill; else per-param via the
        model."""
        comptime if target == "gpu":
            if self.arena.adopted:
                self.arena.zero_grad()
                return
        model.zero_grad[target](ctx)

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def get_lr(self) -> Scalar[DT]:
        return self.lr

    def clip_grads[
        target: StaticString, M: Module
    ](
        mut self, mut model: M, max_norm: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """Global grad-norm clip, returns the pre-clip norm. GPU+adopted → arena
        reduction+scale (capture-safe, no per-param D2H); else → per-param
        `clip_grad_norm`. Symmetric across targets."""
        comptime if target == "gpu":
            if self.arena.adopted:
                return clip_arena_grads(self.arena, max_norm, ctx.value())
        return clip_grad_norm[target](model, max_norm, ctx)

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


# AdamW is Adam with decoupled weight decay (`wd > 0`, gated per param by
# APPLY_DECAY) — both the per-param and arena paths apply `p -= lr·wd·p` before
# the moment update. Construct as `AdamW(lr=..., wd=0.01)`.
comptime AdamW = Adam
