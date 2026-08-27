"""SGD — storage-native SGD optimizer (CPU + GPU).

A stateless `ParamVisitor` (no moments). Two GPU modes, ONE class, identical math:
  - per-param (default, CPU + GPU): `step` walks `for_each_param`.
  - arena (GPU, opt-in via `adopt`): shared `ParamArena` packs params into
    contiguous val/grd; `step` is ONE flat kernel. `adopt` is a NO-OP on CPU, so
    agent code is target-agnostic. Mirrors `Adam` (which adds m/v arenas).

Update: `if decay: g += wd·p ; p -= lr·g`.
"""

from std.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor, ParamVersionBump
from ..core.module import Module
from .param_arena import ParamArena
from .grad_clip import clip_grad_norm, clip_arena_grads
from .optimizer import Optimizer


def _sgd_kernel[
    N: Int
](
    param: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    lr: Scalar[DT],
    wd: Scalar[DT],
    apply_decay_arg: Int64,
):
    """Per-param update (one Param, comptime size N)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var apply_decay = Int(apply_decay_arg)
    var i = Int(global_idx.x)
    if i < N:
        var d = grad[i]
        if apply_decay != 0:
            d += wd * param[i]
        param[i] -= lr * d


def _grouped_sgd_kernel(
    val: Pointer[Scalar[DT], MutAnyOrigin],
    grd: Pointer[Scalar[DT], MutAnyOrigin],
    decay: Pointer[Scalar[DT], MutAnyOrigin],
    total_arg: Int64,
    lr: Scalar[DT],
    wd: Scalar[DT],
):
    """Arena update (all params over runtime-length flat buffers)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var total = Int(total_arg)
    var i = Int(global_idx.x)
    if i >= total:
        return
    var d = grd[unsafe_offset=i]
    if decay[unsafe_offset=i] != Scalar[DT](0.0):
        d += wd * val[unsafe_offset=i]
    val[unsafe_offset=i] -= lr * d


struct SGD(Movable, ParamVisitor, Optimizer):
    var lr: Scalar[DT]
    var wd: Scalar[DT]
    var arena: ParamArena

    def __init__(out self):
        """No-arg default (satisfies Defaultable for the generic Trainer)."""
        self = Self(lr=Scalar[DT](1e-2))

    def __init__(out self, lr: Scalar[DT], wd: Scalar[DT] = 0.0):
        self.lr = lr
        self.wd = wd
        self.arena = ParamArena()

    def adopt[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Engage arena mode (GPU); NO-OP on CPU."""
        self.arena.adopt[target](model, ctx)

    def step[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """GPU+adopted → one arena kernel; CPU or un-adopted GPU → per-param."""
        comptime if target == "cpu":
            model.for_each_param["cpu"](self, ctx)
        else:
            if self.arena.adopted:
                self._grouped_step(ctx.value())
            else:
                model.for_each_param["gpu"](self, ctx)
        # AMP: invalidate cached bf16 weights (see Adam.step). Host-only walk;
        # covers per-param AND arena paths.
        var _bump = ParamVersionBump()
        model.for_each_param[target](_bump, ctx)

    def _grouped_step(mut self, c: DeviceContext) raises:
        if self.arena.total == 0:
            return
        var nblk = (self.arena.total + TPB - 1) // TPB
        c.enqueue_function[_grouped_sgd_kernel](
            self.arena.val.dev.value(),
            self.arena.grd.dev.value(),
            self.arena.decay_mask.dev.value(),
            Int64(self.arena.total),
            self.lr,
            self.wd,
            grid_dim=nblk,
            block_dim=TPB,
        )

    def zero_grad[
        target: StaticString, M: Module
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """GPU+adopted → zero the grad arena in ONE fill; else per-param."""
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
        """Global grad-norm clip (pre-clip norm returned). GPU+adopted → arena
        reduction+scale (capture-safe); else → per-param `clip_grad_norm`."""
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
        mut m: Tensor,  # unused (SGD is stateless)
        mut v: Tensor,  # unused
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
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_sgd_kernel[N]](
                param.lt["gpu", layout](),
                grad.lt["gpu", layout](),
                self.lr,
                self.wd,
                Int64(apply_decay),
                grid_dim=nblk,
                block_dim=256,
            )
