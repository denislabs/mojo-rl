"""ParamS[NAME, DECAY, SIZE] + ParamVisitorS — storage-native params.

The user's idea: "make ParamO a Tensor." A param is just two `Tensor`s
(`val` + `grd`), each carrying CPU + GPU storage — so params are STORAGES,
unified with activations. The optimizer walks them via `ParamVisitorS`, which
receives the `Tensor`s + `target` + `ctx` and updates the active buffer
(`.data` on CPU, `.dev` via a kernel on GPU). No separate CPU-only param type.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .tensor import Tensor


trait ParamVisitorS(ImplicitlyDeletable):
    def visit[target: StaticString, N: Int](
        mut self,
        mut param: Tensor,
        mut grad: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        ...


struct ParamS[NAME: StaticString, APPLY_DECAY: Bool, SIZE: Int](
    Movable & ImplicitlyDeletable
):
    var val: Tensor
    var grd: Tensor

    def __init__(out self):
        self.val = Tensor()
        self.grd = Tensor()

    @staticmethod
    def make_cpu() raises -> Self:
        var p = Self()
        p.val = Tensor.alloc(Self.SIZE)
        p.grd = Tensor.alloc(Self.SIZE)
        return p^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        # `val` keeps a CPU list for host init; the owning leaf fills it and
        # calls `val.upload(ctx)` (which allocates `val.dev`). `grd.dev` is
        # allocated + zeroed here.
        var p = Self()
        p.val = Tensor.alloc(Self.SIZE)
        p.grd = Tensor.alloc(Self.SIZE)
        p.grd.ensure_gpu(ctx, Self.SIZE)
        p.grd.dev.value().enqueue_fill(Scalar[DT](0))
        return p^

    def visit_with[target: StaticString, V: ParamVisitorS](
        mut self, mut visitor: V, ctx: Optional[DeviceContext]
    ) raises:
        visitor.visit[target, Self.SIZE](
            self.val, self.grd, Self.APPLY_DECAY, ctx
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime if target == "cpu":
            for k in range(Self.SIZE):
                self.grd.data[k] = Scalar[DT](0)
        else:
            self.grd.dev.value().enqueue_fill(Scalar[DT](0))
