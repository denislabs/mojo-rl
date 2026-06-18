"""Sequential[*MODULES] — storage-passing orchestrator (N-ary, CPU + GPU).

Threads `comptime target` + `ctx` to its unary children. Inter-module buffers
live in two owning `TensorPack`s; the children lazily allocate their slabs on
the active target (`out.ensure`/`ensure_gpu`). `TensorPack.__getitem__` returns
a `MutAnyOrigin` ref (load-bearing pin, §7.12); each child input is wrapped in
a borrowing `TensorRefs[1]`. Slice scope: N >= 2.
"""

from std.gpu.host import DeviceContext

from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor


struct Sequential[*MODULES: Module](Module):
    comptime ARITY = 1
    comptime N = Self.MODULES.size
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.MODULES[0].IN_DIMS[0])
    comptime OUT_DIM = Self.MODULES[Self.N - 1].OUT_DIM

    var children: Tuple[*Self.MODULES]
    var act: TensorPack[Self.N]
    var grd: TensorPack[Self.N]

    def __init__(out self):
        comptime assert Self.N >= 2, "Sequential slice requires N >= 2"
        comptime for i in range(Self.N):
            comptime assert (
                Self.MODULES[i].ARITY == 1
            ), "Sequential chains UNARY children only (ARITY == 1)"
        comptime for i in range(Self.N - 1):
            comptime assert (
                Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIMS[0]
            ), "Sequential: adjacent child dims must match"
        self.children = Tuple[*Self.MODULES]()
        self.act = TensorPack[Self.N]()
        self.grd = TensorPack[Self.N]()

    @staticmethod
    def make_cpu() raises -> Self:
        var s = Self()
        comptime for i in range(Self.N):
            s.children[i] = Self.MODULES[i].make_cpu()
        return s^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var s = Self()
        comptime for i in range(Self.N):
            s.children[i] = Self.MODULES[i].make_gpu(ctx)
        return s^

    def forward[target: StaticString, B: Int, o: MutOrigin](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime for i in range(Self.N):
            comptime if i == 0:
                self.children[0].forward[target, B](
                    TensorRefs[Self.MODULES[0].ARITY].of1(inputs[0]),
                    self.act[0],
                    ctx,
                )
            elif i == Self.N - 1:
                self.children[Self.N - 1].forward[target, B](
                    TensorRefs[Self.MODULES[Self.N - 1].ARITY].of1(
                        self.act[Self.N - 2]
                    ),
                    out,
                    ctx,
                )
            else:
                self.children[i].forward[target, B](
                    TensorRefs[Self.MODULES[i].ARITY].of1(self.act[i - 1]),
                    self.act[i],
                    ctx,
                )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime for j in range(Self.N):
            comptime i = Self.N - 1 - j
            comptime if i == Self.N - 1:
                self.children[Self.N - 1].vjp[target, B](
                    TensorRefs[Self.MODULES[Self.N - 1].ARITY].of1(
                        self.act[Self.N - 2]
                    ),
                    grad_output,
                    TensorRefs[Self.MODULES[Self.N - 1].ARITY].of1(
                        self.grd[Self.N - 2]
                    ),
                    ctx,
                )
            elif i == 0:
                self.children[0].vjp[target, B](
                    TensorRefs[Self.MODULES[0].ARITY].of1(forward_input[0]),
                    self.grd[0],
                    TensorRefs[Self.MODULES[0].ARITY].of1(grad_inputs[0]),
                    ctx,
                )
            else:
                self.children[i].vjp[target, B](
                    TensorRefs[Self.MODULES[i].ARITY].of1(self.act[i - 1]),
                    self.grd[i],
                    TensorRefs[Self.MODULES[i].ARITY].of1(self.grd[i - 1]),
                    ctx,
                )

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext]
    ) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)
