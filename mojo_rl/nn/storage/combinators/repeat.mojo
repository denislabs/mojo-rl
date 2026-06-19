"""Repeat[N, Inner] — chain N independent copies of `Inner` (storage surface).

`y = Inner_{N-1}(… Inner_1(Inner_0(x)) …)`, each copy with its OWN params. This
is exactly `Sequential` over N homogeneous children — same `TensorPack` mid-slab
wiring (whose `__getitem__` MutAnyOrigin pin lets adjacent slabs alias safely),
just stored in a `List[Inner]` indexed by the comptime stage. Requires
`Inner.IN_DIMS[0] == Inner.OUT_DIM`. `shared=True` is not supported (would need
per-application caches); pass the default `shared=False` (independent copies).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor


struct Repeat[N: Int, Inner: Module, shared: Bool = False](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var children: List[Self.Inner]
    var act: TensorPack[Self.N]
    var grd: TensorPack[Self.N]

    def __init__(out self):
        comptime assert Self.N >= 1, "Repeat requires N >= 1"
        comptime assert (
            not Self.shared
        ), "Repeat: shared=True not supported — use shared=False"
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), "Repeat requires Inner.IN_DIMS[0] == Inner.OUT_DIM"
        self.children = List[Self.Inner]()
        self.act = TensorPack[Self.N]()
        self.grd = TensorPack[Self.N]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var r = Self()
        for _ in range(Self.N):
            r.children.append(Self.Inner.make[target, INIT](ctx))
        return r^

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if Self.N == 1:
            self.children[0].forward[target, B](
                TensorRefs[Self.Inner.ARITY](inputs[0]), out, ctx
            )
        else:
            comptime for i in range(Self.N):
                comptime if i == 0:
                    self.children[0].forward[target, B](
                        TensorRefs[Self.Inner.ARITY](inputs[0]),
                        self.act[0],
                        ctx,
                    )
                elif i == Self.N - 1:
                    self.children[Self.N - 1].forward[target, B](
                        TensorRefs[Self.Inner.ARITY](self.act[Self.N - 2]),
                        out,
                        ctx,
                    )
                else:
                    self.children[i].forward[target, B](
                        TensorRefs[Self.Inner.ARITY](self.act[i - 1]),
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
        comptime if Self.N == 1:
            self.children[0].vjp[target, B](
                TensorRefs[Self.Inner.ARITY](forward_input[0]),
                grad_output,
                TensorRefs[Self.Inner.ARITY](grad_inputs[0]),
                ctx,
            )
        else:
            comptime for j in range(Self.N):
                comptime i = Self.N - 1 - j
                comptime if i == Self.N - 1:
                    self.children[Self.N - 1].vjp[target, B](
                        TensorRefs[Self.Inner.ARITY](self.act[Self.N - 2]),
                        grad_output,
                        TensorRefs[Self.Inner.ARITY](self.grd[Self.N - 2]),
                        ctx,
                    )
                elif i == 0:
                    self.children[0].vjp[target, B](
                        TensorRefs[Self.Inner.ARITY](forward_input[0]),
                        self.grd[0],
                        TensorRefs[Self.Inner.ARITY](grad_inputs[0]),
                        ctx,
                    )
                else:
                    self.children[i].vjp[target, B](
                        TensorRefs[Self.Inner.ARITY](self.act[i - 1]),
                        self.grd[i],
                        TensorRefs[Self.Inner.ARITY](self.grd[i - 1]),
                        ctx,
                    )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        for i in range(Self.N):
            self.children[i].for_each_param[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)
