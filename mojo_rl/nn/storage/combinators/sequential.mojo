"""Sequential[*MODULES] — storage-passing orchestrator (N-ary, CPU + GPU).

Threads `comptime target` + `ctx` to its unary children. Inter-module buffers
live in two owning `TensorPack`s; the children lazily allocate their slabs on
the active target (`out.ensure`/`ensure_gpu`). `TensorPack.__getitem__` returns
a `MutAnyOrigin` ref (load-bearing pin, §7.12); each child input is wrapped in
a borrowing `TensorRefs[1]`. Slice scope: N >= 2.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from ..core.graph_visitor import DisplayStep


struct Sequential[*MODULES: Module](Module):
    comptime ARITY = 1
    comptime N = Self.MODULES.size
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.MODULES[0].IN_DIMS[0])
    comptime OUT_DIM = Self.MODULES[Self.N - 1].OUT_DIM

    @staticmethod
    def display_label() -> String:
        return String("Sequential")

    @staticmethod
    def display_steps() -> List[DisplayStep]:
        """Expand the chain — one step per child (its display label + output
        width) — so `ComputeGraph.describe` exporters open a Sequential node
        instead of showing one opaque box."""
        var steps = List[DisplayStep]()
        comptime for i in range(Self.N):
            steps.append(
                DisplayStep(
                    Self.MODULES[i].display_label(),
                    Self.MODULES[i].OUT_DIM,
                )
            )
        return steps^

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
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        comptime for i in range(Self.N):
            s.children[i] = Self.MODULES[i].make[target, INIT](ctx)
        return s^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime for i in range(Self.N):
            comptime if i == 0:
                self.children[0].forward[target, B, POLICY=POLICY](
                    TensorRefs[Self.MODULES[0].ARITY](inputs[0]),
                    self.act[0],
                    ctx,
                )
            elif i == Self.N - 1:
                self.children[Self.N - 1].forward[target, B, POLICY=POLICY](
                    TensorRefs[Self.MODULES[Self.N - 1].ARITY](
                        self.act[Self.N - 2]
                    ),
                    out,
                    ctx,
                )
            else:
                self.children[i].forward[target, B, POLICY=POLICY](
                    TensorRefs[Self.MODULES[i].ARITY](self.act[i - 1]),
                    self.act[i],
                    ctx,
                )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP
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
                self.children[Self.N - 1].vjp[target, B, POLICY=POLICY](
                    TensorRefs[Self.MODULES[Self.N - 1].ARITY](
                        self.act[Self.N - 2]
                    ),
                    grad_output,
                    TensorRefs[Self.MODULES[Self.N - 1].ARITY](
                        self.grd[Self.N - 2]
                    ),
                    ctx,
                )
            elif i == 0:
                self.children[0].vjp[target, B, POLICY=POLICY](
                    TensorRefs[Self.MODULES[0].ARITY](forward_input[0]),
                    self.grd[0],
                    TensorRefs[Self.MODULES[0].ARITY](grad_inputs[0]),
                    ctx,
                )
            else:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    TensorRefs[Self.MODULES[i].ARITY](self.act[i - 1]),
                    self.grd[i],
                    TensorRefs[Self.MODULES[i].ARITY](self.grd[i - 1]),
                    ctx,
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Recurse a named runtime attribute (e.g. BatchNorm's "training") into
        every child — the plain-Sequential analog of ComputeGraph's
        `set_node_attr`. Leaves without the attr no-op."""
        comptime for i in range(Self.N):
            self.children[i].set_attr[ATTR](value)
