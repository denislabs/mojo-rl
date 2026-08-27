"""DecoderBlock[N, HID, FF] — one lightweight transformer-decoder layer
(storage surface). Transformed from legacy `nn.primitives.decoder_block`
(surface-only change; the math/topology carried VERBATIM).

ARITY=2 Module: forward(x, c) over flattened token sequences x, c ∈
(B, N·HID), interpreted as (B, N, HID). `x` is the query-token stream, `c`
the per-token conditioning — the global ([CLS]/pooled) representation
already replicated to all N positions (see `BroadcastTokens`). The block:

    inj = Tokenwise[Linear[HID, HID]](c)          # cross-attn value/out proj
    xa  = x + inj                                  # cross-attn residual
    out = xa + FFN(LayerNorm(xa))                  # MLP residual

This is the LeWM paper's cross-attention decoder layer, in its exact
mathematical equivalent. The paper uses the single global token as the
attention key+value; with one KV token the softmax is over one element ⇒
weight ≡ 1 ⇒ the attention output is `OutProj(ValueProj(global))`,
**independent of the query**, broadcast to every query token. Composing
value+out projections gives the single `Tokenwise[Linear[HID, HID]]` on the
replicated global `c` above — so no attention kernel is needed.

Implementation: an internal `ComputeGraph[2, *NODES]` (the storage runtime-
edges DAG). NUM_IN=2 external slots (x=0, c=1); three nodes:

    node0 "inj" = Tokenwise[N, Linear[HID, HID]]          edges [1]   → slot 2
    node1 "xa"  = Add[N·HID]                              edges [0,2] → slot 3
    node2 "out" = Residual[Sequential[
                    Tokenwise[N, LayerNorm[HID]],
                    TransformerFFN[N, HID, FF]]]           edges [3]   → slot 4

Because IN0 == IN1 == OUT == N·HID, this block drops straight into a
`Repeat`/`RepeatConditional` of layers. The block OWNS the `ComputeGraph`
(which owns the three node Modules), so for_each_param/zero_grad/for_each_state
/polyak_from recurse into the graph. CPU + GPU.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from ..combinators.compute_graph import ComputeGraph
from ..combinators.graph_decl import InputSlot, Node
from ..combinators.tokenwise import Tokenwise
from ..combinators.residual import Residual
from ..combinators.sequential import Sequential
from ..models.transformer import TransformerFFN
from .linear import Linear
from .layer_norm import LayerNorm
from .add import Add


struct DecoderBlock[N: Int, HID: Int, FF: Int](Module):
    comptime ARITY: Int = 2
    comptime SEQ_DIM = Self.N * Self.HID
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.SEQ_DIM)
    comptime OUT_DIM = Self.SEQ_DIM

    comptime Graph = ComputeGraph[
        InputSlot["x", Self.SEQ_DIM],        # residual stream
        InputSlot["c", Self.SEQ_DIM],        # injection source
        Node["inj", Tokenwise[Self.N, Linear[Self.HID, Self.HID]], "c"],
        Node["xa", Add[Self.SEQ_DIM], "x", "inj"],
        Node[
            "out",
            Residual[
                Sequential[
                    Tokenwise[Self.N, LayerNorm[Self.HID]],
                    TransformerFFN[Self.N, Self.HID, Self.FF],
                ]
            ],
            "xa",
        ],
    ]

    var graph: Self.Graph

    def __init__(out self):
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DecoderBlock: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.graph = Self.Graph.make[target, INIT](ctx)
        return b^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        ref c = inputs[1]
        # Seed the named input slots (the graph copies into its own pool).
        self.graph.set_input["x", B](x, ctx)
        self.graph.set_input["c", B](c, ctx)
        self.graph.forward[B, target, POLICY=POLICY](out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gx = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime DN = B * Self.SEQ_DIM
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        # Copy the graph's named-input grads back into grad_inputs.
        comptime if target == "cpu":
            gx.ensure(DN)
            gc.ensure(DN)
            for q in range(DN):
                gx.data[q] = self.graph.grad_input["x"]().data[q]
            for q in range(DN):
                gc.data[q] = self.graph.grad_input["c"]().data[q]
        else:
            var cc = ctx.value()
            gx.ensure_gpu(cc, DN)
            gc.ensure_gpu(cc, DN)
            # Size-exact sub-buffer copies: the caller's tmp grad slots are
            # reused across nodes and may be larger than `DN`; whole-buffer
            # copies error on the mismatch. Mirrors compute_graph's fix.
            var gx_src = self.graph.grad_input["x"]().dev.value(
            ).create_sub_buffer[DT](0, DN)
            var gx_dst = gx.dev.value().create_sub_buffer[DT](0, DN)
            cc.enqueue_copy(gx_dst, gx_src)
            var gc_src = self.graph.grad_input["c"]().dev.value(
            ).create_sub_buffer[DT](0, DN)
            var gc_dst = gc.dev.value().create_sub_buffer[DT](0, DN)
            cc.enqueue_copy(gc_dst, gc_src)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.graph.for_each_param[target](
            visitor, ctx, join_name(prefix, String("graph"))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.graph.for_each_state[target](
            visitor, ctx, join_name(prefix, String("graph"))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.graph.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.graph.polyak_from[target](src.graph, tau, ctx)
