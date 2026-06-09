"""DecoderBlock[N, HID, FF] — one lightweight transformer-decoder layer.

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
replicated global `c` above — so no attention kernel is needed (the query
projection and its LayerNorm would be dead weights with zero gradient).
The query tokens therefore interact only with the global, exactly as the
paper specifies ("cross-attention layers with residual MLP blocks").

Implementation: an internal `ComputeGraph` (the validated Module-wraps-graph
pattern), identical in spirit to `ConditionalTransformerBlock`. Because
IN0 == IN1 == OUT == N·HID, this block drops straight into
`RepeatConditional[DEPTH, DecoderBlock[...]]` (the global `c` is broadcast to
every layer; grad_c accumulates). CPU + GPU.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for
from ..combinators import (
    ComputeGraph, InputSlot, Node, Tokenwise, Residual, Sequential,
)
from ..models.transformer import TransformerFFN
from .linear import Linear
from .layer_norm import LayerNorm
from .add import Add


def _db_copy_kernel[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


struct DecoderBlock[N: Int, HID: Int, FF: Int](Module):
    comptime ARITY: Int = 2
    comptime SEQ_DIM = Self.N * Self.HID
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.SEQ_DIM)
    comptime OUT_DIM = Self.SEQ_DIM

    @staticmethod
    def display_label() -> String:
        return String("DecoderBlock")

    comptime Graph = ComputeGraph[
        Self.SEQ_DIM,
        InputSlot["x", Self.SEQ_DIM],
        InputSlot["c", Self.SEQ_DIM],
        Node["inj", Tokenwise[Self.N, Linear[Self.HID, Self.HID]], "c"],
        Node["xa", Add[Self.SEQ_DIM, 2], "x", "inj"],
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
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.Graph()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DecoderBlock: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.graph = Self.Graph.make[target=target, INIT=INIT](ctx=ctx)
        comptime if target == "cpu":
            b.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("DecoderBlock.make[gpu]: ctx required")
            b.ts = TargetStorage.make_gpu(ctx.value())
        return b^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["DecoderBlock", target](self.ts.target_tag)
        var x = inputs.tile[0, BATCH, Self.SEQ_DIM]()
        var c = inputs.tile[1, BATCH, Self.SEQ_DIM]()
        var out = typed_view_mut[BATCH, Self.SEQ_DIM](output)
        self.graph.set_input["x", BATCH](x)
        self.graph.set_input["c", BATCH](c)
        self.graph.forward[target, BATCH, POLICY=POLICY](out)

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        assert_tag_for["DecoderBlock", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.SEQ_DIM](grad_output)
        var gx = grad_inputs.tile[0, BATCH, Self.SEQ_DIM]()
        var gc = grad_inputs.tile[1, BATCH, Self.SEQ_DIM]()
        self.graph.vjp[target, BATCH, POLICY=POLICY, mode=mode](go)
        var gx_src = self.graph.grad_input_ptr["x"]()
        var gc_src = self.graph.grad_input_ptr["c"]()
        comptime total = BATCH * Self.SEQ_DIM

        comptime if target == "cpu":
            var gx_p = gx.ptr
            var gc_p = gc.ptr
            for i in range(total):
                gx_p[i] = gx_src[i]
                gc_p[i] = gc_src[i]
        else:
            var ctx = self.ts.ctx.value()
            comptime lay = Layout.row_major(total)
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kern = _db_copy_kernel[total]
            ctx.enqueue_function[kern](
                LayoutTensor[DT, lay, MutAnyOrigin](gx_src),
                LayoutTensor[DT, lay, MutAnyOrigin](gx.ptr),
                grid_dim=n_blocks, block_dim=TPB,
            )
            ctx.enqueue_function[kern](
                LayoutTensor[DT, lay, MutAnyOrigin](gc_src),
                LayoutTensor[DT, lay, MutAnyOrigin](gc.ptr),
                grid_dim=n_blocks, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["DecoderBlock", target](self.ts.target_tag)
        self.graph.for_each_param[target, V](prefix, visitor)

    def for_each_state[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        self.graph.for_each_state[target, V](prefix, visitor)
