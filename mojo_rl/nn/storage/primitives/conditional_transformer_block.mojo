"""ConditionalTransformerBlock[EMB, HEADS, H, FF] — AdaLN-zero DiT block (storage).

ARITY=2 Module: forward(x, c) over flattened sequences x, c ∈ (B, H·EMB),
interpreted as (B, H, EMB). c is the per-token conditioning (the action
embedding in LeWM). The block is the dual-branch pre-LN transformer layer
with AdaLN-zero conditioning:

    c'                          = SiLU(c)
    sh1,sc1,g1,sh2,sc2,g2       = ZeroLinear(c')          (6 per-token proj)
    x = Gate(x, g1, MHA(Modulate(LN(x),  sc1, sh1)))      (causal MSA branch)
    x = Gate(x, g2, FFN(Modulate(LN(x),  sc2, sh2)))      (MLP branch)

Transformed from legacy `nn.primitives.conditional_transformer_block`
(surface-only change). It is a Module-that-wraps-an-internal-ComputeGraph (the
validated pattern — legacy ref `tests/nn/spike_module_wraps_graph.mojo`). The
graph is wired with the name-based storage DX — `InputSlot["x"/"c", DIM]` inputs
+ `Node["name", Op, *predecessor_names]` nodes (edges resolved by name at
compile time, no runtime edge list). `forward` seeds the inputs via
`graph.set_input["x"/"c", B]` (device-to-device copy on GPU) and `vjp` reads the
accumulated input grads back via `graph.grad_input["x"/"c"]()`.

Per-token ops use `Tokenwise[H, ...]`; `Modulate`/`Gate` are elementwise over
(B, H·EMB) so they apply per-token correctly; `MHA` runs at seq_len=H. The 6
modulation projections are `ZeroLinear` so at init shift/scale/gate=0 ⇒ Modulate
is identity and Gate drops both branches ⇒ block(x,c) = x **bitwise**. That
identity is the load-bearing LeWM correctness invariant.

vjp delegates to the graph and copies the accumulated input grads for "x" and
"c" out into the caller's grad-input slots.

Positional slot map (NUM_IN=2 → x=slot 0, c=slot 1; node i writes slot 2+i):

  node 0  cs   = SiLU(c)               edges [1]
  node 1  sh1  = Mod6(cs)              edges [2]
  node 2  sc1  = Mod6(cs)              edges [2]
  node 3  g1   = Mod6(cs)              edges [2]
  node 4  sh2  = Mod6(cs)              edges [2]
  node 5  sc2  = Mod6(cs)              edges [2]
  node 6  g2   = Mod6(cs)              edges [2]
  node 7  ln1  = LN(x)                 edges [0]
  node 8  mod1 = Modulate(ln1,sc1,sh1) edges [9, 4, 3]
  node 9  attn = MHA(mod1)             edges [10]
  node 10 x1   = Gate(x, g1, attn)     edges [0, 5, 11]
  node 11 ln2  = LN(x1)                edges [12]
  node 12 mod2 = Modulate(ln2,sc2,sh2) edges [13, 7, 6]
  node 13 mlp  = FFN(mod2)             edges [14]
  node 14 x2   = Gate(x1, g2, mlp)     edges [12, 8, 15]  (graph output)
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.amp import AMPPolicy, NoAMP
from ..combinators.compute_graph import ComputeGraph
from ..combinators.graph_decl import InputSlot, Node
from ..combinators.tokenwise import Tokenwise
from ..models.transformer import MultiHeadAttentionXL, TransformerFFN
from .silu import SiLU
from .zero_linear import ZeroLinear
from .layer_norm_no_affine import LayerNormNoAffine
from .modulate import Modulate
from .gate import Gate


struct ConditionalTransformerBlock[
    EMB: Int, HEADS: Int, H: Int, FF: Int, HEAD_DIM: Int = 0
](Module):
    comptime ARITY: Int = 2
    comptime SEQ_DIM = Self.H * Self.EMB
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.SEQ_DIM)
    comptime OUT_DIM = Self.SEQ_DIM
    # head_dim 0 ⇒ standard EMB/HEADS (inner == EMB); >0 ⇒ expanded ("XL")
    # attention with inner = HEADS·HEAD_DIM (the paper predictor's 16×64=1024).
    comptime HD = Self.HEAD_DIM if Self.HEAD_DIM > 0 else Self.EMB // Self.HEADS

    comptime Mod6 = Tokenwise[Self.H, ZeroLinear[Self.EMB, Self.EMB]]
    comptime LN = Tokenwise[Self.H, LayerNormNoAffine[Self.EMB]]

    comptime Graph = ComputeGraph[
        InputSlot["x", Self.SEQ_DIM],                             # residual
        InputSlot["c", Self.SEQ_DIM],                             # conditioning
        Node["cs", SiLU[Self.SEQ_DIM], "c"],
        Node["sh1", Self.Mod6, "cs"],
        Node["sc1", Self.Mod6, "cs"],
        Node["g1", Self.Mod6, "cs"],
        Node["sh2", Self.Mod6, "cs"],
        Node["sc2", Self.Mod6, "cs"],
        Node["g2", Self.Mod6, "cs"],
        Node["ln1", Self.LN, "x"],
        Node["mod1", Modulate[Self.SEQ_DIM], "ln1", "sc1", "sh1"],
        Node[
            "attn",
            MultiHeadAttentionXL[Self.EMB, Self.HEADS, Self.HD, Self.H, True],
            "mod1",
        ],
        Node["x1", Gate[Self.SEQ_DIM], "x", "g1", "attn"],
        Node["ln2", Self.LN, "x1"],
        Node["mod2", Modulate[Self.SEQ_DIM], "ln2", "sc2", "sh2"],
        Node["mlp", TransformerFFN[Self.H, Self.EMB, Self.FF], "mod2"],
        Node["x2", Gate[Self.SEQ_DIM], "x1", "g2", "mlp"],        # output
    ]

    var graph: Self.Graph

    def __init__(out self):
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ConditionalTransformerBlock: target must be 'cpu' or 'gpu'"
        )
        comptime if target != "cpu":
            if not ctx:
                raise Error(
                    "ConditionalTransformerBlock.make[gpu]: ctx required"
                )
        var b = Self()
        b.graph = Self.Graph.make[target=target, INIT=INIT](ctx)
        return b^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Seed the named input slots (the graph copies into its own pool).
        ref x = inputs[0]
        ref c = inputs[1]
        self.graph.set_input["x", B](x, ctx)
        self.graph.set_input["c", B](c, ctx)
        self.graph.forward[B, target, POLICY=POLICY](out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime total = B * Self.SEQ_DIM
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        ref gx = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime if target == "cpu":
            gx.ensure(total)
            gc.ensure(total)
            for q in range(total):
                gx.data[q] = self.graph.grad_input["x"]().data[q]
            for q in range(total):
                gc.data[q] = self.graph.grad_input["c"]().data[q]
        else:
            var c = ctx.value()
            gx.ensure_gpu(c, total)
            gc.ensure_gpu(c, total)
            c.enqueue_copy(gx.dev.value(), self.graph.grad_input["x"]().dev.value())
            c.enqueue_copy(gc.dev.value(), self.graph.grad_input["c"]().dev.value())

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.graph.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.graph.for_each_state[target](visitor, ctx, prefix)

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
