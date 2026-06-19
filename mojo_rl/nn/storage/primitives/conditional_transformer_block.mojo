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
legacy file wired the graph with named `InputSlot`/`Node` slots; the storage
`ComputeGraph[NUM_IN, *NODES]` is POSITIONAL with an edge-list, so the only
change is expressing the same DAG as `(*NODES)` in topo order + the matching
`edges: List[List[Int]]` (built in `make`). The `_ctb_copy_kernel` device
grad-copy is carried over VERBATIM (arg surface → `.lt["gpu", layout]()`).

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

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.amp import AMPPolicy, NoAMP
from ..combinators.compute_graph import ComputeGraph
from ..combinators.tokenwise import Tokenwise
from ..models.transformer import MultiHeadAttentionXL, TransformerFFN
from .silu import SiLU
from .zero_linear import ZeroLinear
from .layer_norm_no_affine import LayerNormNoAffine
from .modulate import Modulate
from .gate import Gate


# Carried VERBATIM (arg surface → storage `Tensor.lt`): dst[i] = src[i].
def _ctb_copy_kernel[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


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
        2,
        SiLU[Self.SEQ_DIM],                                       # 0  cs
        Self.Mod6,                                                # 1  sh1
        Self.Mod6,                                                # 2  sc1
        Self.Mod6,                                                # 3  g1
        Self.Mod6,                                                # 4  sh2
        Self.Mod6,                                                # 5  sc2
        Self.Mod6,                                                # 6  g2
        Self.LN,                                                  # 7  ln1
        Modulate[Self.SEQ_DIM],                                   # 8  mod1
        MultiHeadAttentionXL[
            Self.EMB, Self.HEADS, Self.HD, Self.H, True
        ],                                                        # 9  attn
        Gate[Self.SEQ_DIM],                                       # 10 x1
        Self.LN,                                                  # 11 ln2
        Modulate[Self.SEQ_DIM],                                   # 12 mod2
        TransformerFFN[Self.H, Self.EMB, Self.FF],                # 13 mlp
        Gate[Self.SEQ_DIM],                                       # 14 x2
    ]

    var graph: Self.Graph
    var edges: List[List[Int]]

    def __init__(out self):
        self.graph = Self.Graph()
        self.edges = Self._build_edges()

    @staticmethod
    def _build_edges() -> List[List[Int]]:
        # Slot indices: x=0, c=1; node i writes slot 2+i.
        var e = List[List[Int]]()
        e.append([1])           # 0  cs   = SiLU(c)
        e.append([2])           # 1  sh1  = Mod6(cs)
        e.append([2])           # 2  sc1  = Mod6(cs)
        e.append([2])           # 3  g1   = Mod6(cs)
        e.append([2])           # 4  sh2  = Mod6(cs)
        e.append([2])           # 5  sc2  = Mod6(cs)
        e.append([2])           # 6  g2   = Mod6(cs)
        e.append([0])           # 7  ln1  = LN(x)
        e.append([9, 4, 3])     # 8  mod1 = Modulate(ln1, sc1, sh1)
        e.append([10])          # 9  attn = MHA(mod1)
        e.append([0, 5, 11])    # 10 x1   = Gate(x, g1, attn)
        e.append([12])          # 11 ln2  = LN(x1)
        e.append([13, 7, 6])    # 12 mod2 = Modulate(ln2, sc2, sh2)
        e.append([14])          # 13 mlp  = FFN(mod2)
        e.append([12, 8, 15])   # 14 x2   = Gate(x1, g2, mlp)
        return e^

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
        # Seed the graph's TensorPack[2] external inputs from the input refs.
        comptime total = B * Self.SEQ_DIM
        var gin_pack = TensorPack[2]()
        comptime if target == "cpu":
            gin_pack[0].ensure(total)
            gin_pack[1].ensure(total)
            ref x0 = inputs[0].data
            ref x1 = inputs[1].data
            for q in range(total):
                gin_pack[0].data[q] = x0[q]
                gin_pack[1].data[q] = x1[q]
        else:
            var c = ctx.value()
            gin_pack[0].ensure_gpu(c, total)
            gin_pack[1].ensure_gpu(c, total)
            comptime lay = Layout.row_major(total)
            comptime kern = _ctb_copy_kernel[total]
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[kern](
                inputs[0].lt["gpu", lay](),
                gin_pack[0].lt["gpu", lay](),
                grid_dim=n_blocks, block_dim=TPB,
            )
            c.enqueue_function[kern](
                inputs[1].lt["gpu", lay](),
                gin_pack[1].lt["gpu", lay](),
                grid_dim=n_blocks, block_dim=TPB,
            )
        self.graph.forward[B, target, POLICY=POLICY](
            self.edges, gin_pack, out, ctx
        )

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
        var ggrad_pack = TensorPack[2]()
        self.graph.vjp[B, target, POLICY=POLICY](
            self.edges, grad_output, ggrad_pack, ctx
        )
        ref gx = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime if target == "cpu":
            gx.ensure(total)
            gc.ensure(total)
            for q in range(total):
                gx.data[q] = ggrad_pack[0].data[q]
                gc.data[q] = ggrad_pack[1].data[q]
        else:
            var c = ctx.value()
            gx.ensure_gpu(c, total)
            gc.ensure_gpu(c, total)
            comptime lay = Layout.row_major(total)
            comptime kern = _ctb_copy_kernel[total]
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[kern](
                ggrad_pack[0].lt["gpu", lay](),
                gx.lt["gpu", lay](),
                grid_dim=n_blocks, block_dim=TPB,
            )
            c.enqueue_function[kern](
                ggrad_pack[1].lt["gpu", lay](),
                gc.lt["gpu", lay](),
                grid_dim=n_blocks, block_dim=TPB,
            )

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
