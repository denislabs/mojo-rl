"""ConditionalTransformerBlock[EMB, HEADS, H, FF] — AdaLN-zero DiT block.

ARITY=2 Module: forward(x, c) over flattened sequences x, c ∈ (B, H·EMB),
interpreted as (B, H, EMB). c is the per-token conditioning (the action
embedding in LeWM). The block is the dual-branch pre-LN transformer layer
with AdaLN-zero conditioning:

    c'                          = SiLU(c)
    sh1,sc1,g1,sh2,sc2,g2       = ZeroLinear(c')          (6 per-token proj)
    x = Gate(x, g1, MHA(Modulate(LN(x),  sc1, sh1)))      (causal MSA branch)
    x = Gate(x, g2, FFN(Modulate(LN(x),  sc2, sh2)))      (MLP branch)

Implementation: an internal `ComputeGraph` (validated Module-wraps-graph
pattern — `tests/nn2/spike_module_wraps_graph.mojo`). Per-token ops use
`Tokenwise[H, ...]`; `Modulate`/`Gate` are elementwise over (B, H·EMB) so
they apply per-token correctly; `MHA` runs at seq_len=H. The 6 modulation
projections are `ZeroLinear` so at init shift/scale/gate=0 ⇒ Modulate is
identity and Gate drops both branches ⇒ block(x,c) = x **bitwise**. That
identity is the load-bearing LeWM correctness invariant.

vjp delegates to the graph and copies the accumulated input grads for "x"
and "c" out into the caller's grad-input tiles.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for
from ..combinators import ComputeGraph, InputSlot, Node, Tokenwise
from ..composites import MultiHeadAttention, TransformerFFN
from .silu import SiLU
from .zero_linear import ZeroLinear
from .layer_norm_no_affine import LayerNormNoAffine
from .modulate import Modulate
from .gate import Gate


def _ctb_copy_kernel[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


struct ConditionalTransformerBlock[
    EMB: Int, HEADS: Int, H: Int, FF: Int
](Module):
    comptime ARITY: Int = 2
    comptime SEQ_DIM = Self.H * Self.EMB
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.SEQ_DIM)
    comptime OUT_DIM = Self.SEQ_DIM

    @staticmethod
    def display_label() -> String:
        return String("ConditionalTransformerBlock")

    comptime Mod6 = Tokenwise[Self.H, ZeroLinear[Self.EMB, Self.EMB]]
    comptime LN = Tokenwise[Self.H, LayerNormNoAffine[Self.EMB]]

    comptime Graph = ComputeGraph[
        Self.SEQ_DIM,
        InputSlot["x", Self.SEQ_DIM],
        InputSlot["c", Self.SEQ_DIM],
        Node["cs", SiLU[Self.SEQ_DIM], "c"],
        Node["sh1", Self.Mod6, "cs"],
        Node["sc1", Self.Mod6, "cs"],
        Node["g1", Self.Mod6, "cs"],
        Node["sh2", Self.Mod6, "cs"],
        Node["sc2", Self.Mod6, "cs"],
        Node["g2", Self.Mod6, "cs"],
        Node["ln1", Self.LN, "x"],
        Node["mod1", Modulate[Self.SEQ_DIM], "ln1", "sc1", "sh1"],
        Node["attn", MultiHeadAttention[Self.EMB, Self.HEADS, Self.H, True],
             "mod1"],
        Node["x1", Gate[Self.SEQ_DIM], "x", "g1", "attn"],
        Node["ln2", Self.LN, "x1"],
        Node["mod2", Modulate[Self.SEQ_DIM], "ln2", "sc2", "sh2"],
        Node["mlp", TransformerFFN[Self.H, Self.EMB, Self.FF], "mod2"],
        Node["x2", Gate[Self.SEQ_DIM], "x1", "g2", "mlp"],
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
            "ConditionalTransformerBlock: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.graph = Self.Graph.make[target=target, INIT=INIT](ctx=ctx)
        comptime if target == "cpu":
            b.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("ConditionalTransformerBlock.make[gpu]: ctx required")
            b.ts = TargetStorage.make_gpu(ctx.value())
        return b^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["ConditionalTransformerBlock", target](
            self.ts.target_tag
        )
        var x = typed_view[BATCH, Self.SEQ_DIM](inputs[0])
        var c = typed_view[BATCH, Self.SEQ_DIM](inputs[1])
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["ConditionalTransformerBlock", target](
            self.ts.target_tag
        )
        var go = typed_view[BATCH, Self.SEQ_DIM](grad_output)
        var gx = typed_view_mut[BATCH, Self.SEQ_DIM](grad_inputs[0])
        var gc = typed_view_mut[BATCH, Self.SEQ_DIM](grad_inputs[1])
        self.graph.vjp[target, BATCH, POLICY=POLICY, mode=mode](go)
        var gx_src = self.graph.grad_input_ptr["x"]()
        var gc_src = self.graph.grad_input_ptr["c"]()
        comptime total = BATCH * Self.SEQ_DIM

        comptime if target == "cpu":
            var gx_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gx.ptr)
            var gc_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gc.ptr)
            for i in range(total):
                gx_p[i] = gx_src[i]
                gc_p[i] = gc_src[i]
        else:
            var ctx = self.ts.ctx.value()
            comptime lay = Layout.row_major(total)
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kern = _ctb_copy_kernel[total]
            var gx_dst = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gx.ptr)
            var gc_dst = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gc.ptr)
            ctx.enqueue_function[kern](
                LayoutTensor[DT, lay, MutAnyOrigin](gx_src),
                LayoutTensor[DT, lay, MutAnyOrigin](gx_dst),
                grid_dim=n_blocks, block_dim=TPB,
            )
            ctx.enqueue_function[kern](
                LayoutTensor[DT, lay, MutAnyOrigin](gc_src),
                LayoutTensor[DT, lay, MutAnyOrigin](gc_dst),
                grid_dim=n_blocks, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["ConditionalTransformerBlock", target](
            self.ts.target_tag
        )
        self.graph.for_each_param[target, V](prefix, visitor)
