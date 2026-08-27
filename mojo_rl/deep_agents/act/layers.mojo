# +--------------------------------------------------------------------------+ #
# | DETR transformer layers (post-LN), as ACT uses them
# +--------------------------------------------------------------------------+ #
"""`references/act-main/detr/models/transformer.py` layers, 1:1.

Two differences from the repo's existing `models/transformer.py:TransformerBlock`
make it unusable here, and both are structural rather than cosmetic:

1. **Post-LN, not pre-LN.** ACT builds its transformer with
   `normalize_before=False` (`detr/main.py` never passes `--pre_norm`), so the
   layer is `x = LN(x + sublayer(x))`, not `x = x + sublayer(LN(x))`. The two
   are different functions, not two spellings of one.
2. **The positional embedding is added to q and k but NOT v, at EVERY layer**
   (`with_pos_embed`). `TransformerBlock` derives q, k and v from a single
   `Linear[dim, 3*dim]` on one tensor, so it structurally cannot do this.
   `CrossAttention` (ARITY 3/4) is what makes it expressible.

Also: the FFN activation is **ReLU** (`_get_activation_fn("relu")`), where
`TransformerFFN` uses GELU.

## Stacking

Both layers are ARITY=2, dim-preserving on their first input, so
`RepeatConditional[N, Layer]` is exactly the DETR stack: the residual stream
chains layer to layer while the layer-invariant conditioning is broadcast to
every layer and its gradient accumulated. No new combinator.

The conditioning `c` packs everything that is constant across layers into one
tensor, sliced inside the layer:

    DETREncoderLayer        c = pos                            (SEQ*DIM)
    DETREncoderLayerMasked  c = [pos | key_valid]              (SEQ*DIM + SEQ)
    DETRDecoderLayer        c = [query_pos | k_mem | memory]   (Q*DIM + 2*KV*DIM)

`k_mem = memory + pos` is computed ONCE outside the stack rather than per layer
— it is the same tensor at every layer, and the reference recomputes it only
because `with_pos_embed` is written inline.

## Dropout

Present with the reference's `p=0.1` and in the reference's four places. Set
`graph.set_attr["training"](0.0)` for evaluation and for any comparison against
a reference dump — the reference gate runs both sides in eval.
"""

from mojo_rl.nn import (
    Add,
    ComputeGraph,
    Dropout,
    InputSlot,
    LayerNorm,
    Linear,
    Node,
    ReLU,
    Slice,
    Tokenwise,
)
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.combinators.graph_module2 import GraphModule2
from mojo_rl.nn.primitives.cross_attention import CrossAttention


# ── encoder layer ────────────────────────────────────────────────────────
# `TransformerEncoderLayer.forward_post` (transformer.py:160):
#
#     q = k = src + pos
#     src2 = self_attn(q, k, value=src)
#     src  = norm1(src + dropout1(src2))
#     src2 = linear2(dropout(relu(linear1(src))))
#     src  = norm2(src + dropout2(src2))
#
# `self_attn` is `nn.MultiheadAttention`: three input projections then an
# output projection around the attention itself. Spelled out here because our
# `CrossAttention` leaf is the bare attention, with the projections as
# ordinary `Tokenwise[Linear]` nodes — which is also what makes "pos on q,k
# only" expressible at all.

comptime DETREncoderGraph[
    DIM: Int, HEADS: Int, SEQ: Int, FF: Int, P: Float64 = 0.1
] = ComputeGraph[
    InputSlot["x", SEQ * DIM],
    InputSlot["c", SEQ * DIM],  # pos
    Node["xp", Add[SEQ * DIM], "x", "c"],  # with_pos_embed(src, pos)
    Node["q", Tokenwise[SEQ, Linear[DIM, DIM]], "xp"],
    Node["k", Tokenwise[SEQ, Linear[DIM, DIM]], "xp"],
    Node["v", Tokenwise[SEQ, Linear[DIM, DIM]], "x"],  # NO pos on v
    Node["a", CrossAttention[DIM, HEADS, SEQ, SEQ, False], "q", "k", "v"],
    Node["ao", Tokenwise[SEQ, Linear[DIM, DIM]], "a"],  # out_proj
    Node["ad", Dropout[SEQ * DIM, P], "ao"],
    Node["r1", Add[SEQ * DIM], "x", "ad"],
    Node["n1", Tokenwise[SEQ, LayerNorm[DIM]], "r1"],
    Node["f1", Tokenwise[SEQ, Linear[DIM, FF]], "n1"],
    Node["fr", ReLU[SEQ * FF], "f1"],
    Node["fd", Dropout[SEQ * FF, P], "fr"],
    Node["f2", Tokenwise[SEQ, Linear[FF, DIM]], "fd"],
    Node["f2d", Dropout[SEQ * DIM, P], "f2"],
    Node["r2", Add[SEQ * DIM], "n1", "f2d"],
    Node["out", Tokenwise[SEQ, LayerNorm[DIM]], "r2"],
]


# Masked variant — ACT's CVAE encoder, whose `[CLS] | qpos | a_1..a_k` input is
# zero-padded past the end of an episode and passes `src_key_padding_mask`
# (`detr_vae.py:99`). `c` carries `[pos | key_valid]`; `key_valid` is 1.0 =
# attend, the inverse of torch's `key_padding_mask`, matching `ACTDataset`.
comptime DETREncoderMaskedGraph[
    DIM: Int, HEADS: Int, SEQ: Int, FF: Int, P: Float64 = 0.1
] = ComputeGraph[
    InputSlot["x", SEQ * DIM],
    InputSlot["c", SEQ * DIM + SEQ],  # [pos | key_valid]
    Node["pos", Slice[SEQ * DIM + SEQ, 0, SEQ * DIM], "c"],
    Node["msk", Slice[SEQ * DIM + SEQ, SEQ * DIM, SEQ * DIM + SEQ], "c"],
    Node["xp", Add[SEQ * DIM], "x", "pos"],
    Node["q", Tokenwise[SEQ, Linear[DIM, DIM]], "xp"],
    Node["k", Tokenwise[SEQ, Linear[DIM, DIM]], "xp"],
    Node["v", Tokenwise[SEQ, Linear[DIM, DIM]], "x"],
    Node[
        "a",
        CrossAttention[DIM, HEADS, SEQ, SEQ, True],
        "q",
        "k",
        "v",
        "msk",
    ],
    Node["ao", Tokenwise[SEQ, Linear[DIM, DIM]], "a"],
    Node["ad", Dropout[SEQ * DIM, P], "ao"],
    Node["r1", Add[SEQ * DIM], "x", "ad"],
    Node["n1", Tokenwise[SEQ, LayerNorm[DIM]], "r1"],
    Node["f1", Tokenwise[SEQ, Linear[DIM, FF]], "n1"],
    Node["fr", ReLU[SEQ * FF], "f1"],
    Node["fd", Dropout[SEQ * FF, P], "fr"],
    Node["f2", Tokenwise[SEQ, Linear[FF, DIM]], "fd"],
    Node["f2d", Dropout[SEQ * DIM, P], "f2"],
    Node["r2", Add[SEQ * DIM], "n1", "f2d"],
    Node["out", Tokenwise[SEQ, LayerNorm[DIM]], "r2"],
]


# ── decoder layer ────────────────────────────────────────────────────────
# `TransformerDecoderLayer.forward_post` (transformer.py:225):
#
#     q = k = tgt + query_pos
#     tgt2 = self_attn(q, k, value=tgt)
#     tgt  = norm1(tgt + dropout1(tgt2))
#     tgt2 = multihead_attn(query = tgt + query_pos,
#                           key   = memory + pos,
#                           value = memory)
#     tgt  = norm2(tgt + dropout2(tgt2))
#     tgt2 = linear2(dropout(relu(linear1(tgt))))
#     tgt  = norm3(tgt + dropout3(tgt2))
#
# ⚠ `query_pos` is added to the cross-attention QUERY as well as to the
# self-attention q/k — three uses of the same tensor. It is the second of them
# (`with_pos_embed(tgt, query_pos)` on the already-normed tgt) that is easy to
# drop when transcribing, and dropping it degrades the model without breaking
# any shape.

comptime DETRDecoderGraph[
    DIM: Int, HEADS: Int, Q_LEN: Int, KV_LEN: Int, FF: Int, P: Float64 = 0.1
] = ComputeGraph[
    InputSlot["x", Q_LEN * DIM],  # tgt
    InputSlot["c", Q_LEN * DIM + 2 * KV_LEN * DIM],
    # c = [query_pos | k_mem = memory+pos | memory]
    Node[
        "qpos",
        Slice[Q_LEN * DIM + 2 * KV_LEN * DIM, 0, Q_LEN * DIM],
        "c",
    ],
    Node[
        "kmem",
        Slice[
            Q_LEN * DIM + 2 * KV_LEN * DIM,
            Q_LEN * DIM,
            Q_LEN * DIM + KV_LEN * DIM,
        ],
        "c",
    ],
    Node[
        "mem",
        Slice[
            Q_LEN * DIM + 2 * KV_LEN * DIM,
            Q_LEN * DIM + KV_LEN * DIM,
            Q_LEN * DIM + 2 * KV_LEN * DIM,
        ],
        "c",
    ],
    # ── self-attention over the queries ──
    Node["xp", Add[Q_LEN * DIM], "x", "qpos"],
    Node["sq", Tokenwise[Q_LEN, Linear[DIM, DIM]], "xp"],
    Node["sk", Tokenwise[Q_LEN, Linear[DIM, DIM]], "xp"],
    Node["sv", Tokenwise[Q_LEN, Linear[DIM, DIM]], "x"],
    Node[
        "sa", CrossAttention[DIM, HEADS, Q_LEN, Q_LEN, False], "sq", "sk", "sv"
    ],
    Node["sao", Tokenwise[Q_LEN, Linear[DIM, DIM]], "sa"],
    Node["sad", Dropout[Q_LEN * DIM, P], "sao"],
    Node["sr", Add[Q_LEN * DIM], "x", "sad"],
    Node["n1", Tokenwise[Q_LEN, LayerNorm[DIM]], "sr"],
    # ── cross-attention onto the encoder memory ──
    Node["n1p", Add[Q_LEN * DIM], "n1", "qpos"],  # query_pos again
    Node["cq", Tokenwise[Q_LEN, Linear[DIM, DIM]], "n1p"],
    Node["ck", Tokenwise[KV_LEN, Linear[DIM, DIM]], "kmem"],
    Node["cv", Tokenwise[KV_LEN, Linear[DIM, DIM]], "mem"],  # NO pos on v
    Node[
        "ca",
        CrossAttention[DIM, HEADS, Q_LEN, KV_LEN, False],
        "cq",
        "ck",
        "cv",
    ],
    Node["cao", Tokenwise[Q_LEN, Linear[DIM, DIM]], "ca"],
    Node["cad", Dropout[Q_LEN * DIM, P], "cao"],
    Node["cr", Add[Q_LEN * DIM], "n1", "cad"],
    Node["n2", Tokenwise[Q_LEN, LayerNorm[DIM]], "cr"],
    # ── feed-forward ──
    Node["f1", Tokenwise[Q_LEN, Linear[DIM, FF]], "n2"],
    Node["fr", ReLU[Q_LEN * FF], "f1"],
    Node["fd", Dropout[Q_LEN * FF, P], "fr"],
    Node["f2", Tokenwise[Q_LEN, Linear[FF, DIM]], "fd"],
    Node["f2d", Dropout[Q_LEN * DIM, P], "f2"],
    Node["fr2", Add[Q_LEN * DIM], "n2", "f2d"],
    Node["out", Tokenwise[Q_LEN, LayerNorm[DIM]], "fr2"],
]


# ── Module wrappers ──────────────────────────────────────────────────────
# `ComputeGraph` is driven by slot NAME, not by an input pack, so it is not a
# `Module` and cannot be a `RepeatConditional` child. `GraphModule2` is the
# generic ARITY=2 bridge (the parameterized form of `primitives/decoder_block`'s
# hand-written wrapper). These three aliases are what the stacks and the loss
# graph actually use.

# ══════════════════════════════════════════════════════════════════════════
# Module wrappers — NAMED STRUCTS, not aliases
# ══════════════════════════════════════════════════════════════════════════
"""⚠ Each layer is a struct holding its `GraphModule2` as an INTERNAL
`comptime` member, rather than an alias for it.

`GraphModule2[IN0, IN1, OUT, GRAPH]` takes the graph as a PARAMETER, so an
alias puts the layer's entire ~18-node `ComputeGraph` — every `Linear`,
`LayerNorm`, `Dropout` and `CrossAttention` with all their parameters — into
the mangled name of everything that mentions it. `ACTLossGraph` mentions all
three, inside `RepeatConditional`, and `ComputeGraph[*DECLS]` mangles its whole
decl list into `__init__`'s symbol. The expansions multiply.

Behind a named struct the enclosing type sees only
`DETREncoderLayer,DIM=..,HEADS=..,SEQ=..,FF=..,P=..`. Same mechanism as
`ResNet18Backbone` and `primitives/decoder_block.mojo`. Behaviour is
unchanged — every method delegates to the same `GraphModule2`.
"""


struct DETREncoderLayer[
    DIM: Int, HEADS: Int, SEQ: Int, FF: Int, P: Float64 = 0.1
](Module):
    comptime Impl = GraphModule2[
        Self.SEQ * Self.DIM,
        Self.SEQ * Self.DIM,
        Self.SEQ * Self.DIM,
        DETREncoderGraph[Self.DIM, Self.HEADS, Self.SEQ, Self.FF, Self.P],
    ]
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self.Impl.IN_DIMS
    comptime OUT_DIM: Int = Self.SEQ * Self.DIM

    var impl: Self.Impl

    def __init__(out self):
        self.impl = Self.Impl()

    def __init__(out self, *, deinit move: Self):
        self.impl = move.impl^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var b = Self()
        b.impl = Self.Impl.make[target, INIT](ctx)
        return b^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.impl.forward[target, B, POLICY=POLICY](inputs, out, ctx)

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
        self.impl.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.impl.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.impl.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.impl.zero_grad[target](ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.impl.set_attr[ATTR](value)

    def polyak_from[
        target: StaticString
    ](mut self, mut src: Self, tau: Scalar[DT],
      ctx: Optional[DeviceContext]) raises:
        self.impl.polyak_from[target](src.impl, tau, ctx)


struct DETREncoderLayerMasked[
    DIM: Int, HEADS: Int, SEQ: Int, FF: Int, P: Float64 = 0.1
](Module):
    comptime Impl = GraphModule2[
        Self.SEQ * Self.DIM,
        Self.SEQ * Self.DIM + Self.SEQ,
        Self.SEQ * Self.DIM,
        DETREncoderMaskedGraph[
            Self.DIM, Self.HEADS, Self.SEQ, Self.FF, Self.P
        ],
    ]
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self.Impl.IN_DIMS
    comptime OUT_DIM: Int = Self.SEQ * Self.DIM

    var impl: Self.Impl

    def __init__(out self):
        self.impl = Self.Impl()

    def __init__(out self, *, deinit move: Self):
        self.impl = move.impl^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var b = Self()
        b.impl = Self.Impl.make[target, INIT](ctx)
        return b^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.impl.forward[target, B, POLICY=POLICY](inputs, out, ctx)

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
        self.impl.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.impl.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.impl.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.impl.zero_grad[target](ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.impl.set_attr[ATTR](value)

    def polyak_from[
        target: StaticString
    ](mut self, mut src: Self, tau: Scalar[DT],
      ctx: Optional[DeviceContext]) raises:
        self.impl.polyak_from[target](src.impl, tau, ctx)


struct DETRDecoderLayer[
    DIM: Int, HEADS: Int, Q_LEN: Int, KV_LEN: Int, FF: Int,
    P: Float64 = 0.1,
](Module):
    comptime Impl = GraphModule2[
        Self.Q_LEN * Self.DIM,
        Self.Q_LEN * Self.DIM + 2 * Self.KV_LEN * Self.DIM,
        Self.Q_LEN * Self.DIM,
        DETRDecoderGraph[
            Self.DIM, Self.HEADS, Self.Q_LEN, Self.KV_LEN, Self.FF, Self.P
        ],
    ]
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self.Impl.IN_DIMS
    comptime OUT_DIM: Int = Self.Q_LEN * Self.DIM

    var impl: Self.Impl

    def __init__(out self):
        self.impl = Self.Impl()

    def __init__(out self, *, deinit move: Self):
        self.impl = move.impl^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var b = Self()
        b.impl = Self.Impl.make[target, INIT](ctx)
        return b^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.impl.forward[target, B, POLICY=POLICY](inputs, out, ctx)

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
        self.impl.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.impl.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.impl.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.impl.zero_grad[target](ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.impl.set_attr[ATTR](value)

    def polyak_from[
        target: StaticString
    ](mut self, mut src: Self, tau: Scalar[DT],
      ctx: Optional[DeviceContext]) raises:
        self.impl.polyak_from[target](src.impl, tau, ctx)
