# +--------------------------------------------------------------------------+ #
# | SmolVLA — the SmolLM2 text tower (GQA + RoPE + SwiGLU + RMSNorm)
# +--------------------------------------------------------------------------+ #
"""The language half of SmolVLM2-500M, as shipped inside `lerobot/smolvla_base`.

Read from the checkpoint manifest and the backbone `config.json`:

    hidden 960 · 16 layers · 15 query heads over 5 KV heads (head_dim 64)
    SwiGLU ff 2560 · RMSNorm eps 1e-5 · rope_theta 100000 · vocab 49280

⚠ **16 layers, not 32.** The backbone's own `config.json` says
`num_hidden_layers = 32`; SmolVLA ships sixteen (`num_vlm_layers: 16`) and
truncates the tower to its first half. Depth comes from the checkpoint.

⚠ **The prefix pass is NOT causal.** `lerobot`'s `make_att_2d_masks` (an exact
copy of big_vision's) is prefix-LM: token i attends to j iff
`cumsum(att_masks)[j] <= cumsum[i]`. SmolVLA sets `att_masks = 0` across BOTH
the image and the language spans, so image+language is ONE BIDIRECTIONAL BLOCK.
`MaskedAttention`'s default all-zero (all-allow) mask is therefore already
correct here. Blocks only appear once state (`1`) and the action chunk (`1` per
token, hence causal within itself) join, which is the suffix's problem.

## Why the pieces are wired this way

Three places where the obvious composition is wrong, all for the same reason —
`TorchNameMap` is one-of-ours to one-of-theirs, and the checkpoint ships
projections SEPARATELY:

  * **q/k/v are three `Linear`s**, not our fused `Linear[dim, 3*inner]`. Under
    GQA they could not be fused anyway: q is 960-wide and k/v are 320-wide.
  * **gate/up are two `Linear`s** feeding `Concat2` into the existing `SwiGLU`,
    rather than one `Linear[dim, 2*ff]`. A two-input multiply would have been
    the other route and does NOT fit `BinaryElementOp`, whose trait carries one
    cached scalar per element while a product's backward needs both inputs.
  * **RoPE is applied to q and k only, never v**, and to k BEFORE the head
    broadcast — matching the reference, and cheaper by a factor of REP.

⚠ **`Concat2(up, gate)`, in that order.** Our `SwiGLU[H]` reads `[u ‖ v]` and
computes `u · silu(v)`; the reference computes `down(silu(gate) · up)`. So `u`
is *up* and `v` is *gate*. Reversed it computes `gate · silu(up)` — same shape,
same finiteness, a different function.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.combinators.tokenwise import Tokenwise
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.residual import Residual
from mojo_rl.nn.combinators.repeat import Repeat
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.rms_norm import RMSNorm
from mojo_rl.nn.primitives.swiglu import SwiGLU
from mojo_rl.nn.primitives.concat import Concat
from mojo_rl.nn.primitives.masked_attention import MaskedAttention
from mojo_rl.nn.primitives.rope import RoPE
from mojo_rl.nn.primitives.repeat_kv_heads import RepeatKVHeads


# ── the checkpoint's numbers ─────────────────────────────────────────────
comptime SMOLLM_DIM: Int = 960
comptime SMOLLM_HEADS: Int = 15
comptime SMOLLM_KV_HEADS: Int = 5
comptime SMOLLM_HEAD_DIM: Int = SMOLLM_DIM // SMOLLM_HEADS  # 64
comptime SMOLLM_REP: Int = SMOLLM_HEADS // SMOLLM_KV_HEADS  # 3
comptime SMOLLM_FF: Int = 2560
comptime SMOLLM_LAYERS: Int = 16
comptime SMOLLM_VOCAB: Int = 49280
comptime SMOLLM_THETA: Float64 = 100000.0
comptime SMOLLM_EPS: Float64 = 1e-5


struct GQAAttention[
    SEQ: Int,
    DIM: Int,
    N_HEADS: Int,
    N_KV: Int,
    HEAD_DIM: Int,
    THETA: Float64,
](Module):
    """Grouped-query self-attention: three projections, RoPE on q and k, the
    K/V heads broadcast to Q's count, then a packed-QKV attention core."""

    comptime ARITY: Int = 1
    comptime REP: Int = Self.N_HEADS // Self.N_KV
    comptime Q_W: Int = Self.N_HEADS * Self.HEAD_DIM
    comptime KV_W: Int = Self.N_KV * Self.HEAD_DIM
    comptime SEQ_DIM: Int = Self.SEQ * Self.DIM
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ_DIM)
    comptime OUT_DIM: Int = Self.SEQ_DIM

    comptime Graph = ComputeGraph[
        InputSlot["x", Self.SEQ_DIM],
        Node["q", Tokenwise[Self.SEQ, Linear[Self.DIM, Self.Q_W]], "x"],
        Node["k", Tokenwise[Self.SEQ, Linear[Self.DIM, Self.KV_W]], "x"],
        Node["v", Tokenwise[Self.SEQ, Linear[Self.DIM, Self.KV_W]], "x"],
        # RoPE on q and k only — never on v.
        Node["qr", RoPE[Self.SEQ, Self.N_HEADS, Self.HEAD_DIM, Self.THETA], "q"],
        Node["kr", RoPE[Self.SEQ, Self.N_KV, Self.HEAD_DIM, Self.THETA], "k"],
        # Broadcast AFTER the rotation: same result, REP times less rotating.
        Node[
            "kx",
            RepeatKVHeads[Self.SEQ, Self.N_KV, Self.REP, Self.HEAD_DIM],
            "kr",
        ],
        Node[
            "vx",
            RepeatKVHeads[Self.SEQ, Self.N_KV, Self.REP, Self.HEAD_DIM],
            "v",
        ],
        # k and v are now Q-wide, so the packed [Q|K|V] the attention core
        # expects is just a concat — no fused projection, map stays 1:1.
        Node[
            "qkv",
            Concat[Self.SEQ_DIM, Self.SEQ_DIM, Self.SEQ_DIM],
            "qr", "kx", "vx",
        ],
        Node[
            "a",
            MaskedAttention[Self.DIM, Self.N_HEADS, Self.SEQ],
            "qkv",
        ],
        Node["o", Tokenwise[Self.SEQ, Linear[Self.Q_W, Self.DIM]], "a"],
    ]

    var graph: Self.Graph

    def __init__(out self):
        comptime assert Self.N_HEADS % Self.N_KV == 0, (
            "GQAAttention: N_HEADS must be a multiple of N_KV"
        )
        comptime assert Self.DIM == Self.N_HEADS * Self.HEAD_DIM, (
            "GQAAttention: DIM must equal N_HEADS * HEAD_DIM"
        )
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "GQAAttention: target must be 'cpu' or 'gpu'"
        )
        var a = Self()
        a.graph = Self.Graph.make[target, INIT](ctx)
        return a^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.set_input["x", B](inputs[0], ctx)
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
        comptime DN = B * Self.SEQ_DIM
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        comptime if target == "cpu":
            gx.ensure(DN)
            for i in range(DN):
                gx.data[i] = self.graph.grad_input["x"]().data[i]
        else:
            var cc = ctx.value()
            gx.ensure_gpu(cc, DN)
            var src = self.graph.grad_input["x"]().dev.value(
            ).create_sub_buffer[DT](0, DN)
            var dst = gx.dev.value().create_sub_buffer[DT](0, DN)
            cc.enqueue_copy(dst, src)

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
    ](mut self, mut src: Self, tau: Scalar[DT],
      ctx: Optional[DeviceContext]) raises:
        self.graph.polyak_from[target](src.graph, tau, ctx)


struct SmolLMMLP[SEQ: Int, DIM: Int, FF: Int](Module):
    """`down(silu(gate(x)) · up(x))`, with gate and up as separate `Linear`s.

    The fan-out of x into two projections is why this is a graph and not a
    `Sequential`: the graph accumulates x's gradient from both branches.
    """

    comptime ARITY: Int = 1
    comptime SEQ_DIM: Int = Self.SEQ * Self.DIM
    comptime SEQ_FF: Int = Self.SEQ * Self.FF
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ_DIM)
    comptime OUT_DIM: Int = Self.SEQ_DIM

    comptime Graph = ComputeGraph[
        InputSlot["x", Self.SEQ_DIM],
        Node["gate", Tokenwise[Self.SEQ, Linear[Self.DIM, Self.FF]], "x"],
        Node["up", Tokenwise[Self.SEQ, Linear[Self.DIM, Self.FF]], "x"],
        # ⚠ (up, gate), not (gate, up): SwiGLU reads [u ‖ v] -> u · silu(v).
        Node["cat", Concat[Self.SEQ_FF, Self.SEQ_FF], "up", "gate"],
        Node["glu", SwiGLU[Self.SEQ_FF], "cat"],
        Node["down", Tokenwise[Self.SEQ, Linear[Self.FF, Self.DIM]], "glu"],
    ]

    var graph: Self.Graph

    def __init__(out self):
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var m = Self()
        m.graph = Self.Graph.make[target, INIT](ctx)
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.set_input["x", B](inputs[0], ctx)
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
        comptime DN = B * Self.SEQ_DIM
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        comptime if target == "cpu":
            gx.ensure(DN)
            for i in range(DN):
                gx.data[i] = self.graph.grad_input["x"]().data[i]
        else:
            var cc = ctx.value()
            gx.ensure_gpu(cc, DN)
            var src = self.graph.grad_input["x"]().dev.value(
            ).create_sub_buffer[DT](0, DN)
            var dst = gx.dev.value().create_sub_buffer[DT](0, DN)
            cc.enqueue_copy(dst, src)

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
    ](mut self, mut src: Self, tau: Scalar[DT],
      ctx: Optional[DeviceContext]) raises:
        self.graph.polyak_from[target](src.graph, tau, ctx)


# ── layer and tower ──────────────────────────────────────────────────────
comptime SmolLMLayer[
    SEQ: Int, DIM: Int, N_HEADS: Int, N_KV: Int, HEAD_DIM: Int, FF: Int,
    THETA: Float64,
] = Sequential[
    Residual[
        Sequential[
            Tokenwise[SEQ, RMSNorm[DIM]],
            GQAAttention[SEQ, DIM, N_HEADS, N_KV, HEAD_DIM, THETA],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[SEQ, RMSNorm[DIM]],
            SmolLMMLP[SEQ, DIM, FF],
        ]
    ],
]


comptime SmolLMTextTower[
    SEQ: Int,
    DIM: Int = SMOLLM_DIM,
    N_HEADS: Int = SMOLLM_HEADS,
    N_KV: Int = SMOLLM_KV_HEADS,
    HEAD_DIM: Int = SMOLLM_HEAD_DIM,
    FF: Int = SMOLLM_FF,
    LAYERS: Int = SMOLLM_LAYERS,
    THETA: Float64 = SMOLLM_THETA,
] = Sequential[
    Repeat[LAYERS, SmolLMLayer[SEQ, DIM, N_HEADS, N_KV, HEAD_DIM, FF, THETA]],
    Tokenwise[SEQ, RMSNorm[DIM]],
]
