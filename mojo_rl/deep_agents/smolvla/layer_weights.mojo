# +--------------------------------------------------------------------------+ #
# | SmolVLA — one decoder layer's weights, for both towers
# +--------------------------------------------------------------------------+ #
"""`DecoderLayerWeights` — the parameters of one SmolVLA decoder layer.

**The VLM text layer and the expert layer are the same topology at different
widths**, so they are the same struct. Only five numbers differ:

    stream  W    FF    QW   KVW  KV_IN   what it is
    ------  ---  ----  ---  ---  -----   ---------------------------------
    VLM     960  2560  960  320  960     text tower layer
    expert  720  2048  960  320  720     expert, self-attention  (even i)
    expert  720  2048  960  320  320     expert, cross-attention (odd i)

`KV_IN` is the whole story: a layer projects K/V either from its own stream or,
for the expert's odd layers, from the VLM's cached 320-wide K/V. `QW` is 960 for
both because the expert computes queries in the VLM's head geometry and projects
back down through `o` — which is what lets one attention serve two streams of
different widths.

## Why weights and not a `Module`

The fused loop needs each layer's K/V *between* the projection and the attention,
to write into the prefix cache. A composed `Sequential`/`Repeat` tower hands back
only its final activation, so there is nowhere to reach in. These structs own the
parameters and expose them by name; the loop that drives them lives in
`fused.mojo`.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.walkers import join_name
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.rms_norm import RMSNorm


struct DecoderMLP[W: Int, FF: Int](Movable):
    """`down(silu(gate(x)) * up(x))` — three separate `Linear`s, as shipped."""

    var gate: Linear[Self.W, Self.FF]
    var up: Linear[Self.W, Self.FF]
    var down: Linear[Self.FF, Self.W]

    def __init__(out self):
        self.gate = Linear[Self.W, Self.FF]()
        self.up = Linear[Self.W, Self.FF]()
        self.down = Linear[Self.FF, Self.W]()

    def __init__(out self, *, deinit move: Self):
        self.gate = move.gate^
        self.up = move.up^
        self.down = move.down^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var m = Self()
        m.gate = Linear[Self.W, Self.FF].make[target, INIT](ctx)
        m.up = Linear[Self.W, Self.FF].make[target, INIT](ctx)
        m.down = Linear[Self.FF, Self.W].make[target, INIT](ctx)
        return m^

    def walk[
        target: StaticString, V: ParamVisitor
    ](mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String) raises:
        self.gate.for_each_param[target](v, ctx, join_name(prefix, String("gate")))
        self.up.for_each_param[target](v, ctx, join_name(prefix, String("up")))
        self.down.for_each_param[target](v, ctx, join_name(prefix, String("down")))


struct DecoderLayerWeights[
    W: Int, FF: Int, QW: Int, KVW: Int, KV_IN: Int
](Movable):
    """One expert layer. `KV_IN` is the only difference between the two kinds:
    `W` (720) for a self-attention layer, `KVW` (320) for a cross-attention one,
    which reads the VLM's cached K/V instead of its own stream."""

    var input_layernorm: RMSNorm[Self.W]
    var q: Linear[Self.W, Self.QW]
    var k: Linear[Self.KV_IN, Self.KVW]
    var v: Linear[Self.KV_IN, Self.KVW]
    var o: Linear[Self.QW, Self.W]
    var post_attention_layernorm: RMSNorm[Self.W]
    var mlp: DecoderMLP[Self.W, Self.FF]

    def __init__(out self):
        self.input_layernorm = RMSNorm[Self.W]()
        self.q = Linear[Self.W, Self.QW]()
        self.k = Linear[Self.KV_IN, Self.KVW]()
        self.v = Linear[Self.KV_IN, Self.KVW]()
        self.o = Linear[Self.QW, Self.W]()
        self.post_attention_layernorm = RMSNorm[Self.W]()
        self.mlp = DecoderMLP[Self.W, Self.FF]()

    def __init__(out self, *, deinit move: Self):
        self.input_layernorm = move.input_layernorm^
        self.q = move.q^
        self.k = move.k^
        self.v = move.v^
        self.o = move.o^
        self.post_attention_layernorm = move.post_attention_layernorm^
        self.mlp = move.mlp^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var l = Self()
        l.input_layernorm = RMSNorm[Self.W].make[target, INIT](ctx)
        l.q = Linear[Self.W, Self.QW].make[target, INIT](ctx)
        l.k = Linear[Self.KV_IN, Self.KVW].make[target, INIT](ctx)
        l.v = Linear[Self.KV_IN, Self.KVW].make[target, INIT](ctx)
        l.o = Linear[Self.QW, Self.W].make[target, INIT](ctx)
        l.post_attention_layernorm = RMSNorm[Self.W].make[target, INIT](ctx)
        l.mlp = DecoderMLP[Self.W, Self.FF].make[target, INIT](ctx)
        return l^

    def walk[
        target: StaticString, V: ParamVisitor
    ](mut self, mut vis: V, ctx: Optional[DeviceContext],
      prefix: String) raises:
        self.input_layernorm.for_each_param[target](
            vis, ctx, join_name(prefix, String("input_layernorm"))
        )
        var sa = join_name(prefix, String("self_attn"))
        self.q.for_each_param[target](vis, ctx, join_name(sa, String("q")))
        self.k.for_each_param[target](vis, ctx, join_name(sa, String("k")))
        self.v.for_each_param[target](vis, ctx, join_name(sa, String("v")))
        self.o.for_each_param[target](vis, ctx, join_name(sa, String("o")))
        self.post_attention_layernorm.for_each_param[target](
            vis, ctx, join_name(prefix, String("post_attention_layernorm"))
        )
        self.mlp.walk[target](vis, ctx, join_name(prefix, String("mlp")))


