# +--------------------------------------------------------------------------+ #
# | SmolVLA — the fused loop, prefill pass
# +--------------------------------------------------------------------------+ #
"""Run the image+language+state prefix through the 16 VLM layers ONCE, filling
the per-layer KV cache the ten denoising steps then read.

This is mode 2 of the three in `smolvlm_with_expert.py` (`[prefix, None]`,
`fill_kv_cache=True`). Every layer runs ordinary self-attention over the prefix
and stores its **post-RoPE** K/V.

    for i in range(16):
        h  = layers[i].input_layernorm(x)
        q  = q(h);  k = k(h);  v = v(h)
        q  = RoPE(q);  k = RoPE(k)            <- rotate BEFORE caching
        cache.write_prefix(i, k, v)
        a  = attention(q, repeat_kv(k), repeat_kv(v), block_mask)
        x  = x + o(a)
        x  = x + mlp(post_attention_layernorm(x))
    x = norm(x)

⚠ **The K/V written to the cache are the post-RoPE ones**, and they are written
*before* the head broadcast — the cache holds 5 KV heads, not 15. Caching the
broadcast copies would be 3x the memory for the same information, and caching
pre-RoPE keys would be shape-identical and wrong (see `kv_cache.mojo`).

⚠ **The mask is not causal.** It is the prefix-LM block mask from
`attn_mask.mojo`: image and language form one bidirectional block, state its own.
Passing a causal mask here would run, and would change what every prefix token
sees.

Forward-only: this is V1's inference path, and none of the leaves are asked for
a vjp. Training (mode 1, both streams concatenated) reuses these weights through
a different driver.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Initializer, Deterministic
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.rope import RoPE
from mojo_rl.nn.primitives.repeat_kv_heads import RepeatKVHeads
from mojo_rl.nn.primitives.masked_attention import MaskedAttention
from mojo_rl.nn.primitives.concat import Concat
from mojo_rl.nn.primitives.swiglu import SwiGLU
from mojo_rl.nn.primitives.add import Add

from .text import (
    SmolVLMTextLayers, SMOLLM_DIM, SMOLLM_FF, SMOLLM_HEADS, SMOLLM_KV_HEADS,
    SMOLLM_HEAD_DIM, SMOLLM_REP, SMOLLM_LAYERS, SMOLLM_THETA, SMOLLM_KV_W,
)
from .kv_cache import SmolVLAKVCache


struct SmolVLAPrefill[
    P: Int,
    SUFFIX: Int,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    W: Int = SMOLLM_DIM,
    FF: Int = SMOLLM_FF,
    HEADS: Int = SMOLLM_HEADS,
    N_KV: Int = SMOLLM_KV_HEADS,
    HD: Int = SMOLLM_HEAD_DIM,
    THETA: Float64 = SMOLLM_THETA,
](Movable):
    """The prefill driver. Owns the stateless leaves and every scratch slab, so
    one instance runs all 16 layers without allocating in the loop."""

    comptime REP: Int = Self.HEADS // Self.N_KV
    comptime KVW: Int = Self.N_KV * Self.HD
    comptime XN: Int = Self.P * Self.W
    comptime KVN: Int = Self.P * Self.KVW
    comptime FFN: Int = Self.P * Self.FF

    comptime RoPEQ = RoPE[Self.P, Self.HEADS, Self.HD, Self.THETA]
    comptime RoPEK = RoPE[Self.P, Self.N_KV, Self.HD, Self.THETA]
    comptime RepKV = RepeatKVHeads[Self.P, Self.N_KV, Self.REP, Self.HD]
    comptime Attn = MaskedAttention[Self.W, Self.HEADS, Self.P]
    comptime Pack = Concat[Self.XN, Self.XN, Self.XN]
    comptime Glu = SwiGLU[Self.FFN]
    comptime GluCat = Concat[Self.FFN, Self.FFN]
    comptime Res = Add[Self.XN]

    var rope_q: Self.RoPEQ
    var rope_k: Self.RoPEK
    var rep_k: Self.RepKV
    var rep_v: Self.RepKV
    var attn: Self.Attn
    var pack: Self.Pack
    var glu: Self.Glu
    var glu_cat: Self.GluCat
    var res: Self.Res

    # ── scratch ──────────────────────────────────────────────────────────
    # ONE pool, not eighteen fields. `TensorRefs[2]`/`[3]` require every operand
    # to share a single origin, and two `self.` fields do not — `TensorPack`'s
    # subscript returns a `MutAnyOrigin` ref precisely so adjacent slabs can be
    # passed together. Same reason `ComputeGraph` copies its inputs into an
    # owned pool rather than referencing the caller's tensors.
    comptime X = 0        # the running activation
    comptime H = 1
    comptime Q = 2
    comptime K = 3
    comptime V = 4
    comptime QR = 5
    comptime KR = 6
    comptime KX = 7
    comptime VX = 8
    comptime PACKED = 9
    comptime ATT = 10
    comptime AO = 11
    comptime X2 = 12
    comptime GATE = 13
    comptime UP = 14
    comptime CAT = 15
    comptime GLU = 16
    comptime DOWN = 17
    comptime N_SLOTS = 18

    var pool: TensorPack[Self.N_SLOTS]

    def __init__(out self):
        comptime assert Self.W == Self.HEADS * Self.HD, (
            "SmolVLAPrefill: W must equal HEADS * HD"
        )
        self.rope_q = Self.RoPEQ()
        self.rope_k = Self.RoPEK()
        self.rep_k = Self.RepKV()
        self.rep_v = Self.RepKV()
        self.attn = Self.Attn()
        self.pack = Self.Pack()
        self.glu = Self.Glu()
        self.glu_cat = Self.GluCat()
        self.res = Self.Res()
        self.pool = TensorPack[Self.N_SLOTS]()

    def __init__(out self, *, deinit move: Self):
        self.rope_q = move.rope_q^
        self.rope_k = move.rope_k^
        self.rep_k = move.rep_k^
        self.rep_v = move.rep_v^
        self.attn = move.attn^
        self.pack = move.pack^
        self.glu = move.glu^
        self.glu_cat = move.glu_cat^
        self.res = move.res^
        self.pool = move.pool^

    @staticmethod
    def make[
        target: StaticString
    ](
        ref mask: List[Scalar[DT]], ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        """`mask` is the `[P, P]` additive block mask from `attn_mask.mojo`."""
        var s = Self()
        s.rope_q = Self.RoPEQ.make[target, Deterministic](ctx)
        s.rope_k = Self.RoPEK.make[target, Deterministic](ctx)
        s.rep_k = Self.RepKV.make[target, Deterministic](ctx)
        s.rep_v = Self.RepKV.make[target, Deterministic](ctx)
        s.attn = Self.Attn.make[target, Deterministic](ctx)
        s.attn.set_mask(mask.copy(), ctx)
        s.pack = Self.Pack.make[target, Deterministic](ctx)
        s.glu = Self.Glu.make[target, Deterministic](ctx)
        s.glu_cat = Self.GluCat.make[target, Deterministic](ctx)
        s.res = Self.Res.make[target, Deterministic](ctx)
        return s^

    def run[
        target: StaticString
    ](
        mut self,
        mut tower: SmolVLMTextLayers[Self.LAYERS, Self.W, Self.FF, Self.KVW],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, Self.P, Self.SUFFIX, Self.N_KV, Self.HD, Self.B
        ],
        mut x: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`x` is `[B, P*W]` prefix embeddings; `out` receives the final norm."""
        comptime TOK = Self.B * Self.P   # Linear/RMSNorm run per token
        comptime XN = Self.B * Self.XN

        # Seed the pool. The running activation lives in a slot from here on, so
        # every residual pairs two slabs of one origin.
        comptime if target == "cpu":
            self.pool[Self.X].ensure(XN)
            for i in range(XN):
                self.pool[Self.X].data[i] = x.data[i]
        else:
            var c = ctx.value()
            self.pool[Self.X].ensure_gpu(c, XN)
            c.enqueue_copy(
                self.pool[Self.X].dev.value().create_sub_buffer[DT](0, XN),
                x.dev.value().create_sub_buffer[DT](0, XN),
            )

        for i in range(Self.LAYERS):
            # ── attention branch ─────────────────────────────────────────
            tower.layers[i].input_layernorm.forward[target, TOK](
                TensorRefs[1](self.pool[Self.X]), self.pool[Self.H], ctx
            )
            tower.layers[i].q.forward[target, TOK](
                TensorRefs[1](self.pool[Self.H]), self.pool[Self.Q], ctx
            )
            tower.layers[i].k.forward[target, TOK](
                TensorRefs[1](self.pool[Self.H]), self.pool[Self.K], ctx
            )
            tower.layers[i].v.forward[target, TOK](
                TensorRefs[1](self.pool[Self.H]), self.pool[Self.V], ctx
            )
            # RoPE on q and k, never on v.
            self.rope_q.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.Q]), self.pool[Self.QR], ctx
            )
            self.rope_k.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.K]), self.pool[Self.KR], ctx
            )
            # Cache the POST-RoPE, PRE-broadcast K/V: 5 heads, not 15.
            cache.write_prefix[target](
                i, self.pool[Self.KR], self.pool[Self.V], ctx
            )
            self.rep_k.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.KR]), self.pool[Self.KX], ctx
            )
            self.rep_v.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.V]), self.pool[Self.VX], ctx
            )
            self.pack.forward[target, Self.B](
                TensorRefs[3](
                    self.pool[Self.QR], self.pool[Self.KX], self.pool[Self.VX]
                ),
                self.pool[Self.PACKED], ctx,
            )
            self.attn.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.PACKED]), self.pool[Self.ATT], ctx
            )
            tower.layers[i].o.forward[target, TOK](
                TensorRefs[1](self.pool[Self.ATT]), self.pool[Self.AO], ctx
            )
            self.res.forward[target, Self.B](
                TensorRefs[2](self.pool[Self.X], self.pool[Self.AO]),
                self.pool[Self.X2], ctx,
            )

            # ── MLP branch ───────────────────────────────────────────────
            tower.layers[i].post_attention_layernorm.forward[target, TOK](
                TensorRefs[1](self.pool[Self.X2]), self.pool[Self.H], ctx
            )
            tower.layers[i].mlp.gate.forward[target, TOK](
                TensorRefs[1](self.pool[Self.H]), self.pool[Self.GATE], ctx
            )
            tower.layers[i].mlp.up.forward[target, TOK](
                TensorRefs[1](self.pool[Self.H]), self.pool[Self.UP], ctx
            )
            # ⚠ (up, gate): SwiGLU reads [u ‖ v] -> u * silu(v), and the
            # reference computes down(silu(gate) * up). Reversed is the same
            # shape and a different function.
            self.glu_cat.forward[target, Self.B](
                TensorRefs[2](self.pool[Self.UP], self.pool[Self.GATE]),
                self.pool[Self.CAT], ctx,
            )
            self.glu.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.GLU], ctx
            )
            tower.layers[i].mlp.down.forward[target, TOK](
                TensorRefs[1](self.pool[Self.GLU]), self.pool[Self.DOWN], ctx
            )
            self.res.forward[target, Self.B](
                TensorRefs[2](self.pool[Self.X2], self.pool[Self.DOWN]),
                self.pool[Self.X], ctx,
            )

        tower.norm.forward[target, TOK](
            TensorRefs[1](self.pool[Self.X]), out, ctx
        )
