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
from .expert import SmolVLAExpert
from .block_attention import BlockCrossAttention


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


struct SmolVLADenoise[
    P: Int,
    S: Int,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    EW: Int = 720,
    EFF: Int = 2048,
    W: Int = SMOLLM_DIM,
    HEADS: Int = SMOLLM_HEADS,
    N_KV: Int = SMOLLM_KV_HEADS,
    HD: Int = SMOLLM_HEAD_DIM,
    THETA: Float64 = SMOLLM_THETA,
    SELF_EVERY: Int = 2,
    # A parameter, not `N_KV * HD`: Mojo does not unify `5 * (960 // 15)` with
    # the same value written another way, so every container in the loop must
    # name the SAME comptime expression. `SMOLLM_KV_W` is that one definition.
    KVW: Int = SMOLLM_KV_W,
](Movable):
    """One denoising step: the 50-token action suffix through the 16 expert
    layers, reading the prefix K/V the prefill left behind.

    Mode 3 of the three (`[None, suffix]`, cache present). Per layer:

        even i  SELF : k,v from the suffix's own stream; attend over
                       [prefix; suffix] with the block mask
        odd  i  CROSS: k,v are the VLM's CACHED prefix K/V pushed through the
                       expert's `[320, 320]` projections; attend over the prefix,
                       unmasked

    ⚠ **The two kinds rotate q from different origins.** A self layer uses
    absolute positions (the suffix sits after the prefix, so `POS_OFFSET = P`);
    a cross layer re-bases to zero — the reference does
    `expert_position_id -= min(expert_position_id)` before `apply_rope`, while
    the keys it attends to were rotated with their absolute prefix positions.
    Asymmetric, easy to miss, and shape-invisible if you get it wrong.

    ⚠ **A cross layer does NOT touch the cache.** Only the self layers extend
    it, and here they extend a *scratch copy* rather than the cache itself, so
    the reference's `crop(prefix_len)` has no counterpart and ten steps in a row
    see an identical prefix (`kv_cache.mojo`).

    Forward only, like `SmolVLAPrefill`.
    """

    comptime REP: Int = Self.HEADS // Self.N_KV
    comptime FULL: Int = Self.P + Self.S
    comptime XN: Int = Self.S * Self.EW      # the suffix stream, 720 wide
    comptime QN: Int = Self.S * Self.W       # queries, in the VLM's 960 geometry
    comptime KVN_S: Int = Self.S * Self.KVW
    comptime KVN_P: Int = Self.P * Self.KVW
    comptime KVN_F: Int = Self.FULL * Self.KVW
    comptime KX_F: Int = Self.FULL * Self.W
    comptime KX_P: Int = Self.P * Self.W
    comptime FFN: Int = Self.S * Self.EFF

    # q rotated from P for a self layer, from 0 for a cross layer.
    comptime RoPEQSelf = RoPE[Self.S, Self.HEADS, Self.HD, Self.THETA, Self.P]
    comptime RoPEQCross = RoPE[Self.S, Self.HEADS, Self.HD, Self.THETA, 0]
    comptime RoPEKSelf = RoPE[Self.S, Self.N_KV, Self.HD, Self.THETA, Self.P]
    comptime RepKVFull = RepeatKVHeads[
        Self.FULL, Self.N_KV, Self.REP, Self.HD
    ]
    comptime RepKVPre = RepeatKVHeads[Self.P, Self.N_KV, Self.REP, Self.HD]
    comptime AttnSelf = BlockCrossAttention[
        Self.W, Self.HEADS, Self.S, Self.FULL
    ]
    comptime AttnCross = BlockCrossAttention[
        Self.W, Self.HEADS, Self.S, Self.P
    ]
    comptime Glu = SwiGLU[Self.FFN]
    comptime GluCat = Concat[Self.FFN, Self.FFN]
    comptime Res = Add[Self.XN]

    var rope_q_self: Self.RoPEQSelf
    var rope_q_cross: Self.RoPEQCross
    var rope_k_self: Self.RoPEKSelf
    var rep_full_k: Self.RepKVFull
    var rep_full_v: Self.RepKVFull
    var rep_pre_k: Self.RepKVPre
    var rep_pre_v: Self.RepKVPre
    var attn_self: Self.AttnSelf
    var attn_cross: Self.AttnCross
    var glu: Self.Glu
    var glu_cat: Self.GluCat
    var res: Self.Res

    comptime X = 0
    comptime H = 1
    comptime Q = 2
    comptime QR = 3
    comptime KS = 4
    comptime VS = 5
    comptime KRS = 6
    comptime KXF = 7
    comptime VXF = 8
    comptime KP = 9
    comptime VP = 10
    comptime KXP = 11
    comptime VXP = 12
    comptime ATT = 13
    comptime AO = 14
    comptime X2 = 15
    comptime GATE = 16
    comptime UP = 17
    comptime CAT = 18
    comptime GLU = 19
    comptime DOWN = 20
    comptime N_SLOTS = 21

    var pool: TensorPack[Self.N_SLOTS]

    def __init__(out self):
        self.rope_q_self = Self.RoPEQSelf()
        self.rope_q_cross = Self.RoPEQCross()
        self.rope_k_self = Self.RoPEKSelf()
        self.rep_full_k = Self.RepKVFull()
        self.rep_full_v = Self.RepKVFull()
        self.rep_pre_k = Self.RepKVPre()
        self.rep_pre_v = Self.RepKVPre()
        self.attn_self = Self.AttnSelf()
        self.attn_cross = Self.AttnCross()
        self.glu = Self.Glu()
        self.glu_cat = Self.GluCat()
        self.res = Self.Res()
        self.pool = TensorPack[Self.N_SLOTS]()

    def __init__(out self, *, deinit move: Self):
        self.rope_q_self = move.rope_q_self^
        self.rope_q_cross = move.rope_q_cross^
        self.rope_k_self = move.rope_k_self^
        self.rep_full_k = move.rep_full_k^
        self.rep_full_v = move.rep_full_v^
        self.rep_pre_k = move.rep_pre_k^
        self.rep_pre_v = move.rep_pre_v^
        self.attn_self = move.attn_self^
        self.attn_cross = move.attn_cross^
        self.glu = move.glu^
        self.glu_cat = move.glu_cat^
        self.res = move.res^
        self.pool = move.pool^

    @staticmethod
    def make[
        target: StaticString
    ](
        ref self_mask: List[Scalar[DT]], ref cross_mask: List[Scalar[DT]],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """`self_mask` is `[S, P+S]`, `cross_mask` is `[S, P]` — both from
        `attn_mask.att_2d_mask` over the same `ar`, so they cannot disagree."""
        var d = Self()
        d.rope_q_self = Self.RoPEQSelf.make[target, Deterministic](ctx)
        d.rope_q_cross = Self.RoPEQCross.make[target, Deterministic](ctx)
        d.rope_k_self = Self.RoPEKSelf.make[target, Deterministic](ctx)
        d.rep_full_k = Self.RepKVFull.make[target, Deterministic](ctx)
        d.rep_full_v = Self.RepKVFull.make[target, Deterministic](ctx)
        d.rep_pre_k = Self.RepKVPre.make[target, Deterministic](ctx)
        d.rep_pre_v = Self.RepKVPre.make[target, Deterministic](ctx)
        d.attn_self = Self.AttnSelf.make[target](self_mask, ctx)
        d.attn_cross = Self.AttnCross.make[target](cross_mask, ctx)
        d.glu = Self.Glu.make[target, Deterministic](ctx)
        d.glu_cat = Self.GluCat.make[target, Deterministic](ctx)
        d.res = Self.Res.make[target, Deterministic](ctx)
        return d^

    def step[
        target: StaticString
    ](
        mut self,
        mut expert: SmolVLAExpert[
            Self.LAYERS, Self.EW, Self.EFF, Self.W, Self.KVW, Self.SELF_EVERY
        ],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, Self.P, Self.S, Self.N_KV, Self.HD, Self.B
        ],
        mut x: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`x` is `[B, S*EW]` suffix embeddings; `out` receives the final norm."""
        comptime TOK_S = Self.B * Self.S
        comptime TOK_P = Self.B * Self.P
        comptime TOK_F = Self.B * Self.FULL
        comptime XN = Self.B * Self.XN

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
            var is_self = (i % Self.SELF_EVERY) == 0
            var li = i // Self.SELF_EVERY

            # ── attention branch ─────────────────────────────────────────
            if is_self:
                ref L = expert.self_layers[li]
                L.input_layernorm.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.X]), self.pool[Self.H], ctx
                )
                L.q.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.Q], ctx
                )
                L.k.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.KS], ctx
                )
                L.v.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.VS], ctx
                )
                # absolute positions: the suffix sits after the prefix
                self.rope_q_self.forward[target, Self.B](
                    TensorRefs[1](self.pool[Self.Q]), self.pool[Self.QR], ctx
                )
                self.rope_k_self.forward[target, Self.B](
                    TensorRefs[1](self.pool[Self.KS]), self.pool[Self.KRS], ctx
                )
                # [prefix; suffix] into SCRATCH — the cache is not touched.
                cache.build_scratch[target](
                    i, self.pool[Self.KRS], self.pool[Self.VS], ctx
                )
                self.rep_full_k.forward[target, Self.B](
                    TensorRefs[1](cache.sk), self.pool[Self.KXF], ctx
                )
                self.rep_full_v.forward[target, Self.B](
                    TensorRefs[1](cache.sv), self.pool[Self.VXF], ctx
                )
                self.attn_self.forward[target, Self.B](
                    self.pool[Self.QR], self.pool[Self.KXF],
                    self.pool[Self.VXF], self.pool[Self.ATT], ctx,
                )
                L.o.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.ATT]), self.pool[Self.AO], ctx
                )
            else:
                ref L = expert.cross_layers[li]
                L.input_layernorm.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.X]), self.pool[Self.H], ctx
                )
                L.q.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.Q], ctx
                )
                # ⚠ re-based to 0, unlike the self layers
                self.rope_q_cross.forward[target, Self.B](
                    TensorRefs[1](self.pool[Self.Q]), self.pool[Self.QR], ctx
                )
                # k/v are the VLM's CACHED prefix K/V through [320,320] projs
                cache.read_layer_into[target](
                    i, self.pool[Self.KP], self.pool[Self.VP], ctx
                )
                L.k.forward[target, TOK_P](
                    TensorRefs[1](self.pool[Self.KP]), self.pool[Self.KS], ctx
                )
                L.v.forward[target, TOK_P](
                    TensorRefs[1](self.pool[Self.VP]), self.pool[Self.VS], ctx
                )
                self.rep_pre_k.forward[target, Self.B](
                    TensorRefs[1](self.pool[Self.KS]), self.pool[Self.KXP], ctx
                )
                self.rep_pre_v.forward[target, Self.B](
                    TensorRefs[1](self.pool[Self.VS]), self.pool[Self.VXP], ctx
                )
                self.attn_cross.forward[target, Self.B](
                    self.pool[Self.QR], self.pool[Self.KXP],
                    self.pool[Self.VXP], self.pool[Self.ATT], ctx,
                )
                L.o.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.ATT]), self.pool[Self.AO], ctx
                )

            self.res.forward[target, Self.B](
                TensorRefs[2](self.pool[Self.X], self.pool[Self.AO]),
                self.pool[Self.X2], ctx,
            )

            # ── MLP branch (identical for both kinds) ────────────────────
            if is_self:
                ref L = expert.self_layers[li]
                L.post_attention_layernorm.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.X2]), self.pool[Self.H], ctx
                )
                L.mlp.gate.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.GATE], ctx
                )
                L.mlp.up.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.UP], ctx
                )
            else:
                ref L = expert.cross_layers[li]
                L.post_attention_layernorm.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.X2]), self.pool[Self.H], ctx
                )
                L.mlp.gate.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.GATE], ctx
                )
                L.mlp.up.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.H]), self.pool[Self.UP], ctx
                )
            self.glu_cat.forward[target, Self.B](
                TensorRefs[2](self.pool[Self.UP], self.pool[Self.GATE]),
                self.pool[Self.CAT], ctx,
            )
            self.glu.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.GLU], ctx
            )
            if is_self:
                expert.self_layers[li].mlp.down.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.GLU]), self.pool[Self.DOWN],
                    ctx,
                )
            else:
                expert.cross_layers[li].mlp.down.forward[target, TOK_S](
                    TensorRefs[1](self.pool[Self.GLU]), self.pool[Self.DOWN],
                    ctx,
                )
            self.res.forward[target, Self.B](
                TensorRefs[2](self.pool[Self.X2], self.pool[Self.DOWN]),
                self.pool[Self.X], ctx,
            )

        expert.norm.forward[target, TOK_S](
            TensorRefs[1](self.pool[Self.X]), out, ctx
        )
