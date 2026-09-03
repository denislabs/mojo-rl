# +--------------------------------------------------------------------------+ #
# | SmolVLA — the V1 inference policy: cameras + instruction -> action chunk
# +--------------------------------------------------------------------------+ #
"""Ties the pieces together into `sample_actions`.

    embed_prefix  ->  prefill (fills the KV cache)   ONCE
    embed_suffix  ->  denoise -> action_out          x10 Euler steps

⚠ **`lm_head` is not on this path.** It exists in the checkpoint and is mapped,
but SmolVLA's action head never uses it — the language modelling head is dead
weight for control. It is loaded only when something wants it.

⚠ **The prefix order is [images…, language…, state]** and the mask built from
`smolvla_ar` assumes exactly that. Assembling them in another order gives a
prefix of the same length whose block structure no longer matches the mask.
"""

from std.math import sqrt
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Initializer, Deterministic
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.silu import SiLU
from mojo_rl.nn.primitives.pixel_shuffle import PixelShuffle
from mojo_rl.nn.combinators.tokenwise import Tokenwise

from .vision import (
    SigLIPVisionTower, SIGLIP_GRID, SIGLIP_DIM, SIGLIP_IMG, SIGLIP_TOKENS,
)
from .text import (
    SMOLLM_DIM, SMOLLM_KV_W, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_LAYERS,
)
from .heads import SMOLVLA_CONNECTOR_IN, SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM
from .expert import SmolVLAExpert, EXPERT_FF
from .kv_cache import SmolVLAKVCache
from .fused import SmolVLADenoise
from .flow import EulerSchedule, token_concat
from .embed import sinusoidal_time_embedding, embed_language_tokens


def copy_into[
    target: StaticString
](
    mut dst: Tensor, dst_off: Int, mut src: Tensor, n: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Place `n` elements of `src` at `dst[dst_off:]` — prefix assembly's only
    primitive. Kept explicit rather than folded into a `Concat` because the
    prefix is built from a variable number of cameras."""
    comptime if target == "cpu":
        for i in range(n):
            dst.data[dst_off + i] = src.data[i]
    else:
        var c = ctx.value()
        c.enqueue_copy(
            dst.dev.value().create_sub_buffer[DT](dst_off, n),
            src.dev.value().create_sub_buffer[DT](0, n),
        )


struct SmolVLAActionSampler[
    CHUNK: Int,
    ADIM: Int = SMOLVLA_ACTION_DIM,
    EW: Int = SMOLVLA_EXPERT_W,
    STEPS: Int = 10,
    B: Int = 1,
](Movable):
    """The ten-step Euler loop, with `embed_suffix` and `action_out` inside it.

    Each step is: project the current noisy chunk, fuse it with the timestep,
    push it through the expert against the cached prefix, project back to action
    space, and take one negative Euler step.
    """

    comptime XN: Int = Self.B * Self.CHUNK * Self.ADIM
    comptime AN: Int = Self.B * Self.CHUNK * Self.EW
    comptime CN: Int = Self.B * Self.CHUNK * 2 * Self.EW
    comptime Sched = EulerSchedule[Self.STEPS]
    comptime Act = SiLU[Self.CHUNK * Self.EW]

    comptime XT = 0
    comptime AEMB = 1
    comptime TEMB = 2
    comptime CAT = 3
    comptime MID = 4
    comptime SUF = 5
    comptime OUT = 6
    comptime V = 7
    comptime N_SLOTS = 8

    var act: Self.Act
    var pool: TensorPack[Self.N_SLOTS]

    def __init__(out self):
        self.act = Self.Act()
        self.pool = TensorPack[Self.N_SLOTS]()

    def __init__(out self, *, deinit move: Self):
        self.act = move.act^
        self.pool = move.pool^

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.act = Self.Act.make[target, Deterministic](ctx)
        return s^

    def sample[
        target: StaticString, P: Int
    ](
        mut self,
        # ⚠ These name the canonical aliases directly rather than re-deriving
        # the widths. Mojo unifies comptime parameters by EXPRESSION, not by
        # value: `5 * (960 // 15)` and `320` are the same number and different
        # types, so every container in the loop must be spelled from the same
        # definitions.
        mut expert: SmolVLAExpert[
            SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, 2,
        ],
        mut cache: SmolVLAKVCache[
            SMOLLM_LAYERS, P, Self.CHUNK, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
            Self.B,
        ],
        mut denoise: SmolVLADenoise[P, Self.CHUNK, Self.B],
        mut action_in: Linear[Self.ADIM, Self.EW],
        mut time_mlp_in: Linear[2 * Self.EW, Self.EW],
        mut time_mlp_out: Linear[Self.EW, Self.EW],
        mut action_out: Linear[Self.EW, Self.ADIM],
        mut noise: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`noise` is x_1, `[B, CHUNK*ADIM]`; `out` receives x_0."""
        comptime TOK = Self.B * Self.CHUNK
        comptime if target == "cpu":
            self.pool[Self.XT].ensure(Self.XN)
            for i in range(Self.XN):
                self.pool[Self.XT].data[i] = noise.data[i]
        else:
            var c = ctx.value()
            self.pool[Self.XT].ensure_gpu(c, Self.XN)
            c.enqueue_copy(
                self.pool[Self.XT].dev.value().create_sub_buffer[DT](
                    0, Self.XN
                ),
                noise.dev.value().create_sub_buffer[DT](0, Self.XN),
            )

        for step in range(Self.STEPS):
            var t = Self.Sched.time_at(step)

            # ── embed_suffix ─────────────────────────────────────────────
            action_in.forward[target, TOK](
                TensorRefs[1](self.pool[Self.XT]), self.pool[Self.AEMB], ctx
            )
            # The time embedding is one vector, computed in Float64 and shared
            # by every token — `time_emb[:, None, :].expand_as(action_emb)`.
            var te = sinusoidal_time_embedding[Self.EW](t)
            self.pool[Self.TEMB].ensure(Self.EW)
            for i in range(Self.EW):
                self.pool[Self.TEMB].data[i] = te[i]
            comptime if target != "cpu":
                self.pool[Self.TEMB].upload(ctx.value())
            # ⚠ per token: [action_t ‖ time], NOT [all actions ‖ all times]
            token_concat[target, Self.B, Self.CHUNK, Self.EW, Self.EW](
                self.pool[Self.AEMB], self.pool[Self.TEMB],
                self.pool[Self.CAT], ctx,
            )
            time_mlp_in.forward[target, TOK](
                TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.MID], ctx
            )
            # in -> SiLU -> out: a plain two-layer MLP, NOT the decoder's SwiGLU
            self.act.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.MID]), self.pool[Self.SUF], ctx
            )
            time_mlp_out.forward[target, TOK](
                TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.OUT], ctx
            )

            # ── the expert, against the cached prefix ────────────────────
            denoise.step[target](
                expert, cache, self.pool[Self.OUT], self.pool[Self.SUF], ctx
            )
            action_out.forward[target, TOK](
                TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.V], ctx
            )

            # ── one negative Euler step ──────────────────────────────────
            Self.Sched.advance[target, Self.XN](
                self.pool[Self.XT], self.pool[Self.V], ctx
            )

        comptime if target == "cpu":
            out.ensure(Self.XN)
            for i in range(Self.XN):
                out.data[i] = self.pool[Self.XT].data[i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, Self.XN)
            c.enqueue_copy(
                out.dev.value().create_sub_buffer[DT](0, Self.XN),
                self.pool[Self.XT].dev.value().create_sub_buffer[DT](
                    0, Self.XN
                ),
            )


struct SmolVLAPrefixEmbed[
    N_CAM: Int,
    N_LANG: Int,
    B: Int = 1,
    W: Int = SMOLLM_DIM,
    IMG_TOK: Int = 64,
](Movable):
    """`[images…, language…, state]` -> `[B, P * 960]`, the VLM stream.

    Per camera: SigLIP tower -> PixelShuffle -> connector -> **x sqrt(960)**.
    Language: an embedding row gather, also **x sqrt(960)**. State: `state_proj`,
    **not** scaled.

    ⚠ **The order is load-bearing.** `smolvla_ar(N_CAM*IMG_TOK, N_LANG, 1, chunk)`
    describes exactly this layout, and the prefill mask is built from it. Writing
    the segments in another order gives a prefix of the same length whose block
    structure no longer matches its mask — finite, right-shaped, wrong.

    ⚠ **The sqrt(960) applies AFTER the connector**, because the reference takes
    `img_emb_dim` from `embed_image`'s output (960), not from the vision tower's
    768. Scaling by sqrt(768) would be a plausible misreading and ~14% off.

    ⚠ The `add_image_special_tokens = False` branch is assumed. Enabling it would
    put a start and an end token around every camera, changing `P` and every mask
    derived from it.
    """

    comptime IMG_N: Int = Self.IMG_TOK * Self.W
    comptime P: Int = Self.N_CAM * Self.IMG_TOK + Self.N_LANG + 1
    comptime OUT_N: Int = Self.B * Self.P * Self.W
    comptime VIS_IN: Int = 3 * SIGLIP_IMG * SIGLIP_IMG
    comptime VIS_OUT: Int = SIGLIP_TOKENS * SIGLIP_DIM
    comptime Shuffle = PixelShuffle[SIGLIP_GRID, SIGLIP_DIM, 4]

    var shuffle: Self.Shuffle
    var vis: Tensor
    var shuf: Tensor
    var conn: Tensor
    var lang: Tensor
    var st: Tensor

    def __init__(out self):
        comptime assert Self.N_CAM >= 1, "SmolVLAPrefixEmbed: need a camera"
        self.shuffle = Self.Shuffle()
        self.vis = Tensor()
        self.shuf = Tensor()
        self.conn = Tensor()
        self.lang = Tensor()
        self.st = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.shuffle = move.shuffle^
        self.vis = move.vis^
        self.shuf = move.shuf^
        self.conn = move.conn^
        self.lang = move.lang^
        self.st = move.st^

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var p = Self()
        p.shuffle = Self.Shuffle.make[target, Deterministic](ctx)
        return p^

    def run[
        target: StaticString, VOCAB: Int, CONN_IN: Int, SDIM: Int
    ](
        mut self,
        mut vision: SigLIPVisionTower[],
        mut connector: Tokenwise[Self.IMG_TOK, Linear[CONN_IN, Self.W]],
        mut embed_weight: Tensor,
        mut state_proj: Linear[SDIM, Self.W],
        mut images: Tensor,
        ref lang_ids: List[Int],
        mut state: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`images` is `[N_CAM, 3*512*512]` (batch 1 per camera, as SmolVLA
        runs them); `lang_ids` are PRE-TOKENISED ids."""
        if len(lang_ids) != Self.N_LANG:
            raise Error(
                "SmolVLAPrefixEmbed: expected " + String(Self.N_LANG)
                + " language ids, got " + String(len(lang_ids))
            )
        comptime if target == "cpu":
            out.ensure(Self.OUT_N)
        else:
            out.ensure_gpu(ctx.value(), Self.OUT_N)

        var scale = sqrt(Scalar[DT](Self.W))
        var off = 0

        # ── cameras ──────────────────────────────────────────────────────
        for cam in range(Self.N_CAM):
            # one camera's pixels; `images` holds them back to back
            var one = Tensor.alloc(Self.VIS_IN)
            for i in range(Self.VIS_IN):
                one.data[i] = images.data[cam * Self.VIS_IN + i]
            comptime if target != "cpu":
                one.upload(ctx.value())
            vision.forward[target, 1](TensorRefs[1](one), self.vis, ctx)
            self.shuffle.forward[target, 1](
                TensorRefs[1](self.vis), self.shuf, ctx
            )
            connector.forward[target, 1](
                TensorRefs[1](self.shuf), self.conn, ctx
            )
            # ⚠ sqrt of the CONNECTOR's width (960), not the tower's (768)
            comptime if target == "cpu":
                for i in range(Self.IMG_N):
                    self.conn.data[i] = self.conn.data[i] * scale
            else:
                self.conn.download(ctx.value())
                for i in range(Self.IMG_N):
                    self.conn.data[i] = self.conn.data[i] * scale
                self.conn.upload(ctx.value())
            copy_into[target](out, off, self.conn, Self.IMG_N, ctx)
            off += Self.IMG_N

        # ── language: a row gather, scaled the same way ──────────────────
        embed_language_tokens[VOCAB, Self.W](
            embed_weight, lang_ids, self.lang, True
        )
        comptime if target != "cpu":
            self.lang.upload(ctx.value())
        copy_into[target](out, off, self.lang, Self.N_LANG * Self.W, ctx)
        off += Self.N_LANG * Self.W

        # ── state: projected, and NOT scaled ─────────────────────────────
        state_proj.forward[target, Self.B](
            TensorRefs[1](state), self.st, ctx
        )
        copy_into[target](out, off, self.st, Self.W, ctx)
        off += Self.W

        if off != Self.OUT_N:
            raise Error(
                "SmolVLAPrefixEmbed: wrote " + String(off) + " of "
                + String(Self.OUT_N) + " — the segment widths do not sum to P*W"
            )
