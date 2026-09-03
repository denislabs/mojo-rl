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

from .vision import SigLIPVisionTower, SIGLIP_GRID, SIGLIP_DIM
from .text import (
    SMOLLM_DIM, SMOLLM_KV_W, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_LAYERS,
)
from .heads import SMOLVLA_CONNECTOR_IN, SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM
from .expert import SmolVLAExpert, EXPERT_FF
from .kv_cache import SmolVLAKVCache
from .fused import SmolVLADenoise
from .flow import EulerSchedule, token_concat
from .embed import sinusoidal_time_embedding


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
