# +--------------------------------------------------------------------------+ #
# | SmolVLA — one supervised training step over the action heads and expert
# +--------------------------------------------------------------------------+ #
"""Forward to a loss and back to every trainable weight, in one call.

This is the `train_expert_only` + `train_state_proj = False` regime: the
sixteen VLM layers and the SigLIP tower are frozen AND upstream of nothing
that is trained, so the prefix is a constant and the whole gradient lives
below the KV cache. `SmolVLADenoise.backward` forms dL/d(cached prefix K/V)
and drops it; restoring `train_state_proj = True` is what picks it up.

    x_t ──> action_in ──┐
                        ├─> token_concat ─> time_mlp_in ─> SiLU ─> time_mlp_out
    t ──> sinusoidal ───┘                                              │
                                                                       v
                              expert (16 layers, against the KV cache) │
                                                                       v
                                              action_out ──> v_t ──> MSE(u_t)

⚠ **ONE denoising step, at ONE sampled `t`.** Inference runs ten Euler steps;
training runs none. Unrolling the sampler here would be a different objective
(and sixteen times the memory) — see §10.2 of the plan for why prefill + one
step IS the reference's joint training pass.

⚠ **`t` is per SAMPLE.** The time embedding is `[B, EW]` and `token_concat`
gets `B_STRIDE = EW`. Passing a single vector, as inference does, would give
every element of the batch the same timestep, which trains a narrower
distribution than the sampler draws and would never show up as an error.

⚠ **Gradients ACCUMULATE** into every `Param.grd`, per the `nn` convention.
The caller zeroes before the step. Two `run` calls without a zero between them
give a two-batch gradient, which is either exactly what you wanted or a silent
factor of two.

⚠ **`denoise` must be `RECORD = True`** and must be the same instance the
forward used, because the tape it walks is that instance's. `backward` asserts
the first; nothing can assert the second.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.silu import SiLU

from .text import (
    SMOLLM_DIM, SMOLLM_KV_W, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_LAYERS,
    SMOLLM_HEADS, SMOLLM_THETA,
)
from .heads import SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM
from .expert import SmolVLAExpert, EXPERT_FF
from .kv_cache import SmolVLAKVCache
from .fused import SmolVLADenoise
from .flow import token_concat, token_split_a
from .embed import sinusoidal_time_embedding
from .flow_loss import flow_mse, mean_err


struct SmolVLATrainStep[
    CHUNK: Int,
    ADIM_REAL: Int,
    ADIM: Int = SMOLVLA_ACTION_DIM,
    EW: Int = SMOLVLA_EXPERT_W,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    # ⚠ The expert's geometry, parameterised rather than named from the
    # canonical aliases. Fixing them at the checkpoint's widths would make the
    # only testable fixture the real one — 50M parameters, ~700 MFLOP per
    # forward — and a finite-difference gate needs hundreds of forwards. The
    # defaults ARE the checkpoint, so a caller that wants the real model still
    # writes nothing.
    EFF: Int = EXPERT_FF,
    W: Int = SMOLLM_DIM,
    HEADS: Int = SMOLLM_HEADS,
    N_KV: Int = SMOLLM_KV_HEADS,
    HD: Int = SMOLLM_HEAD_DIM,
    THETA: Float64 = SMOLLM_THETA,
    KVW: Int = SMOLLM_KV_W,
](Movable):
    """`ADIM` is the padded action width (32); `ADIM_REAL` is the robot's."""

    comptime XN: Int = Self.B * Self.CHUNK * Self.ADIM
    comptime AN: Int = Self.B * Self.CHUNK * Self.EW
    comptime CN: Int = Self.B * Self.CHUNK * 2 * Self.EW
    comptime TN: Int = Self.B * Self.EW
    comptime Act = SiLU[Self.CHUNK * Self.EW]

    comptime AEMB = 0
    comptime TEMB = 1
    comptime CAT = 2
    comptime MID = 3
    comptime SIL = 4
    comptime OUT = 5
    comptime SUF = 6
    comptime V = 7
    comptime ERR = 8
    comptime GV = 9
    comptime GSUF = 10
    comptime GOUT = 11
    comptime GSIL = 12
    comptime GMID = 13
    comptime GCAT = 14
    comptime GAEMB = 15
    comptime GXT = 16
    comptime N_SLOTS = 17

    var act: Self.Act
    var pool: TensorPack[Self.N_SLOTS]

    def __init__(out self):
        comptime assert Self.ADIM_REAL <= Self.ADIM, (
            "SmolVLATrainStep: ADIM_REAL cannot exceed the padded ADIM"
        )
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

    def set_times[
        target: StaticString
    ](
        mut self, ref times: List[Float64],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Build the `[B, EW]` sinusoidal embedding for this step's timesteps.

        ⚠ Recomputed per step and uploaded, unlike the sampler's table: the
        sampler's ten timesteps are a comptime schedule, these are drawn. One
        upload per training step against a whole forward and backward is not
        where the time goes.
        """
        if len(times) != Self.B:
            raise Error(
                "SmolVLATrainStep.set_times: expected " + String(Self.B)
                + " timesteps, one per batch element, got "
                + String(len(times))
            )
        self.pool[Self.TEMB].ensure(Self.TN)
        for b in range(Self.B):
            var te = sinusoidal_time_embedding[Self.EW](times[b])
            for i in range(Self.EW):
                self.pool[Self.TEMB].data[b * Self.EW + i] = te[i]
        comptime if target != "cpu":
            self.pool[Self.TEMB].upload_resident(ctx.value())

    def run[
        target: StaticString, P: Int
    ](
        mut self,
        mut expert: SmolVLAExpert[
            Self.LAYERS, Self.EW, Self.EFF, Self.W, Self.KVW, 2
        ],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, P, Self.CHUNK, Self.N_KV, Self.HD, Self.B
        ],
        # ⚠ Spelled from Self's parameters, never re-derived. Mojo unifies
        # comptime parameters by EXPRESSION, so `5 * (960 // 15)` and `320`
        # are the same number and different types.
        mut denoise: SmolVLADenoise[
            P, Self.CHUNK, Self.B, Self.LAYERS, Self.EW, Self.EFF, Self.W,
            Self.HEADS, Self.N_KV, Self.HD, Self.THETA, 2, Self.KVW, True,
        ],
        mut action_in: Linear[Self.ADIM, Self.EW],
        mut time_mlp_in: Linear[2 * Self.EW, Self.EW],
        mut time_mlp_out: Linear[Self.EW, Self.EW],
        mut action_out: Linear[Self.EW, Self.ADIM],
        mut x_t: Tensor,
        mut u_t: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Float64:
        """One step. Returns the loss; call `set_times` first.

        ⚠ Returning the loss SYNCHRONISES — `mean_err` brings the per-element
        errors back to the host. That is one drain per training step, which is
        what you already pay to log a loss.
        """
        comptime TOK = Self.B * Self.CHUNK

        # ── forward: embed_suffix ────────────────────────────────────────
        action_in.forward[target, TOK](
            TensorRefs[1](x_t), self.pool[Self.AEMB], ctx
        )
        token_concat[
            target, Self.B, Self.CHUNK, Self.EW, Self.EW, Self.EW
        ](
            self.pool[Self.AEMB], self.pool[Self.TEMB], self.pool[Self.CAT],
            ctx,
        )
        time_mlp_in.forward[target, TOK](
            TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.MID], ctx
        )
        self.act.forward[target, Self.B](
            TensorRefs[1](self.pool[Self.MID]), self.pool[Self.SIL], ctx
        )
        time_mlp_out.forward[target, TOK](
            TensorRefs[1](self.pool[Self.SIL]), self.pool[Self.OUT], ctx
        )

        # ── forward: the expert, recording its tape ──────────────────────
        denoise.step[target](
            expert, cache, self.pool[Self.OUT], self.pool[Self.SUF], ctx
        )
        action_out.forward[target, TOK](
            TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.V], ctx
        )

        # ── the loss ─────────────────────────────────────────────────────
        flow_mse[
            target, Self.B, Self.CHUNK, Self.ADIM, Self.ADIM_REAL
        ](
            self.pool[Self.V], u_t, self.pool[Self.GV], self.pool[Self.ERR],
            ctx,
        )

        # ── backward ─────────────────────────────────────────────────────
        action_out.vjp[target, TOK](
            TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.GV],
            TensorRefs[1](self.pool[Self.GSUF]), ctx,
        )
        denoise.backward[target](
            expert, cache, self.pool[Self.GSUF], self.pool[Self.GOUT], ctx
        )
        time_mlp_out.vjp[target, TOK](
            TensorRefs[1](self.pool[Self.SIL]), self.pool[Self.GOUT],
            TensorRefs[1](self.pool[Self.GSIL]), ctx,
        )
        self.act.vjp[target, Self.B](
            TensorRefs[1](self.pool[Self.MID]), self.pool[Self.GSIL],
            TensorRefs[1](self.pool[Self.GMID]), ctx,
        )
        time_mlp_in.vjp[target, TOK](
            TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.GMID],
            TensorRefs[1](self.pool[Self.GCAT]), ctx,
        )
        # ⚠ Only the action half of the concat carries a gradient anywhere.
        # The time half's would be the sum over tokens, and the sinusoidal
        # embedding has no parameter for it to reach.
        token_split_a[target, Self.B, Self.CHUNK, Self.EW, Self.EW](
            self.pool[Self.GCAT], self.pool[Self.GAEMB], ctx
        )
        # `x_t` is DATA, so GXT is written and dropped — but `action_in.vjp`
        # needs a destination, and forming it is what accumulates action_in's
        # own weight gradient.
        action_in.vjp[target, TOK](
            TensorRefs[1](x_t), self.pool[Self.GAEMB],
            TensorRefs[1](self.pool[Self.GXT]), ctx,
        )

        return mean_err[
            target, Self.B, Self.CHUNK, Self.ADIM, Self.ADIM_REAL
        ](self.pool[Self.ERR], ctx)
