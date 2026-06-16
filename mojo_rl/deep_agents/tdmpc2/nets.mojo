"""TD-MPC2 network aliases (comptime Sequential compositions).

Mirrors `references/tdmpc2-main/tdmpc2/common/layers.py`:

  * `NormedLinear` = Linear → LayerNorm → Mish (reference `NormedLinear`
    with the default `act=nn.Mish`).
  * `NormedLinearSimNorm` = Linear → LayerNorm → SimNorm — the final layer
    of the encoder/dynamics trunks (reference `mlp(..., act=SimNorm(cfg))`).

Trunks (state obs; `num_enc_layers=2`):
  * `TDMPC2Encoder`  : obs → [NL(obs,enc)] → [NL_SimNorm(enc,latent)]
  * `TDMPC2Dynamics` : (z|a) → [NL(.,mlp)] → [NL(mlp,mlp)] → [NL_SimNorm(mlp,latent)]
  * `TDMPC2Reward`   : (z|a) → [NL(.,mlp)] → [NL(mlp,mlp)] → Linear(mlp,bins)
  * `TDMPC2QNet`     : (z|a) → [NL(.,mlp)] → [NL(mlp,mlp)] → Linear(mlp,bins)

`SN` is the SimNorm group **size** (reference `simnorm_dim`, default 8); the
SimNorm primitive takes the group **count** = `DIM // SN`.

Q-trunk dropout (item D, `docs/TDMPC2_DEEP_AGENTS2_PORT.md` §14.4): `TDMPC2QNet`
takes a `QP: Float64 = 0.0` and *always* threads a `Dropout` through its first
`NormedLinear` (reference `layers.mlp` applies `dropout*(i==0)` — first hidden
only, between Linear and LayerNorm). `QP=0.0` makes the Dropout numerically
identity (mask ≡ 1.0, fwd & grad ×1.0) so the default path is bit-identical;
the layer is *structurally always present* (Mojo can't conditionally alias two
type shapes — see memory `feedback_mojo_conditional_type_alias_blocked`).

⚠️ QP>0 caveats (experimental, off by default): (1) all NQ Q-heads share one
`QSEED` (per-head seeds would need per-head types, which can't live in
`List[QNetT]`), so their dropout masks are correlated rather than independent;
(2) the WM reverse-scan *recomputes* each step's forward before its `vjp`
(cache-light BPTT), so the grad-forward draws a *fresh* mask ≠ the loss-forward's
mask — the value-loss gradient is w.r.t. a different mask than the loss. Fine as
a noisy regularizer but not reference-faithful. Enable only to probe value-loss
instability.
"""

from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.primitives.mish import Mish
from mojo_rl.nn.primitives.sim_norm import SimNorm
from mojo_rl.nn.primitives.dropout import Dropout


# Fixed comptime seed for the Q-trunk dropout (shared across the NQ heads — see
# the module docstring's QP>0 caveat). Arbitrary; only used when QP>0.
comptime QDROPOUT_SEED: UInt64 = 0xD40720C2


comptime NormedLinear[I: Int, O: Int] = Sequential[
    Linear[I, O], LayerNorm[O], Mish[O],
]

# NormedLinear with a Dropout between the Linear and the LayerNorm (reference
# `NormedLinear(..., dropout=p)`: Linear → Dropout → LayerNorm → act). At p=0.0
# the Dropout is identity, so this is numerically equal to `NormedLinear`.
comptime NormedLinearDropout[I: Int, O: Int, QP: Float64] = Sequential[
    Linear[I, O], Dropout[O, QP, QDROPOUT_SEED], LayerNorm[O], Mish[O],
]

comptime NormedLinearSimNorm[I: Int, O: Int, SN: Int] = Sequential[
    Linear[I, O], LayerNorm[O], SimNorm[O, O // SN],
]

comptime TDMPC2Encoder[OBS: Int, ENC: Int, LATENT: Int, SN: Int] = Sequential[
    NormedLinear[OBS, ENC],
    NormedLinearSimNorm[ENC, LATENT, SN],
]

comptime TDMPC2Dynamics[
    LATENT: Int, ACT: Int, MLP: Int, SN: Int
] = Sequential[
    NormedLinear[LATENT + ACT, MLP],
    NormedLinear[MLP, MLP],
    NormedLinearSimNorm[MLP, LATENT, SN],
]

comptime TDMPC2Reward[LATENT: Int, ACT: Int, MLP: Int, BINS: Int] = Sequential[
    NormedLinear[LATENT + ACT, MLP],
    NormedLinear[MLP, MLP],
    Linear[MLP, BINS],
]

comptime TDMPC2QNet[
    LATENT: Int, ACT: Int, MLP: Int, BINS: Int, QP: Float64 = 0.0
] = Sequential[
    NormedLinearDropout[LATENT + ACT, MLP, QP],
    NormedLinear[MLP, MLP],
    Linear[MLP, BINS],
]

# Termination head (item B, §14.2): (z|a) → 1 logit (terminate/continue),
# trained with BCE vs the real `terminated` flag. Mirrors the reward trunk
# (predicts the transition's termination from state+action). Always present
# in the WM graph; its BCE loss coefficient defaults to 0 (non-episodic →
# bit-identical), so the head trains only when bce_coef > 0.
comptime TDMPC2Termination[LATENT: Int, ACT: Int, MLP: Int] = Sequential[
    NormedLinear[LATENT + ACT, MLP],
    NormedLinear[MLP, MLP],
    Linear[MLP, 1],
]

# Policy prior π(z): trunk → Linear(2·ACT) = [mean | log_std], consumed by
# RSample (tanh-squashed Gaussian). Reference `mlp(latent, 2*[mlp], 2*act)`.
comptime TDMPC2Policy[LATENT: Int, ACT: Int, MLP: Int] = Sequential[
    NormedLinear[LATENT, MLP],
    NormedLinear[MLP, MLP],
    Linear[MLP, 2 * ACT],
]
