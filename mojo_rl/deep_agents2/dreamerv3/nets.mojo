"""DreamerV3 networks as composable nn2 type aliases (redesign).

The encoder, decoder, RSSM prior, and the reward/cont/value head MLPs are
ALL plain `Sequential` chains of existing nn2 primitives — no hand-written
forward/backward. This is the composable design the spike validated; the
loss heads (twohot / binary / symlog-mse) attach as separate loss ops that
produce the upstream gradient on the logits, after which `Sequential.vjp`
handles the rest.

Activation is `GELU` here (matches the PR4/5b fixtures + reference class
default). The size1m/dmc config uses SiLU — swap `GELU`→`SiLU` (both are
`Elementwise` aliases) when wiring the actual training config.

Layer counts pinned to v1 defaults (enc/dec 2 hidden, prior 2, head MLP 1);
`Repeat[n, Inner]` generalizes the depth later.
"""

from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn2.primitives.symlog import Symlog
from mojo_rl.nn2.core.element_op import ElementOp


# Inter-layer activation is a comptime op `A` (default `GELUOp`, matching the
# PR4/5b JAX fixtures so the validation spikes stay green). The production
# trainer/agent pass `SwishOp` (size1m/dmc config `act: silu`). `Elementwise
# [DIM, A]` == `GELU[DIM]` when A=GELUOp, `SiLU[DIM]` when A=SwishOp.


# Encoder: symlog(obs) → [Linear, RMSNorm, act] × 2 → tokens[U]
comptime DreamerEncoder[OBS: Int, U: Int, A: ElementOp = GELUOp] = Sequential[
    Symlog[OBS],
    Linear[OBS, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, U], RMSNorm[U], Elementwise[U, A],
]

# Decoder: concat([stoch,deter])[FEATIN] → [Linear,RMSNorm,act]×2 → pred[OBS]
comptime DreamerDecoder[FEATIN: Int, OBS: Int, U: Int, A: ElementOp = GELUOp] = Sequential[
    Linear[FEATIN, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, OBS],
]

# RSSM prior: deter[DETER] → [Linear,RMSNorm,act]×2 → logit[SC]
comptime DreamerPrior[DETER: Int, H: Int, SC: Int, A: ElementOp = GELUOp] = Sequential[
    Linear[DETER, H], RMSNorm[H], Elementwise[H, A],
    Linear[H, H], RMSNorm[H], Elementwise[H, A],
    Linear[H, SC],
]

# Reward head MLP (1 hidden): feat[FEAT] → [Linear,RMSNorm,act] → logits[BINS]
comptime DreamerRewardMLP[FEAT: Int, U: Int, BINS: Int, A: ElementOp = GELUOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, BINS],
]

# Cont head MLP (1 hidden): feat[FEAT] → [Linear,RMSNorm,act] → logit[1]
comptime DreamerContMLP[FEAT: Int, U: Int, A: ElementOp = GELUOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, 1],
]

# Value/slowvalue head (1 hidden): feat[FEAT] → twohot logits[BINS].
# Same shape as the reward MLP (symexp_twohot output).
comptime DreamerValue[FEAT: Int, U: Int, BINS: Int, A: ElementOp = GELUOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, BINS],
]

# Policy head (1 hidden): feat[FEAT] → [mean_raw[ACT], std_raw[ACT]] = 2·ACT.
# `bounded_normal` (dists.mojo) maps mean_raw→tanh, std_raw→sigmoid-scaled.
comptime DreamerPolicy[FEAT: Int, U: Int, ACT: Int, A: ElementOp = GELUOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, 2 * ACT],
]

# Discrete policy head (1 hidden): feat[FEAT] → logits[ACT] (ACT = #actions).
# `dists_discrete.mojo`'s unimix categorical (`OneHotDist`) maps logits→probs.
# Same MLP shape as the continuous head but a single ACT-wide logit output
# (vs 2·ACT mean/std). Used when the agent's `DISCRETE` flag is set.
comptime DreamerPolicyDiscrete[FEAT: Int, U: Int, ACT: Int, A: ElementOp = GELUOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, ACT],
]

# Unified policy head selected by the `DISCRETE` flag. The output WIDTH is an
# Int comptime ternary (ACT logits if discrete, else 2·ACT mean/std), which
# resolves to a single concrete `Sequential` type — unlike a type-level
# ternary, which Mojo joins (breaking Movable). Use this when threading a
# compile-time `DISCRETE` flag through the trainer / AC block / agent.
comptime DreamerPolicyHead[
    FEAT: Int, U: Int, ACT: Int, DISCRETE: Bool, A: ElementOp = GELUOp
] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, ACT if DISCRETE else 2 * ACT],
]
