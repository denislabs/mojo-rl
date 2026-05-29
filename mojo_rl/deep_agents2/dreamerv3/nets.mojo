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
from mojo_rl.nn2.primitives.gelu import GELU
from mojo_rl.nn2.primitives.symlog import Symlog


# Encoder: symlog(obs) → [Linear, RMSNorm, GELU] × 2 → tokens[U]
comptime DreamerEncoder[OBS: Int, U: Int] = Sequential[
    Symlog[OBS],
    Linear[OBS, U], RMSNorm[U], GELU[U],
    Linear[U, U], RMSNorm[U], GELU[U],
]

# Decoder: concat([stoch,deter])[FEATIN] → [Linear,RMSNorm,GELU]×2 → pred[OBS]
# (symlog_mse loss attaches separately; pred is the symlog-space output)
comptime DreamerDecoder[FEATIN: Int, OBS: Int, U: Int] = Sequential[
    Linear[FEATIN, U], RMSNorm[U], GELU[U],
    Linear[U, U], RMSNorm[U], GELU[U],
    Linear[U, OBS],
]

# RSSM prior: deter[DETER] → [Linear,RMSNorm,GELU]×2 → logit[SC]
comptime DreamerPrior[DETER: Int, H: Int, SC: Int] = Sequential[
    Linear[DETER, H], RMSNorm[H], GELU[H],
    Linear[H, H], RMSNorm[H], GELU[H],
    Linear[H, SC],
]

# Reward head MLP (1 hidden): feat[FEAT] → [Linear,RMSNorm,GELU] → logits[BINS]
comptime DreamerRewardMLP[FEAT: Int, U: Int, BINS: Int] = Sequential[
    Linear[FEAT, U], RMSNorm[U], GELU[U],
    Linear[U, BINS],
]

# Cont head MLP (1 hidden): feat[FEAT] → [Linear,RMSNorm,GELU] → logit[1]
comptime DreamerContMLP[FEAT: Int, U: Int] = Sequential[
    Linear[FEAT, U], RMSNorm[U], GELU[U],
    Linear[U, 1],
]
