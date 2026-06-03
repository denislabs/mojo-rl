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
SimNorm primitive takes the group **count** = `DIM // SN`. Dropout in the Q
trunk is deferred (per the port plan); add once HalfCheetah converges.
"""

from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.layer_norm import LayerNorm
from mojo_rl.nn2.primitives.mish import Mish
from mojo_rl.nn2.primitives.sim_norm import SimNorm


comptime NormedLinear[I: Int, O: Int] = Sequential[
    Linear[I, O], LayerNorm[O], Mish[O],
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

comptime TDMPC2QNet[LATENT: Int, ACT: Int, MLP: Int, BINS: Int] = Sequential[
    NormedLinear[LATENT + ACT, MLP],
    NormedLinear[MLP, MLP],
    Linear[MLP, BINS],
]

# Policy prior π(z): trunk → Linear(2·ACT) = [mean | log_std], consumed by
# RSample (tanh-squashed Gaussian). Reference `mlp(latent, 2*[mlp], 2*act)`.
comptime TDMPC2Policy[LATENT: Int, ACT: Int, MLP: Int] = Sequential[
    NormedLinear[LATENT, MLP],
    NormedLinear[MLP, MLP],
    Linear[MLP, 2 * ACT],
]
