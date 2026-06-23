"""EfficientZeroV2 continuous-action networks (nn).

The continuous agent reuses MuZero's representation ``h`` and dynamics ``g``
verbatim — for continuous control the dynamics input is ``[z | a]`` with a real
action **vector** (``ACT_DIM`` dims) rather than a one-hot, but ``MZDynNet``'s
``Linear[LATENT+ACT, …]`` torso is identical either way (the planner / unroll just
feed the raw action vector into the ``ACT`` slots). What changes is the
**prediction head**: instead of categorical policy logits it emits the
squashed-Gaussian parameters.

  * ``EZContPredNet[LATENT, ACT_DIM, BINS, H]`` — f: ``z → [μ_raw | σ_raw | value]``.
    Shared trunk → ``Parallel[Linear[H, 2·ACT_DIM], Linear[H, BINS]]``. The first
    ``ACT_DIM`` outputs are the raw mean logits, the next ``ACT_DIM`` the raw std
    logits (decoded by the sampled planner + `loss_ops_continuous`), and the last
    ``BINS`` the categorical value. OUT_DIM = 2·ACT_DIM + BINS — exactly the
    ``SampledGumbelGPUMCTS`` ``PredictionGPU`` contract (μ_raw, σ_raw, value).

The SimSiam projector/predictor (`nets.EZProjectorNet`/`EZPredictorNet`) are
identical to the discrete agent. Rep/Dyn aliases are re-exported for one-stop
import.
"""

from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.parallel import Parallel
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_mish import LinearMish

from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet


# ──────────────────────────────────────────────────────────────────────
# f (continuous) — prediction: z → [μ_raw | σ_raw | value_logits]
# ──────────────────────────────────────────────────────────────────────


comptime EZContPredNet[
    LATENT: Int, ACT_DIM: Int, BINS: Int, H: Int,
] = Sequential[
    LinearMish[LATENT, H],
    Parallel[
        Linear[H, 2 * ACT_DIM],
        Linear[H, BINS],
    ],
]
