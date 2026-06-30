"""EfficientZeroV2 networks (nn).

EZv2's learned model is MuZero's (`deep_agents/muzero/nets.mojo`) — the
representation ``h``, dynamics ``g`` (latent + reward) and prediction ``f``
(policy + categorical value) are **reused verbatim** for the discrete agent.
What EZv2 *adds* is a **SimSiam temporal-consistency** pair (legacy
``efficient_zero_v2/networks.mojo`` ``ProjectionMLP`` / ``PredictionMLP``):

  * ``EZProjectorNet[HIDDEN, PROJ, PROJ_HID]`` — projector ``g_proj``:
    ``Linear → BN → ReLU → Linear → BN → ReLU → Linear → BN``. The **trailing
    BatchNorm (no ReLU)** is the SimSiam collapse-defense (Chen & He 2021);
    keep it. Both the online (dynamics-latent) and target (rep-of-future-obs)
    branches pass through this same projector. IN_DIM=HIDDEN, OUT_DIM=PROJ.

  * ``EZPredictorNet[PROJ, BOTTLENECK]`` — predictor ``h_pred``:
    ``Linear → BN → ReLU → Linear`` (asymmetric bottleneck, **no trailing BN**).
    Only the **online** branch passes through the predictor; the asymmetry +
    the stop-gradient on the target branch are what prevent representational
    collapse. IN_DIM=PROJ, OUT_DIM=PROJ.

Consistency objective (per unroll step ``k = 1..K``):

    online :  p_k = h_pred(g_proj(z_k))          (z_k from the dynamics rollout)
    target :  t_k = sg( g_proj(h(obs_k)) )         (rep of the *real* future obs)
    L_G    =  −cos(p_k, t_k)        (stop-grad on t_k; coeff λ_G, default 2.0)

Both nets are plain ``Sequential`` ``Module`` aliases, so ``make[target,INIT]`` /
``forward[target,B]`` / ``vjp[target]`` / Adam apply, and the BatchNorm
train/eval mode flips through the standard ``set_attr["training"]`` seam (the
self-play driver sets eval before MCTS inference, train around the update).

The discrete EZv2 reuses ``MZRepNet`` / ``MZDynNet`` / ``MZPredNet`` directly
(re-exported here for one-stop import); the continuous variant overrides the
prediction head — see ``nets_continuous.mojo``.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.batch_norm_1d import BatchNorm1D

# Re-export the MuZero learned model — discrete EZv2 uses it unchanged.
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet


# ──────────────────────────────────────────────────────────────────────
# SimSiam projector  g_proj : hidden → projection   (trailing BN, no ReLU)
# ──────────────────────────────────────────────────────────────────────


comptime EZProjectorNet[
    HIDDEN: Int, PROJ: Int, PROJ_HID: Int,
    ADT: DType = DT,
] = Sequential[
    Linear[HIDDEN, PROJ_HID, ADT],
    BatchNorm1D[PROJ_HID, ADT=ADT],
    ReLU[PROJ_HID, ADT],
    Linear[PROJ_HID, PROJ_HID, ADT],
    BatchNorm1D[PROJ_HID, ADT=ADT],
    ReLU[PROJ_HID, ADT],
    Linear[PROJ_HID, PROJ, ADT],
    BatchNorm1D[PROJ, ADT=ADT],
]


# ──────────────────────────────────────────────────────────────────────
# SimSiam predictor  h_pred : projection → projection   (no trailing BN)
# ──────────────────────────────────────────────────────────────────────


comptime EZPredictorNet[
    PROJ: Int, BOTTLENECK: Int,
    ADT: DType = DT,
] = Sequential[
    Linear[PROJ, BOTTLENECK, ADT],
    BatchNorm1D[BOTTLENECK, ADT=ADT],
    ReLU[BOTTLENECK, ADT],
    Linear[BOTTLENECK, PROJ, ADT],
]
