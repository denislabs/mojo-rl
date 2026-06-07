"""Dreamer 4 agent heads — multi-token-prediction (MTP) policy + reward.

Paper §3.3 / eq. 9: the policy and reward heads read the agent task-output
embeddings h_t and predict the next `NMTP = L+1` actions and rewards (MTP of
length L=8 ⇒ NMTP=9 distances n=0..L). "small MLPs with one output layer per
MTP distance" — here a shared 2-layer MLP trunk followed by one wide output
projection whose `NMTP··` columns are the per-distance logits (distance-major:
distance n occupies columns [n·W, (n+1)·W)).

Both heads are pure nn2 `Sequential` modules (Linear→SiLU→Linear), so they are
GPU-ready and autodiff-composable out of the box; the BC loss
(`bc_loss.mojo`) slices per-distance logits and applies the categorical /
symexp-twohot losses from `dreamerv3/{dists_discrete,twohot}.mojo`.

  Policy head : h_t[D_IN] → [NMTP·NACT]   (categorical / unimix per distance)
  Reward head : h_t[D_IN] → [NMTP·NBINS]  (symexp twohot per distance)

The vectorized-binary policy variant (paper alt for keyboard actions) is
deferred; v1 uses the categorical head (e.g. for the discrete Pong lighthouse).
"""

from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.silu import SiLU


# Policy head: NMTP categorical action distributions of NACT classes each.
comptime Dreamer4PolicyHead[
    D_IN: Int, HID: Int, NACT: Int, NMTP: Int
] = Sequential[
    Linear[D_IN, HID],
    SiLU[HID],
    Linear[HID, NMTP * NACT],
]


# Reward head: NMTP symexp-twohot reward distributions of NBINS bins each.
comptime Dreamer4RewardHead[
    D_IN: Int, HID: Int, NBINS: Int, NMTP: Int
] = Sequential[
    Linear[D_IN, HID],
    SiLU[HID],
    Linear[HID, NMTP * NBINS],
]


# Value head (Phase 4 / eq. 10): a SINGLE symexp-twohot value distribution per
# state s_t (NOT multi-token — the value loss predicts the λ-return of the
# current imagined state only). Same trunk as the reward head; one NBINS output
# block. Trained by TD-learning vs sg(R_t^λ) (`imag_rl_loss.mojo`).
comptime Dreamer4ValueHead[
    D_IN: Int, HID: Int, NBINS: Int
] = Sequential[
    Linear[D_IN, HID],
    SiLU[HID],
    Linear[HID, NBINS],
]
