"""OFENet (Online Feature Extractor Network) composite architectures.

Reference: Ota et al., "Can Increasing Input Dimensionality Improve Deep
Reinforcement Learning?" (ICML 2020) and the authors' TF2 implementation
in `references/OFENet-main/teflon/ofe/network.py`.

OFENet builds a DenseNet-style feature extractor trained with an
auxiliary next-state prediction loss, providing a rich representation
that the actor/critic consume instead of raw observations.

Architecture (DenseNet variant — REDQ-OFE convention, no BN):
    - num_layers blocks in state branch, num_layers in action branch
    - per_unit = total_units / num_layers new features per block
    - block = Linear(per_unit) → Swish → concat(input, ·)

The original OFENet (Ota et al., TF2) sandwiches BatchNorm1D between
Linear and Swish, but the REDQ-OFE PyTorch port (Chen et al., 2021)
found that BN destabilises training under REDQ's high UTD ratio:

    > "for some reason adding PyTorch batch norm to OFENet will lead
    >  to divergence. So in the end we did not use batch norm in our
    >  code." — references/REDQ-main/README.md:127

Empirically reproduced in this codebase: with BN, REDQ-OFE diverges on
HalfCheetah (Q-overestimation, entropy collapse, episode reward dropping
to ~-1400 by 20k env steps), independent of inference-mode plumbing or
aux-loss pretraining (10k pretraining aux steps did not fix it). The
DenseNet skip-concat structure already provides most of OFENet's
representational benefit; dropping BN matches the REDQ-OFE paper's
actual implementation.

Full prediction chain (for aux loss):
    concat(s, a) of dim (state_dim + action_dim)
      ↓ SplitApply[StateBranch, Identity, state_dim]
    concat(φ(s), a) of dim (state_dim + num_layers*per_unit + action_dim)
      ↓ ActionBranch (num_layers DenseBlocks)
    φ(s, a) of dim (state_dim + 2*num_layers*per_unit + action_dim)
      ↓ Linear(state_dim)
    predicted next_state

Typical instantiations from `references/OFENet-main/gins/`:
    HalfCheetah: total_units=240, num_layers=6 → per_unit=40
    Ant:         total_units=240, num_layers=8 → per_unit=30
    Hopper:      total_units=240, num_layers=6 → per_unit=40
    Humanoid:    total_units=240, num_layers=8 → per_unit=30
"""

from .model import (
    Sequential,
    Linear,
    Swish,
    Identity,
    SkipConcat,
)
from .autodiff.combinators import SplitApply


# =============================================================================
# DenseBlock — one DenseNet-style feature-expanding block
# =============================================================================

# SkipConcat[Sequential[Linear, Swish]] produces:
#   forward: y = concat(x, Swish(Linear(x)))
#   OUT_DIM = IN + per_unit
# Matches teflon/ofe/blocks.py:DensenetBlock with batchnorm=False, which
# is the REDQ-OFE configuration (see file docstring).
comptime DenseBlock[IN: Int, per_unit: Int] = SkipConcat[
    Sequential[
        Linear[IN, per_unit],
        Swish[per_unit],
    ]
]


# =============================================================================
# StateBranch — 6- or 8-layer stack operating on raw state
# =============================================================================
#
# Each block adds `per_unit` new features concatenated with the running input.
# Because each block has a different IN_DIM (grows by per_unit), we can't use
# Repeat[N, Block] — we must list the blocks explicitly.

comptime StateBranch6[state_dim: Int, per_unit: Int] = Sequential[
    DenseBlock[state_dim + 0 * per_unit, per_unit],
    DenseBlock[state_dim + 1 * per_unit, per_unit],
    DenseBlock[state_dim + 2 * per_unit, per_unit],
    DenseBlock[state_dim + 3 * per_unit, per_unit],
    DenseBlock[state_dim + 4 * per_unit, per_unit],
    DenseBlock[state_dim + 5 * per_unit, per_unit],
]
# StateBranch6.IN_DIM = state_dim
# StateBranch6.OUT_DIM = state_dim + 6 * per_unit  (= φ(s))


comptime StateBranch8[state_dim: Int, per_unit: Int] = Sequential[
    DenseBlock[state_dim + 0 * per_unit, per_unit],
    DenseBlock[state_dim + 1 * per_unit, per_unit],
    DenseBlock[state_dim + 2 * per_unit, per_unit],
    DenseBlock[state_dim + 3 * per_unit, per_unit],
    DenseBlock[state_dim + 4 * per_unit, per_unit],
    DenseBlock[state_dim + 5 * per_unit, per_unit],
    DenseBlock[state_dim + 6 * per_unit, per_unit],
    DenseBlock[state_dim + 7 * per_unit, per_unit],
]
# StateBranch8.OUT_DIM = state_dim + 8 * per_unit


# =============================================================================
# ActionBranch — 6- or 8-layer stack operating on concat(φ(s), a)
# =============================================================================
#
# IN_DIM is φ(s).dim + action_dim. Each block grows the dim by per_unit like
# the state branch.

comptime ActionBranch6[phi_s_dim: Int, action_dim: Int, per_unit: Int] = Sequential[
    DenseBlock[phi_s_dim + action_dim + 0 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 1 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 2 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 3 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 4 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 5 * per_unit, per_unit],
]


comptime ActionBranch8[phi_s_dim: Int, action_dim: Int, per_unit: Int] = Sequential[
    DenseBlock[phi_s_dim + action_dim + 0 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 1 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 2 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 3 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 4 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 5 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 6 * per_unit, per_unit],
    DenseBlock[phi_s_dim + action_dim + 7 * per_unit, per_unit],
]


# =============================================================================
# OFENetPredictor — full prediction chain (state + action branches + Linear head)
# =============================================================================
#
# Input: concat(s, a) of dim (state_dim + action_dim)
# Output: predicted next_state of dim state_dim
# Trained with MSE against true next_state to shape the representation.

# SplitApply[StateBranch, Identity[action_dim], split=state_dim]
# - Left (StateBranch):  input s of dim state_dim → φ(s) of dim state_dim + N*per_unit
# - Right (Identity):    input a of dim action_dim → a
# - Output: concat(φ(s), a)

comptime OFENetPredictor6[state_dim: Int, action_dim: Int, per_unit: Int] = Sequential[
    SplitApply[
        StateBranch6[state_dim, per_unit],
        Identity[action_dim],
        state_dim,
    ],
    # Intermediate: concat(φ(s), a), dim = (state_dim + 6*per_unit) + action_dim
    ActionBranch6[state_dim + 6 * per_unit, action_dim, per_unit],
    # Final dim after action branch:
    #   (state_dim + 6*per_unit + action_dim) + 6*per_unit
    # = state_dim + 12*per_unit + action_dim
    Linear[state_dim + 12 * per_unit + action_dim, state_dim],
]


comptime OFENetPredictor8[state_dim: Int, action_dim: Int, per_unit: Int] = Sequential[
    SplitApply[
        StateBranch8[state_dim, per_unit],
        Identity[action_dim],
        state_dim,
    ],
    ActionBranch8[state_dim + 8 * per_unit, action_dim, per_unit],
    Linear[state_dim + 16 * per_unit + action_dim, state_dim],
]


