"""OFENet composite — DenseNet-style feature extractor for REDQ-OFE (STORAGE).

Reference: Ota et al., "Can Increasing Input Dimensionality Improve Deep
RL?" (ICML 2020); the authors' TF2 implementation in
`references/OFENet-main/teflon/ofe/network.py`; and the REDQ-OFE
PyTorch port in `references/REDQ-main/`.

Architecture — DenseNet variant with LayerNorm:
    block = SkipConcat[Linear → LayerNorm → SiLU]
            (i.e. `y = concat(x, SiLU(LayerNorm(Linear(x))))`)
    state branch  = Sequential[block_0, block_1, …, block_{N-1}]
    action branch = same shape, IN_DIM = phi_s + action_dim

Each block grows the width by `per_unit`, so block `i` has
`IN = state_dim + i * per_unit` (state branch) or
`IN = phi_s_dim + action_dim + i * per_unit` (action branch).
`Repeat[N, Block]` can't express this because every block has a
different IN — we enumerate them explicitly.

STORAGE migration (Stage 5): net defs swapped from the legacy `nn`
combinators/primitives to `nn.storage.{combinators,primitives}`
(`SkipConcat`, `Sequential`, `Linear`, `LayerNorm`, `SiLU`). The
DenseNet math + LayerNorm-not-BatchNorm decision are unchanged.

LayerNorm vs BatchNorm
======================
The original OFENet paper sandwiches BatchNorm1D between Linear and
Swish. The REDQ-OFE PyTorch port (Chen et al., 2021) reports
divergence with PyTorch BN and uses no normalisation; we keep the
LayerNorm decision — LayerNorm has no train/eval split, so the OFE
forward in aux mode and inference mode produce the same output, and
per-sample normalisation bounds the feature scale through the
6–8-block stack.

Predictor head
==============
We expose `OFEStateBranch`, `OFEActionBranch`, and `OFEPredictorHead`
as separate modules (no SplitApply / Identity). The trainer wires them
via the `Concat` primitive to assemble:

    obs (BATCH, OBS)               ──► OFEStateBranch  ──► φ(s)
    Concat[φ(s).dim, ACT](φ(s), a) ──► OFEActionBranch ──► φ(s, a)
    OFEPredictorHead(φ(s, a))      ──► predicted next-obs

Dimensions
==========
    state_branch.OUT_DIM  = OBS + N * per_unit
    action_branch.OUT_DIM = OBS + ACT + 2 * N * per_unit
                          = state_branch.OUT_DIM + ACT + N * per_unit
    predictor_head        = Linear[action_branch.OUT_DIM, OBS]

Typical hyperparams from `references/OFENet-main/gins/`:

    HalfCheetah / Hopper / Walker2d: N=6, per_unit=40, phi_s.dim = OBS+240
    Ant / Humanoid:                  N=8, per_unit=30, phi_s.dim = OBS+240
"""

from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.skip_concat import SkipConcat
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.primitives.silu import SiLU


# =============================================================================
# DenseBlock — one DenseNet-style feature-expanding block
# =============================================================================

# y = concat(x, SiLU(LayerNorm(Linear(x))))
# OUT_DIM = IN + per_unit
comptime OFEDenseBlock[IN: Int, per_unit: Int] = SkipConcat[
    Sequential[
        Linear[IN, per_unit],
        LayerNorm[per_unit],
        SiLU[per_unit],
    ]
]


# =============================================================================
# StateBranch — 6 or 8 DenseBlocks on raw observation
# =============================================================================

comptime OFEStateBranch6[OBS: Int, per_unit: Int] = Sequential[
    OFEDenseBlock[OBS + 0 * per_unit, per_unit],
    OFEDenseBlock[OBS + 1 * per_unit, per_unit],
    OFEDenseBlock[OBS + 2 * per_unit, per_unit],
    OFEDenseBlock[OBS + 3 * per_unit, per_unit],
    OFEDenseBlock[OBS + 4 * per_unit, per_unit],
    OFEDenseBlock[OBS + 5 * per_unit, per_unit],
]
# OUT_DIM = OBS + 6 * per_unit


comptime OFEStateBranch8[OBS: Int, per_unit: Int] = Sequential[
    OFEDenseBlock[OBS + 0 * per_unit, per_unit],
    OFEDenseBlock[OBS + 1 * per_unit, per_unit],
    OFEDenseBlock[OBS + 2 * per_unit, per_unit],
    OFEDenseBlock[OBS + 3 * per_unit, per_unit],
    OFEDenseBlock[OBS + 4 * per_unit, per_unit],
    OFEDenseBlock[OBS + 5 * per_unit, per_unit],
    OFEDenseBlock[OBS + 6 * per_unit, per_unit],
    OFEDenseBlock[OBS + 7 * per_unit, per_unit],
]
# OUT_DIM = OBS + 8 * per_unit


# =============================================================================
# ActionBranch — 6 or 8 DenseBlocks on concat(φ(s), a)
# =============================================================================
#
# `SA_IN` is the input dim *after* the (φ(s), a) concat — typically
# `state_branch.OUT_DIM + ACT`.

comptime OFEActionBranch6[SA_IN: Int, per_unit: Int] = Sequential[
    OFEDenseBlock[SA_IN + 0 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 1 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 2 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 3 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 4 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 5 * per_unit, per_unit],
]
# OUT_DIM = SA_IN + 6 * per_unit


comptime OFEActionBranch8[SA_IN: Int, per_unit: Int] = Sequential[
    OFEDenseBlock[SA_IN + 0 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 1 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 2 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 3 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 4 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 5 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 6 * per_unit, per_unit],
    OFEDenseBlock[SA_IN + 7 * per_unit, per_unit],
]
# OUT_DIM = SA_IN + 8 * per_unit


# =============================================================================
# Predictor head — final Linear that maps φ(s, a) back to the obs space
# =============================================================================

comptime OFEPredictorHead[PHI_SA_DIM: Int, OBS: Int] = Linear[PHI_SA_DIM, OBS]


# =============================================================================
# Comptime helpers — compute the output dim of each branch
# =============================================================================


def state_branch_out_dim(OBS: Int, N_BLOCKS: Int, per_unit: Int) -> Int:
    return OBS + N_BLOCKS * per_unit


def action_branch_out_dim(
    OBS: Int, ACT: Int, N_BLOCKS: Int, per_unit: Int,
) -> Int:
    # state_branch.OUT_DIM + ACT + N_BLOCKS * per_unit
    return state_branch_out_dim(OBS, N_BLOCKS, per_unit) + ACT + N_BLOCKS * per_unit


def predictor_in_dim(
    OBS: Int, ACT: Int, N_BLOCKS: Int, per_unit: Int,
) -> Int:
    return action_branch_out_dim(OBS, ACT, N_BLOCKS, per_unit)
