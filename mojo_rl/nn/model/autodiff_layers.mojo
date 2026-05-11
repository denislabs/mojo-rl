"""RSample and Min Model layers — thin wrappers around DiffOps via AutoDiffChain.

These wrap the RSampleOp and MinOp DiffOps into Model-conforming types
that can be used in Sequential composition.

Usage:
    from mojo_rl.nn.model import Sequential, LinearReLU, RSample, Min
    from mojo_rl.nn.autodiff.combinators import Parallel, SkipConcat, DualPath

    # SAC Actor: obs → [mean, log_std] → RSample → [action, log_prob]
    comptime ActorPath = Sequential[
        LinearReLU[17, 256],
        LinearReLU[256, 256],
        Parallel[Linear[256, 6], LinearTanh[256, 6]],
        RSample[6],  # reparameterized sampling
    ]

    # Twin Critics: critic_input → DualPath → [Q1, Q2] → Min → min_Q
    comptime CriticModel = Sequential[LinearReLU[23, 256], Linear[256, 1]]
    comptime TwinCriticMin = Sequential[DualPath[CriticModel, CriticModel], Min[1]]
"""

from ..autodiff.chain import AutoDiffChain
from ..autodiff.primitives.rsample import RSampleOp
from ..autodiff.primitives.min_op import MinOp
from ..autodiff.primitives.slice_op import SliceOp
from ..autodiff.primitives.negate import NegateOp
from ..autodiff.primitives.gather import GatherOp
from ..autodiff.primitives.ppo_ops import CategoricalLogProbOp, RatioOp, ClipSurrogateOp
from ..autodiff.primitives.gaussian_log_prob import GaussianLogProbOp
from ..autodiff.primitives.mse_op import MSEOp
from ..autodiff.primitives.activations import GELUOp
from ..autodiff.primitives.reshape import Transpose2DOp
from ..autodiff.primitives.reduce import TokenMean as _TokenMeanOp


comptime RSample[
    action_dim: Int,
    log_std_min: Float64 = -5.0,
    log_std_max: Float64 = 2.0,
    action_scale: Float64 = 1.0,
] = AutoDiffChain[
    RSampleOp[action_dim, log_std_min, log_std_max, action_scale]
]

comptime Min[dim: Int] = AutoDiffChain[MinOp[dim]]

comptime Slice[in_dim: Int, start: Int, end: Int] = AutoDiffChain[
    SliceOp[in_dim, start, end]
]

comptime Negate[dim: Int] = AutoDiffChain[NegateOp[dim]]

comptime Gather[dim: Int] = AutoDiffChain[GatherOp[dim]]

comptime CategoricalLogProb[num_actions: Int] = AutoDiffChain[CategoricalLogProbOp[num_actions]]

comptime Ratio[dim: Int = 1] = AutoDiffChain[RatioOp[dim]]

comptime ClipSurrogate[eps: Float64 = 0.2] = AutoDiffChain[ClipSurrogateOp[eps]]

comptime GaussianLogProb[action_dim: Int] = AutoDiffChain[GaussianLogProbOp[action_dim]]

comptime MSELoss = AutoDiffChain[MSEOp]

from ..autodiff.primitives.huber_op import HuberOp
from ..autodiff.primitives.identity import IdentityOp

comptime HuberLoss[delta: Float64 = 1.0] = AutoDiffChain[HuberOp[delta]]

comptime Identity[dim: Int] = AutoDiffChain[IdentityOp[dim]]

# GELU (tanh approximation, GPT-2 / BERT canonical) wrapped as a Model so it
# can be dropped into Sequential / Tokenwise. See autodiff/primitives/activations.mojo.
comptime GELU[dim: Int] = AutoDiffChain[GELUOp[dim]]

# Transpose2D — reshape (BATCH, A * B) viewed as (BATCH, A, B) row-major to
# (BATCH, B, A) row-major. Used to bridge Conv2D (channel-major output) with
# attention (patch-major input) in ViT. See autodiff/primitives/reshape.mojo.
comptime Transpose2D[A: Int, B: Int] = AutoDiffChain[Transpose2DOp[A, B]]

# TokenMean — average over the seq_len axis of a (seq_len, dim) per-sample
# tensor. (BATCH, seq_len * dim) → (BATCH, dim). Used as the ViT
# classification-head pooling step. See autodiff/primitives/reduce.mojo.
comptime TokenMean[seq_len: Int, dim: Int] = AutoDiffChain[
    _TokenMeanOp[seq_len, dim]
]
