# EfficientZero V2 — Gumbel-search MCTS + (later) full agent.
# Phase 1 (this file): CPU Gumbel search reusing MuZero's networks.
# Reference: Wang et al., 2024 (ICML), built on Danihelka et al., 2021
# (Gumbel-MuZero, ICLR 2022).

from .mcts import GumbelMCTS, GumbelMCTSNode
from .gpu_mcts import EZV2GPUMCTSState, run_gumbel_search_gpu
from .strategies import (
    ValueTarget,
    SVETarget,
    MultiStepTDTarget,
    MixedValueTarget,
    PolicyLoss,
    FullCrossEntropy,
    SimpleBestAction,
    compute_sve,
    compute_multistep_td,
)
from .networks import (
    ImproveResidualBlock,
    ActionEmbedding,
    ProjectionMLP,
    PredictionMLP,
    RewardPrefixHeadMLP,
)
from .consistency import (
    cosine_consistency_loss_forward,
    cosine_consistency_loss,
)
from .configs import (
    EZV2DiscreteConfig,
    EZV2DiscreteMLPConfig,
    VALUE_TARGET_SEARCH,
    VALUE_TARGET_SARSA,
    VALUE_TARGET_MIXED,
)
from .state import EZV2DiscreteCPUState, EZV2DiscreteGPUState
from .efficient_zero_v2 import GenericEfficientZeroV2Agent
