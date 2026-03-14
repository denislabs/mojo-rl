# DreamerV3: World Model-Based RL Agent
# Learns RSSM world model + trains actor-critic in imagination
#
# Reference: Hafner et al., 2023 — Mastering Diverse Domains through
# World Models (DreamerV3)

from .rssm import RSSM, categorical_sample, kl_divergence
from .state import DreamerV3CPUState, DreamerV3GPUState
from .dreamer_v3 import DreamerV3Agent
from .imagination import (
    compute_lambda_returns,
    normalize_returns,
    sample_tanh_normal,
    log_prob_tanh_normal,
)
