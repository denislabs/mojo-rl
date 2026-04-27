"""REDQ-OFE agent — REDQ with an OFENet feature extractor.

OFENet is a DenseNet-style feature extractor trained via an auxiliary
next-state-prediction MSE loss. Actor and critics consume OFE features
(phi_s, phi_sa) with stop-gradient — OFE params are updated only by the
aux loss, matching the paper.

Reference: Ota et al., "Can Increasing Input Dimensionality Improve
Deep Reinforcement Learning?" (ICML 2020) and `references/OFENet-main/`.
"""

from .config import (
    REDQOFEConfig,
    DefaultREDQOFEConfig6,
    DefaultREDQOFEConfig8,
)
from .redq_ofe import REDQOFEAgent, REDQOFEGPUState
