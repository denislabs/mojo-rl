"""REDQ-OFE agent — REDQ with an OFENet feature extractor.

Status: config-only. Agent implementation is scheduled for the next
session; see PLAN.md for the surgical modifications to apply to
`../redq/redq.mojo` and the aux training step.

Reference: Ota et al., "Can Increasing Input Dimensionality Improve
Deep Reinforcement Learning?" (ICML 2020) and `references/OFENet-main/`.
"""

from .config import (
    REDQOFEConfig,
    DefaultREDQOFEConfig6,
    DefaultREDQOFEConfig8,
)
