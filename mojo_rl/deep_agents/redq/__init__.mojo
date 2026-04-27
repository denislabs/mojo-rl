"""REDQ agent.

Self-contained module for the Randomized Ensembled Double Q-Learning agent
(Chen et al., ICLR 2021). See `redq.mojo` for the agent, `config.mojo`
for the config structs, and `kernels.mojo` for the ensemble-target kernel.
"""

from .redq import REDQAgent, REDQGPUState
from .config import (
    REDQConfig,
    DefaultREDQConfig,
    DefaultREDQLNConfig,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
    REDQ_TARGET_REM,
)
