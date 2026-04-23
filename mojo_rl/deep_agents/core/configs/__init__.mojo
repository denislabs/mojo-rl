"""Algorithm configuration structs for generic agents."""

from .offpolicy_config import OffPolicyConfig, DDPGConfig, TD3Config, SACConfig, AutodiffSACConfig, AutodiffDDPGConfig, AutodiffTD3Config
from .onpolicy_config import OnPolicyConfig, PPOConfig, A2CConfig, PPOCNNConfig, ContinuousOnPolicyConfig, ContinuousPPOConfig, AutodiffPPOConfig, AutodiffA2CConfig, AutodiffContinuousPPOConfig
from .mbpo_config import MBPOConfig, DefaultMBPOConfig
from .redq_config import (
    REDQConfig,
    DefaultREDQConfig,
    DefaultREDQLNConfig,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
    REDQ_TARGET_REM,
)
