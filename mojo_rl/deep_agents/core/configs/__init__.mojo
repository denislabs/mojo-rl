"""Algorithm configuration structs for generic agents."""

from .offpolicy_config import OffPolicyConfig, DDPGConfig, TD3Config, SACConfig, AutodiffSACConfig, AutodiffDDPGConfig, AutodiffTD3Config
from .onpolicy_config import OnPolicyConfig, PPOConfig, A2CConfig, PPOCNNConfig, ContinuousOnPolicyConfig, ContinuousPPOConfig, AutodiffPPOConfig, AutodiffA2CConfig, AutodiffContinuousPPOConfig
from .mbpo_config import MBPOConfig, DefaultMBPOConfig
