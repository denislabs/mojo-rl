# CPU Deep RL Agents (old architecture - deprecated)
from .cpu import (
    DeepDDPGAgent as LegacyDeepDDPGAgent,
    DeepTD3Agent as LegacyDeepTD3Agent,
    DeepSACAgent as LegacyDeepSACAgent,
    DeepDQNAgent as DeepDQNAgentCPU,
    QNetwork,
    DeepDQNPERAgent,
    DeepDuelingDQNAgent,
    DuelingQNetwork,
)

# GPU Deep RL Agents (old architecture - deprecated)
from .gpu import A2CAgent as GPUA2CAgent

# New architecture agents (use Network wrapper + traits, recommended)
from .dqn import DQNAgent
from .ddpg import DeepDDPGAgent
from .td3 import DeepTD3Agent
from .sac import DeepSACAgent
