# CPU Deep RL Agents (old architecture)
from .cpu import (
    DeepDDPGAgent,
    DeepTD3Agent,
    DeepSACAgent,
    DeepDQNAgent as DeepDQNAgentCPU,
    QNetwork,
    DeepDQNPERAgent,
    DeepDuelingDQNAgent,
    DuelingQNetwork,
)

# GPU Deep RL Agents (old architecture)
from .gpu import A2CAgent as GPUA2CAgent

# New architecture agents (use Network wrapper, support both CPU and GPU)
from .dqn import DQNAgent
