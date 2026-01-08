# CPU Deep RL Agents
from .cpu import (
    DeepDDPGAgent,
    DeepTD3Agent,
    DeepSACAgent,
    DeepDQNAgent,
    QNetwork,
    DeepDQNPERAgent,
    DeepDuelingDQNAgent,
    DuelingQNetwork,
)

# GPU Deep RL Agents
from .gpu import GPUDeepDQNAgent, GPUQNetwork
