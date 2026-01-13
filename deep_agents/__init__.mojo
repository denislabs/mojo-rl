# CPU Deep RL Agents (old architecture - deprecated)
from .cpu import (
    DeepDDPGAgent as LegacyDeepDDPGAgent,
    DeepTD3Agent as LegacyDeepTD3Agent,
    DeepSACAgent as LegacyDeepSACAgent,
    DeepDQNAgent as DeepDQNAgentCPU,
    QNetwork,
    DeepDQNPERAgent,
    DeepDuelingDQNAgent as LegacyDuelingDQNAgent,
    DuelingQNetwork,
)

# GPU Deep RL Agents (old architecture - deprecated)
from .gpu import A2CAgent as GPUA2CAgent

# New architecture agents (use Network wrapper + traits, recommended)
from .dqn import DQNAgent
from .dqn_per import DQNPERAgent
from .ddpg import DeepDDPGAgent
from .td3 import DeepTD3Agent
from .sac import DeepSACAgent
from .dueling_dqn import DuelingDQNAgent
from .a2c import DeepA2CAgent
from .ppo import DeepPPOAgent
