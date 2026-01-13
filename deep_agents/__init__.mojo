"""Deep RL agents using the new trait-based architecture.

All agents use the Network wrapper from deep_rl.training with seq() composition
for building neural networks.

Available agents:
- DQNAgent: Deep Q-Network with Double DQN support
- DQNPERAgent: DQN with Prioritized Experience Replay
- DuelingDQNAgent: Dueling DQN with separate V(s) and A(s,a) streams
- DeepDDPGAgent: Deep Deterministic Policy Gradient
- DeepTD3Agent: Twin Delayed DDPG
- DeepSACAgent: Soft Actor-Critic
- DeepA2CAgent: Advantage Actor-Critic
- DeepPPOAgent: Proximal Policy Optimization
"""

from .dqn import DQNAgent
from .dqn_per import DQNPERAgent
from .ddpg import DeepDDPGAgent
from .td3 import DeepTD3Agent
from .sac import DeepSACAgent
from .dueling_dqn import DuelingDQNAgent
from .a2c import DeepA2CAgent
from .ppo import DeepPPOAgent
