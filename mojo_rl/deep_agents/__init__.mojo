"""Deep RL agents using the new trait-based architecture.

All agents use the Network wrapper from mojo_rl.nn.training with seq() composition
for building neural networks.

Available agents:
- DQNAgent: Deep Q-Network with Double DQN support
- DQNCNNAgent: DQN with CNN for pixel observations (Nature DQN architecture)
- DQNPERAgent: DQN with Prioritized Experience Replay
- DuelingDQNAgent: Dueling DQN with separate V(s) and A(s,a) streams
- DeepDDPGAgent: Deep Deterministic Policy Gradient
- DeepTD3Agent: Twin Delayed DDPG
- DeepSACAgent: Soft Actor-Critic
- DeepA2CAgent: Advantage Actor-Critic
- DeepPPOAgent: Proximal Policy Optimization (discrete actions)
- DeepPPOContinuousAgent: PPO for continuous action spaces
- TDMPC2Agent: TD-MPC2 model-based agent
- DreamerV3Agent: DreamerV3 world model-based agent
"""

from .dqn import DQNAgent
from .dqn_cnn import DQNCNNAgent
from .dqn_per import DQNPERAgent
from .dueling_dqn import DuelingDQNAgent
from .ddpg import DeepDDPGAgent
from .td3 import DeepTD3Agent
from .sac import DeepSACAgent
from .a2c import DeepA2CAgent
from .ppo import DeepPPOAgent, DeepPPOContinuousAgent
from .dreamer_v3 import DreamerV3Agent
