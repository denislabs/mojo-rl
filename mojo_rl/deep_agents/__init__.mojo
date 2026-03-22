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
- GenericMuZeroAgent: MuZero model-based agent with MCTS planning (config-driven)
"""

from .core.agents import (
    DQNAgent,
    DQNCNNAgent,
    DQNPERAgent,
    DuelingDQNAgent,
    DeepDDPGAgent,
    DeepTD3Agent,
    DeepSACAgent,
    DeepA2CAgent,
    DeepPPOAgent,
    DeepPPOContinuousAgent,
    DeepPPOCNNAgent,
)
from .dreamer_v3 import DreamerV3Agent
from .muzero import GenericMuZeroAgent, MuZeroMLPConfig
