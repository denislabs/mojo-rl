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
- MBPOSACAgent: Model-Based Policy Optimization (SAC + dynamics ensemble)
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
    MBPOSACAgent,
)
from .core.agents.mbpo_agent import MBPOAgent
from .core.configs.mbpo_config import MBPOConfig, DefaultMBPOConfig
from .core.training.mbpo_train import run_mbpo_train
from .redq import (
    REDQAgent,
    REDQConfig,
    DefaultREDQConfig,
    DefaultREDQLNConfig,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
    REDQ_TARGET_REM,
)
from .redq_ofe import (
    REDQOFEConfig,
    DefaultREDQOFEConfig6,
    DefaultREDQOFEConfig8,
)
from .dreamer_v3 import DreamerV3Agent
from .muzero import GenericMuZeroAgent, MuZeroMLPConfig
