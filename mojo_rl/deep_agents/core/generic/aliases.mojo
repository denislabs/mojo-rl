"""Convenience aliases matching old agent names to generic agents.

These aliases let you use the familiar old agent names (DQNAgent, DeepDDPGAgent, etc.)
while getting the new generic implementations underneath. Just change the import:

    # Old:
    from mojo_rl.deep_agents.dqn import DQNAgent
    # New:
    from mojo_rl.deep_agents.core.generic import DQNAgent
"""

from mojo_rl.core.logger import Logger, NoOpLogger

from .dqn_agent import (
    GenericDQNAgent,
    GenericDQNPERAgent,
    DQNConfig,
    DoubleDQNConfig,
    DuelingDQNConfig,
    DQNCNNConfig,
    DQNPERConfig,
    AutodiffDQNConfig,
)
from .offpolicy_agent import GenericOffPolicyAgent
from .offpolicy_config import DDPGConfig, TD3Config, SACConfig
from .onpolicy_agent import GenericOnPolicyAgent
from .onpolicy_continuous_agent import GenericOnPolicyContinuousAgent
from .onpolicy_config import PPOConfig, A2CConfig, ContinuousPPOConfig, PPOCNNConfig


# =============================================================================
# DQN family aliases
# =============================================================================


# GenericDQNAgent[Config, n_envs, L]
comptime DQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 120,
    hidden_dim2: Int = 84,
    buffer_capacity: Int = 10000,
    batch_size: Int = 128,
    n_envs: Int = 1024,
    lr: Float64 = 2.5e-4,
    L: Logger = NoOpLogger,
] = GenericDQNAgent[
    DoubleDQNConfig[obs_dim, num_actions, hidden_dim, hidden_dim2, buffer_capacity, batch_size, lr],
    n_envs,
    L,
]


comptime DQNPERAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    hidden_dim2: Int = 128,
    buffer_capacity: Int = 20000,
    batch_size: Int = 64,
    n_envs: Int = 1024,
    lr: Float64 = 0.0005,
    L: Logger = NoOpLogger,
] = GenericDQNPERAgent[
    DQNPERConfig[obs_dim, num_actions, hidden_dim, hidden_dim2, buffer_capacity, batch_size, lr],
    n_envs,
    L,
]


comptime DuelingDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 120,
    stream_hidden_dim: Int = 84,
    buffer_capacity: Int = 10000,
    batch_size: Int = 128,
    n_envs: Int = 1024,
    lr: Float64 = 2.5e-4,
    L: Logger = NoOpLogger,
] = GenericDQNAgent[
    DuelingDQNConfig[obs_dim, num_actions, hidden_dim, stream_hidden_dim, buffer_capacity, batch_size, lr],
    n_envs,
    L,
]


comptime DQNCNNAgent[
    num_actions: Int,
    buffer_capacity: Int = 10000,
    batch_size: Int = 32,
    n_envs: Int = 64,
    lr: Float64 = 0.00025,
    L: Logger = NoOpLogger,
] = GenericDQNAgent[
    DQNCNNConfig[num_actions, buffer_capacity, batch_size, lr],
    n_envs,
    L,
]


comptime AutodiffDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 120,
    hidden_dim2: Int = 84,
    buffer_capacity: Int = 10000,
    batch_size: Int = 128,
    n_envs: Int = 1024,
    lr: Float64 = 2.5e-4,
    L: Logger = NoOpLogger,
] = GenericDQNAgent[
    AutodiffDQNConfig[obs_dim, num_actions, hidden_dim, hidden_dim2, buffer_capacity, batch_size, lr],
    n_envs,
    L,
]


# =============================================================================
# Off-policy continuous aliases
# =============================================================================


# GenericOffPolicyAgent[Config, profile, L]
comptime DeepDDPGAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
    profile: Int = 0,
    L: Logger = NoOpLogger,
    max_n_envs: Int = 64,
] = GenericOffPolicyAgent[
    DDPGConfig[obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size, actor_lr, critic_lr],
    profile,
    L,
    max_n_envs,
]


comptime DeepTD3Agent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
    profile: Int = 0,
    L: Logger = NoOpLogger,
    max_n_envs: Int = 64,
] = GenericOffPolicyAgent[
    TD3Config[obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size, actor_lr, critic_lr],
    profile,
    L,
    max_n_envs,
]


comptime DeepSACAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    profile: Int = 0,
    L: Logger = NoOpLogger,
    max_n_envs: Int = 64,
] = GenericOffPolicyAgent[
    SACConfig[obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size, actor_lr, critic_lr],
    profile,
    L,
    max_n_envs,
]


# =============================================================================
# On-policy aliases
# =============================================================================


# GenericOnPolicyAgent[Config, n_envs, gpu_minibatch_size] (no L param)
comptime DeepA2CAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    rollout_len: Int = 128,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
] = GenericOnPolicyAgent[
    A2CConfig[obs_dim, num_actions, hidden_dim, rollout_len, actor_lr, critic_lr],
]


comptime DeepPPOAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 64,
    rollout_len: Int = 128,
    n_envs: Int = 1024,
    gpu_minibatch_size: Int = 256,
    actor_lr: Float64 = 0.00025,
    critic_lr: Float64 = 0.001,
] = GenericOnPolicyAgent[
    PPOConfig[obs_dim, num_actions, hidden_dim, rollout_len, actor_lr, critic_lr],
    n_envs,
    gpu_minibatch_size,
]


# GenericOnPolicyContinuousAgent[Config, n_envs, gpu_minibatch_size, L]
comptime DeepPPOContinuousAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    rollout_len: Int = 128,
    n_envs: Int = 64,
    gpu_minibatch_size: Int = 256,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
    L: Logger = NoOpLogger,
] = GenericOnPolicyContinuousAgent[
    ContinuousPPOConfig[obs_dim, action_dim, hidden_dim, rollout_len, actor_lr, critic_lr],
    n_envs,
    gpu_minibatch_size,
    L,
]


comptime DeepPPOCNNAgent[
    num_actions: Int,
    rollout_len: Int = 128,
    n_envs: Int = 64,
    gpu_minibatch_size: Int = 256,
    actor_lr: Float64 = 0.00025,
    critic_lr: Float64 = 0.00025,
] = GenericOnPolicyAgent[
    PPOCNNConfig[num_actions, rollout_len, actor_lr, critic_lr],
    n_envs,
    gpu_minibatch_size,
]
