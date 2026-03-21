# deep_agents/ - Deep RL Agents

Neural network-based RL agents with CPU and GPU training support. All agents use the trait-based `nn/` framework for models and the shared `core/` infrastructure for training loops.

## Architecture

**Config-driven generic agents** — algorithms are defined by config structs (~30 lines) that bundle network architectures and algorithm flags. Old per-agent directories have been replaced by composable strategies.

Each agent implements trait combinations from `core/`:
- **Off-policy**: `OffPolicyContinuousAgent` / `OffPolicyDiscreteAgent` + `GPUOffPolicyAgent`
- **On-policy**: `OnPolicyContinuousAgent` / `OnPolicyDiscreteAgent` + `GPUOnPolicyContinuousAgent`
- **Checkpointable**: Save/load via binary checkpoint format

State is separated from agent logic: `*CPUState` holds CPU params/buffers, `*GPUState` holds device memory.

## Directory Structure

```
deep_agents/
├── core/                          # Shared infrastructure
│   ├── agents/                    # Generic agent implementations
│   │   ├── dqn_agent.mojo             # GenericDQNAgent (DQN/Double/Dueling/Noisy/PER/CNN)
│   │   ├── c51_agent.mojo             # GenericC51Agent (categorical distributional DQN)
│   │   ├── rainbow_agent.mojo         # GenericRainbowAgent (6 improvements combined)
│   │   ├── offpolicy_agent.mojo       # GenericOffPolicyAgent (DDPG/TD3/SAC)
│   │   ├── onpolicy_agent.mojo        # GenericOnPolicyAgent (PPO/A2C discrete)
│   │   ├── onpolicy_continuous_agent.mojo  # GenericOnPolicyContinuousAgent (PPO continuous)
│   │   └── aliases.mojo               # Convenience aliases (DQNAgent, DeepSACAgent, etc.)
│   ├── configs/                   # Compile-time algorithm configurations
│   │   ├── offpolicy_config.mojo      # DDPGConfig, TD3Config, SACConfig
│   │   └── onpolicy_config.mojo       # PPOConfig, A2CConfig, ContinuousPPOConfig, PPOCNNConfig
│   ├── strategies/                # Composable building blocks
│   │   ├── exploration.mojo           # GaussianNoise, StochasticSample
│   │   ├── update_schedule.mojo       # EveryStep, DelayedAll, DelayedActorOnly
│   │   ├── target_value.mojo          # SingleQTarget, TwinQTarget, EntropicTwinQTarget
│   │   ├── target_action.mojo         # DeterministicTarget, SmoothedTarget, ReparamTarget
│   │   ├── actor_loss.mojo            # DPGLoss, MaxEntLoss, AutodiffDPGLoss, AutodiffMaxEntLoss
│   │   ├── policy_gradient.mojo       # VanillaPG, ClippedSurrogate, AutodiffClippedSurrogate
│   │   ├── epoch_schedule.mojo        # SinglePass, MultiEpochMinibatch
│   │   ├── q_target.mojo              # StandardQTarget, DoubleQTarget
│   │   ├── q_output.mojo              # DirectQ, DuelingQ
│   │   └── q_gradient.mojo            # ManualQGradient, AutodiffQGradient
│   ├── training/                  # Training loops (CPU/GPU)
│   │   ├── offpolicy_train.mojo       # CPU off-policy loop
│   │   ├── gpu_offpolicy_train.mojo   # GPU off-policy loop
│   │   ├── onpolicy_train.mojo        # CPU on-policy loop
│   │   ├── gpu_onpolicy_train.mojo    # GPU on-policy loop
│   │   ├── offpolicy_helpers.mojo     # Action selection, transition storage
│   │   └── onpolicy_helpers.mojo      # GAE computation, advantage normalization
│   ├── replay/                    # Experience replay buffers
│   │   ├── replay_buffer.mojo         # HeapReplayBuffer, PrioritizedReplayBuffer
│   │   ├── gpu_replay_buffer.mojo     # GPUReplayBuffer (device-side circular)
│   │   ├── gpu_per_replay_buffer.mojo # GPUPrioritizedReplayBuffer
│   │   ├── nstep_buffer.mojo          # N-step returns (for Rainbow)
│   │   ├── sequence_replay_buffer.mojo     # CPU sequence buffer (world models)
│   │   └── gpu_sequence_replay_buffer.mojo # GPU sequence buffer
│   ├── kernels.mojo               # 80+ shared GPU kernels
│   ├── eval.mojo                  # Evaluation loops for all agent types
│   ├── checkpoint_trait.mojo      # Checkpointable trait
│   └── utils.mojo                 # Helpers, progress bar
├── dreamer_v3/                    # DreamerV3 (world model-based) [experimental]
│   ├── dreamer_v3.mojo            # Main DreamerV3Agent
│   ├── rssm.mojo                  # Recurrent State Space Model
│   ├── state.mojo                 # DreamerV3CPUState, DreamerV3GPUState
│   ├── imagination.mojo           # Imagination rollout utilities
│   └── kernels.mojo               # DreamerV3-specific GPU kernels
└── tdmpc2/                        # TD-MPC2 (model-based) [experimental]
    ├── tdmpc2.mojo                # TDMPC2Agent (world model + MPPI)
    ├── world_model.mojo           # Encoder, dynamics, reward, policy ensemble
    ├── mppi.mojo                  # Model Predictive Path Integral planner
    ├── state.mojo                 # TDMPC2CPUState, TDMPC2GPUState
    └── kernels.mojo               # 80+ TDMPC2-specific GPU kernels
```

## Agent Summary

| Agent | Config | Generic Struct | GPU | Key Features |
|-------|--------|---------------|-----|-------------|
| **DQN** | `DQNConfig` / `DoubleDQNConfig` | `GenericDQNAgent` | Yes | Target network, epsilon-greedy |
| **Dueling DQN** | `DuelingDQNConfig` | `GenericDQNAgent` | Yes | V(s) + A(s,a) architecture |
| **DQN+PER** | `DQNPERConfig` | `GenericDQNPERAgent` | Yes | Prioritized replay (sum-tree) |
| **Noisy DQN** | `NoisyDQNConfig` | `GenericDQNAgent` | Yes | NoisyLinear, no epsilon-greedy |
| **DQN CNN** | `DQNCNNConfig` | `GenericDQNAgent` | Yes | NatureDQN for pixel observations |
| **C51** | `C51Config` | `GenericC51Agent` | Yes | Categorical distributional (51 atoms) |
| **Rainbow** | `RainbowConfig` | `GenericRainbowAgent` | Yes | C51 + Double + PER + Dueling + Noisy + N-step |
| **DDPG** | `DDPGConfig` | `GenericOffPolicyAgent` | Yes | Deterministic actor, Gaussian noise |
| **TD3** | `TD3Config` | `GenericOffPolicyAgent` | Yes | Twin critics, delayed policy, target smoothing |
| **SAC** | `SACConfig` | `GenericOffPolicyAgent` | Yes | Stochastic policy, max entropy, auto alpha |
| **A2C** | `A2CConfig` | `GenericOnPolicyAgent` | CPU | GAE, softmax policy |
| **PPO** | `PPOConfig` / `PPOCNNConfig` | `GenericOnPolicyAgent` | Yes | Clipped surrogate, multi-epoch |
| **PPO Continuous** | `ContinuousPPOConfig` | `GenericOnPolicyContinuousAgent` | Yes | Unbounded Gaussian (CleanRL-style) |
| **TD-MPC2** | — | `TDMPC2Agent` | Yes | World model ensemble, MPPI, distributional |
| **DreamerV3** | — | `DreamerV3Agent` | Yes | RSSM, imagination rollouts |

## Shared GPU Kernels (core/kernels.mojo)

- **Network ops**: `soft_update_kernel`, `zero_buffer_kernel`, `copy_buffer_kernel`
- **Episode tracking**: `accumulate_rewards_kernel`, `extract_completed_episodes_kernel`
- **Replay buffer**: `store_transitions_kernel`, `sample_indices_kernel`, `gather_batch_kernel`
- **Continuous control**: `td_target_continuous_kernel`, `actor_grad_from_critic_kernel`, `ddpg_exploration_kernel`
- **Distributional RL**: Bellman projection, cross-entropy loss kernels
- **Action sampling**: reparameterization, tanh squashing
- **Advantage computation**: GAE, normalization
- **Gradient clipping**: norm computation, scaling

## Usage

```mojo
# Convenience aliases
from mojo_rl.deep_agents import DQNAgent, DeepSACAgent, RainbowAgent, DreamerV3Agent

# Or use configs directly
from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent, GenericDQNAgent
from mojo_rl.deep_agents.core.configs import TD3Config, SACConfig
from mojo_rl.deep_agents.core.agents import C51Config, RainbowConfig
```
