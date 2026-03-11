# deep_agents/ - Deep RL Agents

Neural network-based RL agents with CPU and GPU training support. All agents use the trait-based `nn/` framework for models and the shared `core/` infrastructure for training loops.

## Architecture

Each agent implements trait combinations from `core/`:
- **Off-policy**: `OffPolicyContinuousAgent` / `OffPolicyDiscreteAgent` + `GPUOffPolicyAgent`
- **On-policy**: `OnPolicyContinuousAgent` / `OnPolicyDiscreteAgent` + `GPUOnPolicyContinuousAgent`
- **Checkpointable**: Save/load via binary checkpoint format

State is separated from agent logic: `*CPUState` holds CPU params/buffers, `*GPUState` holds device memory.

## Agents

```
deep_agents/
├── core/                   # Shared infrastructure
│   ├── checkpoint_trait.mojo    # Checkpointable trait
│   ├── eval.mojo                # Evaluation loops (all agent types)
│   ├── offpolicy_train.mojo     # CPU off-policy training loop
│   ├── onpolicy_train.mojo      # CPU on-policy training loop
│   ├── gpu_offpolicy_train.mojo # GPU off-policy training loop
│   ├── gpu_onpolicy_train.mojo  # GPU on-policy training loop
│   ├── offpolicy_helpers.mojo   # Action selection, transition storage
│   ├── onpolicy_helpers.mojo    # GAE computation, advantage normalization
│   ├── kernels.mojo             # 80+ shared GPU kernels
│   ├── utils.mojo               # InlineArray helpers, progress bar
│   └── replay/                  # Experience replay buffers
│       ├── replay_buffer.mojo       # HeapReplayBuffer, PrioritizedReplayBuffer
│       ├── gpu_replay_buffer.mojo   # GPUReplayBuffer (device-side circular buffer)
│       └── sequence_replay_buffer.mojo # SequenceReplayBuffer (for TD-MPC2)
├── dqn/                    # Deep Q-Network
│   ├── dqn.mojo            # DQNAgent (Double DQN, epsilon-greedy, GPU support)
│   ├── state.mojo          # DQNGPUState
│   └── kernels.mojo        # TD target kernels
├── dqn_per/                # DQN + Prioritized Experience Replay
│   └── dqn_per.mojo        # DQNPERAgent (sum-tree sampling, CPU only)
├── dueling_dqn/            # Dueling DQN
│   └── dueling_dqn.mojo    # DuelingDQNAgent: Q(s,a) = V(s) + A(s,a) - mean(A)
├── ddpg/                   # Deep Deterministic Policy Gradient
│   ├── ddpg.mojo           # DeepDDPGAgent (deterministic actor, GPU support)
│   └── state.mojo          # DDPGCPUState, DDPGGPUState
├── td3/                    # Twin Delayed DDPG
│   ├── td3.mojo            # DeepTD3Agent (twin critics, delayed policy, GPU)
│   ├── state.mojo          # TD3CPUState, TD3GPUState
│   └── kernels.mojo        # Clipped noise kernel
├── sac/                    # Soft Actor-Critic
│   ├── sac.mojo            # DeepSACAgent (max entropy, auto alpha, GPU)
│   ├── state.mojo          # SACCPUState, SACGPUState
│   └── kernels.mojo        # Reparameterization + tanh squashing kernels
├── a2c/                    # Advantage Actor-Critic
│   ├── a2c.mojo            # DeepA2CAgent (GAE, softmax policy)
│   └── kernels.mojo        # GAE + softmax sampling kernels
├── ppo/                    # Proximal Policy Optimization
│   ├── ppo.mojo            # DeepPPOAgent (discrete, clipped surrogate)
│   ├── ppo_continuous.mojo # DeepPPOContinuousAgent (unbounded Gaussian, GPU)
│   ├── state.mojo          # PPO*State, PPO*GPUState
│   └── kernels.mojo        # Action sampling, gradient clipping kernels
└── tdmpc2/                 # TD-MPC2 (Model-Based)
    ├── tdmpc2.mojo         # TDMPC2Agent (world model + MPPI planning)
    ├── world_model.mojo    # Encoder, dynamics, reward, termination, policy, Q-ensemble
    ├── mppi.mojo           # Model Predictive Path Integral planner
    ├── state.mojo          # TDMPC2CPUState, TDMPC2GPUState
    └── kernels.mojo        # 80+ TDMPC2-specific GPU kernels
```

## Agent Summary

| Agent | Type | Actions | GPU | Key Features |
|-------|------|---------|-----|-------------|
| **DQN** | Off-policy | Discrete | Yes | Double DQN, target network, epsilon-greedy |
| **DQN+PER** | Off-policy | Discrete | No | Prioritized replay (sum-tree), importance sampling |
| **Dueling DQN** | Off-policy | Discrete | No | V(s) + A(s,a) architecture |
| **DDPG** | Off-policy | Continuous | Yes | Deterministic actor, Gaussian noise |
| **TD3** | Off-policy | Continuous | Yes | Twin critics, delayed policy, target smoothing |
| **SAC** | Off-policy | Continuous | Yes | Stochastic policy, max entropy, auto alpha |
| **A2C** | On-policy | Discrete | No | GAE, softmax policy |
| **PPO** | On-policy | Discrete | No | Clipped surrogate, multi-epoch |
| **PPO Continuous** | On-policy | Continuous | Yes | Unbounded Gaussian (CleanRL-style) |
| **TD-MPC2** | Off-policy (model-based) | Continuous | Yes | World model ensemble, MPPI, distributional RL |

## Shared GPU Kernels (core/kernels.mojo)

- **Network ops**: `soft_update_kernel`, `zero_buffer_kernel`, `copy_buffer_kernel`
- **Episode tracking**: `accumulate_rewards_kernel`, `extract_completed_episodes_kernel`
- **Replay buffer**: `store_transitions_kernel`, `sample_indices_kernel`, `gather_batch_kernel`
- **Continuous control**: `td_target_continuous_kernel`, `actor_grad_from_critic_kernel`, `ddpg_exploration_kernel`
