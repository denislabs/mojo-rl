# mojo-rl Roadmap

## Completed

### Environments - Native Mojo (Tabular)

- [x] GridWorld - 2D navigation (5x5 default)
- [x] FrozenLake - Slippery grid with holes (4x4)
- [x] CliffWalking - Cliff avoidance task (4x12)
- [x] Taxi - Pickup/dropoff with 500 states

### Environments - Native Mojo (Classic Control)

- [x] CartPole (Native) - Pure Mojo, 145x faster than Gymnasium, GPU batch support
- [x] MountainCar (Native) - Physics matching Gymnasium MountainCar-v0, tile coding support
- [x] Acrobot (Native) - Two-link pendulum with RK4 integration, tile coding + polynomial features
- [x] Pendulum (Native) - Continuous actions, V1 (CPU) and V2 (GPU batch) implementations

### Environments - Native Mojo (2D Physics Engine)

- [x] LunarLander - Custom physics2d engine, discrete + continuous actions, GPU batch, terrain + flame particles
- [x] BipedalWalker - Custom physics2d with revolute joints, 24D obs, 4D continuous actions, lidar, normal/hardcore modes, GPU batch
- [x] CarRacing - Custom physics2d with slip-based tire friction, procedural tracks, 3D continuous actions, GPU batch

### Environments - Native Mojo (MuJoCo-Style, physics3d Engine)

- [x] HalfCheetah - 9 bodies, 8 joints, 6 actuators, 17D obs, GPU batch
- [x] Hopper - 5 bodies, 4 joints, 3 actuators, 11D obs, GPU batch
- [x] Ant - 13 bodies, 9 joints, 8 actuators, 27D obs, GPU batch
- [x] Walker2d - 7 bodies, 6 joints, 6 actuators, 17D obs, GPU batch
- [x] Swimmer - 5 bodies, 2 joints, 2 actuators, 8D obs, GPU batch
- [x] Humanoid - 17 bodies, 16 joints, 17 actuators, 376D obs, GPU batch
- [x] HumanoidStandup - Humanoid with standing reward, GPU batch
- [x] InvertedPendulum - 2 bodies, 1 joint, 1 actuator, 4D obs, GPU batch
- [x] InvertedDoublePendulum - 3 bodies, 2 joints, 2 actuators, 11D obs, GPU batch

### Environments - Gymnasium Wrappers (`envs/gymnasium/`)

- [x] Generic Gymnasium wrapper - Works with any Gymnasium environment
- [x] **Classic Control**: CartPole, MountainCar, Pendulum, Acrobot
- [x] **Box2D**: LunarLander, BipedalWalker, CarRacing
- [x] **Toy Text**: FrozenLake, Taxi, Blackjack, CliffWalking
- [x] **MuJoCo**: HalfCheetah, Ant, Humanoid, Walker2d, Hopper, Swimmer, InvertedPendulum, InvertedDoublePendulum, Reacher, Pusher

### Algorithms - TD Methods

- [x] Q-Learning - Off-policy TD learning
- [x] SARSA - On-policy TD learning
- [x] Expected SARSA - Lower variance than SARSA
- [x] Double Q-Learning - Reduces overestimation bias

### Algorithms - Multi-step Methods

- [x] N-step SARSA - Configurable n-step returns
- [x] SARSA(lambda) - Eligibility traces with replacing traces
- [x] Monte Carlo - First-visit MC

### Algorithms - Model-based

- [x] Dyna-Q - Q-Learning with model-based planning
- [x] Priority Sweeping - Prioritized updates by TD error

### Infrastructure

- [x] Replay Buffer - Circular buffer for experience replay
- [x] Prioritized Replay Buffer - Samples by TD error priority (sum-tree)
- [x] Q-Learning with Replay - Off-policy learning with replay buffer
- [x] Q-Learning with PER - Off-policy learning with prioritized replay
- [x] Continuous Replay Buffer - For continuous state/action algorithms
- [x] GPU Replay Buffer - Device-side circular buffer for GPU training
- [x] Sequence Replay Buffer - Contiguous sequence sampling for model-based RL (TD-MPC2)

### Function Approximation

- [x] Tile Coding - Multi-dimensional overlapping tilings for continuous states
- [x] Linear Function Approximation - Polynomial and RBF feature extractors
- [x] Tiled Q-Learning, SARSA, SARSA(lambda) agents
- [x] Linear Q-Learning, SARSA, SARSA(lambda) agents

### Policy Gradient Methods

- [x] REINFORCE - Monte Carlo policy gradient with optional baseline + entropy variant
- [x] Actor-Critic - One-step TD, Actor-Critic(lambda) with traces, A2C with n-step returns
- [x] GAE (Generalized Advantage Estimation) - Exponentially-weighted TD residuals
- [x] PPO - Clipped surrogate objective, minibatch variant

### Continuous Control (Linear FA)

- [x] DDPG - Deterministic actor + Q-critic with target networks
- [x] TD3 - Twin critics, delayed policy, target smoothing
- [x] SAC - Stochastic Gaussian policy, max entropy, auto alpha tuning

### Deep Learning Framework (`nn/`)

- [x] **Model Trait** - Stateless layers with compile-time dimensions
  - Linear, LinearReLU, LinearTanh, ReLU, Tanh, Sigmoid, Softmax, Mish
  - LayerNorm, SimNorm, Dropout, NormedLinear (Linear->LayerNorm->Mish)
  - StochasticActor (Gaussian policy with reparameterization trick)
  - Sequential[*LAYERS] - Variadic N-layer composition
- [x] **Optimizer Trait** - SGD, Adam, AdamW, RMSprop, Muon
- [x] **Loss Function Trait** - MSELoss, HuberLoss, CrossEntropyLoss, SoftCrossEntropyLoss, TwoHot (C51 distributional)
- [x] **Initializer Trait** - Xavier, Kaiming, LeCun, Zeros, Ones, Constant, Uniform, Normal
- [x] **Training Infrastructure**
  - Trainer[MODEL, OPT, LOSS] with CPU and GPU training loops
  - NetworkState (CPU) and GPUNetworkState (device memory)
  - NetworkPair / GPUNetworkPair (online + target networks with soft updates)
- [x] **Checkpointing** - Text-based and binary checkpoint formats
  - Save/load for agents, networks, trainers
- [x] **GPU Kernels** - Tiled matmul (shared memory + MMA tensor cores), elementwise ops, Apple Silicon optimizations
  - Fused matmul+bias+activation kernels for inference and training
  - Matmul backward kernels (dx, dW)

### Automatic Differentiation (`nn/autodiff/`)

- [x] **DiffOp Trait** - Fine-grained differentiable operations
  - MatMul, BiasAdd, ReLUOp, TanhOp, SigmoidOp, MishOp
  - SoftmaxOp, LayerNormOp, RMSNormOp
  - Scale, ElemMul, ReduceSum, ReduceMean
  - DropoutOp, Flatten, Embedding
  - Conv2D, MaxPool2D, AvgPool2D
  - ScaledDotProductAttention
- [x] **AutoDiffChain[*OPS]** - Variadic composition of N DiffOps into a Model
- [x] **Fused Operations** - FusedMatMulBias, FusedMatMulBiasReLU, FusedMatMulBiasTanh, generic FusedMatMulBiasActivation
- [x] **AutoFused[*OPS]** - Automatic compile-time greedy fusion (MatMul+Bias+Act -> fused kernel)
- [x] **Combinators** - Residual[Inner], Parallel[*BRANCHES], Repeat[n, Inner]
- [x] **Composites** - Pre-built architectures: ResBlock, ResNet, LeNet, FFN
- [x] **Convenience Aliases** - Dense, DenseReLU, DenseTanh (AutoDiffChain shortcuts)

### Deep RL Agents (`deep_agents/`)

- [x] **DQN / Double DQN** - Deep Q-Network with target network, epsilon-greedy, GPU training
- [x] **DQN + PER** - Prioritized replay with sum-tree, importance sampling, beta annealing
- [x] **Dueling DQN** - V(s) + A(s,a) architecture with shared backbone
- [x] **DDPG** - Deterministic actor, Gaussian noise, target networks, GPU training
- [x] **TD3** - Twin critics, delayed policy updates, target smoothing, GPU training
- [x] **SAC** - Stochastic Gaussian policy, max entropy, auto alpha tuning, GPU training
- [x] **A2C** - Advantage Actor-Critic with GAE
- [x] **PPO (Discrete)** - Clipped surrogate, multi-epoch, entropy bonus
- [x] **PPO (Continuous)** - Unbounded Gaussian policy (CleanRL-style), GPU training, LR annealing, KL early stopping, gradient clipping
- [x] **TD-MPC2** - Model-based RL with world model ensemble, MPPI planning, distributional RL (two-hot), sequence replay buffer

### Deep RL Shared Infrastructure (`deep_agents/core/`)

- [x] **Trait-based agent design** - OffPolicyContinuousAgent, OffPolicyDiscreteAgent, OnPolicyContinuousAgent, OnPolicyDiscreteAgent, GPUOffPolicyAgent, GPUOnPolicyContinuousAgent, Checkpointable
- [x] **Unified training loops** - CPU and GPU variants for off-policy and on-policy agents
- [x] **Shared GPU kernels** - 80+ reusable kernels (soft update, episode tracking, replay, TD targets, etc.)
- [x] **Replay buffer variants** - HeapReplayBuffer, PrioritizedReplayBuffer, GPUReplayBuffer, SequenceReplayBuffer

### 3D Physics Engine (`physics3d/`)

- [x] **Generalized coordinates dynamics** - MuJoCo-inspired joint-space representation
- [x] **Mass matrix** - CRBA (Composite Rigid Body Algorithm), sparse variants, LDL/LU decomposition
- [x] **Bias forces** - RNE (Recursive Newton-Euler) for Coriolis + gravity
- [x] **Jacobians** - Contact Jacobians, analytical Jacobians, composite inertia
- [x] **Joint types** - FREE (7 DOF), BALL (4 DOF), SLIDE (1 DOF), HINGE (1 DOF)
- [x] **Integrators** - Euler, ImplicitFast (default), Implicit (full RNE velocity derivative), RK4
- [x] **Constraint solvers** - PGS, Newton, CG, Island-based PGS with early termination
- [x] **Collision detection** - Sphere/capsule/box narrow-phase, Sweep-and-Prune broadphase, CPU + GPU
- [x] **Constraint building** - Contact, equality, tendon constraints, CPU + GPU
- [x] **Compile-time model specs** - BodySpec, JointSpec, GeomSpec, ActuatorSpec, EqualitySpec, TendonSpec traits
- [x] **ModelDef compositor** - Variadic iteration for N-body composition
- [x] **MJCF XML parser** - XML -> FlatModelDef -> Model/Data pipeline
- [x] **GPU support** - Forward kinematics, body velocities, collision, constraint building all have GPU paths
- [x] **Validation** - 75 test files comparing against MuJoCo reference

### 2D Physics Engine (`physics2d/`)

- [x] **Batched GPU physics** - Strided [BATCH, STATE_SIZE] layout for parallel simulation
- [x] **Impulse solver** - Velocity + position level contact resolution with warm-starting
- [x] **Revolute joints** - Motor control, spring damping, angle limits
- [x] **Terrain collision** - Flat ground and edge terrain detection
- [x] **Articulated chains** - Multi-body support for Hopper, Walker, Cheetah configurations
- [x] **Car physics** - Slip-based tire friction, track tile lookup, fused GPU kernel
- [x] **Lidar sensors** - Ray-cast distance sensing
- [x] **PhysicsKernel** - One-call step_gpu() for full physics step

### 3D Math Library (`math3d/`)

- [x] Vec3, Quat, Mat3, Mat4 with GPU variants

### Rendering (`render/`)

- [x] **SDL3 FFI bindings** - Complete bindings (38 files: video, render, GPU, events, keyboard, mouse, audio, etc.)
- [x] **Renderer2D** - SDL3 2D CPU rasterizer (rect, line, circle, polygon, text)
- [x] **Renderer3D** - GPU-accelerated 3D renderer using SDL3 GPU API
  - Blinn-Phong lighting with up to 4 lights
  - Shadow mapping
  - Procedural checkerboard ground with reflections
  - Gradient skybox
  - GPU bitmap font atlas for HUD text
  - Mesh caching (sphere, box, capsule LRU cache)
  - Deferred draw commands
  - MSL Metal shaders (solid, ground, line, shadow, reflection, skybox, text)
- [x] **Interactive camera** - Orbit, pan, zoom (mouse), camera switching (1-9 keys)
- [x] **Playback control** - Pause (Space), step (->), reset camera (R)
- [x] **Video recording** - MP4/GIF via Python imageio (V key toggle)
- [x] **Screenshot** - GPU readback (S key)
- [x] **Colors and shapes** - 30+ named colors, 10+ 2D shape factories, 3D wireframe generators
- [x] **Camera types** - 2D Camera + RotatingCamera, 3D Camera3D with perspective projection

### Vectorized Environments

- [x] VecCartPoleEnv - SIMD-based parallel CartPole (SoA layout, auto-reset)

### Infrastructure

- [x] Logging/Metrics - EpisodeMetrics, TrainingMetrics with convergence/success tracking
- [x] Hyperparameter Search - Grid and random search with CSV export
- [x] Learning rate scheduling - Linear annealing in Deep PPO

## In Progress / Next Steps

### GPU Optimization

- [ ] MMA tensor core matmul for all platforms (currently Apple Silicon optimized)
- [ ] Mixed precision training (fp16/bf16)
- [ ] Multi-GPU support

### Infrastructure Improvements

- [ ] Cosine annealing and warmup LR schedulers
- [ ] Curriculum learning framework (CurriculumScheduler trait exists)
- [ ] Population-based training
- [ ] TensorBoard-style metric visualization
- [ ] Parallel training across multiple environments

### Environments

- [ ] Custom environment builder
- [ ] MinAtar environments (simplified Atari for CNN testing)
- [ ] POMDP benchmark environments

## Future Exploration

> Ideas for future development, roughly prioritized by potential impact.

### Offline / Batch RL

Learning from fixed datasets without environment interaction.

- [ ] **CQL (Conservative Q-Learning)** - Penalizes Q-values for out-of-distribution actions
- [ ] **IQL (Implicit Q-Learning)** - Avoids explicit policy evaluation
- [ ] **Decision Transformer** - Treats RL as sequence modeling (returns-conditioned)
- [ ] **BCQ (Batch-Constrained Q-Learning)** - Constrains policy to data support
- [ ] **AWR (Advantage Weighted Regression)** - Simple offline-compatible algorithm
- [ ] Dataset infrastructure - D4RL format loading/saving

### Distributional RL

Model full distribution of returns instead of just expected value.

- [ ] **C51** - Categorical distribution over returns (51 atoms)
- [ ] **QR-DQN** - Quantile regression for distributional RL
- [ ] **IQN (Implicit Quantile Networks)** - Sample quantile fractions
- [ ] **Rainbow** - Combines DQN improvements (already have Double, Dueling, PER - add C51, NoisyNets)

### Exploration Enhancements

For sparse reward and hard exploration problems.

- [ ] **ICM (Intrinsic Curiosity Module)** - Prediction error as intrinsic reward
- [ ] **RND (Random Network Distillation)** - Simpler curiosity-driven exploration
- [ ] **NoisyNets** - Learnable parametric noise in network weights
- [ ] **Bootstrapped DQN** - Ensemble for uncertainty estimation

### Recurrent Policies (POMDPs)

For partial observability and memory-dependent tasks.

- [ ] **LSTM layer** in nn Model trait
- [ ] **GRU layer** - Simpler recurrent alternative
- [ ] **R2D2** - Recurrent DQN with burn-in and stored hidden states

### Multi-Agent RL

Cooperative and competitive multi-agent settings.

- [ ] **MADDPG** - Multi-agent DDPG with centralized critic
- [ ] **QMIX** - Value decomposition for cooperative agents
- [ ] **IPPO** - Independent PPO baseline
- [ ] Simple multi-agent environments

### Model-Based Deep RL (Beyond TD-MPC2)

- [ ] **Dreamer** - Latent imagination with actor-critic
- [ ] **MBPO (Model-Based Policy Optimization)** - Short rollouts from learned model
- [ ] **World Models** - VAE + MDN-RNN for latent dynamics

### Architecture Extensions

- [ ] Transformer blocks for sequence modeling
- [ ] Imitation learning (Behavioral Cloning, DAgger)

### Quick Wins

- [ ] **HER (Hindsight Experience Replay)** - Works with existing replay infrastructure
- [ ] **Soft Q-Learning** - Max entropy with discrete actions
- [ ] **n-step DQN** - Multi-step returns for DQN

## Algorithm Summary

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| Q-Learning | TD | Off-policy, max Q(s',a') |
| SARSA | TD | On-policy, Q(s',a') |
| Expected SARSA | TD | E[Q(s',a')], lower variance |
| Double Q-Learning | TD | Two Q-tables, reduces overestimation |
| N-step SARSA | Multi-step | n-step returns |
| SARSA(lambda) | Eligibility | Trace decay |
| Monte Carlo | Episode | Complete episode returns |
| Dyna-Q | Model-based | Planning with learned model |
| Priority Sweeping | Model-based | Prioritized planning |
| Q-Learning + Replay | Replay | Experience replay buffer |
| Q-Learning + PER | Replay | Prioritized experience replay |
| Tiled Q-Learning/SARSA | Function Approx | Tile coding for continuous states |
| Linear Q-Learning/SARSA | Function Approx | Polynomial, RBF features |
| REINFORCE | Policy Gradient | Monte Carlo policy gradient |
| Actor-Critic / A2C | Policy Gradient | TD-based + n-step returns |
| PPO | Policy Gradient | Clipped surrogate, stable updates |
| DDPG (Linear) | Continuous | Deterministic policy + Q-critic |
| TD3 (Linear) | Continuous | Twin critics + delayed updates |
| SAC (Linear) | Continuous | Stochastic + entropy + auto alpha |
| Deep DQN | Deep RL | Neural Q-function + target network |
| Deep Double DQN | Deep RL | Reduced overestimation |
| Deep Dueling DQN | Deep RL | V(s) + A(s,a) streams |
| Deep DQN + PER | Deep RL | Priority sampling by TD error |
| Deep DDPG | Deep RL | Deterministic actor + Q-critic |
| Deep TD3 | Deep RL | Twin critics + delayed + smoothing |
| Deep SAC | Deep RL | Stochastic + entropy + auto alpha |
| Deep A2C | Deep RL | Actor-Critic with GAE |
| Deep PPO | Deep RL | Clipped surrogate + LR anneal + KL stop |
| Deep PPO Continuous | Deep RL | Unbounded Gaussian, GPU training |
| TD-MPC2 | Model-Based RL | World model + MPPI + distributional |
