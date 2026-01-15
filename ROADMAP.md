# mojo-rl Roadmap

## Completed

### Environments - Native Mojo
- [x] GridWorld - 2D navigation (5x5 default)
- [x] FrozenLake - Slippery grid with holes (4x4)
- [x] CliffWalking - Cliff avoidance task (4x12)
- [x] Taxi - Pickup/dropoff with 500 states
- [x] CartPole (Native) - Pure Mojo implementation with 145x performance boost over Python
  - `cartpole_native.mojo` - Physics engine matching Gymnasium CartPole-v1
  - `cartpole_renderer.mojo` - Pygame-based visualization (optional)
- [x] MountainCar (Native) - Pure Mojo implementation
  - `mountain_car_native.mojo` - Physics matching Gymnasium MountainCar-v0
  - Includes tile coding factory function
- [x] Pendulum (Native) - Pure Mojo implementation with continuous actions
  - `pendulum.mojo` - Physics matching Gymnasium Pendulum-v1
  - Supports both discrete (tabular) and continuous (DDPG) action interfaces
  - SDL2 rendering with pivot, rod, and bob visualization
- [x] LunarLander (Native) - Pure Mojo implementation with custom 2D physics engine
  - `lunar_lander.mojo` - Physics matching Gymnasium LunarLander-v3
  - Custom minimal Box2D-style physics (rigid bodies, joints, collision detection)
  - Both discrete (4 actions) and continuous (2D throttle) action spaces
  - Wind and turbulence effects
  - SDL2 rendering with terrain, lander, legs, helipad, flags, and flame particles
  - Implements `BoxDiscreteActionEnv` trait for RL agent compatibility
- [x] BipedalWalker (Native) - Pure Mojo implementation with custom 2D physics engine
  - `bipedal_walker.mojo` - Physics matching Gymnasium BipedalWalker-v3
  - Both normal and hardcore modes (stumps, pits, stairs obstacles)
  - Continuous action space (4D: hip and knee torques for both legs)
  - 24D observation: hull state, joint angles/speeds, leg contacts, 10-ray lidar
  - Custom Box2D-style physics with revolute joints and motors
  - SDL2 rendering with scrolling viewport following the walker
  - Implements `BoxContinuousActionEnv` trait for RL agent compatibility
- [x] CarRacing (Native) - Pure Mojo implementation with procedural track generation
  - `car_racing.mojo` - Physics matching Gymnasium CarRacing-v3
  - Procedural track generation with random bezier checkpoints
  - Top-down car dynamics with 4-wheel friction model
  - Continuous action space (3D: steering, gas, brake)
  - Discrete action space (5: nothing, left, right, gas, brake)
  - 12D observation: position, velocity, angle, wheel states, waypoint direction
  - Tile-based reward system (+1000/N per tile, -0.1 per frame)
  - SDL2 rendering with rotating camera following the car
  - Implements `BoxContinuousActionEnv` trait for RL agent compatibility

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
- [x] SARSA(λ) - Eligibility traces with replacing traces
- [x] Monte Carlo - First-visit MC

### Algorithms - Model-based
- [x] Dyna-Q - Q-Learning with model-based planning
- [x] Priority Sweeping - Prioritized updates by TD error

### Infrastructure
- [x] Replay Buffer - Circular buffer for experience replay
- [x] Prioritized Replay Buffer - Samples by TD error priority
- [x] Q-Learning with Replay - Off-policy learning with replay buffer
- [x] Continuous Replay Buffer - For continuous state/action algorithms (DDPG, TD3, SAC)
  - `core/continuous_replay_buffer.mojo` - ContinuousTransition, ContinuousReplayBuffer

### Function Approximation
- [x] Tile Coding - Linear function approximation for continuous state spaces
  - `core/tile_coding.mojo` - TileCoding and TiledWeights structs
  - `agents/tiled_qlearning.mojo` - TiledQLearningAgent, TiledSARSAAgent, TiledSARSALambdaAgent
  - `examples/cartpole_tiled.mojo` - CartPole training with tile coding

### Policy Gradient Methods
- [x] REINFORCE - Monte Carlo policy gradient with softmax policy
  - `agents/reinforce.mojo` - REINFORCEAgent, REINFORCEWithEntropyAgent
  - Supports optional learned baseline for variance reduction
  - Entropy regularization variant for improved exploration
- [x] Actor-Critic - Combine policy and value learning
  - `agents/actor_critic.mojo` - ActorCriticAgent, ActorCriticLambdaAgent, A2CAgent
  - One-step TD Actor-Critic with online updates
  - Actor-Critic(λ) with eligibility traces for both actor and critic
  - A2C with n-step returns and entropy bonus
- [x] Examples - `examples/cartpole_policy_gradient.mojo`
- [x] GAE (Generalized Advantage Estimation)
  - `agents/ppo.mojo` - `compute_gae()` function
  - Computes advantages using exponentially-weighted TD residuals
  - Parameterized by λ (0=TD, 1=MC, typically 0.95)
- [x] PPO (Proximal Policy Optimization)
  - `agents/ppo.mojo` - PPOAgent, PPOAgentWithMinibatch
  - Clipped surrogate objective for stable policy updates
  - Multiple optimization epochs per rollout
  - Entropy bonus for exploration
  - `examples/cartpole_ppo.mojo` - CartPole training with PPO

### Linear Function Approximation
- [x] Linear Q-Learning - Q-Learning with arbitrary feature vectors
  - `core/linear_fa.mojo` - LinearWeights, PolynomialFeatures, RBFFeatures
  - `agents/linear_qlearning.mojo` - LinearQLearningAgent, LinearSARSAAgent, LinearSARSALambdaAgent
  - `examples/mountain_car_linear.mojo` - MountainCar training with polynomial and RBF features

### Continuous Control (Deterministic Policy Gradient)
- [x] DDPG - Deep Deterministic Policy Gradient with linear function approximation
  - `agents/ddpg.mojo` - DDPGAgent with deterministic actor and Q-function critic
  - Target networks with soft updates for stability
  - Gaussian exploration noise
  - Works with polynomial features for continuous state/action spaces
  - `examples/pendulum_ddpg.mojo` - Pendulum swing-up with DDPG
- [x] TD3 - Twin Delayed DDPG with linear function approximation
  - `agents/td3.mojo` - TD3Agent with twin Q-networks
  - Twin critics to reduce overestimation bias (uses min(Q1, Q2))
  - Delayed policy updates (actor updates every N critic updates)
  - Target policy smoothing (clipped noise added to target actions)
  - `examples/pendulum_td3.mojo` - Pendulum swing-up with TD3
- [x] SAC - Soft Actor-Critic with linear function approximation
  - `agents/sac.mojo` - SACAgent with stochastic Gaussian policy
  - Maximum entropy RL: maximizes reward + α * entropy
  - Stochastic policy for better exploration
  - Twin Q-networks (like TD3)
  - Automatic entropy temperature (α) tuning
  - `examples/pendulum_sac.mojo` - Pendulum swing-up with SAC

### Deep RL Infrastructure (`deep_rl/`) - PyTorch-like Trait-Based Architecture
- [x] **Model Trait** - Stateless neural network layers with compile-time dimensions
  - `deep_rl/model/model.mojo` - Base `Model` trait with `forward()`, `backward()`, GPU variants
  - Parameters managed externally via `LayoutTensor` for zero-copy composition
  - Compile-time constants: `IN_DIM`, `OUT_DIM`, `PARAM_SIZE`, `CACHE_SIZE`
- [x] **Layer Implementations** - Pluggable neural network layers
  - `deep_rl/model/linear.mojo` - `Linear[in_dim, out_dim]` with tiled matmul GPU kernels
  - `deep_rl/model/relu.mojo` - `ReLU[dim]` activation layer
  - `deep_rl/model/tanh.mojo` - `Tanh[dim]` activation layer
  - `deep_rl/model/sigmoid.mojo` - `Sigmoid[dim]` activation layer
  - `deep_rl/model/softmax.mojo` - `Softmax[dim]` activation layer (for policy networks)
  - `deep_rl/model/layer_norm.mojo` - `LayerNorm[dim]` layer normalization
  - `deep_rl/model/dropout.mojo` - `Dropout[dim, p, training]` regularization (compile-time training flag)
  - `deep_rl/model/sequential.mojo` - `Seq2[L0, L1]` composition + `seq()` helpers (up to 8 layers)
- [x] **Optimizer Trait** - Pluggable parameter update rules
  - `deep_rl/optimizer/optimizer.mojo` - Base `Optimizer` trait with `step()`, `step_gpu()`
  - `deep_rl/optimizer/sgd.mojo` - `SGD` optimizer
  - `deep_rl/optimizer/adam.mojo` - `Adam` optimizer with bias correction
  - `deep_rl/optimizer/rmsprop.mojo` - `RMSprop` optimizer with adaptive learning rates
  - `deep_rl/optimizer/adamw.mojo` - `AdamW` optimizer with decoupled weight decay
- [x] **Loss Function Trait** - Pluggable loss computation
  - `deep_rl/loss/loss.mojo` - Base `LossFunction` trait
  - `deep_rl/loss/mse.mojo` - `MSELoss` with block reduction for GPU
  - `deep_rl/loss/huber.mojo` - `HuberLoss` (Smooth L1) for DQN variants
  - `deep_rl/loss/cross_entropy.mojo` - `CrossEntropyLoss` for classification/policy gradients
- [x] **Initializer Trait** - Pluggable weight initialization
  - `deep_rl/initializer/initializers.mojo` - Xavier, Kaiming, LeCun, Zeros, Ones, Uniform, Normal
- [x] **Stochastic Actor** - Gaussian policy network for SAC/PPO
  - `deep_rl/model/stochastic_actor.mojo` - `StochasticActor[in_dim, action_dim]` with learned mean and log_std
  - Two linear heads: mean_head and log_std_head with shared input
  - log_std clamping to [-20, 2] for numerical stability
  - Utility functions: `rsample()`, `sample_action()`, `compute_log_prob()`, `get_deterministic_action()`
  - Reparameterization trick: action = tanh(mean + exp(log_std) * noise)
  - Log probability with tanh squashing correction
  - GPU kernels with tiled matmul for both heads
  - Compatible with `seq()` composition for backbone networks
- [x] **Training Utilities** - High-level training management
  - `deep_rl/training/trainer.mojo` - `Trainer[MODEL, OPTIMIZER, LOSS, INITIALIZER]`
  - Manages params, grads, optimizer state externally
  - `train[BATCH]()` (CPU) and `train_gpu[BATCH]()` (GPU) methods
- [x] **Constants** - Global configuration
  - `deep_rl/constants.mojo` - `dtype`, `TILE` (matmul tile size), `TPB` (threads per block)

### Deep RL Agents (`deep_agents/`)
- [x] **Deep DQN / Double DQN** - Deep Q-Network using new trait-based architecture  - `deep_agents/dqn.mojo` - DeepDQNAgent using `seq()` composition
  - Uses `Linear`, `ReLU` layers with `Adam` optimizer and `Kaiming` initializer
  - Network via `seq(Linear[obs, h1](), ReLU[h1](), Linear[h1, h2](), ReLU[h2](), Linear[h2, actions]())`
  - GPU kernels for DQN-specific operations (TD targets, soft updates, epsilon-greedy)
  - Target network with soft updates for stability
  - Experience replay buffer
  - Epsilon-greedy exploration with decay
  - **Double DQN** (compile-time flag): Uses online network to select actions, target to evaluate
  - Works with `BoxDiscreteActionEnv` trait
  - `examples/lunar_lander_dqn.mojo` - LunarLander training with Double DQN
- [x] **Deep SAC** - Soft Actor-Critic using new trait-based architecture  - `deep_agents/sac.mojo` - DeepSACAgent using `seq()` composition
  - Actor: `seq(Linear, ReLU, Linear, ReLU, StochasticActor)` for Gaussian policy
  - Twin critics: `seq(Linear, ReLU, Linear, ReLU, Linear)` for Q-functions
  - Target networks for critics (no target actor, per SAC design)
  - Automatic entropy temperature (α) tuning (optional)
  - Maximum entropy RL: maximizes reward + α * entropy
  - **Proper gradient backpropagation**: dQ/da through critic, chain rule through reparameterization
  - `rsample_backward()` utility for reparameterization trick gradients (CPU + GPU)
  - `backward_input()` in Network wrapper for gradient chaining between networks
  - Works with `BoxContinuousActionEnv` trait
  - `examples/pendulum_deep_sac.mojo` - Pendulum training with Deep SAC (achieves ~-140 reward)
- [x] **Deep DDPG** - Deep Deterministic Policy Gradient using new trait-based architecture  - `deep_agents/ddpg.mojo` - DeepDDPGAgent using `seq()` composition
  - Actor: `seq(Linear, ReLU, Linear, ReLU, Linear, Tanh)` for deterministic policy
  - Critic: `seq(Linear, ReLU, Linear, ReLU, Linear)` for Q-function
  - Target networks for both actor and critic with soft updates
  - Gaussian exploration noise with decay
  - Works with `BoxContinuousActionEnv` trait
  - `examples/pendulum_deep_ddpg.mojo` - Pendulum training with Deep DDPG
- [x] **Deep TD3** - Twin Delayed DDPG using new trait-based architecture  - `deep_agents/td3.mojo` - DeepTD3Agent using `seq()` composition
  - Actor: `seq(Linear, ReLU, Linear, ReLU, Linear, Tanh)` for deterministic policy
  - Twin critics: `seq(Linear, ReLU, Linear, ReLU, Linear)` for Q1 and Q2 functions
  - Target networks for actor and both critics with soft updates
  - Twin Q-networks: uses min(Q1, Q2) targets to reduce overestimation
  - Delayed policy updates: actor updates every N critic updates
  - Target policy smoothing: adds clipped noise to target actions
  - Gaussian exploration noise with decay
  - Works with `BoxContinuousActionEnv` trait
  - `examples/pendulum_deep_td3.mojo` - Pendulum training with Deep TD3
- [x] **Deep Dueling DQN** - Dueling DQN using new trait-based architecture  - `deep_agents/dueling_dqn.mojo` - DuelingDQNAgent using `seq()` composition
  - Split architecture: backbone + value_head + advantage_head (3 Network instances)
  - Backbone: `seq(Linear, ReLU, Linear, ReLU)` for shared features
  - Value head: `seq(Linear, ReLU, Linear)` → V(s)
  - Advantage head: `seq(Linear, ReLU, Linear)` → A(s,a)
  - Q(s,a) = V(s) + (A(s,a) - mean(A)) for value decomposition
  - Double DQN support via compile-time flag
  - Works with `BoxDiscreteActionEnv` trait
  - `examples/lunar_lander_dueling_dqn.mojo` - LunarLander training
- [x] **Deep DQN with PER** - DQN with Prioritized Experience Replay using new architecture  - `deep_agents/dqn_per.mojo` - DQNPERAgent using `seq()` composition
  - Q-network: `seq(Linear, ReLU, Linear, ReLU, Linear)` with Kaiming initialization
  - Prioritized replay buffer with sum-tree for O(log n) priority sampling
  - Importance sampling weights correct for non-uniform sampling bias
  - Beta annealing from beta_start to 1.0 over training
  - Double DQN support via compile-time flag
  - Works with `BoxDiscreteActionEnv` trait
  - `examples/lunar_lander_per_demo.mojo` - DQN vs DQN+PER comparison
- [x] **Deep A2C** - Advantage Actor-Critic using new architecture  - `deep_agents/a2c.mojo` - DeepA2CAgent using `seq()` composition
  - Actor: `seq(Linear, ReLU, Linear, ReLU, Linear)` → softmax policy
  - Critic: `seq(Linear, ReLU, Linear, ReLU, Linear)` → value function
  - GAE (Generalized Advantage Estimation) for variance reduction
  - Works with `BoxDiscreteActionEnv` trait
  - `examples/cartpole_deep_a2c_ppo.mojo` - CartPole training demo
- [x] **Deep PPO** - Proximal Policy Optimization using new architecture  - `deep_agents/ppo.mojo` - DeepPPOAgent using `seq()` composition
  - Clipped surrogate objective for stable policy updates
  - Multiple optimization epochs per rollout
  - GAE for advantage estimation
  - **Advanced training features (environment-agnostic):**
    - Linear learning rate annealing (`anneal_lr=True`)
    - KL divergence early stopping (`target_kl=0.015`) - stops epoch if policy changes too much
    - Gradient norm clipping (`max_grad_norm=0.5`) - prevents exploding gradients
    - Entropy coefficient annealing (`anneal_entropy=False`) - optional exploration decay
  - Works with `BoxDiscreteActionEnv` trait
  - `examples/cartpole_deep_a2c_ppo.mojo` - CartPole training demo
  - `tests/test_ppo_gpu.mojo` - GPU training test with advanced features

## In Progress / Next Steps

### GPU Support
- [x] GPU tensor operations - Tiled matmul, elementwise ops (implemented in Linear, ReLU, Tanh)
- [x] GPU-accelerated training - Trainer with `train_gpu[BATCH]()` method
- [ ] GPU-accelerated deep RL agents - Full batch processing on GPU for DQN, SAC, etc.
- [ ] Mixed CPU/GPU pipeline - Environment stepping on CPU, network forward/backward on GPU
- [ ] GPU replay buffer - Store and sample transitions on GPU to avoid CPU-GPU transfers

### Vectorized Environments
- [x] VecCartPoleEnv - Vectorized CartPole running N environments in parallel
  - `core/vec_env.mojo` - VecStepResult struct and SIMD helper functions
  - `envs/vec_cartpole.mojo` - VecCartPoleEnv[num_envs] with SIMD state storage
  - `examples/vec_cartpole_demo.mojo` - Demo and performance benchmark
  - Uses Structure of Arrays (SoA) layout for SIMD-friendly memory access
  - Auto-reset: done environments automatically reset (with early-exit optimization)
  - **Performance**: Uses native SIMD methods (.eq(), .lt(), .gt()), @always_inline,
    and @parameter for compile-time loop unrolling. Achieves ~14-16M steps/sec.
  - **Note on SIMD efficiency**: For simple environments like CartPole, scalar code
    is faster (~34M steps/sec) because the physics is too simple to benefit from
    SIMD parallelism - the overhead exceeds the gains. Vectorized environments
    shine for: (1) batched data collection for neural network training (PPO, A2C),
    (2) more complex environments with heavier physics, (3) future GPU support.

### Infrastructure
- [x] Logging/Metrics - Export learning curves for visualization
- [x] Hyperparameter Search - Grid search / random search
  - `core/hyperparam/` - Hyperparameter optimization module
  - `param_space.mojo` - Parameter spaces for all agent types (Tabular, N-step, Lambda, Model-based, Replay, PolicyGradient, PPO, Continuous)
  - `search_result.mojo` - TrialResult and SearchResults with CSV export
  - `grid_search.mojo` - Grid search utilities
  - `random_search.mojo` - Random search utilities
  - `agent_factories.mojo` - Factory functions for all tabular agents
  - `examples/hyperparam_search_demo.mojo` - Demo with Q-Learning on GridWorld
  - Supports multiple metrics: mean/max/final reward, convergence speed
- [ ] Parallel Training - Multiple environments (vectorized envs provide foundation)

### Environments - Native Ports (Next)
- [x] LunarLander (Native) - Port Box2D physics to pure Mojo (COMPLETED - see above)
- [x] Pendulum (Native) - Port continuous action environment (COMPLETED - see above)
- [x] BipedalWalker (Native) - Port Box2D walker physics to pure Mojo (COMPLETED - see above)
- [x] CarRacing (Native) - Port Box2D car racing with procedural tracks (COMPLETED - see above)
- [ ] Custom environment builder

## Future Exploration

> Ideas for future development, roughly prioritized by potential impact.

### Offline / Batch RL
Learning from fixed datasets without environment interaction - practical for real-world applications.
- [ ] **CQL (Conservative Q-Learning)** - Penalizes Q-values for out-of-distribution actions
- [ ] **IQL (Implicit Q-Learning)** - Avoids explicit policy evaluation, simpler than CQL
- [ ] **Decision Transformer** - Treats RL as sequence modeling (returns-conditioned)
- [ ] **BCQ (Batch-Constrained Q-Learning)** - Constrains policy to data support
- [ ] **AWR (Advantage Weighted Regression)** - Simple offline-compatible algorithm
- [ ] Dataset infrastructure - Loading/saving offline RL datasets (D4RL format)

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
- [ ] **NoisyNets** - Learnable parametric noise in network weights (replaces ε-greedy)
- [ ] **Bootstrapped DQN** - Ensemble for uncertainty estimation
- [ ] **UCB-based exploration** - Upper confidence bound for action selection

### Recurrent Policies (POMDPs)
For partial observability and memory-dependent tasks.
- [ ] **LSTM layer** in deep_rl Model trait
- [ ] **GRU layer** - Simpler recurrent alternative
- [ ] **R2D2** - Recurrent DQN with burn-in and stored hidden states
- [ ] **DRQN** - Deep Recurrent Q-Network (simpler than R2D2)
- [ ] POMDP benchmark environments

### Multi-Agent RL
Cooperative and competitive multi-agent settings.
- [ ] **MADDPG** - Multi-agent DDPG with centralized critic
- [ ] **QMIX** - Value decomposition for cooperative agents
- [ ] **VDN (Value Decomposition Networks)** - Simpler than QMIX
- [ ] **IPPO** - Independent PPO baseline
- [ ] Simple multi-agent environments (predator-prey, cooperative navigation)

### Model-Based Deep RL
Learning and using environment dynamics models.
- [ ] **MBPO (Model-Based Policy Optimization)** - Short rollouts from learned model
- [ ] **World Models** - VAE + MDN-RNN for latent dynamics
- [ ] **Dreamer** - Latent imagination with actor-critic
- [ ] **PETS** - Probabilistic ensemble trajectory sampling
- [ ] Ensemble dynamics models for uncertainty quantification

### Architecture Extensions
New neural network components for deep_rl.
- [ ] **Conv2D layer** - For image-based observations
- [ ] **Attention mechanism** - For complex observations
- [ ] **Transformer blocks** - For sequence modeling
- [ ] MinAtar environments - Simplified Atari for CNN testing

### Infrastructure Improvements
Developer experience and training efficiency.
- [x] **Checkpointing** - Save/load agent state for resuming training
  - `deep_rl/checkpoint/` - Checkpoint utilities module (text-based format)
  - `Network.save_checkpoint(path)` / `load_checkpoint(path)` - Single file for params + optimizer state
  - `Trainer.save_checkpoint(path)` / `load_checkpoint(path)` - Single file for trainer state
  - `DQNAgent.save_checkpoint(path)` / `load_checkpoint(path)` - Single file for full agent (both networks + hyperparams)
  - `examples/checkpoint_demo.mojo` - Usage demo
- [x] **Learning rate schedulers** - Linear decay in Deep PPO (`anneal_lr=True`)
  - [ ] Cosine annealing, warmup (future)
- [ ] **Curriculum learning** - Progressive environment difficulty
- [ ] **Population-based training** - Evolutionary hyperparameter optimization
- [ ] **Imitation learning** - Behavioral cloning, DAgger
- [ ] Improved visualization - TensorBoard-style metric logging

### Quick Wins
Lower-effort additions that extend existing work.
- [ ] **HER (Hindsight Experience Replay)** - Works with existing replay infrastructure
- [ ] **Soft Q-Learning** - Precursor to SAC, maximum entropy with discrete actions
- [ ] **Categorical DQN** - Discrete action + continuous state baseline
- [ ] **n-step DQN** - Multi-step returns for DQN (combine n-step SARSA ideas)

## Algorithm Summary

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| Q-Learning | TD | Off-policy, uses max Q(s',a') |
| SARSA | TD | On-policy, uses Q(s',a') |
| Expected SARSA | TD | Uses E[Q(s',a')], lower variance |
| Double Q-Learning | TD | Two Q-tables, reduces overestimation |
| N-step SARSA | Multi-step | n-step returns, tunable bias-variance |
| SARSA(λ) | Eligibility | Trace decay, efficient credit assignment |
| Monte Carlo | Episode | Complete episode returns |
| Dyna-Q | Model-based | Planning with learned model |
| Priority Sweeping | Model-based | Prioritized planning updates |
| Q-Learning + Replay | Replay | Experience replay buffer |
| Tiled Q-Learning | Function Approx | Tile coding for continuous states |
| Tiled SARSA | Function Approx | On-policy with tile coding |
| Tiled SARSA(λ) | Function Approx | Eligibility traces + tile coding |
| Linear Q-Learning | Function Approx | Arbitrary feature vectors (poly, RBF) |
| Linear SARSA | Function Approx | On-policy with arbitrary features |
| Linear SARSA(λ) | Function Approx | Eligibility traces + arbitrary features |
| REINFORCE | Policy Gradient | Monte Carlo policy gradient |
| REINFORCE + Baseline | Policy Gradient | Variance reduction with value baseline |
| Actor-Critic | Policy Gradient | Online TD-based policy gradient |
| Actor-Critic(λ) | Policy Gradient | Eligibility traces for actor and critic |
| A2C | Policy Gradient | N-step returns + entropy bonus |
| PPO | Policy Gradient | Clipped surrogate objective, stable updates |
| GAE | Advantage Est. | Exponentially-weighted TD residuals |
| DDPG | Continuous Control | Deterministic policy + Q-function critic |
| TD3 | Continuous Control | Twin critics + delayed updates + target smoothing |
| SAC | Continuous Control | Stochastic policy + entropy regularization + auto α |
| Deep DQN | Deep RL ✅ | Neural network Q-function with target network + replay (NEW architecture) |
| Deep Double DQN | Deep RL ✅ | DQN with reduced overestimation (NEW architecture) |
| Deep SAC | Deep RL ✅ | Stochastic policy + entropy maximization (NEW architecture) |
| Deep DDPG | Deep RL ✅ | Deterministic policy + Q-critic + target networks (NEW architecture) |
| Deep TD3 | Deep RL ✅ | Twin critics + delayed updates + target smoothing (NEW architecture) |
| Deep Dueling DQN | Deep RL ✅ | Separate V(s) and A(s,a) streams (NEW architecture) |
| Deep DQN + PER | Deep RL ✅ | Priority sampling by TD error + IS weights (NEW architecture) |
| Deep A2C | Deep RL ✅ | Advantage Actor-Critic with GAE (NEW architecture) |
| Deep PPO | Deep RL ✅ | Clipped surrogate + LR annealing + KL early stop + grad clipping (NEW architecture) |

## Architecture Notes

### New PyTorch-like Deep RL Architecture (`deep_rl/`)

The new architecture follows a **stateless, trait-based design** similar to PyTorch:

```
Model Trait          → Linear, ReLU, Tanh, Sigmoid, Softmax, LayerNorm, Dropout, StochasticActor, Sequential (seq())
Optimizer Trait      → SGD, Adam, RMSprop, AdamW
LossFunction Trait   → MSELoss, HuberLoss, CrossEntropyLoss
Initializer Trait    → Xavier, Kaiming, LeCun, etc.
```

**Key Design Principles:**
1. **Stateless Models** - Models describe computation, not storage. Parameters managed externally.
2. **Zero-Copy Composition** - `LayoutTensor` views enable efficient layer stacking without data copies.
3. **Compile-Time Dimensions** - All sizes known at compile-time for maximum performance.
4. **Pluggable Components** - Mix any Model + Optimizer + Loss + Initializer combination.

**Example Network Construction:**
```mojo
from deep_rl import seq, Linear, ReLU, Adam, Kaiming, Trainer

# Build: obs -> 64 (ReLU) -> 64 (ReLU) -> actions
var model = seq(
    Linear[obs_dim, 64](), ReLU[64](),
    Linear[64, 64](), ReLU[64](),
    Linear[64, action_dim](),
)
```

### Deep RL Agents Reference

| Agent | Location | Notes |
|-------|----------|-------|
| DQN / Double DQN | `deep_agents/dqn.mojo` | Discrete actions, GPU support |
| SAC | `deep_agents/sac.mojo` | Continuous actions, entropy regularization |
| DDPG | `deep_agents/ddpg.mojo` | Continuous actions, deterministic policy |
| TD3 | `deep_agents/td3.mojo` | Continuous actions, twin critics |
| Dueling DQN | `deep_agents/dueling_dqn.mojo` | Discrete actions, V(s) + A(s,a) split |
| DQN + PER | `deep_agents/dqn_per.mojo` | Discrete actions, prioritized replay |
| A2C | `deep_agents/a2c.mojo` | Discrete actions, actor-critic with GAE |
| PPO | `deep_agents/ppo.mojo` | Discrete actions, clipped surrogate, LR annealing, KL early stop, grad clip |
