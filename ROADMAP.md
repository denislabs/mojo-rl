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

### Deep RL Infrastructure (`deep_rl/`) - CPU with SIMD
- [x] Tensor operations - SIMD-optimized tensor ops with compile-time dimensions
  - `deep_rl/tensor.mojo` - matmul, transpose, relu, tanh, sigmoid, gradients
  - FMA (fused multiply-add) and register blocking for performance
  - Xavier initialization, element-wise ops, reductions
  - **Note**: Currently CPU-only; GPU support planned (see In Progress section)
- [x] Linear layer - Fully connected layer with forward/backward
  - `deep_rl/linear.mojo` - Linear layer with compile-time dimensions
  - Gradient computation for backpropagation
- [x] Adam optimizer - Adaptive moment estimation optimizer
  - `deep_rl/adam.mojo` - AdamState, LinearAdam with SIMD optimization
  - Soft update support for target networks
- [x] MLP networks - Multi-layer perceptrons
  - `deep_rl/mlp.mojo` - MLP2 (2-layer), MLP3 (3-layer) networks
  - Forward/backward with activation caching
- [x] Actor-Critic networks - Networks for continuous control
  - `deep_rl/actor_critic.mojo` - Actor and Critic with ReLU hidden, tanh output
  - Built-in Adam optimizer and soft update methods
- [x] Replay buffer - Experience replay for deep RL
  - `deep_rl/replay_buffer.mojo` - Fixed-capacity buffer with compile-time dimensions

### Deep RL Agents (`deep_agents/`)
- [x] Deep DDPG - DDPG with neural network function approximation
  - `deep_agents/ddpg.mojo` - DeepDDPGAgent
  - 2-layer MLP actor (obs -> 256 -> 256 -> action with tanh)
  - 2-layer MLP critic (obs+action -> 256 -> 256 -> Q-value)
  - Target networks with soft updates
  - Gaussian exploration noise with decay
  - `train()` and `evaluate()` methods for BoxContinuousActionEnv
- [x] Deep TD3 - TD3 with neural network function approximation
  - `deep_agents/td3.mojo` - DeepTD3Agent
  - Twin critics (Q1, Q2) to reduce overestimation bias
  - Delayed policy updates (actor updates every N critic updates)
  - Target policy smoothing (clipped noise on target actions)
  - All three key TD3 improvements over DDPG

## In Progress / Next Steps

### Deep RL Agents (Neural Networks)
- [ ] Deep DQN - Deep Q-Network with target network
- [ ] Deep Double DQN - DQN with double Q-learning
- [ ] Deep Dueling DQN - Separate value and advantage streams
- [ ] Deep SAC - Soft Actor-Critic with neural networks (stochastic policy + entropy)

### GPU Support
- [ ] GPU tensor operations - Port tensor.mojo ops to GPU (when Mojo GPU support matures)
- [ ] GPU-accelerated training - Batch processing on GPU for deep RL agents
- [ ] Mixed CPU/GPU pipeline - Environment stepping on CPU, network forward/backward on GPU

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
- [ ] LunarLander (Native) - Port Box2D physics to pure Mojo
- [x] Pendulum (Native) - Port continuous action environment (COMPLETED - see above)
- [ ] Custom environment builder

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
| Deep DDPG | Deep RL | Neural network actor/critic with Adam optimizer |
| Deep TD3 | Deep RL | Twin critics + delayed updates + target smoothing (neural networks) |
