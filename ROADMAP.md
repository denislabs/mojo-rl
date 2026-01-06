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

### Function Approximation
- [x] Tile Coding - Linear function approximation for continuous state spaces
  - `core/tile_coding.mojo` - TileCoding and TiledWeights structs
  - `agents/tiled_qlearning.mojo` - TiledQLearningAgent, TiledSARSAAgent, TiledSARSALambdaAgent
  - `examples/cartpole_tiled.mojo` - CartPole training with tile coding

## In Progress / Next Steps

### Function Approximation (continued)
- [ ] Linear Q-Learning - Q-Learning with arbitrary feature vectors

### Deep RL (depends on Mojo tensor support)
- [ ] DQN - Deep Q-Network with target network
- [ ] Double DQN - DQN with double Q-learning
- [ ] Dueling DQN - Separate value and advantage streams

### Policy Gradient Methods
- [ ] REINFORCE - Monte Carlo policy gradient
- [ ] Actor-Critic - Combine policy and value learning
- [ ] A2C/A3C - Advantage Actor-Critic

### Infrastructure
- [x] Logging/Metrics - Export learning curves for visualization
- [ ] Hyperparameter Search - Grid search / random search
- [ ] Parallel Training - Multiple environments

### Environments - Native Ports (Next)
- [ ] MountainCar (Native) - Port continuous state space env to pure Mojo
- [ ] LunarLander (Native) - Port Box2D physics to pure Mojo
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
