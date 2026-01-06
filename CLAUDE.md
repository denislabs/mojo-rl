# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mojo-rl is a reinforcement learning framework written in Mojo, featuring trait-based design for extensibility, 20 RL algorithms (including tile-coded function approximation, policy gradient methods, and PPO with GAE), 6 native environments (CartPole, MountainCar, GridWorld, FrozenLake, CliffWalking, Taxi) with integrated SDL2 rendering for continuous-state environments, 20+ Gymnasium wrappers, and experience replay infrastructure.

## Build and Run Commands

This project uses **pixi** for dependency management and to ensure the latest Mojo version is used. All Mojo commands should be run through pixi:

```bash
# Run the main entry point (Q-Learning on GridWorld)
pixi run mojo run main.mojo

# Run benchmarks comparing algorithms
pixi run mojo run benchmark.mojo

# Build a binary
pixi run mojo build main.mojo

# Run native renderer demos (requires SDL2)
pixi run mojo run examples/native_renderer_demo.mojo
pixi run mojo run examples/cartpole_native_demo.mojo
pixi run mojo run examples/mountain_car_tiled.mojo
```

### Installing pixi

If you don't have pixi installed:
```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Or with Homebrew
brew install pixi
```

Then install dependencies:
```bash
pixi install
```

### SDL2 Requirements (for visualization)

```bash
# macOS
brew install sdl2 sdl2_ttf

# Ubuntu/Debian
sudo apt-get install libsdl2-dev libsdl2-ttf-dev
```

## Architecture

The codebase follows a trait-based design for RL components with clear separation of concerns:

### Core Traits (`core/`)

- **`state.mojo`** - `State` trait: Base trait for environment states (requires equality, copyability)
- **`action.mojo`** - `Action` trait: Marker trait for environment actions
- **`env.mojo`** - `Env` trait: Generic environment interface parameterized over StateType and ActionType
  - Methods: `step()`, `reset()`, `get_state()`, `render()`, `close()`
- **`agent.mojo`** - `Agent` trait: Generic agent interface
  - Methods: `select_action()`, `update()`, `reset()`
- **`space.mojo`** - Space abstractions: `DiscreteSpace`, `BoxSpace[dim]` for action/observation spaces
- **`tabular_agent.mojo`** - `TabularAgent` trait: Specialized for discrete state/action spaces
  - Works with state/action indices (Int) for Q-table lookup
  - Methods: `select_action()`, `update()`, `get_best_action()`, `decay_epsilon()`, `get_epsilon()`
- **`replay_buffer.mojo`** - Experience replay infrastructure
  - `Transition`: Struct for (s, a, r, s', done) tuples
  - `ReplayBuffer`: Fixed-size circular buffer
  - `PrioritizedReplayBuffer`: Samples proportional to TD error
- **`tile_coding.mojo`** - Tile coding for function approximation
  - `TileCoding`: Multi-dimensional tile coding with asymmetric offsets
  - `TiledWeights`: Weight storage for linear value function
  - `make_cartpole_tile_coding()`: Factory for CartPole configuration
- **`linear_fa.mojo`** - Linear function approximation with arbitrary features
  - `LinearWeights`: Weight storage for dense feature vectors
  - `PolynomialFeatures`: Polynomial feature extractor (x, y, x², xy, etc.)
  - `RBFFeatures`: Radial Basis Function features
  - `make_grid_rbf_centers()`: Create RBF center grid
- **`sdl2.mojo`** - SDL2 FFI bindings for native rendering
  - `SDL2Renderer`: Low-level SDL2 wrapper with FFI calls
  - Functions: `init()`, `quit()`, `create_window()`, `set_color()`, `fill_rect()`, `draw_line()`, etc.

### Agents (`agents/`)

All agents implement `TabularAgent` trait and use a shared `QTable` structure:

**TD Methods:**
- **`qlearning.mojo`** - `QLearningAgent`: Off-policy TD learning
- **`sarsa.mojo`** - `SARSAAgent`: On-policy TD learning
- **`expected_sarsa.mojo`** - `ExpectedSARSAAgent`: Uses expected Q-value, lower variance
- **`double_qlearning.mojo`** - `DoubleQLearningAgent`: Two Q-tables, reduces overestimation

**Multi-step Methods:**
- **`nstep_sarsa.mojo`** - `NStepSARSAAgent`: Configurable n-step returns
- **`sarsa_lambda.mojo`** - `SARSALambdaAgent`: Eligibility traces with replacing traces
- **`monte_carlo.mojo`** - `MonteCarloAgent`: First-visit Monte Carlo

**Model-based:**
- **`dyna_q.mojo`** - `DynaQAgent`: Q-Learning with model-based planning
- **`priority_sweeping.mojo`** - `PrioritySweepingAgent`: Prioritized updates by TD error

**With Replay:**
- **`qlearning_replay.mojo`** - `QLearningReplayAgent`: Q-Learning with experience replay buffer

**Function Approximation (Tile Coding):**
- **`tiled_qlearning.mojo`** - Tile-coded agents for continuous state spaces:
  - `TiledQLearningAgent`: Q-Learning with tile coding
  - `TiledSARSAAgent`: On-policy SARSA with tile coding
  - `TiledSARSALambdaAgent`: SARSA(λ) with eligibility traces and tile coding

**Function Approximation (Linear with Arbitrary Features):**
- **`linear_qlearning.mojo`** - Linear agents with arbitrary feature vectors:
  - `LinearQLearningAgent`: Q-Learning with polynomial, RBF, or custom features
  - `LinearSARSAAgent`: On-policy SARSA with arbitrary features
  - `LinearSARSALambdaAgent`: SARSA(λ) with eligibility traces and arbitrary features

**Policy Gradient Methods:**
- **`reinforce.mojo`** - Monte Carlo policy gradient agents:
  - `REINFORCEAgent`: Basic REINFORCE with optional baseline
  - `REINFORCEWithEntropyAgent`: REINFORCE with entropy regularization
- **`actor_critic.mojo`** - Actor-Critic family of algorithms:
  - `ActorCriticAgent`: One-step TD Actor-Critic (online updates)
  - `ActorCriticLambdaAgent`: Actor-Critic with eligibility traces
  - `A2CAgent`: Advantage Actor-Critic with n-step returns
- **`ppo.mojo`** - PPO with Generalized Advantage Estimation:
  - `compute_gae()`: Function to compute GAE advantages
  - `PPOAgent`: PPO with clipped surrogate objective
  - `PPOAgentWithMinibatch`: PPO with minibatch sampling for larger rollouts

**Continuous Control (Deterministic Policy Gradient):**
- **`ddpg.mojo`** - Deep Deterministic Policy Gradient with linear function approximation:
  - `DDPGAgent`: Deterministic actor + Q-function critic with target networks
  - Gaussian exploration noise, soft target updates
- **`td3.mojo`** - Twin Delayed DDPG with linear function approximation:
  - `TD3Agent`: Improved DDPG with twin Q-networks
  - Key improvements: min(Q1, Q2) targets, delayed policy updates, target smoothing
- **`sac.mojo`** - Soft Actor-Critic with linear function approximation:
  - `SACAgent`: Maximum entropy RL with stochastic Gaussian policy
  - Key features: entropy bonus, twin critics, automatic α tuning

### Environments (`envs/`)

**Native Mojo Environments:**
- **`gridworld.mojo`** - `GridWorld`: 2D navigation (5x5 default)
  - Reward: -1 per step, +10 for goal
- **`frozenlake.mojo`** - `FrozenLake`: Slippery grid with holes (4x4)
  - Reward: +1 for goal, 0 otherwise (sparse)
- **`cliffwalking.mojo`** - `CliffWalking`: Cliff avoidance (4x12)
  - Reward: -1 per step, -100 for falling off cliff
- **`taxi.mojo`** - `Taxi`: Pickup/dropoff with 500 states
  - Reward: +20 dropoff, -10 illegal action, -1 per step
- **`cartpole_native.mojo`** - `CartPoleNative`: Pure Mojo CartPole (145x faster than Gymnasium)
  - Physics matching Gymnasium CartPole-v1
  - State: SIMD[DType.float64, 4] = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
  - Actions: 0 (left), 1 (right)
  - Integrated SDL2 rendering via `render()` method
- **`mountain_car_native.mojo`** - `MountainCarNative`: Pure Mojo MountainCar
  - Physics matching Gymnasium MountainCar-v0
  - State: SIMD[DType.float64, 2] = [position, velocity]
  - Actions: 0 (left), 1 (no push), 2 (right)
  - `make_mountain_car_tile_coding()`: Factory for tile coding config
  - Integrated SDL2 rendering via `render()` method
- **`native_renderer_base.mojo`** - `NativeRendererBase`: SDL2 rendering infrastructure
  - Used internally by CartPoleNative and MountainCarNative
  - Provides: `init_display()`, `clear()`, `present()`, `set_color()`, `fill_rect()`, `draw_line()`, etc.

**Gymnasium Wrappers (`envs/gymnasium/`):**
- **`gymnasium_wrapper.mojo`** - Generic wrapper for any Gymnasium environment
- **`gymnasium_classic_control.mojo`** - CartPole, MountainCar, Pendulum, Acrobot
- **`gymnasium_box2d.mojo`** - LunarLander, BipedalWalker, CarRacing
- **`gymnasium_toy_text.mojo`** - FrozenLake, Taxi, Blackjack, CliffWalking
- **`gymnasium_mujoco.mojo`** - HalfCheetah, Ant, Humanoid, Walker2d, Hopper, Swimmer, etc.

## Key Design Patterns

### Trait-Based Generics
Environments and agents use compile-time type parameters. Training functions work with ANY DiscreteEnv + TabularAgent combination.

### Discrete State Indexing
Environments convert states to 1D indices for tabular storage:
- GridWorld: `index = y * width + x`
- FrozenLake: `index = position`
- CliffWalking: `index = y * width + x`
- Taxi: `index = ((row * 5 + col) * 5 + passenger_loc) * 4 + destination`
- CartPoleNative: Uses `CartPoleNative.discretize_obs()` to bin continuous state into 10^4 discrete bins (or use trait-based API where state is already discretized)
- MountainCarNative: Uses `discretize_obs_mountain_car()` for 2D state (20^2 = 400 bins default)

### Epsilon-Greedy Exploration
All agents use epsilon-greedy with decay: `ε *= ε_decay` each episode (default decay: 0.995, min: 0.01).

## Adding New Components

### New Environment
1. Define state struct implementing `State` trait
2. Define action struct implementing `Action` trait
3. Implement `DiscreteEnv` trait with `state_to_index()` and `action_from_index()`

### New Agent
1. Implement `TabularAgent` trait
2. Use `QTable` for value storage or define custom structure
3. Implement epsilon-greedy exploration pattern

## Algorithm Parameters

| Algorithm | Key Parameters |
|-----------|---------------|
| Q-Learning | `learning_rate`, `discount_factor`, `epsilon`, `epsilon_decay` |
| N-step SARSA | Above + `n` (number of steps) |
| SARSA(λ) | Above + `lambda_` (trace decay) |
| Dyna-Q | Above + `n_planning` (planning steps per real step) |
| Priority Sweeping | Above + `priority_threshold` |
| Q-Learning + Replay | Above + `buffer_size`, `batch_size`, `min_buffer_size` |
| REINFORCE | `learning_rate`, `discount_factor`, `use_baseline`, `baseline_lr` |
| Actor-Critic | `actor_lr`, `critic_lr`, `discount_factor`, `entropy_coef` |
| Actor-Critic(λ) | Above + `lambda_` (trace decay) |
| A2C | Above + `n_steps` (for n-step returns) |
| PPO | `actor_lr`, `critic_lr`, `clip_epsilon`, `gae_lambda`, `num_epochs`, `entropy_coef` |
| DDPG | `actor_lr`, `critic_lr`, `discount_factor`, `tau`, `noise_std`, `action_scale` |
| TD3 | Above + `policy_delay`, `target_noise_std`, `target_noise_clip` |
| SAC | `actor_lr`, `critic_lr`, `discount_factor`, `tau`, `alpha`, `auto_alpha`, `target_entropy` |
