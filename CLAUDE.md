# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mojo-rl is a reinforcement learning framework written in Mojo, featuring trait-based design for extensibility, 10 tabular RL algorithms, 5 native environments (including CartPole with 145x speedup over Python), 20+ Gymnasium wrappers, and experience replay infrastructure.

## Build and Run Commands

```bash
# Run the main entry point (Q-Learning on GridWorld)
mojo run main.mojo

# Run benchmarks comparing algorithms
mojo run benchmark.mojo

# Build a binary
mojo build main.mojo
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
- **`training.mojo`** - `DiscreteEnv` trait + generic training functions
  - `train_tabular()`: Generic training loop for any DiscreteEnv + TabularAgent
  - `evaluate_tabular()`: Evaluation with greedy policy
- **`replay_buffer.mojo`** - Experience replay infrastructure
  - `Transition`: Struct for (s, a, r, s', done) tuples
  - `ReplayBuffer`: Fixed-size circular buffer
  - `PrioritizedReplayBuffer`: Samples proportional to TD error

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
  - State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
  - Actions: 0 (left), 1 (right)
- **`cartpole_renderer.mojo`** - `CartPoleRenderer`: Pygame visualization for CartPoleNative

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
- CartPoleNative: Uses `discretize_obs_native()` to bin continuous state into 10^4 discrete bins

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
