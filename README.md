# mojo-rl

A reinforcement learning framework written in Mojo, featuring trait-based design for extensibility and comprehensive tabular RL algorithms.

## Features

- **Trait-based architecture**: Generic interfaces for environments, agents, states, and actions
- **10 RL algorithms**: TD methods, multi-step, eligibility traces, model-based planning
- **5 native environments**: GridWorld, FrozenLake, CliffWalking, Taxi, CartPole (145x faster than Python)
- **20+ Gymnasium wrappers**: Classic Control, Box2D, Toy Text, MuJoCo environments
- **Experience replay**: Uniform and prioritized replay buffers
- **Generic training utilities**: Works with any compatible environment/agent combination

## Quick Start

```bash
# Run the main example (Q-Learning on GridWorld)
mojo run main.mojo

# Run benchmarks comparing algorithms
mojo run benchmark.mojo

# Build a binary
mojo build main.mojo
```

## Algorithms

### TD Methods
| Algorithm | Description |
|-----------|-------------|
| **Q-Learning** | Off-policy TD learning: `Q(s,a) += α(r + γ·max Q(s',a') - Q(s,a))` |
| **SARSA** | On-policy TD learning: `Q(s,a) += α(r + γ·Q(s',a') - Q(s,a))` |
| **Expected SARSA** | Uses expected value: `Q(s,a) += α(r + γ·E[Q(s',a')] - Q(s,a))` |
| **Double Q-Learning** | Two Q-tables to reduce overestimation bias |

### Multi-step & Eligibility Traces
| Algorithm | Description |
|-----------|-------------|
| **N-step SARSA** | n-step returns, configurable bias-variance tradeoff |
| **SARSA(λ)** | Eligibility traces for efficient credit assignment |
| **Monte Carlo** | First-visit MC with complete episode returns |

### Model-based Planning
| Algorithm | Description |
|-----------|-------------|
| **Dyna-Q** | Q-Learning + simulated experience from learned model |
| **Priority Sweeping** | Prioritized updates by TD error magnitude |

### With Experience Replay
| Algorithm | Description |
|-----------|-------------|
| **Q-Learning + Replay** | Off-policy learning with replay buffer |

## Environments

### Native Mojo Environments
| Environment | States | Actions | Description |
|-------------|--------|---------|-------------|
| **GridWorld** | 25 (5x5) | 4 | Navigate to goal, -1/step, +10 goal |
| **FrozenLake** | 16 (4x4) | 4 | Avoid holes on slippery ice |
| **CliffWalking** | 48 (4x12) | 4 | Avoid cliff, -100 penalty |
| **Taxi** | 500 | 6 | Pickup/dropoff passenger |
| **CartPole** | Continuous | 2 | Balance pole on cart (145x faster than Gymnasium) |

### Gymnasium Wrappers (`envs/gymnasium/`)
Wrap any Gymnasium environment with Python interop:
- **Classic Control**: CartPole, MountainCar, Pendulum, Acrobot
- **Box2D**: LunarLander, BipedalWalker, CarRacing
- **Toy Text**: FrozenLake, Taxi, Blackjack, CliffWalking
- **MuJoCo**: HalfCheetah, Ant, Humanoid, Walker2d, Hopper, Swimmer, and more

## Project Structure

```
mojo-rl/
├── main.mojo              # Entry point (Q-Learning on GridWorld)
├── benchmark.mojo         # Algorithm comparison
├── core/                  # Core RL abstractions
│   ├── state.mojo         # State trait
│   ├── action.mojo        # Action trait
│   ├── env.mojo           # Environment trait
│   ├── agent.mojo         # Agent trait
│   ├── tabular_agent.mojo # TabularAgent trait
│   ├── training.mojo      # Training/evaluation functions
│   ├── replay_buffer.mojo # Experience replay buffers
│   └── space.mojo         # Space abstractions
├── agents/                # Algorithm implementations
│   ├── qlearning.mojo
│   ├── sarsa.mojo
│   ├── expected_sarsa.mojo
│   ├── nstep_sarsa.mojo
│   ├── sarsa_lambda.mojo
│   ├── monte_carlo.mojo
│   ├── double_qlearning.mojo
│   ├── dyna_q.mojo
│   ├── priority_sweeping.mojo
│   └── qlearning_replay.mojo
└── envs/                  # Environment implementations
    ├── gridworld.mojo
    ├── frozenlake.mojo
    ├── cliffwalking.mojo
    ├── taxi.mojo
    ├── cartpole_native.mojo   # Native CartPole (145x faster)
    ├── cartpole_renderer.mojo # Pygame visualization
    └── gymnasium/             # Gymnasium wrappers
        ├── gymnasium_wrapper.mojo
        ├── gymnasium_classic_control.mojo
        ├── gymnasium_box2d.mojo
        ├── gymnasium_toy_text.mojo
        └── gymnasium_mujoco.mojo
```

## Usage Example

```mojo
from core import train_tabular, evaluate_tabular
from agents import QLearningAgent, SARSALambdaAgent, DynaQAgent
from envs import GridWorld, CliffWalking

fn main():
    # Train Q-Learning on GridWorld
    var env = GridWorld(width=5, height=5)
    var agent = QLearningAgent(num_states=25, num_actions=4)
    _ = train_tabular(env, agent, num_episodes=500, verbose=True)
    var reward = evaluate_tabular(env, agent, num_episodes=10)
    print("Q-Learning reward:", reward)

    # Train SARSA(λ) on CliffWalking
    var cliff_env = CliffWalking(width=12, height=4)
    var sarsa_agent = SARSALambdaAgent(
        num_states=48, num_actions=4, lambda_=0.9
    )
    _ = train_tabular(cliff_env, sarsa_agent, num_episodes=500)
```

## Extending the Framework

### Adding a New Environment
```mojo
struct MyEnv(DiscreteEnv):
    comptime StateType = MyState
    comptime ActionType = MyAction

    fn step(mut self, action: MyAction) -> Tuple[MyState, Float64, Bool]: ...
    fn reset(mut self) -> MyState: ...
    fn state_to_index(self, state: MyState) -> Int: ...
    fn action_from_index(self, idx: Int) -> MyAction: ...
```

### Adding a New Agent
```mojo
struct MyAgent(TabularAgent):
    fn select_action(self, state_idx: Int) -> Int: ...
    fn update(mut self, state: Int, action: Int, reward: Float64,
              next_state: Int, done: Bool): ...
    fn get_best_action(self, state_idx: Int) -> Int: ...
    fn decay_epsilon(mut self): ...
    fn get_epsilon(self) -> Float64: ...
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including:
- Function approximation (tile coding)
- Deep RL (DQN, when Mojo tensor ops mature)
- Policy gradient methods (REINFORCE, Actor-Critic)
