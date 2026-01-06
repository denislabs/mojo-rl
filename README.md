# mojo-rl

A reinforcement learning framework written in Mojo, featuring trait-based design for extensibility, comprehensive tabular RL algorithms, and policy gradient methods.

## Features

- **Trait-based architecture**: Generic interfaces for environments, agents, states, and actions
- **18 RL algorithms**: TD methods, multi-step, eligibility traces, model-based planning, function approximation, policy gradients
- **6 native environments**: GridWorld, FrozenLake, CliffWalking, Taxi, CartPole, MountainCar
- **Integrated SDL2 rendering**: Native visualization for CartPole and MountainCar
- **20+ Gymnasium wrappers**: Classic Control, Box2D, Toy Text, MuJoCo environments
- **Experience replay**: Uniform and prioritized replay buffers
- **Policy gradient methods**: REINFORCE, Actor-Critic, A2C with tile coding
- **Generic training utilities**: Works with any compatible environment/agent combination

## Quick Start

This project uses **pixi** for dependency management and to ensure the latest Mojo version is used.

### Installing pixi

```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Or with Homebrew
brew install pixi
```

### Install dependencies and run

```bash
# Install all dependencies (Mojo, Python packages, etc.)
pixi install

# Run the main example (Q-Learning on GridWorld)
pixi run mojo run main.mojo

# Run benchmarks comparing algorithms
pixi run mojo run benchmark.mojo

# Build a binary
pixi run mojo build main.mojo

# Run native renderer demo (requires SDL2)
pixi run mojo run examples/native_renderer_demo.mojo
```

### SDL2 Requirements (for visualization)

To use the integrated SDL2 rendering for CartPole and MountainCar:

```bash
# macOS
brew install sdl2 sdl2_ttf

# Ubuntu/Debian
sudo apt-get install libsdl2-dev libsdl2-ttf-dev
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

### Function Approximation (Tile Coding)
| Algorithm | Description |
|-----------|-------------|
| **Tiled Q-Learning** | Q-Learning with tile coding for continuous states |
| **Tiled SARSA** | On-policy SARSA with tile coding |
| **Tiled SARSA(λ)** | Eligibility traces + tile coding |

### Policy Gradient Methods
| Algorithm | Description |
|-----------|-------------|
| **REINFORCE** | Monte Carlo policy gradient with softmax policy |
| **REINFORCE + Baseline** | Variance reduction using learned value baseline |
| **Actor-Critic** | One-step TD policy gradient with online updates |
| **Actor-Critic(λ)** | Actor-Critic with eligibility traces for actor and critic |
| **A2C** | Advantage Actor-Critic with n-step returns and entropy bonus |

## Environments

### Native Mojo Environments
| Environment | States | Actions | Description |
|-------------|--------|---------|-------------|
| **GridWorld** | 25 (5x5) | 4 | Navigate to goal, -1/step, +10 goal |
| **FrozenLake** | 16 (4x4) | 4 | Avoid holes on slippery ice |
| **CliffWalking** | 48 (4x12) | 4 | Avoid cliff, -100 penalty |
| **Taxi** | 500 | 6 | Pickup/dropoff passenger |
| **CartPole** | Continuous | 2 | Balance pole on cart (145x faster than Gymnasium), integrated SDL2 rendering |
| **MountainCar** | Continuous | 3 | Drive car up mountain using momentum, integrated SDL2 rendering |

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
│   ├── space.mojo         # Space abstractions
│   └── tile_coding.mojo   # Tile coding for function approximation
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
│   ├── qlearning_replay.mojo
│   ├── tiled_qlearning.mojo   # Tile coding agents
│   ├── reinforce.mojo         # REINFORCE policy gradient
│   └── actor_critic.mojo      # Actor-Critic family
└── envs/                  # Environment implementations
    ├── gridworld.mojo
    ├── frozenlake.mojo
    ├── cliffwalking.mojo
    ├── taxi.mojo
    ├── cartpole_native.mojo      # Native CartPole with integrated SDL2 rendering
    ├── mountain_car_native.mojo  # Native MountainCar with integrated SDL2 rendering
    ├── native_renderer_base.mojo # SDL2 rendering infrastructure
    └── gymnasium/                # Gymnasium wrappers
        ├── gymnasium_wrapper.mojo
        ├── gymnasium_classic_control.mojo
        ├── gymnasium_box2d.mojo
        ├── gymnasium_toy_text.mojo
        └── gymnasium_mujoco.mojo
```

## Usage Example

```mojo
from agents import QLearningAgent, SARSALambdaAgent, DynaQAgent
from envs import GridWorldEnv, CliffWalkingEnv

fn main():
    # Train Q-Learning on GridWorld
    var env = GridWorldEnv(width=5, height=5)
    var agent = QLearningAgent(num_states=25, num_actions=4)
    _ = agent.train(env, num_episodes=500, verbose=True)
    var reward = agent.evaluate(env, num_episodes=10)
    print("Q-Learning reward:", reward)

    # Train SARSA(λ) on CliffWalking
    var cliff_env = CliffWalkingEnv(width=12, height=4)
    var sarsa_agent = SARSALambdaAgent(
        num_states=48, num_actions=4, lambda_=0.9
    )
    _ = sarsa_agent.train(cliff_env, num_episodes=500)
```

### CartPole with SDL2 Visualization

```mojo
from envs import CartPoleNative, CartPoleAction
from agents import QLearningAgent

fn main() raises:
    var num_bins = 10
    var env = CartPoleEnv(num_bins=num_bins)
    var agent = QLearningAgent(
        num_states=CartPoleEnv.get_num_states(num_bins),
        num_actions=2,
    )

    # Train using generic training function
    _ = agent.train(env, num_episodes=1000, max_steps_per_episode=500)

    # Visualize trained agent
    var state = env.reset()
    for _ in range(500):
        var action_idx = agent.get_best_action(state.index)
        var result = env.step(CartPoleAction(direction=action_idx))
        env.render()  # Opens SDL2 window
        state = result[0]
        if result[2]:
            break
    env.close()
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

### Policy Gradient on CartPole

```mojo
from core.tile_coding import make_cartpole_tile_coding
from agents import REINFORCEAgent, ActorCriticAgent
from envs import CartPoleNative

fn main() raises:
    # Create tile coding for continuous state space
    var tc = make_cartpole_tile_coding(num_tilings=8, tiles_per_dim=8)

    # REINFORCE agent with baseline
    var agent = REINFORCEAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=0.001,
        use_baseline=True,
    )

    var env = CartPoleNative()

    for episode in range(1000):
        var obs = env.reset()
        agent.reset()

        for step in range(500):
            var tiles = tc.get_tiles_simd4(obs)
            var action = agent.select_action(tiles)
            var result = env.step(action)
            agent.store_transition(tiles, action, result[1])
            obs = result[0]
            if result[2]:
                break

        # Update at end of episode
        agent.update_from_episode()
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including:
- Deep RL (DQN, when Mojo tensor ops mature)
- PPO (Proximal Policy Optimization)
- GAE (Generalized Advantage Estimation)
