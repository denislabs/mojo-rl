# mojo-rl

A reinforcement learning framework written in Mojo, featuring trait-based design for extensibility, comprehensive tabular RL algorithms, policy gradient methods, and continuous control.

## Features

- **Trait-based architecture**: Generic interfaces for environments, agents, states, and actions
- **28+ RL algorithms**: TD methods, multi-step, eligibility traces, model-based planning, function approximation, policy gradients, PPO, continuous control (DDPG, TD3, SAC), and deep RL (DQN, Double DQN)
- **Deep RL infrastructure**: Trait-based neural network framework with Model, Optimizer, LossFunction, and Initializer traits; Linear/ReLU/Tanh layers; SGD/Adam optimizers; Sequential composition via `seq()`; GPU kernels with tiled matmul
- **9 native environments**: GridWorld, FrozenLake, CliffWalking, Taxi, CartPole, MountainCar, Acrobot, Pendulum, LunarLander
- **Integrated SDL2 rendering**: Native visualization for continuous-state environments (CartPole, MountainCar, Acrobot, Pendulum, LunarLander)
- **20+ Gymnasium wrappers**: Classic Control, Box2D, Toy Text, MuJoCo environments
- **Experience replay**: Uniform and prioritized replay buffers for both discrete and continuous actions
- **Policy gradient methods**: REINFORCE, Actor-Critic, A2C, PPO with GAE
- **Continuous control**: DDPG, TD3, SAC with linear function approximation
- **Generic training utilities**: Works with any compatible environment/agent combination

## Acknowledgments

This project uses [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) as a reference implementation for environment physics and reward structures. The native Mojo environments (CartPole, MountainCar, Acrobot, Pendulum) are faithful ports of their Gymnasium counterparts, ensuring compatibility and correctness.

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

### GPU Support

GPU-accelerated code requires specifying the target environment with the `-e` flag:

```bash
# Apple Silicon (Metal)
pixi run -e apple mojo run examples/lunar_lander_gpu_dqn.mojo

# NVIDIA GPUs (CUDA)
pixi run -e nvidia mojo run examples/lunar_lander_gpu_dqn.mojo
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

### Function Approximation (Linear with Arbitrary Features)
| Algorithm | Description |
|-----------|-------------|
| **Linear Q-Learning** | Q-Learning with polynomial, RBF, or custom features |
| **Linear SARSA** | On-policy SARSA with arbitrary feature vectors |
| **Linear SARSA(λ)** | Eligibility traces with linear function approximation |

### Policy Gradient Methods
| Algorithm | Description |
|-----------|-------------|
| **REINFORCE** | Monte Carlo policy gradient with softmax policy |
| **REINFORCE + Baseline** | Variance reduction using learned value baseline |
| **Actor-Critic** | One-step TD policy gradient with online updates |
| **Actor-Critic(λ)** | Actor-Critic with eligibility traces for actor and critic |
| **A2C** | Advantage Actor-Critic with n-step returns and entropy bonus |
| **PPO** | Proximal Policy Optimization with clipped surrogate objective |
| **PPO + Minibatch** | PPO with minibatch sampling for larger rollouts |

### Continuous Control (Deterministic Policy Gradient)
| Algorithm | Description |
|-----------|-------------|
| **DDPG** | Deep Deterministic Policy Gradient with linear function approximation |
| **TD3** | Twin Delayed DDPG with twin Q-networks, delayed policy updates, target smoothing |
| **SAC** | Soft Actor-Critic with maximum entropy RL and automatic temperature tuning |

### Deep RL (Neural Networks)
| Algorithm | Description |
|-----------|-------------|
| **Deep DDPG** | DDPG with 2-layer MLP actor/critic networks, Adam optimizer, SIMD-optimized tensor ops |
| **Deep TD3** | TD3 with twin critics, delayed policy updates, target policy smoothing |
| **Deep DQN** | Deep Q-Network with target network, experience replay, epsilon-greedy exploration |
| **Deep Double DQN** | DQN with reduced overestimation: online network selects actions, target evaluates |
| **GPU Deep DQN** | GPU-accelerated DQN with tiled matmul kernels (Metal/CUDA), Double DQN support |

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
| **Acrobot** | Continuous | 3 | Swing two-link pendulum above threshold, integrated SDL2 rendering |
| **Pendulum** | Continuous | Continuous | Swing up and balance inverted pendulum, integrated SDL2 rendering |
| **LunarLander** | Continuous (8D) | 4 (discrete) | Land spacecraft on helipad, custom 2D physics engine, flame particles, SDL2 rendering |

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
│   ├── env_traits.mojo    # Environment trait hierarchy (DiscreteEnv, BoxDiscreteActionEnv, etc.)
│   ├── agent.mojo         # Agent trait
│   ├── tabular_agent.mojo # TabularAgent trait
│   ├── replay_buffer.mojo # Experience replay buffers (discrete actions)
│   ├── continuous_replay_buffer.mojo # Replay buffer for continuous actions
│   ├── space.mojo         # Space abstractions (DiscreteSpace, BoxSpace)
│   ├── tile_coding.mojo   # Tile coding for function approximation
│   ├── linear_fa.mojo     # Linear function approximation (polynomial, RBF features)
│   ├── vec_env.mojo       # Vectorized environment support
│   ├── metrics.mojo       # Training metrics and logging
│   └── sdl2.mojo          # SDL2 FFI bindings for rendering
├── agents/                # Algorithm implementations (tabular & linear)
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
│   ├── linear_qlearning.mojo  # Linear function approximation agents
│   ├── reinforce.mojo         # REINFORCE policy gradient
│   ├── actor_critic.mojo      # Actor-Critic, Actor-Critic(λ), A2C
│   ├── ppo.mojo               # PPO with GAE
│   ├── ddpg.mojo              # Deep Deterministic Policy Gradient (linear)
│   ├── td3.mojo               # Twin Delayed DDPG (linear)
│   └── sac.mojo               # Soft Actor-Critic (linear)
├── deep_rl/               # Deep RL infrastructure (trait-based neural networks)
│   ├── constants.mojo     # Global constants (dtype=float32, TILE=16, TPB=256)
│   ├── model/             # Neural network layers (Model trait)
│   │   ├── model.mojo     # Model trait: stateless forward/backward with LayoutTensor
│   │   ├── linear.mojo    # Linear[in_dim, out_dim]: y = x @ W + b with GPU kernels
│   │   ├── relu.mojo      # ReLU[dim]: y = max(0, x) activation
│   │   ├── tanh.mojo      # Tanh[dim]: y = tanh(x) activation
│   │   └── sequential.mojo # Seq2[L0, L1] + seq() helpers for composing layers
│   ├── optimizer/         # Optimizers (Optimizer trait)
│   │   ├── optimizer.mojo # Optimizer trait: step() with external state
│   │   ├── sgd.mojo       # SGD: param -= lr * grad
│   │   └── adam.mojo      # Adam: adaptive learning with momentum
│   ├── loss/              # Loss functions (LossFunction trait)
│   │   ├── loss.mojo      # LossFunction trait: forward/backward
│   │   └── mse.mojo       # MSELoss: mean squared error
│   ├── initializer/       # Weight initializers (Initializer trait)
│   │   └── initializers.mojo # Xavier, Kaiming, LeCun, Zeros, Ones, Constant, Uniform, Normal
│   ├── training/          # Training utilities
│   │   └── trainer.mojo   # Trainer[MODEL, OPT, LOSS, INIT]: CPU/GPU training loop
│   ├── cpu/               # Legacy CPU implementations (deprecated)
│   └── gpu/               # Legacy GPU implementations (deprecated)
├── deep_agents/           # Deep RL agents (neural networks)
│   ├── cpu/               # CPU-based deep RL agents
│   │   ├── ddpg.mojo      # DeepDDPGAgent with MLP networks
│   │   ├── td3.mojo       # DeepTD3Agent with twin critics
│   │   └── dqn.mojo       # DeepDQNAgent with Double DQN support
│   └── gpu/               # GPU-accelerated deep RL agents
│       └── dqn.mojo       # GPUDeepDQNAgent with Metal/CUDA kernels
├── physics/               # 2D physics engine for LunarLander
│   ├── vec2.mojo          # 2D vector math
│   ├── body.mojo          # Rigid body dynamics
│   ├── shape.mojo         # Polygon, circle, edge shapes
│   ├── fixture.mojo       # Shape-body attachment + collision filtering
│   ├── collision.mojo     # Edge-polygon collision detection
│   ├── joint.mojo         # Revolute joint with motor/limits
│   └── world.mojo         # Physics world simulation
└── envs/                  # Environment implementations
    ├── gridworld.mojo
    ├── frozenlake.mojo
    ├── cliffwalking.mojo
    ├── taxi.mojo
    ├── cartpole.mojo          # Native CartPole with integrated SDL2 rendering
    ├── mountain_car.mojo      # Native MountainCar with integrated SDL2 rendering
    ├── acrobot.mojo           # Native Acrobot with integrated SDL2 rendering
    ├── pendulum.mojo          # Native Pendulum with integrated SDL2 rendering
    ├── lunar_lander.mojo      # Native LunarLander with custom physics + SDL2 rendering
    ├── vec_cartpole.mojo      # Vectorized CartPole for parallel training
    ├── renderer_base.mojo     # SDL2 rendering infrastructure
    └── gymnasium/             # Gymnasium wrappers
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

### Deep Learning with Trainer

```mojo
from deep_rl import (
    Model, seq, Linear, ReLU, Tanh,
    Optimizer, SGD, Adam,
    LossFunction, MSELoss,
    Initializer, Xavier, Kaiming,
    Trainer, TrainResult,
)

fn main() raises:
    # Build a 2-layer MLP: 2 -> 16 (ReLU) -> 1
    var model = seq(
        Linear[2, 16](),
        ReLU[16](),
        Linear[16, 1](),
    )

    # Create trainer with Adam optimizer and MSE loss
    var trainer = Trainer[
        typeof(model),  # Model type
        Adam,           # Optimizer type
        MSELoss,        # Loss function type
        Kaiming,        # Initializer type (good for ReLU)
    ](
        model,
        Adam(lr=0.001),
        MSELoss(),
        Kaiming(),
        epochs=1000,
        print_every=100,
    )

    # Prepare training data (XOR problem)
    var input = InlineArray[Scalar[DType.float32], 8](
        0, 0,  # Sample 1
        0, 1,  # Sample 2
        1, 0,  # Sample 3
        1, 1,  # Sample 4
    )
    var target = InlineArray[Scalar[DType.float32], 4](
        0,  # 0 XOR 0 = 0
        1,  # 0 XOR 1 = 1
        1,  # 1 XOR 0 = 1
        0,  # 1 XOR 1 = 0
    )

    # Train on CPU
    var result = trainer.train[4](input, target)
    print("Final loss:", result.final_loss)

    # Or train on GPU (Metal/CUDA)
    from gpu.host import DeviceContext
    var ctx = DeviceContext()
    var gpu_result = trainer.train_gpu[4](ctx, input, target)
```

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
| Deep DDPG | `actor_lr`, `critic_lr`, `gamma`, `tau`, `noise_std`, `noise_decay`, `action_scale`, `hidden_dim`, `batch_size` |
| Deep TD3 | Above + `policy_delay`, `target_noise_std`, `target_noise_clip` |
| Deep DQN | `lr`, `gamma`, `tau`, `epsilon`, `epsilon_min`, `epsilon_decay`, `hidden_dim`, `batch_size`, `buffer_capacity` |
| Deep Double DQN | Same as Deep DQN (enabled by default via `double_dqn=True` compile-time parameter) |
| GPU Deep DQN | Same as Deep DQN, GPU-accelerated forward pass with tiled matmul kernels |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including:
- Deep RL algorithms (Dueling DQN, Deep SAC)
- Prioritized Experience Replay
- ~~GPU support for deep RL~~ ✅ GPU DQN now available (Metal/CUDA)
- More native environment ports (BipedalWalker)
