# mojo-rl

A reinforcement learning framework written in Mojo, featuring trait-based design, 40+ RL algorithms, GPU-accelerated deep RL, custom 2D/3D physics engines, native arcade game engines, and SDL3 rendering.

## Features

- **Trait-based architecture**: Generic interfaces for environments, agents, states, actions, models, optimizers, and physics
- **40+ RL algorithms**: TD methods, multi-step, eligibility traces, model-based planning, function approximation, policy gradients, PPO, continuous control (DDPG, TD3, SAC), deep RL (DQN family including Noisy DQN, C51, Rainbow; A2C, PPO), and model-based RL (TD-MPC2, DreamerV3, MuZero)
- **Deep learning framework** (`mojo_rl/nn/`): Trait-based neural networks with autodiff, 20+ layer types, 5 optimizers (SGD, Adam, AdamW, RMSprop, Muon), automatic compile-time fusion, CPU/GPU support
- **Autodiff system** (`mojo_rl/nn/autodiff/`): Composition-based automatic differentiation with 27+ DiffOp primitives, AutoDiffChain, fused kernels, 7 combinators (Residual, Parallel, Repeat, SkipConcat, DualPath, SplitApply, FanOut), ComputeGraph named-node DAG builder
- **3D physics engine** (`mojo_rl/physics3d/`): MuJoCo-inspired generalized coordinates engine with CRBA, RNE, constraint solvers (PGS, Newton, CG), collision detection, MJCF XML parsing, CPU/GPU support
- **2D physics engine** (`mojo_rl/physics2d/`): GPU-accelerated batched physics for LunarLander, BipedalWalker, CarRacing with impulse solving and tire friction
- **25 native environments**: Tabular, classic control, 2D physics, MuJoCo-style 3D, and GPU-accelerated arcade games
- **Arcade game engines** (`mojo_rl/envs/arcade_games/`): Native GPU-accelerated Pong, Breakout, Space Invaders with clean obs + pixel obs modes
- **Atari 2600 emulator** (`mojo_rl/envs/atari/`): Full 6502 CPU, TIA, RIOT emulation for ROM-based training
- **SDL3 rendering** (`mojo_rl/render/`): 2D CPU rasterizer + GPU-accelerated 3D renderer with Blinn-Phong lighting, shadows, skybox, interactive camera, video recording
- **20+ Gymnasium wrappers**: Classic Control, Box2D, Toy Text, MuJoCo environments
- **GPU training**: All deep agents (DQN, C51, Rainbow, DDPG, TD3, SAC, PPO, TD-MPC2, DreamerV3, MuZero) support GPU-accelerated training

## Acknowledgments

This project uses [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) as a reference for environment physics. Native Mojo environments are faithful ports ensuring compatibility. MuJoCo-style environments reference [MuJoCo](https://mujoco.org/) for physics and model definitions.

## Quick Start

This project uses **pixi** for dependency management.

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

# Run an example (note: -I . is required for module resolution)
pixi run mojo run -I . examples/solve_gridworld.mojo

# Run benchmarks
pixi run mojo run -I . benchmarks/benchmark_matmul_apple.mojo
```

### GPU Support

GPU-accelerated code requires specifying the target environment with the `-e` flag:

```bash
# Apple Silicon (Metal)
pixi run -e apple mojo run -I . examples/lunar_lander_dqn.mojo

# NVIDIA GPUs (CUDA)
pixi run -e nvidia mojo run -I . examples/lunar_lander_dqn.mojo
```

## Project Structure

```
mojo-rl/
├── mojo_rl/                     # Main Mojo package
│   ├── core/                    #   Core RL abstractions (traits, replay buffers, tile coding)
│   ├── agents/                  #   Tabular & linear RL algorithms (20+ agents)
│   ├── deep_agents/             #   Deep RL agents (config-driven generic architecture)
│   │   ├── core/                #     Shared infrastructure
│   │   │   ├── agents/          #       Generic agents (DQN, C51, Rainbow, off/on-policy)
│   │   │   ├── configs/         #       Algorithm configs (DQN/TD3/SAC/PPO/A2C variants)
│   │   │   ├── strategies/      #       Composable building blocks (exploration, target, loss)
│   │   │   ├── training/        #       CPU/GPU training loops
│   │   │   └── replay/          #       Replay buffers (heap, PER, GPU, N-step, sequence)
│   │   ├── dreamer_v3/          #     DreamerV3 (RSSM world model, imagination)
│   │   ├── tdmpc2/              #     TD-MPC2 (world model + MPPI planning)
│   │   └── muzero/              #     MuZero (learned model + MCTS planning)
│   ├── nn/                      #   Deep learning framework
│   │   ├── model/               #     20+ layers: Linear, Conv2D, NoisyLinear, LayerNorm, etc.
│   │   ├── optimizer/           #     SGD, Adam, AdamW, RMSprop, Muon
│   │   ├── loss/                #     MSE, Huber, CrossEntropy, SoftCrossEntropy, TwoHot
│   │   ├── initializer/         #     Xavier, Kaiming, LeCun, etc.
│   │   ├── training/            #     Trainer, NetworkState, GPUNetworkState, NetworkPair
│   │   ├── checkpoint/          #     Model serialization (text + binary)
│   │   ├── autodiff/            #     Automatic differentiation framework
│   │   │   ├── primitives/      #       27+ DiffOps (MatMul, Conv2D, Attention, RSample, etc.)
│   │   │   ├── fused/           #       Fused kernels (MatMul+Bias+Act, Conv2D+Act)
│   │   │   ├── combinators/     #       Residual, Parallel, Repeat, SkipConcat, DualPath, etc.
│   │   │   └── compute_graph.mojo #    Named-node DAG builder (ComputeGraph)
│   │   ├── composites.mojo      #     Pre-built architectures (ResBlock, ResNet, LeNet, NatureDQN)
│   │   └── gpu/                 #     GPU kernels (matmul, elementwise, random)
│   ├── physics3d/               #   3D MuJoCo-inspired physics engine
│   │   ├── model/               #     Compile-time model specs (BodySpec, JointSpec, GeomSpec)
│   │   ├── dynamics/            #     Mass matrix (CRBA), bias forces (RNE), Jacobians
│   │   ├── integrator/          #     Euler, ImplicitFast, Implicit, RK4
│   │   ├── solver/              #     PGS, Newton, CG, Island-based solvers
│   │   ├── collision/           #     Narrow-phase + Sweep-and-Prune broadphase
│   │   ├── constraints/         #     Constraint building + solving
│   │   ├── kinematics/          #     Forward kinematics + quaternion math
│   │   └── parser/              #     MJCF XML model loading
│   ├── physics2d/               #   GPU-accelerated 2D physics engine
│   │   ├── integrators/         #     Semi-implicit Euler
│   │   ├── collision/           #     Flat/edge terrain detection
│   │   ├── solvers/             #     Impulse + unified constraint solver
│   │   ├── joints/              #     Revolute joint solver
│   │   ├── articulated/         #     Multi-body chain support
│   │   ├── car/                 #     CarRacing slip-based tire physics
│   │   └── lidar/               #     Distance sensing
│   ├── math3d/                  #   3D math library (Vec3, Quat, Mat3, Mat4)
│   ├── render/                  #   SDL3 rendering infrastructure
│   │   ├── renderer2d.mojo      #     2D CPU rasterizer
│   │   ├── renderer3d.mojo      #     GPU-accelerated 3D renderer (Metal shaders)
│   │   ├── gpu_shaders.mojo     #     MSL shaders (solid, shadow, skybox, text)
│   │   ├── video_recorder.mojo  #     MP4/GIF recording
│   │   └── sdl/                 #     SDL3 FFI bindings (38 files)
│   └── envs/                    #   Environment implementations
│       ├── gridworld.mojo       #     Tabular environments
│       ├── cartpole.mojo        #     Classic control (GPU-capable)
│       ├── lunar_lander/        #     Custom 2D physics (GPU batch)
│       ├── bipedal_walker/      #     Custom 2D physics (GPU batch)
│       ├── car_racing/          #     Tire slip physics (GPU batch)
│       ├── half_cheetah/        #     MuJoCo-style (physics3d)
│       ├── hopper/              #     MuJoCo-style (physics3d)
│       ├── ant/                 #     MuJoCo-style (physics3d)
│       ├── walker2d/            #     MuJoCo-style (physics3d)
│       ├── humanoid/            #     MuJoCo-style (physics3d)
│       ├── arcade_games/        #     Native GPU game engines (Pong, Breakout, Space Invaders)
│       ├── atari/               #     Atari 2600 emulator (6502 CPU, TIA, RIOT)
│       └── gymnasium/           #     Python Gymnasium wrappers
├── tests/                       # Test suite (166+ files)
│   ├── physics3d/               #   Physics engine validation tests (73 files)
│   ├── nn/                      #   Neural network + autodiff tests (20 files)
│   ├── deep_agents/             #   Deep RL agent tests (39 files)
│   └── arcade_games/            #   Arcade/Atari environment tests (6 files)
├── examples/                    # Demo scripts organized by environment
│   ├── cartpole/                #   CartPole demos and benchmarks
│   ├── half_cheetah/            #   HalfCheetah training (PPO, SAC, TD3, TD-MPC2)
│   ├── hopper/                  #   Hopper training (PPO)
│   ├── ant/                     #   Ant training (PPO)
│   ├── acrobot/                 #   Acrobot demos
│   ├── arcade_games/            #   Pong/Breakout/SpaceInvaders (DQN, PPO, playable)
│   └── *.mojo                   #   Various environment demos
├── benchmarks/                  # Performance benchmarks
└── pixi.toml                    # Dependency management
```

## Algorithms

### Tabular & Linear Methods

| Category | Algorithms |
|----------|-----------|
| **TD Methods** | Q-Learning, SARSA, Expected SARSA, Double Q-Learning |
| **Multi-step** | N-step SARSA, SARSA(lambda), Monte Carlo |
| **Model-based** | Dyna-Q, Priority Sweeping |
| **With Replay** | Q-Learning + Replay, Q-Learning + PER |
| **Tile Coding** | Tiled Q-Learning, Tiled SARSA, Tiled SARSA(lambda) |
| **Linear FA** | Linear Q-Learning, Linear SARSA, Linear SARSA(lambda) |
| **Policy Gradient** | REINFORCE, Actor-Critic, Actor-Critic(lambda), A2C, PPO |
| **Continuous (Linear)** | DDPG, TD3, SAC |

### Deep RL (Neural Networks)

| Algorithm | Actions | GPU | Description |
|-----------|---------|-----|-------------|
| **DQN** | Discrete | Yes | Double DQN, target network, epsilon-greedy |
| **DQN + PER** | Discrete | Yes | Prioritized replay with sum-tree |
| **Dueling DQN** | Discrete | Yes | V(s) + A(s,a) architecture |
| **Noisy DQN** | Discrete | Yes | NoisyLinear layers, no epsilon-greedy |
| **DQN CNN** | Discrete | Yes | NatureDQN CNN for pixel observations |
| **C51** | Discrete | Yes | Categorical distributional (51 atoms) |
| **Rainbow** | Discrete | Yes | C51 + Double + PER + Dueling + Noisy + N-step |
| **DDPG** | Continuous | Yes | Deterministic actor, Gaussian noise |
| **TD3** | Continuous | Yes | Twin critics, delayed policy, target smoothing |
| **SAC** | Continuous | Yes | Max entropy, stochastic policy, auto alpha |
| **A2C** | Discrete | CPU | GAE, softmax policy |
| **PPO** | Both | Yes | Clipped surrogate, GAE, multi-epoch, CNN variant |
| **TD-MPC2** | Continuous | Yes | World model, MPPI planning, distributional RL |
| **DreamerV3** | Continuous | Yes | RSSM world model, imagination rollouts |
| **MuZero** | Discrete | Yes | Learned model, MCTS planning, distributional |

## Environments

### Native Mojo Environments

| Environment | Obs Dim | Actions | Physics Engine | GPU Batch |
|-------------|---------|---------|----------------|-----------|
| GridWorld | 25 | 4 (discrete) | Grid | No |
| FrozenLake | 16 | 4 (discrete) | Grid | No |
| CliffWalking | 48 | 4 (discrete) | Grid | No |
| Taxi | 500 | 6 (discrete) | Grid | No |
| CartPole | 4 | 2 (discrete) | Gymnasium-matching | Yes |
| MountainCar | 2 | 3 (discrete) | Gymnasium-matching | No |
| Acrobot | 6 | 3 (discrete) | RK4 | No |
| Pendulum | 3 | 1 (continuous) | Direct | Yes |
| LunarLander | 8 | 4 / continuous | physics2d (impulse) | Yes |
| BipedalWalker | 24 | 4 (continuous) | physics2d (impulse + joints) | Yes |
| CarRacing | 12 | 3 (continuous) | physics2d (tire slip) | Yes |
| HalfCheetah | 17 | 6 (continuous) | physics3d (GC) | Yes |
| Hopper | 11 | 3 (continuous) | physics3d (GC) | Yes |
| Ant | 27 | 8 (continuous) | physics3d (GC) | Yes |
| Walker2d | 17 | 6 (continuous) | physics3d (GC) | Yes |
| Swimmer | 8 | 2 (continuous) | physics3d (GC) | Yes |
| Humanoid | 376 | 17 (continuous) | physics3d (GC) | Yes |
| HumanoidStandup | 376 | 17 (continuous) | physics3d (GC) | Yes |
| InvertedPendulum | 4 | 1 (continuous) | physics3d (GC) | Yes |
| InvDoublePendulum | 9 | 1 (continuous) | physics3d (GC) | Yes |
| Pong | 6 / 4x84x84 | 3 (discrete) | Native GPU engine | Yes |
| Breakout | 7 / 4x84x84 | 4 (discrete) | Native GPU engine | Yes |
| Space Invaders | 10 / 4x84x84 | 4 (discrete) | Native GPU engine | Yes |

### Atari 2600 Emulator

Full 6502 CPU + TIA + RIOT emulation. CPU-only (Pong, Breakout, Space Invaders ROMs). Used for validation against native GPU engines.

### Gymnasium Wrappers

- **Classic Control**: CartPole, MountainCar, Pendulum, Acrobot
- **Box2D**: LunarLander, BipedalWalker, CarRacing
- **Toy Text**: FrozenLake, Taxi, Blackjack, CliffWalking
- **MuJoCo**: HalfCheetah, Ant, Humanoid, Walker2d, Hopper, Swimmer, and more

## Usage Examples

### Tabular RL

```mojo
from mojo_rl.agents import QLearningAgent
from mojo_rl.envs import GridWorldEnv

fn main():
    var env = GridWorldEnv(width=5, height=5)
    var agent = QLearningAgent(num_states=25, num_actions=4)
    _ = agent.train(env, num_episodes=500, verbose=True)
```

### Deep RL with GPU Training

```mojo
from mojo_rl.deep_agents import DeepPPOContinuousAgent
from mojo_rl.envs.half_cheetah import HalfCheetahEnv
from gpu.host import DeviceContext

fn main() raises:
    var agent = DeepPPOContinuousAgent[
        obs_dim=17, action_dim=6, hidden_dim=64,
    ]()
    var ctx = DeviceContext()
    var metrics = agent.train_gpu[HalfCheetahEnv](ctx, num_updates=1000)
```

### Neural Network Training

```mojo
from mojo_rl.nn import Sequential, Linear, ReLU, Adam, MSELoss, Kaiming, Trainer

fn main() raises:
    # Define model at compile time: 2 -> 16 (ReLU) -> 1
    comptime MLP = Sequential[Linear[2, 16], ReLU[16], Linear[16, 1]]

    var trainer = Trainer[MLP, Adam, MSELoss, Kaiming](
        MLP(), Adam(lr=0.001), MSELoss(), Kaiming(), epochs=1000,
    )
    var result = trainer.train[4](input, target)
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
