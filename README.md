# mojo-rl

An educational reinforcement learning framework written in Mojo, featuring trait-based design, 40+ RL algorithms, GPU-accelerated deep RL, custom 2D/3D physics engines, native arcade game engines, and SDL3 rendering.

> **Note:** This is a personal learning project, not a production-grade library. While the tabular agents and core deep RL algorithms (DQN, PPO, SAC, TD3) are well-tested, the 3D physics engine, complex deep agents (DreamerV3, TD-MPC2, MuZero), and some advanced features are still experimental and may contain bugs. Contributions and bug reports are welcome!

## Features

- **Trait-based architecture**: Generic interfaces for environments, agents, states, actions, models, optimizers, and physics
- **40+ RL algorithms**: TD methods, multi-step, eligibility traces, model-based planning, function approximation, policy gradients, PPO, continuous control (DDPG, TD3, SAC, REDQ), deep RL (DQN family including Noisy DQN, C51, Rainbow; A2C, PPO), and model-based RL (MBPO, TD-MPC2, DreamerV3, MuZero)
- **Deep learning framework** (`mojo_rl/nn/`): Module/Param neural networks with autodiff (each `Param` owns `val`+`grad` tensors), 20+ primitive layer types, SGD/Adam/AdamW optimizers, automatic compile-time fusion (MatMul+Bias+Act, Conv2D+Act), checkpoint v2, CPU/GPU support
- **Composable models** (`mojo_rl/nn/`): Sequential/Residual/Parallel/Repeat combinators, pre-built architectures (ResNet, GPT, ViT, LSTM), and a ComputeGraph named-node DAG builder for complex loss graphs
- **3D physics engine** (`mojo_rl/physics3d/`): MuJoCo-inspired generalized coordinates engine with CRBA, RNE, constraint solvers (PGS, Newton, CG), collision detection, MJCF XML parsing, CPU/GPU support
- **2D physics engine** (`mojo_rl/physics2d/`): GPU-accelerated batched physics for LunarLander, BipedalWalker, CarRacing with impulse solving and tire friction
- **25 native environments**: Tabular, classic control, 2D physics, MuJoCo-style 3D, and GPU-accelerated arcade games
- **Arcade game engines** (`mojo_rl/envs/arcade_games/`): Native GPU-accelerated Pong, Breakout, Space Invaders with clean obs + pixel obs modes
- **Atari 2600 emulator** (`mojo_rl/envs/atari/`): Full 6502 CPU, TIA, RIOT emulation for ROM-based training
- **SDL3 rendering** (`mojo_rl/render/`): 2D CPU rasterizer + GPU-accelerated 3D renderer with Blinn-Phong lighting, shadows, skybox, interactive camera, video recording
- **20+ Gymnasium wrappers**: Classic Control, Box2D, Toy Text, MuJoCo environments
- **GPU training**: All deep agents (DQN, C51, Rainbow, DDPG, TD3, SAC, REDQ, PPO, MBPO, TD-MPC2, DreamerV3, MuZero) support GPU-accelerated training

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

### Installing SDL3 (optional, for rendering)

SDL3 is required for environment visualization and video recording. It is **not** needed for training.

```bash
# macOS (Homebrew)
brew install sdl3

# Ubuntu/Debian
sudo apt install libsdl3-dev

# Fedora
sudo dnf install SDL3-devel

# From source (all platforms): https://github.com/libsdl-org/SDL/releases
```

### Install dependencies and run

```bash
# Install all dependencies (Mojo, Python packages, etc.)
pixi install

# Run an example (note: -I . is required for module resolution)
pixi run mojo run -I . examples/solve_gridworld.mojo

# Run a test
pixi run mojo run -I . tests/physics3d/test_half_cheetah_match.mojo
```

### GPU Support

GPU-accelerated code requires specifying the target environment with the `-e` flag:

```bash
# Apple Silicon (Metal)
pixi run -e apple mojo run -I . examples/half_cheetah/ppo_half_cheetah_training_gpu.mojo

# NVIDIA GPUs (CUDA)
pixi run -e nvidia mojo run -I . examples/half_cheetah/ppo_half_cheetah_training_gpu.mojo
```

## Project Structure

```
mojo-rl/
├── mojo_rl/                     # Main Mojo package
│   ├── core/                    #   Core RL abstractions (traits, replay buffers, tile coding)
│   ├── agents/                  #   Tabular & linear RL algorithms (20+ agents)
│   ├── deep_agents/             #   Deep RL agents (per-algorithm facade packages)
│   │   ├── dqn/ c51/            #     Value-based (DQN, target net; C51/Rainbow distributional)
│   │   ├── ddpg/ td3/ sac/      #     Off-policy continuous (twin critics, max-entropy)
│   │   ├── redq/ mbpo/          #     Ensemble / model-based continuous
│   │   ├── ppo/ ppo_discrete/ a2c/ #  On-policy (clipped surrogate, GAE)
│   │   ├── tdmpc2/              #     TD-MPC2 (world model + MPPI planning)
│   │   ├── dreamerv3/ dreamer4/ #     Latent world models (RSSM / transformer)
│   │   ├── alphazero/ muzero/ efficient_zero_v2/ zero/ #  Zero-series (MCTS planning)
│   │   ├── core/               #     Module/Trainer/agent traits + shared infra
│   │   ├── training/           #     Off/on-policy CPU/GPU drivers + BatchedEnv wrappers
│   │   └── primitives/ loss/ data/ #  GaussianHead/rsample, losses, replay buffers
│   ├── nn/                      #   Deep learning framework (Module + Param)
│   │   ├── core/                #     Module trait + Param (val+grad tensors), checkpoint v2
│   │   ├── primitives/          #     20+ leaves: Linear, Conv2D, NoisyLinear, LSTMCell, attention
│   │   ├── combinators/         #     Sequential, Residual, Parallel, Repeat, ...
│   │   ├── models/              #     Pre-built architectures (ResNet, GPT, ViT, ...)
│   │   ├── optimizer/           #     SGD, Adam, AdamW (+ grouped multi-tensor apply)
│   │   ├── loss/                #     MSE, Huber, CrossEntropy, SoftCrossEntropy, TwoHot
│   │   ├── initializer/         #     Xavier, Kaiming, LeCun, Normal, ...
│   │   ├── training/            #     Supervised Trainer (AMP / CUDA-graph capable)
│   │   ├── datasets/            #     MNIST, CIFAR10, TinyShakespeare, lewm_pusht loaders
│   │   └── random/              #     Host RNG (box_muller, gaussian_noise)
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
│   ├── nn/                      #   Neural network + autodiff tests
│   ├── deep_agents/             #   Deep RL agent tests
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
| **REDQ** | Continuous | Yes | N critic ensemble, subset-min target, high UTD (~20), LayerNorm variant |
| **A2C** | Discrete | CPU | GAE, softmax policy |
| **PPO** | Both | Yes | Clipped surrogate, GAE, multi-epoch, CNN variant |
| **MBPO** | Continuous | Yes | SAC + probabilistic dynamics ensemble + synthetic rollouts |
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

def main():
    var env = GridWorldEnv(width=5, height=5)
    var agent = QLearningAgent(num_states=25, num_actions=4)
    _ = agent.train(env, num_episodes=500, verbose=True)
```

### Deep RL with GPU Training

```mojo
from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM      # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM   #  6
comptime HIDDEN = 256
comptime N_ENVS = 32

# Networks are nn Modules passed as compile-time params to the agent facade.
comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN], LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN], LinearReLU[HIDDEN, HIDDEN], Linear[HIDDEN, 1],
]
comptime EnvT = BatchedGpuEnv[HalfCheetah[DT], N_ENVS, OBS_DIM, ACT_DIM]

def main() raises:
    with DeviceContext() as ctx:
        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, 256, 1_000_000],  # OBS, ACT, BATCH, CAPACITY
            ActorNet,
            CriticNet,
        ](ctx=ctx, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005)

        # One batched GPU off-policy driver call (CUDA-graph capture on by default).
        _ = agent.train[EnvT, N_ENVS=N_ENVS](
            EnvT(ctx), 600_000, updates_per_step=N_ENVS, verbose=True,
        )
```

See [`examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`](examples/half_cheetah/sac_half_cheetah_training_gpu.mojo) for the full GPU training script (with logging), and [`examples/half_cheetah/sac_half_cheetah_training.mojo`](examples/half_cheetah/sac_half_cheetah_training.mojo) for the single-process CPU version.

### Neural Network GPU Training

```mojo
from std.gpu.host import DeviceContext
from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer import Kaiming

# Model at compile time: 784 -> 128 (ReLU) -> 128 (ReLU) -> 10
comptime Net = Sequential[
    Linear[784, 128], ReLU[128],
    Linear[128, 128], ReLU[128],
    Linear[128, 10],
]

def main() raises:
    var ds = MNIST()
    with DeviceContext() as ctx:
        # The Trainer owns the Module's Params (each a val+grad tensor),
        # the optimizer state, and the loss — make() runs the initializer.
        var trainer = Trainer[Net, Adam, CrossEntropyLoss].make[INIT=Kaiming](ctx)
        # ... upload MNIST into device tensors, then run the whole-dataset loop:
        var result = trainer.train_gpu[MNIST.N_TRAIN, MNIST.N_TEST](
            ctx, train_x, train_y, test_x, test_y, epochs=20, print_every=1,
        )
        print("Final loss:", result.final_loss)
```

See [`examples/nn/mlp/mlp_mnist_training_gpu.mojo`](examples/nn/mlp/mlp_mnist_training_gpu.mojo) for the full working example.

## Extending the Framework

### Adding a New Environment

```mojo
struct MyEnv(DiscreteEnv):
    comptime StateType = MyState
    comptime ActionType = MyAction

    def step(mut self, action: MyAction) -> Tuple[MyState, Float64, Bool]: ...
    def reset(mut self) -> MyState: ...
    def state_to_index(self, state: MyState) -> Int: ...
    def action_from_index(self, idx: Int) -> MyAction: ...
```

### Adding a New Agent

```mojo
struct MyAgent(TabularAgent):
    def select_action(self, state_idx: Int) -> Int: ...
    def update(mut self, state: Int, action: Int, reward: Float64,
              next_state: Int, done: Bool): ...
    def get_best_action(self, state_idx: Int) -> Int: ...
    def decay_epsilon(mut self): ...
    def get_epsilon(self) -> Float64: ...
```
