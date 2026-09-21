# envs/ - Environment Implementations

Native Mojo environments and Gymnasium wrappers for reinforcement learning.

## Native Tabular Environments

| Environment | File | States | Actions | Reward |
|-------------|------|--------|---------|--------|
| **GridWorld** | `gridworld.mojo` | 25 (5x5) | 4 (UDLR) | -1/step, +10 goal |
| **FrozenLake** | `frozenlake.mojo` | 16 (4x4) | 4 (UDLR) | +1 goal (sparse) |
| **CliffWalking** | `cliffwalking.mojo` | 48 (4x12) | 4 (UDLR) | -1/step, -100 cliff |
| **Taxi** | `taxi.mojo` | 500 | 6 | +20 dropoff, -10 illegal |

## Native Continuous-State Environments

| Environment | Package | Obs Dim | Actions | Physics | GPU |
|-------------|---------|---------|---------|---------|-----|
| **CartPole** | `cartpole.mojo` | 4 | 2 (discrete) | Gymnasium-matching | Yes |
| **MountainCar** | `mountain_car.mojo` | 2 | 3 (discrete) | Gymnasium-matching | No |
| **Acrobot** | `acrobot.mojo` | 6 | 3 (discrete) | RK4 integration | No |
| **Pendulum** | `pendulum/` | 3 | Continuous (1D) | CPU (V1) + GPU (V2) | Yes |

## Complex Environments (Custom Physics)

| Environment | Package | Obs Dim | Action Dim | Physics Engine | GPU |
|-------------|---------|---------|------------|----------------|-----|
| **LunarLander** | `lunar_lander/` | 8 | 4 (discrete) or continuous | physics2d (impulse) | Yes |
| **BipedalWalker** | `bipedal_walker/` | 24 | 4 (continuous) | physics2d (impulse + joints) | Yes |
| **CarRacing** | `car_racing/` | 12 | 3 (continuous) | physics2d (tire slip) | Yes |

## MuJoCo-Style Environments (physics3d)

All use the `Phyics3dEnv[MODEL_DEF, CONFIG]` generic wrapper with compile-time model definitions.

| Environment | Package | Obs Dim | Action Dim | Bodies | Joints |
|-------------|---------|---------|------------|--------|--------|
| **HalfCheetah** | `half_cheetah/` | 17 | 6 | 9 | 8 |
| **Hopper** | `hopper/` | 11 | 3 | 5 | 4 |
| **Ant** | `ant/` | 27 | 8 | 13 | 9 |
| **Walker2d** | `walker2d/` | 17 | 6 | 7 | 6 |
| **Swimmer** | `swimmer/` | 8 | 2 | 5 | 2 |
| **Humanoid** | `humanoid/` | 376 | 17 | 17 | 16 |
| **HumanoidStandup** | `humanoid_standup/` | 376 | 17 | 17 | 16 |
| **InvertedPendulum** | `inverted_pendulum/` | 4 | 1 | 2 | 1 |
| **InvertedDoublePendulum** | `inverted_double_pendulum/` | 11 | 2 | 3 | 2 |

## Gymnasium Wrappers (`gymnasium/`)

Bridge to Python Gymnasium environments via FFI:

| Wrapper | File | Environments |
|---------|------|-------------|
| **Classic Control** | `gymnasium_classic_control.mojo` | CartPole, MountainCar, Pendulum, Acrobot |
| **Box2D** | `gymnasium_box2d.mojo` | LunarLander, BipedalWalker, CarRacing |
| **Toy Text** | `gymnasium_toy_text.mojo` | FrozenLake, Taxi, Blackjack, CliffWalking |
| **MuJoCo** | `gymnasium_mujoco.mojo` | HalfCheetah, Ant, Humanoid, Walker2d, Hopper, Swimmer, etc. |

## Shared Infrastructure

- `phyics3d_env.mojo` - Generic MuJoCo environment wrapper
- `phyics3d_env_config.mojo` - Config trait (reward, termination, integrator)

## Environment Trait Hierarchy

```
RenderableEnv
├── DiscreteEnv           (tabular: GridWorld, FrozenLake, etc.)
├── BoxDiscreteActionEnv  (continuous obs + discrete actions: CartPole, LunarLander)
├── BoxContinuousActionEnv (continuous obs + continuous actions: Pendulum, DDPG/SAC envs)
├── GPUDiscreteEnv        (GPU batch: CartPole, LunarLander discrete)
└── GPUContinuousEnv      (GPU batch: Pendulum, LunarLander continuous, MuJoCo envs)
```
