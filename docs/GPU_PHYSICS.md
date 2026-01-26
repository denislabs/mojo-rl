# GPU Physics Engine Project

This directory contains documentation for the GPU physics engine project for Mojo-RL.

## Documentation Index

| Document | Description |
|----------|-------------|
| [GPU_PHYSICS_ARCHITECTURE.md](./GPU_PHYSICS_ARCHITECTURE.md) | High-level architecture, memory layout, implementation roadmap |
| [GPU_PHYSICS_IMPLEMENTATION.md](./GPU_PHYSICS_IMPLEMENTATION.md) | Detailed code patterns, integrator/collision/solver implementations |
| [GPU_PHYSICS_RESEARCH.md](./GPU_PHYSICS_RESEARCH.md) | Research summary, comparison of approaches, differentiability, lessons learned |

---

## ✅ Current Implementation Status

The GPU physics engine is **fully functional** with the following features implemented:

### Core Physics Engine (`physics2d/`)

| Component | Status | Description |
|-----------|--------|-------------|
| **PhysicsLayout** | ✅ Done | Compile-time memory layout configuration |
| **PhysicsKernel** | ✅ Done | GPU kernel orchestration (integration, collision, solving) |
| **PhysicsState** | ✅ Done | Runtime state management with CPU/GPU sync |
| **Semi-Implicit Euler** | ✅ Done | Velocity-first integration matching Box2D |
| **Edge Terrain Collision** | ✅ Done | Polygon vs edge chain collision (varying terrain heights) |
| **Contact Impulse Solver** | ✅ Done | Sequential impulse with friction and position correction |
| **Revolute Joints** | ✅ Done | Point-to-point constraints with springs and angle limits |

### LunarLanderV2 Environment (`envs/lunar_lander_v2.mojo`)

| Feature | Status | Description |
|---------|--------|-------------|
| **3-Body Physics** | ✅ Done | Main lander + 2 leg bodies |
| **Revolute Joints** | ✅ Done | Legs attached with spring stiffness/damping |
| **Joint Angle Limits** | ✅ Done | Left: 0.4-0.9 rad, Right: -0.9 to -0.4 rad |
| **Terrain Smoothing** | ✅ Done | 3-point average smoothing (matches Gymnasium) |
| **Edge Collision** | ✅ Done | Polygon vs terrain edge chain |
| **Wind/Turbulence** | ✅ Done | Optional environmental effects |
| **CPU/GPU Parity** | ✅ Done | <0.01 max error between backends |
| **Discrete Actions** | ✅ Done | 0=nop, 1=left, 2=main, 3=right |
| **Continuous Actions** | ✅ Done | [main_throttle, lateral_throttle] |

### Comparison with Gymnasium LunarLander

| Feature | Gymnasium | LunarLanderV2 | Notes |
|---------|-----------|---------------|-------|
| Physics engine | Box2D | Custom GPU | Equivalent behavior |
| Bodies | 3 | 3 | Lander + 2 legs |
| Revolute joints | ✅ | ✅ | With springs |
| Joint limits | ✅ | ✅ | Same values |
| Terrain smoothing | ✅ | ✅ | 3-point average |
| Particle effects | ✅ | ❌ | Cosmetic only, no physics impact |
| Rendering | Pygame | Not yet | Future work |

---

## Project Goal

Create a **modular physics engine** that:
- Works seamlessly on both **CPU and GPU**
- Uses LayoutTensor for zero-copy tensor views
- Enables RL environments to run physics on GPU for 100x+ speedup
- Eventually supports **differentiable physics** for model-based RL

## Quick Start

```mojo
from physics2d import PhysicsLayout, PhysicsKernel, PhysicsState
from envs.lunar_lander_v2 import LunarLanderV2

# Create environment with batched physics
var env = LunarLanderV2[BATCH=256](seed=42)

# Reset all environments
env.reset_all()

# CPU step
var obs = env.step_batch_cpu(actions)

# GPU step (same physics, different backend)
var ctx = DeviceContext()
var obs = env.step_batch_gpu(ctx, actions)
```

---

## External Resources

### GPU Physics Engines

- **Rust GPU Physics Engine** (wgpu-based)
  https://github.com/MarcVivas/gpu-physics-engine

- **Unity GPU Physics** (Compute shaders)
  - https://www.reddit.com/r/Unity3D/comments/7pa6bq/drawing_mandelbrot_fractal_using_gpu_compute/
  - https://www.reddit.com/r/Unity3D/comments/7ppldz/physics_simulation_on_gpu_with_compute_shader_in/

### NVIDIA Resources

- **GPU Gems 3: Physics Simulation** (Chapters 29-35)
  - [Chapter 29: Real-Time Rigid Body Simulation](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus)
  - [Chapter 30: 3D Fluids](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)
  - [Chapter 31: N-Body Simulation](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
  - [Chapter 32: Broad-Phase Collision Detection](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
  - [Chapter 33: LCP Algorithms for Collision](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-33-lcp-algorithms-collision-detection-using-cuda)
  - [Chapter 34: Signed Distance Fields](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-34-signed-distance-fields-using-single-pass-gpu)

### Modern GPU Physics Projects

- **MuJoCo Warp** (Google DeepMind) - Batched GPU physics for RL
  https://github.com/google-deepmind/mujoco_warp

- **Newton** (NVIDIA) - Differentiable GPU physics
  https://github.com/newton-physics/newton

- **Brax** (Google) - JAX-based differentiable physics
  https://github.com/google/brax

- **IsaacGym** (NVIDIA) - Production GPU physics for RL
  https://developer.nvidia.com/isaac-gym

### Differentiable Physics

- **Physics-Based Deep Learning** - Comprehensive book/tutorial
  https://www.physicsbaseddeeplearning.org/diffphys.html

- **Differentiable Physics Engine for Deep Learning** (Frontiers paper)
  https://www.frontiersin.org/articles/10.3389/fnbot.2019.00006/full

### Mojo GPU Programming

- **Mojo GPU Puzzles** (Modular)
  https://puzzles.modular.com