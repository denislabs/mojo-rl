# GPU Physics Research Summary

This document summarizes research on GPU physics approaches and how they inform the Mojo-RL physics engine design.

---

## 1. Comparison of GPU Physics Approaches

### 1.1 Approaches Evaluated

| Approach | Source | Parallelization Strategy | Differentiable | RL-Optimized |
|----------|--------|-------------------------|----------------|--------------|
| **Particle-Based** | NVIDIA GPU Gems 3 | One thread per particle | No | No |
| **Batched Worlds** | MuJoCo Warp | One thread per environment | Yes | Yes |
| **Spatial Hashing** | Academic | One thread per cell | No | No |
| **Scene Graph** | IsaacGym | Complex hierarchy | Yes | Yes |

### 1.2 Particle-Based (GPU Gems 3)

**Philosophy**: Represent rigid bodies as collections of spheres.

**Pros**:
- Simple uniform collision detection
- Naturally parallel (all particles equal)
- Good for fluids, granular materials

**Cons**:
- Poor shape fidelity for rigid bodies
- Many particles needed for accurate shape
- Not suitable for constraint-heavy RL environments

**Verdict**: Not suitable for LunarLander (needs precise polygon collision).

### 1.3 Batched Worlds (MuJoCo Warp)

**Philosophy**: Run N complete physics simulations in parallel.

**Pros**:
- Perfect for RL (batched training)
- Each environment is independent
- No synchronization between envs
- Simple to implement correctly

**Cons**:
- Limited by slowest environment per step
- No intra-environment parallelism
- Memory scales with batch size

**Verdict**: **Best fit for Mojo-RL**. Matches our use case exactly.

### 1.4 Spatial Hashing

**Philosophy**: Divide space into cells, parallelize per cell.

**Pros**:
- O(n) collision detection (vs O(n²))
- Good for many-body simulations
- Cache-efficient memory access

**Cons**:
- Overhead for small body counts
- Load imbalance when bodies cluster
- Complex cell-to-cell handling

**Verdict**: Future optimization for CarRacing (many terrain edges), but overkill for LunarLander (3 bodies).

### 1.5 Scene Graph (IsaacGym)

**Philosophy**: Complex hierarchical scene with shared actors.

**Pros**:
- Efficient memory reuse across envs
- Supports articulated bodies
- Production-ready

**Cons**:
- Very complex implementation
- Proprietary CUDA kernels
- Hard to port to Mojo

**Verdict**: Too complex for initial implementation.

---

## 2. Differentiable Physics Approaches

### 2.1 Forward-Mode Autodiff

Compute Jacobians alongside forward pass.

```
For each operation y = f(x):
  Compute dy/dx alongside y
  Propagate Jacobian through chain: dy/dinput = dy/dx * dx/dinput
```

**Pros**:
- Single pass (no tape)
- Memory efficient
- Good for few inputs, many outputs

**Cons**:
- Cost scales with input dimension
- Need to manually implement Jacobians

**Implementation Strategy**:
```mojo
trait DifferentiableIntegrator(Integrator):
    fn integrate_with_jacobian[BATCH: Int, NUM_BODIES: Int](
        self,
        bodies: LayoutTensor[...],
        forces: LayoutTensor[...],
        gravity: Vec2[dtype],
        dt: Scalar[dtype],
        # Additional outputs
        mut d_bodies_d_forces: LayoutTensor[...],  # [BATCH, NUM_BODIES, 12, 3]
    )
```

### 2.2 Reverse-Mode Autodiff (Backpropagation)

Record operations, replay backwards.

```
Forward pass: record operations to tape
Backward pass:
  For each operation in reverse:
    Compute local gradient
    Chain with downstream gradient
```

**Pros**:
- Cost scales with output dimension (typically 1 for RL loss)
- Standard deep learning pattern

**Cons**:
- Memory scales with trajectory length
- Need tape storage
- Complex for iterative solvers

**Implementation Strategy**:
```mojo
struct DifferentiablePhysicsWorld[...]:
    var tape: PhysicsTape  # Records operations

    fn step_and_record(mut self):
        """Record operations for backward pass."""
        self.tape.record_start()
        # ... physics step
        self.tape.record_end()

    fn backward(
        self,
        grad_output: LayoutTensor[...],  # dL/d_state_final
        mut grad_actions: LayoutTensor[...],  # dL/d_actions
    ):
        """Backpropagate through recorded operations."""
        self.tape.backward(grad_output, grad_actions)
```

### 2.3 Implicit Differentiation (Advanced)

For constraint solvers, use implicit function theorem.

**Problem**: Constraint solver finds x* such that g(x*, θ) = 0.
**Question**: What is dx*/dθ?

**Solution**: Implicit differentiation.
```
d/dθ [g(x*, θ)] = 0
∂g/∂x * dx*/dθ + ∂g/∂θ = 0
dx*/dθ = -(∂g/∂x)^(-1) * ∂g/∂θ
```

This avoids differentiating through all solver iterations.

**Implementation Sketch**:
```mojo
fn solve_with_gradient[BATCH: Int](
    self,
    bodies: LayoutTensor[...],
    contacts: LayoutTensor[...],
    # Outputs
    mut grad_bodies: LayoutTensor[...],  # dx*/d_input
) -> LayoutTensor[...]:
    # Forward: solve constraints
    var x_star = self.solve[BATCH](bodies, contacts)

    # Gradient: use implicit differentiation
    # Compute Jacobian ∂g/∂x at solution
    var dg_dx = self._compute_constraint_jacobian(x_star, contacts)

    # Solve linear system for gradient
    # dx*/d_input = -(dg/dx)^(-1) * dg/d_input
    grad_bodies = self._solve_linear_system(dg_dx, dg_d_input)

    return x_star
```

---

## 3. Integration with Deep RL

### 3.1 Current Architecture

```
Environment  →  Agent  →  Policy Network  →  Action
     ↓            ↑
   (state)    (gradient via RL loss)
```

The policy network gets gradients from RL loss (PPO, SAC, etc.), not from physics.

### 3.2 Differentiable Physics Path

```
Environment  →  Agent  →  Policy Network  →  Action
     ↓            ↑              ↑
   (state)    (RL loss)    (physics gradient)
                                 ↑
                           differentiable
                            physics
```

With differentiable physics, gradients can flow:
1. **Through time**: dL/d_action = Σ_t (dL/d_state_t * d_state_t/d_action)
2. **Through physics**: d_state_next/d_action via physics Jacobians

### 3.3 Use Cases

**Model-Based RL**:
```mojo
fn train_model_based(
    agent: DeepPPOAgent,
    physics: DifferentiablePhysicsWorld,
):
    # Collect real experience
    var states, actions, rewards = collect_rollout(physics)

    # Fit dynamics model (optional if physics is known)
    # ...

    # Optimize policy using differentiable simulation
    for step in range(optimization_steps):
        # Simulate trajectory with differentiable physics
        var trajectory = physics.simulate_trajectory(agent.policy, initial_state)

        # Compute loss (e.g., sum of rewards)
        var loss = -sum(trajectory.rewards)

        # Backprop through physics to get policy gradients
        var grad_policy = physics.backward(loss)

        # Update policy
        agent.optimizer.step(grad_policy)
```

**System Identification**:
```mojo
fn learn_physics_parameters(
    physics: DifferentiablePhysicsWorld,
    real_trajectories: List[Trajectory],
):
    """Learn physics parameters (mass, friction) from real data."""
    var params = physics.get_learnable_params()  # mass, friction, etc.

    for epoch in range(epochs):
        var total_loss = 0.0

        for traj in real_trajectories:
            # Simulate with current params
            var sim_states = physics.simulate(traj.initial_state, traj.actions)

            # Loss: difference from real states
            var loss = mse_loss(sim_states, traj.states)
            total_loss += loss

            # Backprop through physics
            var grad_params = physics.backward_to_params(loss)
            params -= learning_rate * grad_params

        print("Epoch", epoch, "Loss:", total_loss)
```

---

## 4. Lessons from Failed GPU LunarLander Attempts (V1-V4)

### 4.1 Version History

| Version | Approach | Issue |
|---------|----------|-------|
| V1 | Direct physics translation | Leg contact wrong (point vs box) |
| V2 | Sub-stepping + hidden state | Verlet integration order wrong |
| V3 | Improved contact physics | Still 10-15% drift over episode |
| V4 | Domain randomization | Hid problem instead of fixing |
| **V2 (new)** | **Unified physics engine** | **✅ SOLVED - <0.01 max error** |

### 4.2 Root Causes (Now Fixed)

**1. Two Different Physics Engines** → **FIXED**

The fundamental problem was: CPU used Box2D, GPU used hand-written physics.

**Solution**: Created unified `physics2d/` module with identical CPU and GPU code paths. Same `PhysicsKernel` functions run on both backends.

**2. No Shared Code Path** → **FIXED**

CPU and GPU had completely different codebases.

**Solution**: `PhysicsState` manages both CPU tensors and GPU buffers. `PhysicsKernel` provides static methods that work on LayoutTensor (CPU) or DeviceBuffer (GPU).

**3. Testing Was Trajectory-Based** → **FIXED**

Tested full episodes, not single-step physics.

**Solution**: `test_lunar_lander_v2.mojo` includes step-by-step CPU vs GPU comparison showing <0.01 max error at every step.

### 4.3 Solutions Applied and Verified

**1. Single Source of Truth** ✅

```mojo
# Same kernel, different backend
PhysicsKernel.step_cpu[BATCH, BODIES, SHAPES](...)  # CPU
PhysicsKernel.step_gpu[BATCH, BODIES, SHAPES](ctx, ...)  # GPU
```

**2. Unit Tests Per Operation** ✅

The `test_lunar_lander_v2.mojo` test suite includes:
- `test_basic_functionality()` - Environment works correctly
- `test_revolute_joints()` - Joints constrain leg movement
- `test_cpu_gpu_equivalence()` - Step-by-step comparison
- `test_wind_effects()` - Wind affects trajectory
- `test_terrain_generation()` - Terrain smoothing and helipad
- `test_landing_detection()` - Crash and success detection

**3. Deterministic Seeding** ✅

```mojo
# PhiloxRandom for reproducible random numbers on CPU and GPU
var rng = PhiloxRandom(seed=self.rng_seed + env, offset=self.rng_counter)
```

---

## 5. Performance Expectations

### 5.1 Target Benchmarks

Based on MuJoCo Warp and IsaacGym performance data:

| Batch Size | CPU (single-thread) | GPU (A100) | Speedup |
|------------|---------------------|------------|---------|
| 1 | 0.5 ms | 1.0 ms | 0.5x (slower due to launch overhead) |
| 64 | 32 ms | 1.5 ms | 21x |
| 256 | 128 ms | 2.0 ms | 64x |
| 1024 | 512 ms | 5.0 ms | 102x |
| 4096 | 2048 ms | 15 ms | 136x |

### 5.2 Memory Budget

For LunarLander with 3 bodies:

```
Per-environment state:
  Bodies: 3 * 12 floats = 36 floats = 144 bytes
  Shapes: 3 * 20 floats = 60 floats = 240 bytes (shared)
  Forces: 3 * 3 floats = 9 floats = 36 bytes
  Contacts: 12 * 9 floats = 108 floats = 432 bytes
  Total: ~600 bytes per environment

For 4096 parallel environments:
  Bodies: 4096 * 144 = 576 KB
  Shapes: 240 bytes (shared)
  Forces: 4096 * 36 = 144 KB
  Contacts: 4096 * 432 = 1.7 MB
  Total: ~2.5 MB
```

This easily fits in GPU memory. The bottleneck will be compute, not memory.

### 5.3 Kernel Optimization Opportunities

**1. Reduce Global Memory Access**

```mojo
# Bad: multiple global reads per body
var x = bodies[env, body, 0]
var y = bodies[env, body, 1]
var vx = bodies[env, body, 3]
# ...

# Good: single read into local variables
var body_state = bodies[env, body, :]  # Coalesced read
var x = body_state[0]
var y = body_state[1]
# ...
```

**2. Use Shared Memory for Collision**

```mojo
# Load environment's bodies into shared memory
var shared_bodies = LayoutTensor[...].stack_allocation()
for i in range(NUM_BODIES):
    shared_bodies[i, :] = bodies[env, i, :]
barrier()

# Now collision checks use fast shared memory
for i in range(NUM_BODIES):
    for j in range(i+1, NUM_BODIES):
        if overlap(shared_bodies[i], shared_bodies[j]):
            # ...
```

**3. Unroll Small Loops**

```mojo
# Use @parameter for compile-time unrolling
@parameter
for body in range(NUM_BODIES):  # NUM_BODIES is comptime
    # ...

# Mojo will unroll this completely, no loop overhead
```

---

## 6. References

### Academic Papers

1. **Efficient Simulation of Inextensible Cloth** (Goldenthal et al., 2007)
   - Constraint projection methods for fast convergence

2. **Position Based Dynamics** (Müller et al., 2007)
   - Simplified constraint solving without explicit forces

3. **A Unified Approach for Rigid Body Dynamics** (Catto, 2005)
   - Box2D constraint solver design (Sequential Impulse)

### Industry Resources

4. **NVIDIA GPU Gems 3, Chapter 29-35** (2007)
   - Classic GPU physics techniques
   - https://developer.nvidia.com/gpugems/gpugems3

5. **MuJoCo Warp Documentation** (Google DeepMind, 2024)
   - Modern batched GPU physics
   - https://mujoco.readthedocs.io/en/latest/mjwarp/

6. **Newton Physics Engine** (NVIDIA, 2024)
   - Differentiable physics on GPU
   - https://github.com/newton-physics/newton

7. **IsaacGym** (NVIDIA, 2021)
   - Production GPU physics for RL
   - https://developer.nvidia.com/isaac-gym

### Mojo-Specific

8. **Mojo GPU Puzzles** (Modular, 2024)
   - GPU programming patterns in Mojo
   - https://puzzles.modular.com

9. **Mojo MAX Graphs** (Modular, 2024)
   - Graph-based computation for potential autodiff
   - https://docs.modular.com/max/graph

---

## 7. Implementation Progress

### ✅ Completed

1. [x] Create `physics2d/` directory structure
2. [x] Implement Semi-Implicit Euler integration (CPU + GPU)
3. [x] Write tests for CPU-GPU equivalence
4. [x] Implement `EdgeTerrainCollision` for varying terrain heights
5. [x] Implement contact impulse solver with friction
6. [x] Create `PhysicsState` orchestrator with CPU/GPU sync
7. [x] Create LunarLanderV2 with new physics
8. [x] Add `RevoluteJoint` constraint with:
   - [x] Point-to-point constraint
   - [x] Spring stiffness and damping
   - [x] Angle limits (matching Gymnasium values)
9. [x] Add terrain smoothing (3-point average)
10. [x] Validate CPU matches GPU (<0.01 error)

### 🔄 In Progress / Future Work

11. [ ] Add rendering support (SDL2 integration)
12. [ ] Port CarRacing environment
13. [ ] Benchmark GPU vs CPU performance
14. [ ] Explore forward-mode Jacobians
15. [ ] Prototype differentiable physics
16. [ ] Integrate with `deep_rl` for learned dynamics
17. [ ] Add motor constraints to revolute joints
18. [ ] Consider 3D extension for MuJoCo-like envs
