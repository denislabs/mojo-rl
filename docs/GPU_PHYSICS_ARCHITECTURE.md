# GPU Physics Engine Architecture for Mojo-RL

## Executive Summary

This document outlines an ambitious plan to create a **trait-based, modular physics engine** that works seamlessly on both CPU and GPU, following the successful pattern established in the `deep_rl` module.

### Goals

1. **Dual CPU/GPU execution** - Same trait interface, different backends
2. **Modular composition** - Swap integrators, solvers, collision detection
3. **RL-optimized** - Batched environments, fast reset, deterministic seeding
4. **Differentiable (stretch)** - Gradient flow through physics for learning

### Non-Goals (Initially)

- Full Box2D feature parity (complex joints, motors, continuous collision)
- 3D physics (focus on 2D for LunarLander, CarRacing first)
- Soft-body or fluid simulation

---

## 1. Lessons Learned from Failed GPU LunarLander Attempts

The existing GPU LunarLander implementations (v1-v4) failed to match CPU physics because:

| Issue | Root Cause | Solution |
|-------|------------|----------|
| Contact detection mismatch | GPU used point contact, CPU uses polygon SAT | Unified contact model |
| Integration drift | Different integration order CPU vs GPU | Single integration trait |
| Torque calculation errors | Manual torque formulas diverged from Box2D | Physics primitives with tests |
| Reward shaping differences | Computed from different state representations | State normalization in one place |
| Domain randomization hack | V4 added noise to hide physics mismatch | Fix physics, not hide it |

**Key Insight**: The problem wasn't GPU physics being hard - it was having *two different physics implementations*. The solution is **one implementation with CPU/GPU backends**.

---

## 2. Architecture Overview

### 2.1 Trait Hierarchy

Following the `deep_rl` pattern where `Model` trait enables both CPU and GPU execution:

```
                     ┌─────────────────────┐
                     │   PhysicsEngine     │
                     │   (Main Trait)      │
                     └─────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌────────────────┐
│  Integrator   │    │ CollisionSystem │    │ ConstraintSolver│
│    Trait      │    │     Trait       │    │     Trait      │
└───────────────┘    └─────────────────┘    └────────────────┘
        │                      │                      │
   ┌────┴────┐            ┌────┴────┐           ┌────┴────┐
   │         │            │         │           │         │
   ▼         ▼            ▼         ▼           ▼         ▼
Euler    Verlet     SpatialHash  BVH      Sequential  Parallel
```

### 2.2 Core Traits

```mojo
# Stateless trait - no stored parameters, all state external
trait Integrator(Movable & ImplicitlyCopyable):
    """Velocity and position integration."""

    # CPU path
    fn integrate_velocities[BATCH: Int](
        self,
        bodies: LayoutTensor[...],      # [BATCH, BODY_STATE_SIZE]
        forces: LayoutTensor[...],       # [BATCH, 3] - fx, fy, torque
        gravity: Vec2,
        dt: Scalar[dtype],
    )

    fn integrate_positions[BATCH: Int](
        self,
        mut bodies: LayoutTensor[...],
        dt: Scalar[dtype],
    )

    # GPU path
    @staticmethod
    fn integrate_velocities_gpu[BATCH: Int](
        ctx: DeviceContext,
        bodies_buf: DeviceBuffer[dtype],
        forces_buf: DeviceBuffer[dtype],
        gravity: Vec2,
        dt: Scalar[dtype],
    ) raises

    @staticmethod
    fn integrate_positions_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        dt: Scalar[dtype],
    ) raises
```

```mojo
trait CollisionSystem(Movable & ImplicitlyCopyable):
    """Broad + narrow phase collision detection."""

    # Compile-time sizes
    comptime MAX_CONTACTS_PER_BODY: Int
    comptime CONTACT_DATA_SIZE: Int  # Per-contact storage

    fn detect[BATCH: Int, NUM_BODIES: Int](
        self,
        bodies: LayoutTensor[...],       # [BATCH, NUM_BODIES, BODY_SIZE]
        shapes: LayoutTensor[...],       # [NUM_SHAPES, SHAPE_SIZE]
        mut contacts: LayoutTensor[...], # [BATCH, MAX_CONTACTS, CONTACT_SIZE]
        mut contact_counts: LayoutTensor[...],  # [BATCH]
    )

    @staticmethod
    fn detect_gpu[BATCH: Int, NUM_BODIES: Int](
        ctx: DeviceContext,
        bodies_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
    ) raises
```

```mojo
trait ConstraintSolver(Movable & ImplicitlyCopyable):
    """Resolve penetration and apply impulses."""

    comptime VELOCITY_ITERATIONS: Int
    comptime POSITION_ITERATIONS: Int

    fn solve_velocity[BATCH: Int](
        self,
        mut bodies: LayoutTensor[...],
        contacts: LayoutTensor[...],
        contact_counts: LayoutTensor[...],
    )

    fn solve_position[BATCH: Int](
        self,
        mut bodies: LayoutTensor[...],
        contacts: LayoutTensor[...],
        contact_counts: LayoutTensor[...],
    )

    @staticmethod
    fn solve_velocity_gpu[BATCH: Int](
        ctx: DeviceContext,
        bodies_buf: DeviceBuffer[dtype],
        contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
    ) raises

    @staticmethod
    fn solve_position_gpu[BATCH: Int](
        ctx: DeviceContext,
        bodies_buf: DeviceBuffer[dtype],
        contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
    ) raises
```

### 2.3 PhysicsWorld - The "Trainer" of Physics

Like `Trainer` orchestrates model/optimizer/loss, `PhysicsWorld` orchestrates integrator/collision/solver:

```mojo
struct PhysicsWorld[
    INTEGRATOR: Integrator,
    COLLISION: CollisionSystem,
    SOLVER: ConstraintSolver,
    NUM_BODIES: Int,        # Maximum bodies per environment
    NUM_SHAPES: Int,        # Total shape definitions
    BATCH: Int = 1,         # Parallel environments
]:
    """Central physics simulation manager."""

    # Compile-time derived sizes
    comptime BODY_STATE_SIZE: Int = 12  # x, y, angle, vx, vy, omega, fx, fy, tau, mass, inv_mass, inv_inertia
    comptime BODIES_SIZE: Int = BATCH * NUM_BODIES * BODY_STATE_SIZE
    comptime CONTACTS_SIZE: Int = BATCH * COLLISION.MAX_CONTACTS_PER_BODY * NUM_BODIES * COLLISION.CONTACT_DATA_SIZE

    # Components (stateless)
    var integrator: Self.INTEGRATOR
    var collision: Self.COLLISION
    var solver: Self.SOLVER

    # State storage (CPU, copied to GPU when needed)
    var bodies: List[Scalar[dtype]]   # Flattened body state
    var shapes: List[Scalar[dtype]]   # Shape definitions (polygons, circles)
    var contacts: List[Scalar[dtype]] # Contact cache

    # Configuration
    var gravity: Vec2
    var dt: Scalar[dtype]

    fn step(mut self):
        """CPU physics step."""
        var bodies_tensor = self._bodies_tensor()
        var forces_tensor = self._forces_tensor()
        var contacts_tensor = self._contacts_tensor()
        var counts_tensor = self._contact_counts_tensor()

        # 1. Integrate velocities (apply forces + gravity)
        self.integrator.integrate_velocities[BATCH](
            bodies_tensor, forces_tensor, self.gravity, self.dt
        )

        # 2. Collision detection
        self.collision.detect[BATCH, NUM_BODIES](
            bodies_tensor, self._shapes_tensor(), contacts_tensor, counts_tensor
        )

        # 3. Velocity constraint solving
        for _ in range(SOLVER.VELOCITY_ITERATIONS):
            self.solver.solve_velocity[BATCH](
                bodies_tensor, contacts_tensor, counts_tensor
            )

        # 4. Integrate positions
        self.integrator.integrate_positions[BATCH](bodies_tensor, self.dt)

        # 5. Position constraint solving
        for _ in range(SOLVER.POSITION_ITERATIONS):
            self.solver.solve_position[BATCH](
                bodies_tensor, contacts_tensor, counts_tensor
            )

        # 6. Clear forces
        self._clear_forces()

    fn step_gpu(mut self, ctx: DeviceContext) raises:
        """GPU physics step - same logic, different backend."""
        # Create device buffers
        var bodies_buf = ctx.enqueue_create_buffer[dtype](Self.BODIES_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](...)
        var contacts_buf = ctx.enqueue_create_buffer[dtype](Self.CONTACTS_SIZE)
        var counts_buf = ctx.enqueue_create_buffer[dtype](BATCH)

        # Copy to GPU
        ctx.enqueue_copy(bodies_buf, self.bodies.unsafe_ptr())
        ctx.enqueue_copy(shapes_buf, self.shapes.unsafe_ptr())

        # 1. Integrate velocities
        Self.INTEGRATOR.integrate_velocities_gpu[BATCH](
            ctx, bodies_buf, forces_buf, self.gravity, self.dt
        )

        # 2. Collision detection
        Self.COLLISION.detect_gpu[BATCH, NUM_BODIES](
            ctx, bodies_buf, shapes_buf, contacts_buf, counts_buf
        )

        # 3. Velocity solving
        for _ in range(SOLVER.VELOCITY_ITERATIONS):
            Self.SOLVER.solve_velocity_gpu[BATCH](
                ctx, bodies_buf, contacts_buf, counts_buf
            )

        # 4. Integrate positions
        Self.INTEGRATOR.integrate_positions_gpu[BATCH](ctx, bodies_buf, self.dt)

        # 5. Position solving
        for _ in range(SOLVER.POSITION_ITERATIONS):
            Self.SOLVER.solve_position_gpu[BATCH](
                ctx, bodies_buf, contacts_buf, counts_buf
            )

        # Copy back
        ctx.enqueue_copy(self.bodies.unsafe_ptr(), bodies_buf)
        ctx.synchronize()
```

---

## 3. Memory Layout for GPU Efficiency

### 3.1 Batch-Major Layout

Following MuJoCo Warp pattern - batch dimension first for coalesced memory access:

```
Body State Buffer Layout (row-major, batch-first):
┌─────────────────────────────────────────────────────────────────────┐
│ Env 0, Body 0: [x, y, angle, vx, vy, omega, fx, fy, tau, m, im, ii] │
│ Env 0, Body 1: [x, y, angle, vx, vy, omega, fx, fy, tau, m, im, ii] │
│ ...                                                                  │
│ Env 1, Body 0: [x, y, angle, vx, vy, omega, fx, fy, tau, m, im, ii] │
│ Env 1, Body 1: [x, y, angle, vx, vy, omega, fx, fy, tau, m, im, ii] │
│ ...                                                                  │
└─────────────────────────────────────────────────────────────────────┘

LayoutTensor view: LayoutTensor[dtype, Layout.row_major(BATCH, NUM_BODIES, 12), ...]
```

### 3.2 Shape Definitions (Shared Across Envs)

Shapes are defined once and shared across all environments:

```
Shape Buffer Layout:
┌────────────────────────────────────────────────────────────────────┐
│ Shape 0 (Polygon): [type=0, n_verts, v0x, v0y, v1x, v1y, ...]     │
│ Shape 1 (Circle):  [type=1, radius, center_x, center_y, 0, ...]    │
│ Shape 2 (Edge):    [type=2, v0x, v0y, v1x, v1y, normal_x, normal_y]│
│ ...                                                                 │
└────────────────────────────────────────────────────────────────────┘

Max shape size: 20 floats (polygon with 8 vertices)
```

### 3.3 Contact Buffer (Per-Environment)

```
Contact Buffer Layout:
┌──────────────────────────────────────────────────────────────────────┐
│ Env 0: [contact_0, contact_1, ..., contact_MAX-1]                    │
│ Env 1: [contact_0, contact_1, ..., contact_MAX-1]                    │
│ ...                                                                   │
└──────────────────────────────────────────────────────────────────────┘

Per-contact: [body_a, body_b, point_x, point_y, normal_x, normal_y,
              penetration, normal_impulse, tangent_impulse]
Size: 9 floats per contact
```

---

## 4. GPU Kernel Design Patterns

### 4.1 One Thread Per Environment

For RL with batched environments, the simplest and most efficient pattern:

```mojo
@always_inline
@staticmethod
fn integrate_velocities_kernel[BATCH: Int, NUM_BODIES: Int](
    bodies: LayoutTensor[dtype, Layout.row_major(BATCH, NUM_BODIES, 12), MutAnyOrigin],
    forces: LayoutTensor[dtype, Layout.row_major(BATCH, NUM_BODIES, 3), ImmutAnyOrigin],
    gravity_x: Scalar[dtype],
    gravity_y: Scalar[dtype],
    dt: Scalar[dtype],
):
    """One thread processes one entire environment."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= BATCH:
        return

    # Process all bodies in this environment
    for body_idx in range(NUM_BODIES):
        # Read state
        var vx = bodies[env_idx, body_idx, 3]
        var vy = bodies[env_idx, body_idx, 4]
        var omega = bodies[env_idx, body_idx, 5]
        var inv_mass = bodies[env_idx, body_idx, 10]
        var inv_inertia = bodies[env_idx, body_idx, 11]

        # Read forces
        var fx = forces[env_idx, body_idx, 0]
        var fy = forces[env_idx, body_idx, 1]
        var tau = forces[env_idx, body_idx, 2]

        # Integrate: v += (F/m + g) * dt
        vx = vx + (fx * inv_mass + gravity_x) * dt
        vy = vy + (fy * inv_mass + gravity_y) * dt
        omega = omega + tau * inv_inertia * dt

        # Write back
        bodies[env_idx, body_idx, 3] = vx
        bodies[env_idx, body_idx, 4] = vy
        bodies[env_idx, body_idx, 5] = omega
```

### 4.2 Shared Memory for Collision Detection

When checking collisions between bodies within an environment:

```mojo
@always_inline
@staticmethod
fn detect_collisions_kernel[BATCH: Int, NUM_BODIES: Int, MAX_CONTACTS: Int](
    bodies: LayoutTensor[...],
    shapes: LayoutTensor[...],
    mut contacts: LayoutTensor[dtype, Layout.row_major(BATCH, MAX_CONTACTS, 9), MutAnyOrigin],
    mut contact_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Detect collisions within each environment."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= BATCH:
        return

    # Load environment's bodies into shared memory for faster access
    var shared_bodies = LayoutTensor[
        dtype, Layout.row_major(NUM_BODIES, 12), MutAnyOrigin
    ].stack_allocation()

    for i in range(NUM_BODIES):
        for j in range(12):
            shared_bodies[i, j] = bodies[env_idx, i, j]

    # Check all pairs (O(n²) but small n for RL environments)
    var contact_idx = 0
    for i in range(NUM_BODIES):
        for j in range(i + 1, NUM_BODIES):
            # Simplified AABB check first
            if _aabb_overlap(shared_bodies, i, j):
                # Narrow phase
                var manifold = _compute_contact(shared_bodies, shapes, i, j)
                if manifold.penetration > 0 and contact_idx < MAX_CONTACTS:
                    contacts[env_idx, contact_idx, 0] = Scalar[dtype](i)
                    contacts[env_idx, contact_idx, 1] = Scalar[dtype](j)
                    contacts[env_idx, contact_idx, 2] = manifold.point_x
                    # ... store rest of contact data
                    contact_idx += 1

    contact_counts[env_idx] = Scalar[dtype](contact_idx)
```

---

## 5. Differentiability Strategy

### 5.1 Forward-Mode Autodiff (Initial)

For simple gradient computation, use forward-mode differentiation:

```mojo
trait DifferentiableIntegrator(Integrator):
    """Integrator with Jacobian computation."""

    fn integrate_with_jacobian[BATCH: Int](
        self,
        bodies: LayoutTensor[...],
        forces: LayoutTensor[...],
        gravity: Vec2,
        dt: Scalar[dtype],
        # Outputs
        mut jacobian: LayoutTensor[...],  # dstate_next / dstate_current
    )
```

### 5.2 Reverse-Mode (Stretch Goal)

For RL, we need gradients w.r.t. actions. This requires:

1. **Tape-based autodiff**: Record operations, replay backwards
2. **Implicit function theorem**: For constraint solvers, compute dL/dstate analytically
3. **Checkpointing**: Trade memory for compute by recomputing forward pass during backward

```mojo
trait BackpropagablePhysics:
    """Physics that can backpropagate gradients."""

    fn step_and_cache[BATCH: Int](
        mut self,
        actions: LayoutTensor[...],
        mut cache: LayoutTensor[...],  # For backward pass
    )

    fn backward[BATCH: Int](
        self,
        grad_output: LayoutTensor[...],  # dL/dstate_next
        cache: LayoutTensor[...],
        mut grad_actions: LayoutTensor[...],  # dL/dactions
    )
```

---

## 6. Implementation Roadmap

### Phase 1: Core Framework ✅ COMPLETED

- [x] Define `PhysicsLayout`, `PhysicsKernel`, `PhysicsState` architecture
- [x] Implement Semi-Implicit Euler integrator (CPU + GPU)
- [x] Implement edge terrain collision for varying heights (CPU + GPU)
- [x] Implement contact impulse solver with friction (CPU + GPU)
- [x] Create `PhysicsState` orchestrator with CPU/GPU sync

### Phase 2: LunarLander Validation ✅ COMPLETED

- [x] Create LunarLanderV2 with new physics framework
- [x] Implement 3-body physics (lander + 2 legs)
- [x] Validate GPU matches CPU (<0.01 max error)
- [x] Implement discrete and continuous actions
- [x] Add wind and turbulence effects

### Phase 3: Modular Extensions ✅ COMPLETED

- [x] Add `EdgeTerrainCollision` for polygon vs edge chain
- [x] Add `RevoluteJoint` constraint with:
  - [x] Point-to-point constraint (anchor coincidence)
  - [x] Spring stiffness and damping
  - [x] Angle limits (lower/upper bounds)
- [x] Add terrain smoothing (3-point average, matches Gymnasium)
- [x] Implement leg joint angle limits matching original LunarLander

### Phase 4: Future Work

- [ ] Add rendering support (SDL2 integration)
- [ ] Port CarRacing environment to new physics
- [ ] Implement forward-mode Jacobians for differentiable physics
- [ ] Explore integration with `deep_rl` for learned dynamics
- [ ] Add motor constraints to revolute joints
- [ ] Implement spatial hashing for many-body collision

---

## 7. Testing Strategy

### 7.1 Unit Tests

```mojo
fn test_integrator_matches_cpu_gpu():
    """Verify CPU and GPU integration produce identical results."""
    var integrator = SemiImplicitEuler()

    # Initialize bodies
    var bodies_cpu = ...
    var bodies_gpu = ...  # Copy of CPU

    # Step both
    integrator.integrate_velocities[BATCH](bodies_cpu, ...)
    SemiImplicitEuler.integrate_velocities_gpu[BATCH](ctx, bodies_gpu, ...)

    # Compare
    for i in range(BATCH * NUM_BODIES * 12):
        assert abs(bodies_cpu[i] - bodies_gpu[i]) < 1e-6
```

### 7.2 Integration Tests

```mojo
fn test_lunar_lander_trajectory():
    """Verify LunarLander trajectory matches reference."""
    var world = PhysicsWorld[...](...)

    # Run 100 steps with fixed actions
    var trajectory = List[State]()
    for step in range(100):
        world.step()
        trajectory.append(world.get_state())

    # Compare with reference trajectory
    var reference = load_reference_trajectory("lunar_lander_100_steps.json")
    for i in range(100):
        assert trajectory[i].matches(reference[i], tolerance=0.01)
```

### 7.3 Drift Tests

```mojo
fn test_accumulated_drift():
    """Verify drift stays within acceptable bounds over long episodes."""
    var world_cpu = PhysicsWorld[...](...)
    var world_gpu = PhysicsWorld[...](...)  # Same initial state

    # Run 1000 steps
    for step in range(1000):
        world_cpu.step()
        world_gpu.step_gpu(ctx)

    # Check relative error
    var error = compute_relative_error(world_cpu.bodies, world_gpu.bodies)
    assert error < 0.05  # 5% max drift after 1000 steps
```

---

## 8. File Structure (Current Implementation)

```
mojo-rl/
├── physics_gpu/                      # GPU-compatible physics engine
│   ├── __init__.mojo                 # Module exports
│   ├── constants.mojo                # dtype, TILE, TPB, body/shape/contact indices
│   ├── layout.mojo                   # PhysicsLayout - compile-time memory configuration
│   ├── kernel.mojo                   # PhysicsKernel - GPU kernel orchestration
│   ├── state.mojo                    # PhysicsState - runtime state management
│   │
│   ├── collision/                    # Collision detection implementations
│   │   ├── __init__.mojo
│   │   └── edge_terrain.mojo        # EdgeTerrainCollision - polygon vs edge chain
│   │
│   └── joints/                       # Joint constraint implementations
│       ├── __init__.mojo
│       └── revolute.mojo            # RevoluteJointSolver - springs + angle limits
│
├── envs/
│   ├── lunar_lander_v2.mojo         # LunarLanderV2 using physics_gpu
│   │                                  # - 3 bodies (lander + 2 legs)
│   │                                  # - 2 revolute joints with springs/limits
│   │                                  # - Terrain smoothing
│   │                                  # - Wind/turbulence effects
│   └── ...
│
├── tests/
│   └── test_lunar_lander_v2.mojo    # Comprehensive tests:
│                                      # - Basic functionality
│                                      # - Revolute joints
│                                      # - CPU/GPU equivalence
│                                      # - Wind effects
│                                      # - Terrain generation
│                                      # - Landing detection
│
└── docs/
    ├── GPU_PHYSICS.md                # Overview and status (this index)
    ├── GPU_PHYSICS_ARCHITECTURE.md   # Architecture design
    ├── GPU_PHYSICS_IMPLEMENTATION.md # Implementation details
    └── GPU_PHYSICS_RESEARCH.md       # Research and lessons learned
```

---

## 9. Success Criteria

| Criterion | Target | Achieved | Notes |
|-----------|--------|----------|-------|
| **Correctness** | GPU matches CPU within 0.1% per step | ✅ | <0.01 max observation error |
| **CPU/GPU Parity** | Identical physics behavior | ✅ | Same constraint solver, same integration |
| **Modularity** | Swap components without env changes | ✅ | PhysicsLayout/Kernel/State separation |
| **Joint Support** | Revolute joints with limits | ✅ | Springs, damping, angle limits |
| **Gymnasium Parity** | Match original LunarLander | ✅ | All features except particles (cosmetic) |
| **Performance** | GPU 10x+ faster for batch 256+ | 🔄 | Not yet benchmarked |
| **Differentiability** | Gradient flow through physics | ⏳ | Future work |
