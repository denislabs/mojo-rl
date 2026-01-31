# Physics Engine Rebuild Plan

## Goal
Build a minimal, mathematically correct 3D physics engine following MuJoCo's computational foundations, designed for incremental extension.

## Design Principles

1. **MuJoCo-style separation**: Model (static) vs Data (runtime state)
2. **Mojo trait-based**: Compile-time dimensions, stateless computation graphs
3. **Incremental validation**: Each phase has analytical tests
4. **CPU/GPU parity**: Same algorithms on both targets, colocated in same files
5. **Parameterized dtype**: `float32` for GPU (Metal), configurable for debugging

---

## Current Status

| Phase | Description | CPU | GPU | Status |
|-------|-------------|-----|-----|--------|
| Phase 1 | Free fall (single body) | ✅ | ✅ | Complete |
| Phase 2 | Ground contact (sphere-plane) | ✅ | ✅ | Complete |
| Phase 3 | Multi-body + sphere-sphere | ✅ | ✅ | Complete |
| Phase 4 | Single hinge joint (pendulum) | ❌ | ❌ | Not started |
| Phase 5 | Two-link chain | ❌ | ❌ | Not started |
| Phase 6 | Friction model | ❌ | ❌ | Not started |
| Phase 7 | Simple walker environment | ❌ | ❌ | Not started |

---

## Current Architecture (Phase 1-3 Complete)

### File Structure

```
physics3d_v2/
├── __init__.mojo              # Module exports (Model, Data, Integrators)
├── PLAN.md                    # This file
├── constants.mojo             # Global constants (TILE, TPB, physics defaults)
├── types.mojo                 # Model, Data, ContactInfo structs
├── render.mojo                # SDL2 rendering for visualization
│
├── traits/                    # Trait definitions
│   ├── __init__.mojo
│   ├── collision.mojo         # CollisionSystem trait
│   └── integrator.mojo        # Integrator trait
│
├── collision/                 # Collision detection (CPU + GPU colocated)
│   ├── __init__.mojo
│   ├── collision_primitives.mojo  # Pure functions: sphere_sphere, sphere_plane
│   └── collision.mojo         # CollisionDetector with CPU and GPU methods
│
├── solver/                    # Constraint solvers (CPU + GPU colocated)
│   ├── __init__.mojo
│   ├── impulse_solver.mojo    # Bullet/Box2D style impulse solver
│   ├── pgs_solver.mojo        # MuJoCo style PGS solver
│   └── gravity_solver.mojo    # Gravity application (GPU helper)
│
├── integrator/                # Physics integrators (CPU + GPU colocated)
│   ├── __init__.mojo
│   ├── impulse_integrator.mojo    # ImpulseIntegrator (Bullet/Box2D style)
│   ├── pgs_integrator.mojo        # PGSIntegrator (MuJoCo style)
│   └── integrate_positions.mojo   # Position integration (GPU helper)
│
├── gpu/                       # GPU-specific utilities
│   ├── __init__.mojo          # Buffer utilities exports
│   ├── constants.mojo         # GPU buffer layout constants
│   └── buffer_utils.mojo      # Host/device buffer creation and access
│
└── tests/                     # Validation tests
    ├── __init__.mojo
    ├── test_freefall.mojo         # Phase 1: free fall validation
    ├── test_multi_body_impulse.mojo  # Phase 3: impulse solver tests
    ├── test_multi_body_pgs.mojo      # Phase 3: PGS solver tests
    ├── test_gpu.mojo                 # GPU parity tests
    ├── test_render_simple.mojo       # Rendering test
    ├── test_render_multi_body_impulse.mojo
    └── test_render_multi_body_pgs.mojo
```

### Core Data Structures

#### Model (Static Configuration)
```mojo
struct Model[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int]:
    # Global physics parameters
    var gravity_z: Scalar[DTYPE]     # -9.81 default
    var timestep: Scalar[DTYPE]      # 0.01 default
    var ground_z: Scalar[DTYPE]      # Ground plane height
    var restitution: Scalar[DTYPE]   # Coefficient of restitution
    var friction: Scalar[DTYPE]      # Friction coefficient

    # Per-body properties (compile-time sized arrays)
    var masses: InlineArray[Scalar[DTYPE], NUM_BODIES]
    var inv_masses: InlineArray[Scalar[DTYPE], NUM_BODIES]
    var radii: InlineArray[Scalar[DTYPE], NUM_BODIES]
    var inertias: InlineArray[Scalar[DTYPE], NUM_BODIES * 3]      # Diagonal
    var inv_inertias: InlineArray[Scalar[DTYPE], NUM_BODIES * 3]
```

#### Data (Mutable State)
```mojo
struct Data[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int]:
    # Per-body state (flattened for GPU compatibility)
    var positions: InlineArray[Scalar[DTYPE], NUM_BODIES * 3]
    var quaternions: InlineArray[Scalar[DTYPE], NUM_BODIES * 4]
    var velocities: InlineArray[Scalar[DTYPE], NUM_BODIES * 3]
    var angular_velocities: InlineArray[Scalar[DTYPE], NUM_BODIES * 3]
    var accelerations: InlineArray[Scalar[DTYPE], NUM_BODIES * 3]
    var angular_accelerations: InlineArray[Scalar[DTYPE], NUM_BODIES * 3]

    # Contact buffer
    var contacts: InlineArray[ContactInfo[DTYPE], MAX_CONTACTS]
    var num_contacts: Int
```

### Two Integrator Approaches

#### 1. ImpulseIntegrator (Bullet/Box2D Style)
Split Impulse approach for stable stacking:
- Velocity constraints handle collision response
- Position constraints directly correct penetration
- Resting contact detection prevents drift

**CPU Pipeline:**
1. Collision detection (pre-step)
2. Apply gravity to velocities
3. Solve velocity constraints (30 iterations)
4. Handle resting contacts
5. Integrate positions
6. Collision detection (post-step)
7. Solve position constraints (15 iterations)
8. Final resting contact handling

#### 2. PGSIntegrator (MuJoCo Style)
Projected Gauss-Seidel with spring-damper constraints:
- Contacts are soft constraints with spring-damper dynamics
- solref = [timeconst, dampratio] controls stiffness
- Reference acceleration: aref = -k * pos - b * vel

**CPU Pipeline:**
1. Collision detection (pre-step)
2. Apply gravity to velocities
3. Solve contact constraints using PGS (30 iterations)
4. Integrate positions
5. Collision detection (post-step)
6. Position correction (10 iterations)

### GPU Buffer Layout

For batched GPU simulation, state is stored in flat buffers:

```
Buffer shape: [BATCH, STATE_SIZE]
STATE_SIZE = NUM_BODIES * BODY_STATE_SIZE + MAX_CONTACTS * CONTACT_STATE_SIZE + METADATA_SIZE

BODY_STATE_SIZE = 22 floats per body:
  [0-2]   Position (px, py, pz)
  [3-6]   Quaternion (qx, qy, qz, qw)
  [7-9]   Linear velocity (vx, vy, vz)
  [10-12] Angular velocity (wx, wy, wz)
  [13-15] Linear acceleration (ax, ay, az)
  [16-18] Angular acceleration (αx, αy, αz)
  [19-21] Applied forces (fx, fy, fz)

CONTACT_STATE_SIZE = 12 floats per contact:
  [0-1]   Body indices (body_a, body_b)
  [2-4]   Contact position (px, py, pz)
  [5-7]   Contact normal (nx, ny, nz)
  [8]     Signed distance (negative = penetration)
  [9-11]  Impulses for warm starting (normal, tangent1, tangent2)

METADATA_SIZE = 4 floats:
  [0]     Number of active contacts
  [1-3]   Padding
```

### Colocated CPU/GPU Design

Each module contains both CPU and GPU implementations in the same file:

```mojo
struct CollisionDetector(CollisionSystem):
    # CPU method
    @staticmethod
    fn detect_all_contacts[DTYPE, NUM_BODIES, MAX_CONTACTS](
        model: Model[...], mut data: Data[...]
    ):
        ...

    # GPU method (same algorithm, LayoutTensor access)
    @staticmethod
    fn detect_all_contacts_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BATCH](
        env: Int,
        state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        model: LayoutTensor[DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin],
        ground_z: Scalar[DTYPE],
    ):
        ...
```

### Usage Examples

#### CPU Simulation
```mojo
from physics3d_v2 import Model, Data, ImpulseIntegrator, PGSIntegrator

# Create a 2-body system
var model = Model[DType.float64, 2, 10](gravity_z=-9.81, restitution=0.6)
model.set_body(0, mass=1.0, radius=0.1)
model.set_body(1, mass=1.0, radius=0.1)

var data = Data[DType.float64, 2, 10]()
data.set_body_position(0, 0, 0, 1.0)  # Body 0 at height 1m
data.set_body_position(1, 0, 0, 0.3)  # Body 1 at height 0.3m

# Simulate using ImpulseIntegrator
for i in range(100):
    ImpulseIntegrator.step(model, data)

# Or use PGSIntegrator
PGSIntegrator.step(model, data)
```

#### GPU Simulation (Batched)
```mojo
from physics3d_v2.gpu import (
    init_state_host_buffer, create_model_host_buffer,
    set_body_position, get_body_z
)
from physics3d_v2 import ImpulseIntegrator

var ctx = DeviceContext()

# Create buffers for 256 parallel environments
var host_state = init_state_host_buffer[float32, 2, 10, 256](ctx)
var host_model = create_model_host_buffer[float32, 2, 10](ctx, model)

# Set initial positions for all environments
for env in range(256):
    set_body_position[float32, 2, 10](host_state, env, 0, x=0, y=0, z=1.0)
    set_body_position[float32, 2, 10](host_state, env, 1, x=0, y=0, z=0.3)

# Transfer to GPU
var state_buf = ctx.enqueue_create_buffer(host_state)
var model_buf = ctx.enqueue_create_buffer(host_model)

# Simulate all 256 environments in parallel
for _ in range(100):
    ImpulseIntegrator.step_gpu[float32, 2, 10, 256](
        ctx, state_buf, model_buf,
        dt=0.01, gravity_z=-9.81, ground_z=0.0, restitution=0.6, friction=0.5
    )

# Transfer back and read results
ctx.enqueue_copy(host_state, state_buf)
ctx.synchronize()
var z = get_body_z[float32, 2, 10](host_state, env=0, body=0)
```

---

## Validation Tests

### Phase 1-3 Test Commands
```bash
cd mojo-rl

# Phase 1: Free fall
pixi run mojo run physics3d_v2/tests/test_freefall.mojo

# Phase 3: Multi-body CPU
pixi run mojo run physics3d_v2/tests/test_multi_body_impulse.mojo
pixi run mojo run physics3d_v2/tests/test_multi_body_pgs.mojo

# GPU tests (requires GPU environment)
pixi run -e apple mojo run physics3d_v2/tests/test_gpu.mojo

# Rendering tests (requires SDL2)
pixi run mojo run physics3d_v2/tests/test_render_multi_body_impulse.mojo
pixi run mojo run physics3d_v2/tests/test_render_multi_body_pgs.mojo
```

### Test Coverage
- Free fall: Analytical validation (z(t) = h - 0.5*g*t²)
- Ball drop: Sphere stops at correct height (radius above ground)
- Ball at rest: No drift over 1000 steps
- Two spheres collision: Proper bounce with restitution
- Sphere stack: Bodies settle without sinking
- GPU parity: Same results as CPU for all tests
- Batched simulation: 256 environments in parallel

---

## Phase 4: Single Hinge Joint (Pendulum)

### Goal
Add joint constraints to connect bodies. Start with a simple hinge joint creating a pendulum.

### New Concepts
1. **Joint constraint**: Position-level constraint connecting two bodies
2. **Jacobian computation**: J matrix relating joint to body velocities
3. **Constraint stabilization**: Baumgarte for position error correction

### Proposed Changes

#### New Files
- `joints/joint.mojo` - Joint trait and common utilities
- `joints/hinge_joint.mojo` - Hinge joint implementation
- `tests/test_pendulum.mojo` - Validation tests

#### Modified Types
```mojo
struct Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]:
    # ... existing fields ...
    var joints: InlineArray[HingeJoint[DTYPE], MAX_JOINTS]
    var num_joints: Int

struct HingeJoint[DTYPE: DType]:
    var parent_body: Int      # -1 for world anchor
    var child_body: Int
    var anchor_parent: SIMD[DTYPE, 4]  # Anchor in parent frame
    var anchor_child: SIMD[DTYPE, 4]   # Anchor in child frame
    var axis: SIMD[DTYPE, 4]           # Rotation axis
```

### Validation
- Simple pendulum: Period matches analytical T = 2π√(L/g)
- Energy conservation: Total energy constant (within integration error)
- Multi-link chain: 2-3 pendulums connected

---

## Future Phases (Outline)

### Phase 5: Two-Link Chain
- Extend joint system for multiple connected bodies
- Add constraint solver for joint + contact combined

### Phase 6: Friction Model
- Coulomb friction cone approximation
- Tangent impulses at contacts
- Static vs dynamic friction

### Phase 7: Simple Walker Environment
- Capsule/box bodies for legs and torso
- Hinge joints at hips, knees, ankles
- Contact with ground
- RL-ready observation/action interface
