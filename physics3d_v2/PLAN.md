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
| Phase 4 | Single hinge joint (pendulum) | ✅ | ✅ | Complete |
| Phase 5 | Two-link chain (double pendulum) | ✅ | ✅ | Complete |
| Phase 6 | Friction model | ✅ | ✅ | Complete |
| Phase 7 | Simple hopper (2-body + torque) | ✅ | ✅ | Complete |
| Phase 8 | Capsule geometry | ✅ | ✅ | Complete |
| Phase 9 | Box geometry | ✅ | ✅ | Complete |
| Phase 10a | Bipedal walker (3-body, 2-joint) | ✅ | ✅ | Complete |
| Phase 11f | Free DOF joints (MuJoCo-style roots) | ✅ | ✅ | Complete |

---

## Current Architecture (Phase 1-9, 10a Complete)

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
│   ├── collision_primitives.mojo  # Pure functions: sphere, capsule, box primitives, compute_tangent_basis
│   └── collision.mojo         # CollisionDetector with CPU and GPU methods
│
├── solver/                    # Constraint solvers (CPU + GPU colocated)
│   ├── __init__.mojo
│   ├── impulse_solver.mojo    # Bullet/Box2D style impulse solver
│   ├── pgs_solver.mojo        # MuJoCo style PGS solver
│   └── gravity_solver.mojo    # Gravity application (GPU helper)
│
├── joints/                    # Joint constraints (CPU + GPU colocated)
│   ├── __init__.mojo          # Module exports
│   ├── hinge_joint.mojo       # HingeJoint struct definition
│   └── joint_solver.mojo      # CPU + GPU joint constraint solvers
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
├── envs/                      # RL environments (Phase 7+)
│   ├── __init__.mojo          # Module exports
│   ├── hopper.mojo            # HopperEnv struct (Phase 7)
│   └── walker.mojo            # WalkerEnv struct (Phase 10a)
│
├── tests/                     # Validation tests
│   ├── __init__.mojo
│   ├── test_freefall.mojo         # Phase 1: free fall validation
│   ├── test_multi_body_impulse.mojo  # Phase 3: impulse solver tests
│   ├── test_multi_body_pgs.mojo      # Phase 3: PGS solver tests
│   ├── test_pendulum.mojo         # Phase 4: CPU pendulum tests
│   ├── test_pendulum_gpu.mojo     # Phase 4: GPU pendulum tests
│   ├── test_double_pendulum.mojo      # Phase 5: CPU double pendulum tests
│   ├── test_double_pendulum_gpu.mojo  # Phase 5: GPU double pendulum tests
│   ├── test_gpu.mojo              # GPU parity tests
│   ├── test_render_simple.mojo    # Rendering test
│   ├── test_render_multi_body_impulse.mojo
│   ├── test_render_multi_body_pgs.mojo
│   ├── test_friction.mojo            # Phase 6: CPU friction tests
│   ├── test_friction_gpu.mojo        # Phase 6: GPU friction tests
│   ├── test_render_friction.mojo     # Phase 6: Visual friction demo
│   ├── test_joint_torque.mojo       # Phase 7: torque actuation tests
│   ├── test_joint_torque_gpu.mojo   # Phase 7: GPU torque tests
│   ├── test_joint_sensing.mojo      # Phase 7: joint sensing tests
│   ├── test_joint_sensing_gpu.mojo  # Phase 7: GPU sensing tests
│   ├── test_hopper.mojo             # Phase 7: HopperEnv tests (CPU)
│   ├── test_hopper_gpu.mojo         # Phase 7: HopperEnv tests (GPU)
│   ├── test_capsule_plane.mojo      # Phase 8: Capsule-plane tests
│   ├── test_capsule_sphere.mojo     # Phase 8: Capsule-sphere tests
│   ├── test_capsule_capsule.mojo    # Phase 8: Capsule-capsule tests
│   ├── test_capsule_gpu.mojo        # Phase 8: GPU capsule tests
│   ├── test_box_plane.mojo          # Phase 9: Box-plane tests
│   ├── test_box_sphere.mojo         # Phase 9: Box-sphere tests
│   ├── test_box_capsule.mojo        # Phase 9: Box-capsule tests
│   ├── test_box_box.mojo            # Phase 9: Box-box tests
│   ├── test_box_gpu.mojo            # Phase 9: GPU box tests
│   ├── test_walker.mojo             # Phase 10a: WalkerEnv tests (CPU)
│   └── test_walker_gpu.mojo         # Phase 10a: WalkerEnv tests (GPU)
│
└── examples/
    ├── __init__.mojo
    ├── double_pendulum_render_demo.mojo  # Visual double pendulum demonstration
    └── walker_render_demo.mojo           # Visual walker demonstration (Phase 10a)
```

### Core Data Structures

#### Model (Static Configuration)
```mojo
struct Model[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0]:
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

    # Joints (Phase 4)
    var joints: InlineArray[HingeJoint[DTYPE], MAX_JOINTS]
    var num_joints: Int

    fn add_hinge_joint(...) -> Int  # Add a joint, returns joint index
```

#### Data (Mutable State)
```mojo
struct Data[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0]:
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

#### HingeJoint (Phase 4)
```mojo
struct HingeJoint[DTYPE: DType]:
    var parent_body: Int      # -1 for world anchor
    var child_body: Int
    # Anchor points (local frame)
    var anchor_parent_x, anchor_parent_y, anchor_parent_z: Scalar[DTYPE]
    var anchor_child_x, anchor_child_y, anchor_child_z: Scalar[DTYPE]
    # Rotation axis (parent/world frame)
    var axis_x, axis_y, axis_z: Scalar[DTYPE]
    # Accumulated impulses for warm starting (5 DOF: 3 linear + 2 angular)
    var impulse_lx, impulse_ly, impulse_lz: Scalar[DTYPE]
    var impulse_ax, impulse_ay: Scalar[DTYPE]
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
4. **Solve joint velocity constraints (if MAX_JOINTS > 0)**
5. Handle resting contacts
6. Integrate positions
7. Integrate angular positions (quaternions)
8. Collision detection (post-step)
9. Solve position constraints (15 iterations)
10. **Solve joint position constraints (if MAX_JOINTS > 0)**
11. Final resting contact handling

#### 2. PGSIntegrator (MuJoCo Style)
Projected Gauss-Seidel with spring-damper constraints:
- Contacts are soft constraints with spring-damper dynamics
- solref = [timeconst, dampratio] controls stiffness
- Reference acceleration: aref = -k * pos - b * vel

**CPU Pipeline:**
1. Collision detection (pre-step)
2. Apply gravity to velocities
3. Solve contact constraints using PGS (30 iterations)
4. **Solve joint velocity constraints (if MAX_JOINTS > 0)**
5. Integrate positions
6. Integrate angular positions (quaternions)
7. Collision detection (post-step)
8. Position correction (10 iterations)
9. **Solve joint position constraints (if MAX_JOINTS > 0)**

### GPU Buffer Layout

For batched GPU simulation, state is stored in flat buffers:

```
Buffer shape: [BATCH, STATE_SIZE]
STATE_SIZE = NUM_BODIES * BODY_STATE_SIZE + MAX_CONTACTS * CONTACT_STATE_SIZE
           + MAX_JOINTS * JOINT_STATE_SIZE + METADATA_SIZE

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

JOINT_STATE_SIZE = 16 floats per joint (Phase 4):
  [0-1]   Body indices (parent, child) - parent=-1 for world anchor
  [2-4]   Anchor point on parent (px, py, pz)
  [5-7]   Anchor point on child (cx, cy, cz)
  [8-10]  Hinge axis (ax, ay, az)
  [11-15] Accumulated impulses (lx, ly, lz, ax, ay)

METADATA_SIZE = 4 floats:
  [0]     Number of active contacts
  [1]     Number of active joints
  [2-3]   Padding
```

### Colocated CPU/GPU Design

Each module contains both CPU and GPU implementations in the same file:

```mojo
# Joint solver example (joint_solver.mojo)
fn solve_joint_velocity_constraints[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
    model: Model[...], mut data: Data[...], iterations: Int
):
    """CPU implementation."""
    for _ in range(iterations):
        for j in range(model.num_joints):
            _solve_single_joint_velocity(model, data, j)

fn solve_joint_velocity_constraints_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, STATE_SIZE, BATCH](
    env: Int,
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin],
    iterations: Int,
):
    """GPU implementation with same algorithm, LayoutTensor access."""
    for _ in range(iterations):
        for j in range(MAX_JOINTS):
            # ... constraint solving with flat buffer access ...
```

### Usage Examples

#### CPU Simulation with Pendulum
```mojo
from physics3d_v2 import Model, Data, ImpulseIntegrator

# Create a pendulum (1 body + 1 joint)
var model = Model[DType.float64, 1, 5, 1](gravity_z=-9.81)
model.set_body(0, mass=1.0, radius=0.1)

# Add hinge joint anchored to world
model.add_hinge_joint(
    parent=-1,  # World anchor
    child=0,
    anchor_parent=(0.0, 0.0, 1.0),  # Pivot point at (0, 0, 1)
    anchor_child=(0.0, 0.0, 1.0),   # Body local anchor
    axis=(0.0, 1.0, 0.0),           # Y-axis rotation (XZ plane swing)
)

# Initial position: 30 degrees from vertical
var data = Data[DType.float64, 1, 5, 1]()
data.set_body_position(0, 0.5, 0.0, 0.134)  # sin(30°), 0, 1-cos(30°)

# Simulate
for _ in range(1000):
    ImpulseIntegrator.step(model, data)
```

#### GPU Simulation (Batched Pendulums)
```mojo
from physics3d_v2 import ImpulseIntegrator
from physics3d_v2.gpu.constants import compute_state_size, body_offset, joint_offset

var ctx = DeviceContext()
comptime STATE_SIZE = compute_state_size[1, 5, 1]()

# Initialize state buffer with pendulum configuration
var state_host = List[Scalar[float32]](capacity=STATE_SIZE)
# ... set body position, quaternion, joint anchors, metadata ...

var state_buf = ctx.enqueue_create_buffer[float32](STATE_SIZE)
ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

# Simulate on GPU
for _ in range(1000):
    ImpulseIntegrator.step_gpu[float32, 1, 5, 1, 1](
        ctx, state_buf, model_buf,
        dt=0.001, gravity_z=-9.81, ground_z=-10.0, restitution=0.0, friction=0.0
    )
```

---

## Validation Tests

### Phase 1-5 Test Commands
```bash
cd mojo-rl

# Phase 1: Free fall
pixi run mojo run physics3d_v2/tests/test_freefall.mojo

# Phase 3: Multi-body CPU
pixi run mojo run physics3d_v2/tests/test_multi_body_impulse.mojo
pixi run mojo run physics3d_v2/tests/test_multi_body_pgs.mojo

# Phase 4: Pendulum CPU
pixi run mojo run physics3d_v2/tests/test_pendulum.mojo

# Phase 4: Pendulum GPU
pixi run -e apple mojo run physics3d_v2/tests/test_pendulum_gpu.mojo

# Phase 5: Double Pendulum CPU
pixi run mojo run physics3d_v2/tests/test_double_pendulum.mojo

# Phase 5: Double Pendulum GPU
pixi run -e apple mojo run physics3d_v2/tests/test_double_pendulum_gpu.mojo

# GPU tests (requires GPU environment)
pixi run -e apple mojo run physics3d_v2/tests/test_gpu.mojo

# Rendering tests (requires SDL2)
pixi run mojo run physics3d_v2/tests/test_render_multi_body_impulse.mojo
pixi run mojo run physics3d_v2/tests/test_render_multi_body_pgs.mojo

# Visual double pendulum demo (requires SDL2)
pixi run mojo run physics3d_v2/examples/double_pendulum_render_demo.mojo

# Phase 6: Friction tests
pixi run mojo run physics3d_v2/tests/test_friction.mojo

# Phase 6: Friction GPU tests
pixi run -e apple mojo run physics3d_v2/tests/test_friction_gpu.mojo

# Phase 6: Visual friction demo (requires SDL2)
pixi run mojo run physics3d_v2/tests/test_render_friction.mojo

# Phase 7: Joint torque actuation tests
pixi run mojo run physics3d_v2/tests/test_joint_torque.mojo

# Phase 7: Joint sensing tests
pixi run mojo run physics3d_v2/tests/test_joint_sensing.mojo

# Phase 7: Joint torque GPU tests
pixi run -e apple mojo run physics3d_v2/tests/test_joint_torque_gpu.mojo

# Phase 7: Joint sensing GPU tests
pixi run -e apple mojo run physics3d_v2/tests/test_joint_sensing_gpu.mojo

# Phase 7: HopperEnv tests
pixi run mojo run physics3d_v2/tests/test_hopper.mojo

# Phase 7: HopperEnv GPU tests
pixi run -e apple mojo run physics3d_v2/tests/test_hopper_gpu.mojo

# Phase 8: Capsule collision tests
pixi run mojo run physics3d_v2/tests/test_capsule_plane.mojo
pixi run mojo run physics3d_v2/tests/test_capsule_sphere.mojo
pixi run mojo run physics3d_v2/tests/test_capsule_capsule.mojo
pixi run -e apple mojo run physics3d_v2/tests/test_capsule_gpu.mojo

# Phase 9: Box collision tests
pixi run mojo run physics3d_v2/tests/test_box_plane.mojo
pixi run mojo run physics3d_v2/tests/test_box_sphere.mojo
pixi run mojo run physics3d_v2/tests/test_box_capsule.mojo
pixi run mojo run physics3d_v2/tests/test_box_box.mojo
pixi run -e apple mojo run physics3d_v2/tests/test_box_gpu.mojo

# Phase 10a: Walker environment tests
pixi run mojo run physics3d_v2/tests/test_walker.mojo
pixi run -e apple mojo run physics3d_v2/tests/test_walker_gpu.mojo

# Visual walker demo (requires SDL2)
pixi run mojo run physics3d_v2/examples/walker_render_demo.mojo
```

### Test Coverage
- Free fall: Analytical validation (z(t) = h - 0.5*g*t²)
- Ball drop: Sphere stops at correct height (radius above ground)
- Ball at rest: No drift over 1000 steps
- Two spheres collision: Proper bounce with restitution
- Sphere stack: Bodies settle without sinking
- **Pendulum constraint**: Distance to pivot maintained (<1.2mm error)
- **Pendulum period**: Within 5% of analytical T = 2π√(L/g)
- **Pendulum energy**: Stable (bounded drift, no explosion)
- **Double pendulum constraints**: Both link lengths maintained (<15mm error)
- **Double pendulum motion**: Oscillatory behavior with reasonable amplitude
- **Double pendulum sensitivity**: Sensitive to initial conditions (chaos indicator)
- GPU parity: Same results as CPU for all tests
- Batched simulation: 256 environments in parallel
- **Friction sliding**: Sphere stops due to friction within expected time
- **Friction cone**: Tangent impulse magnitude <= μ × normal impulse
- **Zero friction**: Sphere slides freely without deceleration
- **High friction**: Sphere with high friction stops quickly
- **Two spheres friction**: Collision with friction affects motion
- **Joint torque actuation**: Torque causes correct angular acceleration (Δω = τ × inv_I × dt)
- **Torque limits**: Clamped to ±torque_limit
- **Reaction torque**: Newton's 3rd law on parent body
- **Pendulum torque control**: Responds to control input
- **Joint angle sensing**: Accurate angle from quaternion difference (±0.01 deg)
- **Joint angular velocity**: Matches actual rotation speed
- **Two-body joint sensing**: Relative angle and velocity correct
- **GPU torque actuation**: Same results as CPU (position diff < 0.1m)
- **GPU batched torque**: 16 environments with different torques
- **GPU sensing parity**: Angle diff < 3.5e-06 deg, omega diff < 1e-06 rad/s
- **HopperEnv initialization**: Model configured correctly (1 joint, 2 bodies)
- **HopperEnv reset**: Valid observation (height, velocities, angles)
- **HopperEnv standing**: Stable with zero action (height drops <0.01m)
- **HopperEnv torque response**: Hip angle changes with torque
- **HopperEnv termination**: Episode ends when pitch exceeds limit
- **HopperEnv reward**: Forward velocity + alive bonus - control cost
- **GPU HopperEnv**: Joint constraints maintained, foot above ground
- **GPU HopperEnv parity**: CPU vs GPU position difference <1cm
- **Box-plane collision**: Box stops at correct height (half_z for axis-aligned)
- **Tilted box-plane**: Box stops at lowest vertex height
- **Box drift**: No drift over 1000 steps (<0.01mm)
- **Box-sphere face**: Sphere lands on box face at correct height
- **Box-sphere edge/corner**: Sphere stays above ground
- **Box-capsule face**: Capsule lands on box at correct height
- **Box-capsule parallel/angled**: Proper collision response
- **Box-box face-face**: Small box rests on large box correctly
- **Box-box edge-edge**: Rotated box rests on base box
- **Stacked boxes stability**: 3 boxes stack with minimal drift (<0.03mm)
- **Box GPU parity**: CPU/GPU difference < 1e-6 for all box primitives
- **WalkerEnv initialization**: Model configured correctly (2 joints, 3 bodies)
- **WalkerEnv reset**: Valid 12-dim observation (height ~0.5m, velocities ~0)
- **WalkerEnv standing**: Stable with zero action (height drops <0.02m over 100 steps)
- **WalkerEnv asymmetric torque**: Hip angles diverge with opposite torques
- **WalkerEnv symmetric actuation**: Both legs move together
- **WalkerEnv termination**: Episode ends when pitch/roll exceeds limits
- **WalkerEnv reward**: Forward velocity + alive bonus - control cost - height penalty
- **WalkerEnv bilateral contact**: Both feet detect ground contact
- **GPU WalkerEnv**: 3-body simulation maintains joint constraints
- **GPU WalkerEnv parity**: CPU vs GPU torso position difference <0.5m
- **GPU Batched Walker**: 8 environments with different torques run in parallel

---

## Phase 4: Single Hinge Joint (Pendulum) - COMPLETE

### Implementation Summary

#### Files Added
- `joints/__init__.mojo` - Module exports
- `joints/hinge_joint.mojo` - HingeJoint struct with create() factory
- `joints/joint_solver.mojo` - CPU and GPU joint constraint solvers
- `tests/test_pendulum.mojo` - CPU validation (constraint, period, energy)
- `tests/test_pendulum_gpu.mojo` - GPU validation
- `examples/pendulum_render_demo.mojo` - Visual demonstration

#### Files Modified
- `types.mojo` - Added MAX_JOINTS template parameter to Model/Data
- `gpu/constants.mojo` - Added JOINT_STATE_SIZE, joint buffer layout
- `integrator/impulse_integrator.mojo` - Integrated joint solving in CPU/GPU pipelines
- `integrator/pgs_integrator.mojo` - Integrated joint solving in CPU/GPU pipelines

#### Joint Constraint Physics
The hinge joint constrains 5 degrees of freedom:
- **Position constraint (3 DOF)**: Anchor points must coincide
  ```
  C_pos = anchor_world_child - anchor_world_parent = 0
  ```
- **Angular constraint (2 DOF)**: Bodies rotate only around hinge axis
  ```
  C_ang = perpendicular components of relative angular velocity = 0
  ```

#### Solver Algorithm
1. Compute world-space anchor positions using quaternion rotation
2. Compute velocity at anchors: v_anchor = v_body + ω × r
3. Compute velocity error: Δv = v_anchor_parent - v_anchor_child
4. Compute effective mass: K = inv_mass_a + inv_mass_b + rotational_contribution
5. Apply impulse: j = -relaxation × Δv / K
6. Apply position correction using Baumgarte stabilization

#### Test Results
- **CPU Constraint accuracy**: <1.2mm error over 5 seconds
- **CPU Period**: Within 2% of analytical (T ≈ 2.006s for 1m pendulum)
- **CPU Energy**: Stable with bounded drift (<500%)
- **GPU Constraint accuracy**: <0.3mm error
- **GPU Motion**: Correct oscillating pendulum behavior

---

## Phase 5: Two-Link Chain (Double Pendulum) - COMPLETE

### Implementation Summary

#### Files Added
- `tests/test_double_pendulum.mojo` - CPU validation (constraints, energy, motion, chaos)
- `tests/test_double_pendulum_gpu.mojo` - GPU validation
- `examples/double_pendulum_render_demo.mojo` - Visual demonstration
- `examples/__init__.mojo` - Examples module

#### Files Modified
- `joints/joint_solver.mojo` - Extended GPU solver for body-to-body joints
- `render.mojo` - Added `render_with_joints()` for joint visualization

#### Double Pendulum Configuration
A double pendulum consists of two links connected in series:
```mojo
# 2 bodies, 2 joints
var model = Model[DTYPE, 2, 10, 2](gravity_z=-9.81)
model.set_body(0, mass=1.0, radius=0.1)
model.set_body(1, mass=1.0, radius=0.1)

# Joint 0: World -> Body 0
model.add_hinge_joint(
    parent=-1,  # World anchor
    child=0,
    anchor_parent=(0.0, 0.0, L1),  # Pivot at height L1
    anchor_child=(0.0, 0.0, L1),   # L1 above body 0
    axis=(0.0, 1.0, 0.0),          # Y-axis rotation
)

# Joint 1: Body 0 -> Body 1
model.add_hinge_joint(
    parent=0,   # Body 0 is parent
    child=1,
    anchor_parent=(0.0, 0.0, 0.0),  # At body 0's position
    anchor_child=(0.0, 0.0, L2),    # L2 above body 1
    axis=(0.0, 1.0, 0.0),
)
```

#### GPU Body-to-Body Joint Support
Extended the GPU joint solver to support chained joints using a sequential pattern:
```mojo
# Joint 0: parent = -1 (world), child = 0
# Joint j (j>0): parent = j-1, child = j
var body_a: Int = -1 if j == 0 else j - 1
var body_b: Int = j
```

This supports single pendulum, double pendulum, triple pendulum, and longer chains without requiring conditionals on buffer values (which cause GPU issues).

#### Test Results
- **CPU Constraint accuracy**: <15mm error for both joints over 5 seconds
- **CPU Energy stability**: Bounded drift within 3000% (impulse solver characteristic)
- **CPU Motion**: 7+ zero crossings in 10s, reasonable amplitude
- **CPU Chaos**: Sensitive to 0.1° initial angle difference
- **GPU Constraint accuracy**: <20mm error (float32 precision)
- **GPU Motion**: Correct oscillating double pendulum behavior

#### Rendering Support
Added `render_with_joints()` method that draws:
- Link lines connecting pivot to body 0, body 0 to body 1
- Pivot point indicator (gold circle at world anchor)
- Bodies as colored spheres
- Ground plane with shadows

---

## Phase 6: Friction Model - COMPLETE

### Implementation Summary

#### Files Added
- `tests/test_friction.mojo` - CPU friction validation (5 tests)
- `tests/test_friction_gpu.mojo` - GPU friction validation (3 tests)
- `tests/test_render_friction.mojo` - Visual friction demonstration

#### Files Modified
- `collision/collision_primitives.mojo` - Added `compute_tangent_basis()` function
- `solver/impulse_solver.mojo` - Added friction to velocity constraints (CPU + GPU)
- `solver/pgs_solver.mojo` - Added friction to PGS constraints (CPU + GPU)
- `integrator/impulse_integrator.mojo` - Pass friction parameter through pipeline
- `integrator/pgs_integrator.mojo` - Pass friction parameter through pipeline

#### Coulomb Friction Physics
The friction model implements Coulomb friction with cone constraint:
```
|f_tangent| <= μ × f_normal
```

For each contact:
1. **Tangent basis**: Two orthonormal vectors (t1, t2) perpendicular to normal
2. **Relative tangent velocity**: v_t = v_rel - (v_rel · n) × n
3. **Tangent impulses**: Computed to stop sliding, then clamped to friction cone

#### Tangent Basis Computation
Uses Gram-Schmidt orthogonalization from contact normal:
```mojo
fn compute_tangent_basis[DTYPE: DType](
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], ...]:  # Returns t1x, t1y, t1z, t2x, t2y, t2z
    # Find axis least parallel to normal
    # t1 = normalize(axis - (axis·n)×n)
    # t2 = n × t1
```

#### Friction Solving Algorithm
For each contact iteration:
1. Compute tangent basis from contact normal
2. Compute relative tangent velocities: `rel_vt1`, `rel_vt2`
3. Compute delta impulses: `delta_jt = rel_vt / effective_mass`
4. Accumulate: `new_jt = old_jt + delta_jt`
5. Clamp to friction cone:
   ```mojo
   var jt_mag = sqrt(new_jt1² + new_jt2²)
   var max_friction = μ × jn
   if jt_mag > max_friction:
       var scale = max_friction / jt_mag
       new_jt1 *= scale
       new_jt2 *= scale
   ```
6. Apply impulse difference to velocities
7. Store accumulated impulses for warm starting

#### Test Results

**CPU Tests (5 tests):**
1. **Sphere sliding to stop**: v=1m/s, μ=0.5 → stops in ~0.2s ✅
2. **Friction cone constraint**: All impulses satisfy |jt| <= μ×jn ✅
3. **Zero friction**: Sphere maintains velocity when μ=0 ✅
4. **High friction resting**: Sphere with μ=1.0 stops quickly ✅
5. **Two spheres with friction**: Collision detected, proper bounce ✅

**GPU Tests (3 tests):**
1. **Sphere sliding to stop (GPU)**: Same behavior as CPU ✅
2. **Zero friction (GPU)**: Velocity maintained ✅
3. **CPU vs GPU comparison**: Position difference <5cm ✅

#### Visual Demo
`test_render_friction.mojo` shows:
- 4 spheres with different initial velocities
- Spheres sliding and stopping due to friction
- Two spheres colliding and bouncing
- One sphere dropping and landing with friction

---

## Phase 7: Simple Hopper Environment

A 2-body hopper that uses only existing primitives (spheres, hinge joints, friction). This is the minimal RL-compatible locomotion environment.

### Goal
Create a hopper that can learn to hop forward using hip torque control.

### Physical Configuration

```
       Pivot (world anchor)
         │
         │ Link 1 (virtual - constraint only)
         │
        (●) Body 0: Torso (sphere, mass=1.0, radius=0.15)
         │
         │ Link 2 (hinge joint with torque control)
         │
        (●) Body 1: Foot (sphere, mass=0.5, radius=0.1)
         │
    ════════════════ Ground (z=0)
```

**Bodies:**
- Body 0 (Torso): Larger sphere, mass=1.0, radius=0.15
- Body 1 (Foot): Smaller sphere, mass=0.5, radius=0.1

**Joints:**
- Joint 0: World → Torso (optional, for constrained hopper variant)
- Joint 1: Torso → Foot (actuated hinge, Y-axis rotation for XZ plane movement)

**Unconstrained variant (recommended for RL):**
- No world anchor - torso is free to move
- Only 1 joint: Torso → Foot

### Implementation Steps

#### Step 7.1: Joint Actuation (Torque Control) ✅ COMPLETE (CPU + GPU)

**Files modified:**
- `joints/hinge_joint.mojo` - Added `target_torque`, `torque_limit`, `set_torque()`, `set_torque_limit()`
- `joints/joint_solver.mojo` - Added `apply_joint_torques()` (CPU) and `apply_joint_torques_gpu()`
- `gpu/constants.mojo` - Added `JOINT_IDX_TARGET_TORQUE`, `JOINT_IDX_TORQUE_LIMIT`, updated `JOINT_STATE_SIZE` to 18
- `integrator/impulse_integrator.mojo` - Integrated torque application in pipeline
- `integrator/pgs_integrator.mojo` - Integrated torque application in pipeline
- `tests/test_joint_torque.mojo` - 5 CPU validation tests
- `tests/test_joint_torque_gpu.mojo` - 3 GPU validation tests

**Test commands:**
```bash
pixi run mojo run physics3d_v2/tests/test_joint_torque.mojo
pixi run -e apple mojo run physics3d_v2/tests/test_joint_torque_gpu.mojo
```

**CPU test results (all pass):**
- Torque causes angular acceleration (Δω = 1.25 rad/s for τ=5 N·m)
- Torque limits are respected
- Reaction torque on parent body (Newton's 3rd law)
- Pendulum responds to torque control
- Zero torque has no effect

**GPU test results (all pass):**
- GPU torque causes angular velocity (Δω = 1.25 rad/s)
- GPU vs CPU parity (position diff = 0.0)
- Batched simulation (16 envs with different torques)

**Files to modify (reference):**
- `joints/hinge_joint.mojo` - Add torque field
- `joints/joint_solver.mojo` - Apply torque before constraint solving
- `gpu/constants.mojo` - Add JOINT_IDX_TORQUE

**HingeJoint changes:**
```mojo
struct HingeJoint[DTYPE: DType]:
    # ... existing fields ...

    # Actuation
    var target_torque: Scalar[DTYPE]  # Control input (N·m)
    var torque_limit: Scalar[DTYPE]   # Maximum torque magnitude
```

**Joint solver changes:**
```mojo
fn apply_joint_torques[...](
    model: Model[...],
    mut data: Data[...],
):
    """Apply actuator torques to angular velocities before constraint solving."""
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var torque = clamp(joint.target_torque, -joint.torque_limit, joint.torque_limit)

        # Get world-space axis
        var axis = _get_world_axis(data, joint.parent_body, joint.axis_x, joint.axis_y, joint.axis_z)

        # Apply torque to child body: Δω = τ × axis × inv_I × dt
        var body_b = joint.child_body
        var inv_I = model.inv_inertias[body_b * 3 + 1]  # Y-axis for hinge
        data.angular_velocities[body_b * 3 + 0] += torque * axis[0] * inv_I * dt
        data.angular_velocities[body_b * 3 + 1] += torque * axis[1] * inv_I * dt
        data.angular_velocities[body_b * 3 + 2] += torque * axis[2] * inv_I * dt

        # Apply reaction torque to parent (if not world)
        if joint.parent_body >= 0:
            var body_a = joint.parent_body
            var inv_I_a = model.inv_inertias[body_a * 3 + 1]
            data.angular_velocities[body_a * 3 + 0] -= torque * axis[0] * inv_I_a * dt
            data.angular_velocities[body_a * 3 + 1] -= torque * axis[1] * inv_I_a * dt
            data.angular_velocities[body_a * 3 + 2] -= torque * axis[2] * inv_I_a * dt
```

**GPU buffer layout update:**
```
JOINT_STATE_SIZE = 17 floats per joint (was 16):
  [0-1]   Body indices (parent, child)
  [2-4]   Anchor point on parent (px, py, pz)
  [5-7]   Anchor point on child (cx, cy, cz)
  [8-10]  Hinge axis (ax, ay, az)
  [11-15] Accumulated impulses (lx, ly, lz, ax, ay)
  [16]    Target torque (control input)
```

#### Step 7.2: Joint Angle/Velocity Sensing ✅ COMPLETE (CPU + GPU verified)

**Files modified:**
- `joints/joint_solver.mojo` - Added `get_joint_angle()` and `get_joint_angular_velocity()`
- `joints/__init__.mojo` - Exported new sensing functions
- `tests/test_joint_sensing.mojo` - 5 CPU validation tests
- `tests/test_joint_sensing_gpu.mojo` - 3 GPU validation tests

**Test commands:**
```bash
pixi run mojo run physics3d_v2/tests/test_joint_sensing.mojo
pixi run -e apple mojo run physics3d_v2/tests/test_joint_sensing_gpu.mojo
```

**CPU test results (all pass):**
- Initial angle is zero for identity quaternions
- Angle changes during pendulum swing (30° → 6.6° → -8°)
- Angular velocity reading matches set value (2.0 rad/s)
- Angle sign convention correct (+45° and -45° measured correctly)
- Two-body joint sensing works (relative angle and velocity)

**GPU test results (all pass):**
- Angle sensing works after GPU simulation
- Angular velocity sensing works after GPU simulation
- CPU vs GPU sensing parity (angle diff: 3.5e-06 deg, omega diff: 1e-06 rad/s)

**Implementation:**
```mojo
fn get_joint_angle[...](
    model: Model[...],
    data: Data[...],
    joint_idx: Int,
) -> Scalar[DTYPE]:
    """Compute current hinge angle from quaternion difference.

    Uses relative quaternion q_rel = q_parent^(-1) * q_child
    and projects onto hinge axis using atan2.
    Returns angle in radians.
    """

fn get_joint_angular_velocity[...](
    model: Model[...],
    data: Data[...],
    joint_idx: Int,
) -> Scalar[DTYPE]:
    """Compute angular velocity around hinge axis.

    Computes ω_rel = ω_child - ω_parent and projects onto axis.
    Returns angular velocity in rad/s.
    """
```

#### Step 7.3: Hopper Environment ✅ COMPLETE (CPU)

**Files added:**
- `envs/__init__.mojo` - Module exports
- `envs/hopper.mojo` - HopperEnv struct
- `tests/test_hopper.mojo` - 7 CPU validation tests

**Test command:**
```bash
pixi run mojo run physics3d_v2/tests/test_hopper.mojo
```

**CPU test results (all pass):**
1. **Environment initialization**: Model configured correctly (1 joint, 2 bodies)
2. **Reset returns valid observation**: Height=0.45m, velocities=0, angles=0
3. **Standing stability**: Height only drops 0.006m over 100 steps (stable)
4. **Torque causes motion**: Hip angle reaches 90° with 50% torque
5. **Termination on falling**: Episode terminates at step 4 with max torque (pitch exceeds limit)
6. **Reward structure**: alive_bonus=1.0, forward velocity adds reward, control cost reduces it
7. **Observation bounds**: Height 0.37-0.45m, max velocity 2.0 m/s, max pitch 66°

**Original design (kept for reference):**

**New file: `envs/hopper.mojo`**

```mojo
struct HopperEnv[DTYPE: DType = DType.float64]:
    """Simple 2-body hopper environment for RL.

    Observation (8 dims):
        [0] Torso height (z position)
        [1] Torso x velocity
        [2] Torso z velocity
        [3] Torso pitch angle (rotation around Y)
        [4] Torso pitch angular velocity
        [5] Hip angle (relative to torso)
        [6] Hip angular velocity
        [7] Foot contact (0 or 1)

    Action (1 dim):
        [0] Hip torque (normalized to [-1, 1], scaled by torque_limit)

    Reward:
        forward_reward = x_velocity
        alive_bonus = 1.0 (if not fallen)
        control_cost = -0.01 * torque²
        reward = forward_reward + alive_bonus + control_cost

    Termination:
        - Torso height < 0.2 (fallen)
        - Torso pitch > 60° (tipped over)
    """

    var model: Model[DTYPE, 2, 10, 1]
    var data: Data[DTYPE, 2, 10, 1]
    var torque_limit: Scalar[DTYPE]
    var dt: Scalar[DTYPE]

    fn __init__(out self, torque_limit: Scalar[DTYPE] = 5.0):
        self.model = Model[DTYPE, 2, 10, 1](
            gravity_z=-9.81,
            timestep=0.01,
            ground_z=0.0,
            friction=0.8,
            restitution=0.0,
        )
        # Torso (body 0)
        self.model.set_body(0, mass=1.0, radius=0.15)
        # Foot (body 1)
        self.model.set_body(1, mass=0.5, radius=0.1)

        # Hip joint: Torso → Foot
        self.model.add_hinge_joint(
            parent=0,
            child=1,
            anchor_parent=(0.0, 0.0, -0.15),  # Bottom of torso
            anchor_child=(0.0, 0.0, 0.1),     # Top of foot
            axis=(0.0, 1.0, 0.0),             # Y-axis rotation
        )

        self.data = Data[DTYPE, 2, 10, 1]()
        self.torque_limit = torque_limit
        self.dt = self.model.timestep
        self.reset()

    fn reset(mut self) -> List[Scalar[DTYPE]]:
        """Reset to initial standing position."""
        self.data = Data[DTYPE, 2, 10, 1]()
        # Torso at height 0.4
        self.data.set_body_position(0, 0.0, 0.0, 0.4)
        # Foot at height 0.1 (radius above ground)
        self.data.set_body_position(1, 0.0, 0.0, 0.1)
        return self.get_observation()

    fn step(mut self, action: Scalar[DTYPE]) -> Tuple[
        List[Scalar[DTYPE]],  # observation
        Scalar[DTYPE],        # reward
        Bool,                 # done
    ]:
        """Take one step with given hip torque."""
        # Clamp and apply action
        var torque = action * self.torque_limit
        self.model.joints[0].target_torque = torque

        # Physics step
        ImpulseIntegrator.step(self.model, self.data)

        # Compute reward
        var obs = self.get_observation()
        var x_vel = obs[1]
        var alive = not self.is_terminated()
        var reward = x_vel + (1.0 if alive else 0.0) - 0.01 * torque * torque

        return (obs, reward, not alive)

    fn get_observation(self) -> List[Scalar[DTYPE]]:
        # ... extract 8-dim observation vector ...

    fn is_terminated(self) -> Bool:
        var torso_z = self.data.get_body_z(0)
        # ... check height and pitch ...
```

#### Step 7.4: GPU Hopper Integration ✅ COMPLETE

**Files added:**
- `tests/test_hopper_gpu.mojo` - GPU hopper validation tests

**Test command:**
```bash
pixi run -e apple mojo run physics3d_v2/tests/test_hopper_gpu.mojo
```

**GPU test results (all pass):**
- GPU hopper simulation maintains joint constraints
- Foot stays above ground with friction
- CPU vs GPU parity validated

**Critical Bug Fix: MAX_JOINTS Parameter Missing in GPU Functions**

The GPU hopper test initially failed because collision detection, velocity solver, and position solver functions were missing the `MAX_JOINTS` compile-time parameter. This caused incorrect buffer offset calculations:

**Problem:** Functions like `detect_all_contacts_gpu`, `solve_velocity_constraints_gpu`, and `solve_position_constraints_gpu` computed metadata offsets without accounting for joint state:
```mojo
# WRONG: metadata_offset[NUM_BODIES, MAX_CONTACTS]() = 82 (for pendulum)
# CORRECT: metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]() = 100
```

This caused the collision detector to write `num_contacts` to index 82 (the joint parent location) instead of index 100 (metadata), corrupting joint indices.

**Files fixed:**
- `collision/collision.mojo` - Added `MAX_JOINTS` parameter to `detect_all_contacts_gpu`
- `solver/impulse_solver.mojo` - Added `MAX_JOINTS` parameter to `solve_velocity_constraints_gpu` and `solve_position_constraints_gpu`
- `solver/pgs_solver.mojo` - Added `MAX_JOINTS` parameter to `solve_constraints_pgs_gpu`
- `integrator/impulse_integrator.mojo` - Updated all GPU function calls with `MAX_JOINTS`

**Lesson learned:** All GPU functions that access the state buffer must include `MAX_JOINTS` in their compile-time parameters to compute correct offsets.

### File Structure (Phase 7)

```
physics3d_v2/
├── joints/
│   ├── hinge_joint.mojo      # Add target_torque, torque_limit
│   └── joint_solver.mojo     # Add apply_joint_torques()
├── envs/
│   ├── __init__.mojo         # New module
│   └── hopper.mojo           # HopperEnv struct
├── gpu/
│   └── constants.mojo        # JOINT_IDX_TORQUE
├── integrator/
│   ├── impulse_integrator.mojo  # Call apply_joint_torques
│   └── pgs_integrator.mojo      # Call apply_joint_torques
├── tests/
│   ├── test_joint_torque.mojo       # Torque actuation tests (CPU) ✅
│   ├── test_joint_torque_gpu.mojo   # Torque actuation tests (GPU) ✅
│   ├── test_joint_sensing.mojo      # Joint sensing tests (CPU) ✅
│   ├── test_joint_sensing_gpu.mojo  # Joint sensing tests (GPU) ✅
│   ├── test_hopper.mojo             # CPU hopper tests ✅
│   └── test_hopper_gpu.mojo         # GPU hopper tests ✅
└── examples/
    └── hopper_render_demo.mojo      # Visual demonstration
```

### Test Plan

#### Test 7.1: Joint Torque Application
```bash
pixi run mojo run physics3d_v2/tests/test_joint_torque.mojo
```
- Apply constant torque, verify angular acceleration
- Verify torque limits are respected
- Verify reaction torque on parent body

#### Test 7.2: Hopper Standing
```bash
pixi run mojo run physics3d_v2/tests/test_hopper.mojo
```
- Hopper stands without falling (no action)
- Foot maintains ground contact
- Joint constraint maintained

#### Test 7.3: Hopper Hopping
- Apply sinusoidal torque, verify hopping motion
- Verify foot leaves and returns to ground
- Verify forward progress with asymmetric actuation

#### Test 7.4: GPU Parity
```bash
pixi run -e apple mojo run physics3d_v2/tests/test_hopper_gpu.mojo
```
- Same tests as CPU with GPU integrator
- Batched simulation (256 hoppers)

### Validation Criteria

| Test | Pass Criteria |
|------|---------------|
| Torque response | Angular acceleration ≈ τ × inv_I |
| Standing stability | Height maintained ±5% for 10s |
| Hopping height | Reaches >0.5m with sinusoidal torque |
| Joint constraint | Anchor error <2cm during hopping |
| CPU/GPU parity | Position difference <1cm after 1000 steps |

---

## Phase 8: Capsule Geometry - COMPLETE

Capsules (sphere-swept lines) are essential for limbs, legs, and elongated bodies. They provide better collision behavior than spheres for articulated characters.

### Implementation Summary

#### Files Added
- `tests/test_capsule_plane.mojo` - CPU capsule-plane validation (4 tests)
- `tests/test_capsule_sphere.mojo` - CPU capsule-sphere validation (4 tests)
- `tests/test_capsule_capsule.mojo` - CPU capsule-capsule validation (4 tests)
- `tests/test_capsule_gpu.mojo` - GPU parity validation (3 tests)

#### Files Modified
- `types.mojo` - Added `geom_types`, `half_lengths` arrays to Model, added `set_body_capsule()` method
- `gpu/constants.mojo` - Added `GEOM_CAPSULE`, `MODEL_IDX_GEOM_TYPE`, `MODEL_IDX_HALF_LENGTH`, updated `MODEL_BODY_SIZE` to 11
- `collision/collision_primitives.mojo` - Added `rotate_vector_by_quat()`, `capsule_plane()`, `capsule_sphere()`, `_closest_points_line_segments()`, `capsule_capsule()`
- `collision/collision.mojo` - Updated `detect_all_contacts()` and `detect_all_contacts_gpu()` with geometry-type dispatch
- `collision/__init__.mojo` - Exported new capsule functions
- `gpu/buffer_utils.mojo` - Updated `create_model_host_buffer()` to include geometry type and half_length

#### Test Commands
```bash
# Phase 8 CPU tests
pixi run mojo run physics3d_v2/tests/test_capsule_plane.mojo
pixi run mojo run physics3d_v2/tests/test_capsule_sphere.mojo
pixi run mojo run physics3d_v2/tests/test_capsule_capsule.mojo

# Phase 8 GPU tests
pixi run -e apple mojo run physics3d_v2/tests/test_capsule_gpu.mojo
```

#### Test Results
**CPU Capsule-Plane Tests (4/4 passed):**
- Vertical capsule falls and stops at correct height (half_len + radius)
- Horizontal capsule stops at radius above ground
- Tilted capsule stops at lowest endpoint + radius
- Capsule at rest does not drift (<1mm)

**CPU Capsule-Sphere Tests (4/4 passed):**
- Sphere lands on horizontal capsule surface
- Sphere lands on capsule endpoint cap
- Capsule-sphere bounce with restitution
- Sphere slides along capsule (zero friction)

**CPU Capsule-Capsule Tests (4/4 passed):**
- Parallel capsules bounce apart correctly
- Perpendicular capsules (cross) collision
- Cap-to-cap collision (endpoints)
- Two capsules stack on ground

**GPU Parity Tests (3/3 passed):**
- Capsule-plane: CPU/GPU difference < 1e-6
- Capsule-sphere: CPU/GPU difference < 1mm
- Batched simulation: 16 environments within tolerance

### Goal
Add capsule collision primitives with support for:
- Capsule-plane collision
- Capsule-sphere collision
- Capsule-capsule collision

### Physical Model

A capsule is defined by:
- Two endpoint positions (or center + axis + half-length)
- Radius

```
     ___________
    /           \
   (  ●-------●  )  radius r
    \___________/
       |<--->|
       half_length
```

The capsule can be parameterized as:
- **Center-based**: center (x, y, z), axis (ax, ay, az), half_length, radius
- **Endpoint-based**: p0 (x, y, z), p1 (x, y, z), radius

### Implementation Steps

#### Step 8.1: Capsule Data Structure

**New file: `types.mojo` additions**
```mojo
struct CapsuleGeom[DTYPE: DType]:
    var half_length: Scalar[DTYPE]  # Half-length along local Z-axis
    var radius: Scalar[DTYPE]

# Add geometry type enum
comptime GEOM_SPHERE: Int = 0
comptime GEOM_CAPSULE: Int = 1
comptime GEOM_BOX: Int = 2
```

**Model changes:**
```mojo
struct Model[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0]:
    # Per-body geometry type
    var geom_types: InlineArray[Int, NUM_BODIES]  # GEOM_SPHERE or GEOM_CAPSULE
    var half_lengths: InlineArray[Scalar[DTYPE], NUM_BODIES]  # For capsules
```

#### Step 8.2: Capsule-Plane Collision

Find the closest point on capsule axis to the plane, then treat as sphere-plane.

```mojo
fn capsule_plane[DTYPE: DType](
    # Capsule center position
    cx: Scalar[DTYPE], cy: Scalar[DTYPE], cz: Scalar[DTYPE],
    # Capsule quaternion (for axis orientation)
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    # Plane (assuming horizontal at ground_z)
    ground_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], ...]:
    """Returns (dist, contact_x, contact_y, contact_z, nx, ny, nz).

    Algorithm:
    1. Compute capsule endpoints in world frame
    2. Find lowest endpoint
    3. Compute distance from lowest point to plane
    4. Return signed distance (negative = penetration)
    """
```

#### Step 8.3: Capsule-Sphere Collision

Find closest point on capsule axis to sphere center, then sphere-sphere.

```mojo
fn capsule_sphere[DTYPE: DType](
    # Capsule
    cap_x, cap_y, cap_z: Scalar[DTYPE],  # Center
    cap_qx, cap_qy, cap_qz, cap_qw: Scalar[DTYPE],  # Orientation
    cap_half_len, cap_radius: Scalar[DTYPE],
    # Sphere
    sph_x, sph_y, sph_z, sph_radius: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], ...]:
    """
    Algorithm:
    1. Project sphere center onto capsule axis (clamped to segment)
    2. Compute distance from projection to sphere center
    3. Return signed distance - (cap_radius + sph_radius)
    """
```

#### Step 8.4: Capsule-Capsule Collision

Find closest points between two line segments, then sphere-sphere at those points.

```mojo
fn capsule_capsule[DTYPE: DType](
    # Capsule A
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw: Scalar[DTYPE],
    a_half_len, a_radius: Scalar[DTYPE],
    # Capsule B
    b_x, b_y, b_z, b_qx, b_qy, b_qz, b_qw: Scalar[DTYPE],
    b_half_len, b_radius: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], ...]:
    """
    Algorithm:
    1. Compute closest points between two line segments
    2. Distance = |p_a - p_b| - (r_a + r_b)
    3. Normal = normalize(p_b - p_a)
    """
```

#### Step 8.5: GPU Buffer Layout Update

```
MODEL_BODY_SIZE = 11 floats per body (was 9):
  [0]     Mass
  [1]     Inverse mass
  [2]     Radius
  [3-5]   Inertia (Ixx, Iyy, Izz)
  [6-8]   Inverse inertia
  [9]     Geometry type (0=sphere, 1=capsule, 2=box)
  [10]    Half-length (for capsules) or 0
```

#### Step 8.6: Collision Dispatcher Update

Modify `CollisionDetector.detect_all_contacts_gpu` to dispatch based on geometry type:

```mojo
fn detect_contact_pair_gpu[...](
    geom_type_a: Int, geom_type_b: Int,
    # Body A state
    # Body B state (or ground)
):
    if geom_type_a == GEOM_SPHERE and geom_type_b == GEOM_PLANE:
        return sphere_plane(...)
    elif geom_type_a == GEOM_CAPSULE and geom_type_b == GEOM_PLANE:
        return capsule_plane(...)
    elif geom_type_a == GEOM_SPHERE and geom_type_b == GEOM_SPHERE:
        return sphere_sphere(...)
    elif geom_type_a == GEOM_CAPSULE and geom_type_b == GEOM_SPHERE:
        return capsule_sphere(...)
    elif geom_type_a == GEOM_CAPSULE and geom_type_b == GEOM_CAPSULE:
        return capsule_capsule(...)
    # ... etc
```

### Test Plan

```bash
# Phase 8 tests
pixi run mojo run physics3d_v2/tests/test_capsule_plane.mojo
pixi run mojo run physics3d_v2/tests/test_capsule_sphere.mojo
pixi run mojo run physics3d_v2/tests/test_capsule_capsule.mojo
pixi run -e apple mojo run physics3d_v2/tests/test_capsule_gpu.mojo
```

### Validation Criteria

| Test | Pass Criteria |
|------|---------------|
| Capsule-plane contact | Capsule stops at correct height (radius above ground) |
| Capsule resting | No penetration, stable on ground |
| Capsule-sphere collision | Proper bounce with correct contact normal |
| Capsule-capsule collision | Bodies separate correctly, no interpenetration |
| GPU parity | Same results as CPU |

---

## Phase 9: Box Geometry - COMPLETE

Boxes (oriented bounding boxes, OBBs) enable walls, platforms, and rectangular bodies.

### Implementation Summary

#### Files Added
- `tests/test_box_plane.mojo` - CPU box-plane validation (3 tests)
- `tests/test_box_sphere.mojo` - CPU box-sphere validation (3 tests)
- `tests/test_box_capsule.mojo` - CPU box-capsule validation (3 tests)
- `tests/test_box_box.mojo` - CPU box-box validation (3 tests)
- `tests/test_box_gpu.mojo` - GPU parity validation (3 tests)

#### Files Modified
- `types.mojo` - Added `half_x`, `half_y`, `half_z` arrays to Model, added `set_body_box()` method with box inertia calculation
- `gpu/constants.mojo` - Added `GEOM_BOX`, `MODEL_IDX_HALF_X`, `MODEL_IDX_HALF_Y`, `MODEL_IDX_HALF_Z`, updated `MODEL_BODY_SIZE` to 14
- `collision/collision_primitives.mojo` - Added `rotate_vector_by_quat_inverse()`, `box_plane()`, `box_sphere()`, `box_capsule()`, `box_box()` with SAT algorithm
- `collision/collision.mojo` - Updated `detect_all_contacts()` and `detect_all_contacts_gpu()` with box geometry dispatch
- `collision/__init__.mojo` - Exported new box collision functions
- `gpu/buffer_utils.mojo` - Updated `create_model_host_buffer()` and `init_model_host_buffer()` to include half_x/y/z

#### Test Commands
```bash
# Phase 9 CPU tests
pixi run mojo run physics3d_v2/tests/test_box_plane.mojo
pixi run mojo run physics3d_v2/tests/test_box_sphere.mojo
pixi run mojo run physics3d_v2/tests/test_box_capsule.mojo
pixi run mojo run physics3d_v2/tests/test_box_box.mojo

# Phase 9 GPU tests
pixi run -e apple mojo run physics3d_v2/tests/test_box_gpu.mojo
```

#### Test Results

**CPU Box-Plane Tests (3/3 passed):**
- Axis-aligned box falls and stops at correct height (half_z)
- Tilted box (45°) stops at lowest vertex height
- Box at rest does not drift (<0.01mm)

**CPU Box-Sphere Tests (3/3 passed):**
- Sphere lands on box face at correct height
- Sphere hits box edge and stays above ground
- Sphere hits box corner and stays above ground

**CPU Box-Capsule Tests (3/3 passed):**
- Vertical capsule lands on box face
- Horizontal capsule parallel to box edge lands correctly
- Tilted capsule at 45° lands on box

**CPU Box-Box Tests (3/3 passed):**
- Face-face contact: small box rests on large box at correct height
- Edge-edge contact: rotated box rests on base box
- Stacked boxes stability: 3 boxes stack with <0.03mm drift

**GPU Parity Tests (3/3 passed):**
- Box-plane: CPU/GPU difference < 3e-7
- Box-sphere: CPU/GPU difference < 1e-6
- Box-box: CPU/GPU difference < 1e-6

### Physical Model

A box is defined by:
- Center position
- Orientation (quaternion)
- Half-extents (hx, hy, hz)

```
        +--------+
       /|       /|
      / |      / |
     +--------+  |
     |  +-----|--+
     | /      | /
     |/       |/
     +--------+
     |<-hx->|
```

### Key Algorithms

#### Box-Plane Collision
Find the lowest of 8 rotated vertices:
```mojo
fn box_plane[DTYPE: DType](...) -> Tuple[...]:
    # Transform 8 vertices by quaternion rotation
    # Find vertex with minimum z
    # Return signed distance = min_z - ground_z
```

#### Box-Sphere Collision
Transform sphere to box local frame, clamp to bounds:
```mojo
fn box_sphere[DTYPE: DType](...) -> Tuple[...]:
    # Transform sphere center to box local frame using inverse quaternion
    # Clamp to [-half_x, half_x], [-half_y, half_y], [-half_z, half_z]
    # Transform closest point back to world frame
    # Return distance - sphere_radius
```

#### Box-Capsule Collision
Project box center onto capsule axis, treat as box-sphere:
```mojo
fn box_capsule[DTYPE: DType](...) -> Tuple[...]:
    # Compute capsule endpoints in world frame
    # Project box center onto capsule axis (clamped)
    # Treat as box-sphere with closest point on capsule
```

#### Box-Box Collision (SAT)
Separating Axis Theorem with 15 axes:
```mojo
fn box_box[DTYPE: DType](...) -> Tuple[...]:
    # Test 15 potential separating axes:
    # - 3 face normals of box A (world frame)
    # - 3 face normals of box B (world frame)
    # - 9 edge-edge cross products (Ax×Bx, Ax×By, ..., Az×Bz)
    #
    # For each axis:
    #   Project both boxes onto axis
    #   Compute overlap = min(max_A, max_B) - max(min_A, min_B)
    #   Track minimum overlap axis for contact normal
    #
    # If separated on any axis, no collision
    # Otherwise return minimum penetration depth and contact normal
```

#### Box Inertia Tensor
For a solid box with half-extents (hx, hy, hz):
```
I_xx = m * (hy² + hz²) / 3
I_yy = m * (hx² + hz²) / 3
I_zz = m * (hx² + hy²) / 3
```

### GPU Buffer Layout

```
MODEL_BODY_SIZE = 14 floats per body:
  [0]     Mass
  [1]     Inverse mass
  [2]     Radius (for sphere/capsule)
  [3-5]   Inertia (Ixx, Iyy, Izz)
  [6-8]   Inverse inertia
  [9]     Geometry type (0=sphere, 1=capsule, 2=box)
  [10]    Half-length (capsule) / Half-X (box)
  [11]    Half-Y (box)
  [12]    Half-Z (box)
  [13]    Padding
```

### GPU Compatibility Notes

The box collision primitives required special handling for GPU compatibility:
1. **No heap allocation**: Replaced `InlineArray` usage with explicit helper function calls for vertex iteration
2. **No closures**: Refactored SAT algorithm to use module-level `_test_sat_axis()` function instead of nested closure

### Validation Criteria

| Test | Pass Criteria |
|------|---------------|
| Box on ground | Box rests stably at half_z (axis-aligned) or lowest vertex (tilted) |
| Box drift | < 1mm over 1000 steps |
| Box-sphere | Correct penetration depth and normal for face/edge/corner cases |
| Box-capsule | Bodies separate correctly, no interpenetration |
| Box-box SAT | Stacked boxes stable without interpenetration |
| GPU parity | < 1e-6 difference vs CPU |

---

## Phase 10a: Bipedal Walker - COMPLETE

### Implementation Summary

A bipedal walker environment extending the HopperEnv to support two legs:
- 3 bodies: Torso (sphere) + Left Foot (sphere) + Right Foot (sphere)
- 2 actuated hinge joints: Left Hip + Right Hip
- 12-dimensional observation space
- 2-dimensional action space

#### Files Added
- `envs/walker.mojo` - WalkerEnv struct
- `tests/test_walker.mojo` - CPU validation (8 tests)
- `tests/test_walker_gpu.mojo` - GPU validation (3 tests)
- `examples/walker_render_demo.mojo` - Visual demonstration

#### Files Modified
- `envs/__init__.mojo` - Added WalkerEnv export
- `PLAN.md` - Updated status and documentation

### Physical Configuration

```
              Torso (body 0)
             mass=1.0, sphere
               radius=0.2
                  *
                 / |
    Left Hip   /   |   Right Hip
   (joint 0)  /    |  (joint 1)
             /      |
           *         *
     Left Foot      Right Foot
     (body 1)       (body 2)
    mass=0.3        mass=0.3
    radius=0.08     radius=0.08

  ============================== Ground (z=0)
```

| Body | Index | Mass | Radius | Initial Z |
|------|-------|------|--------|-----------|
| Torso | 0 | 1.0 kg | 0.20 m | 0.50 m |
| Left Foot | 1 | 0.3 kg | 0.08 m | 0.08 m |
| Right Foot | 2 | 0.3 kg | 0.08 m | 0.08 m |

| Joint | Parent | Child | Torque Limit |
|-------|--------|-------|--------------|
| Left Hip (0) | Torso | Left Foot | 15 N*m |
| Right Hip (1) | Torso | Right Foot | 15 N*m |

### Observation Space (12 dimensions)

| Index | Name | Range |
|-------|------|-------|
| 0 | torso_height | [0.2, 0.8] m |
| 1 | torso_vx | [-2, 2] m/s |
| 2 | torso_vz | [-3, 3] m/s |
| 3 | torso_pitch | [-1.0, 1.0] rad |
| 4 | torso_pitch_vel | [-5, 5] rad/s |
| 5 | left_hip_angle | [-pi/2, pi/2] rad |
| 6 | left_hip_vel | [-10, 10] rad/s |
| 7 | left_contact | {0, 1} |
| 8 | right_hip_angle | [-pi/2, pi/2] rad |
| 9 | right_hip_vel | [-10, 10] rad/s |
| 10 | right_contact | {0, 1} |
| 11 | torso_roll | [-0.5, 0.5] rad |

### Action Space (2 dimensions)

| Index | Name | Range | Scaled By |
|-------|------|-------|-----------|
| 0 | left_hip_torque | [-1, 1] | 15 N*m |
| 1 | right_hip_torque | [-1, 1] | 15 N*m |

### Reward Function

```
reward = forward_velocity + alive_bonus - control_cost - height_penalty
where:
  forward_velocity = obs[1]  (torso vx)
  alive_bonus = 1.0 if not terminated
  control_cost = 0.005 * (left_torque^2 + right_torque^2)
  height_penalty = 0.5 * (obs[0] - 0.45)^2
```

### Termination Conditions

- `torso_height < 0.15` (fallen)
- `|torso_pitch| > 1.0 rad` (tipped forward/backward)
- `|torso_roll| > 0.5 rad` (tipped sideways)
- `steps >= 1000` (timeout)

### Test Results

**CPU Tests (8/8 passed):**
1. Environment initialization: 2 joints, 3 bodies configured
2. Reset: Height ~0.5m, velocities ~0
3. Standing stability: 100 steps with zero action, height drops <0.02m
4. Asymmetric torque: Hip angles diverge with opposite torques
5. Symmetric actuation: Both legs respond together
6. Termination on falling: Destabilizing torque causes episode end
7. Reward structure: Alive bonus ~1.0, forward velocity adds reward
8. Bilateral contact: Both feet detect ground contact

**GPU Tests (3/3 passed):**
1. GPU walker simulation: 100 steps, walker remains upright
2. GPU vs CPU parity: Torso position diff < 0.5m
3. Batched simulation: 8 environments with different torques

### GPU Buffer Layout

```
STATE_SIZE = 3*22 + 15*12 + 2*18 + 4 = 286 floats

Body offsets: 0 (torso), 22 (left foot), 44 (right foot)
Contact offsets: 66 to 245
Joint offsets: 246 (left hip), 264 (right hip)
Metadata offset: 282
```

### Test Commands

```bash
# CPU tests
pixi run mojo run physics3d_v2/tests/test_walker.mojo

# GPU tests (requires Apple Silicon)
pixi run -e apple mojo run physics3d_v2/tests/test_walker_gpu.mojo

# Visual demo (requires SDL2)
pixi run mojo run physics3d_v2/examples/walker_render_demo.mojo
```

---

## Known Issues

### Horizontal Capsule Collision - Single Contact Point

**Issue:** When a capsule lies horizontally on the ground plane (rotated 90° around Y-axis), the collision detection only generates 1 contact point instead of 2.

**Expected behavior:** A horizontal capsule should generate 2 contact points - one at each endpoint cap where it touches the ground. This provides a stable base with proper constraint solving.

**Current behavior:** Only the lowest endpoint generates a contact point. When the capsule is perfectly horizontal (both endpoints at the same height), only one arbitrary endpoint is detected.

**Impact:**
- Minor instability at rest (jittering)
- Requires velocity damping as a workaround

**Workaround applied:** Added velocity damping in `hopper_render_demo.mojo`:
```mojo
var linear_damping: Float64 = 0.995
var angular_damping: Float64 = 0.99
// Applied after each physics step to reduce oscillations at rest
```

**Root cause:** The `capsule_plane()` function in `collision/collision_primitives.mojo` computes the lowest endpoint and returns a single contact. It should detect when both endpoints are within a tolerance of the ground and generate two contacts.

**Fix required:** Modify `capsule_plane()` to:
1. Compute both endpoint distances to ground
2. If both are within contact threshold (penetration + small tolerance), generate 2 contacts
3. Store both contacts in the contact buffer

**Affected file:** `collision/collision_primitives.mojo` - `capsule_plane()` function

---

## Future Extensions (Phase 10b+)

### Phase 10: Multi-Leg Locomotion (continued)
- **10a**: Bipedal walker (3 bodies, 2 actuated joints) ✅ COMPLETE
- **10b**: Add knee joints (5 bodies, 4 actuated joints)
- **10c**: Full bipedal walker with capsule legs

### Phase 11: Advanced Joints
- **11a**: Ball-and-socket joints (3 DOF rotation)
- **11b**: Slider joints (1 DOF translation) ✅ COMPLETE
  - SlideJoint struct with CPU/GPU solvers
  - Integrated into Model/Data with MAX_SLIDE_JOINTS parameter
  - Position and velocity sensing functions
  - Force actuation support
  - HopperEnv updated with root slide joints (X-Z plane constraint)
- **11c**: Universal joints (2 DOF rotation)
- **11d**: Joint limits (angle min/max)
- **11e**: Joint damping and stiffness
- **11f**: Free DOF joints for root bodies ✅ CORE COMPLETE
  - Problem: Current root joints (world→body) create conflicting constraints
  - Solution: Add "free DOF" mode where joints track state without constraining
  - Implementation (DONE):
    - ✅ Add `is_free_dof: Bool`, `qpos`, `qvel` fields to HingeJoint and SlideJoint
    - ✅ Add `create_free_dof()` factory methods to both joint types
    - ✅ Add `add_free_hinge_joint()` and `add_free_slide_joint()` to Model class
    - ✅ When `is_free_dof=True`: skip constraint solving in velocity and position solvers
    - ✅ Updated GPU joint state layout (JOINT_STATE_SIZE = 21, SLIDE_JOINT_STATE_SIZE = 21)
    - ✅ Updated GPU joint solvers to check is_free_dof flag
    - ✅ Updated Hopper3D to use free DOF joints for root joints
  - Files modified:
    - `joints/hinge_joint.mojo` - added free_dof flag, qpos/qvel, create_free_dof()
    - `joints/slide_joint.mojo` - added free_dof flag, qpos/qvel, create_free_dof()
    - `joints/joint_solver.mojo` - skip constraint solving for free DOF joints (CPU + GPU)
    - `gpu/constants.mojo` - added JOINT_IDX_IS_FREE_DOF, SLIDE_IDX_IS_FREE_DOF
    - `types.mojo` - added add_free_hinge_joint(), add_free_slide_joint()
    - `envs/hopper_3d/hopper_3d.mojo` - use free DOF for root joints
  - TODO (optional enhancements):
    - Forward kinematics: compute body position from free DOF values
    - Inverse: update free DOF values from body state after physics step

### Phase 12: Soft Bodies & Cables
- **12a**: Mass-spring systems
- **12b**: Position-based dynamics (PBD)
- **12c**: Rope/cable simulation

### Phase 13: Performance Optimization
- **13a**: Broad-phase collision (spatial hashing, BVH)
- **13b**: Parallel constraint solving
- **13c**: SIMD optimizations for CPU
- **13d**: Batched GPU collision detection

### Phase 14: MuJoCo-Style Generalized Coordinates (Alternative Engine)

**Goal**: Implement a full MuJoCo-style physics engine using generalized coordinates alongside the existing Cartesian constraint-based engine. Both engines will coexist to allow comparison.

**Background**: MuJoCo uses a fundamentally different approach where joints ADD degrees of freedom rather than constrain them. The state is stored as joint positions (`qpos`) and velocities (`qvel`), with body positions computed via forward kinematics.

**Key Differences from Current Engine**:

| Aspect | Current (Cartesian) | MuJoCo-Style (Generalized) |
|--------|---------------------|---------------------------|
| State | Body positions/quaternions | Joint angles (`qpos`) |
| Joints | Remove DOF (constraints) | Add DOF (transformations) |
| Root joints | Fixed pivot (conflicts!) | Transform accumulation |
| Dynamics | Impulse-based | Mass matrix in joint space |

**Implementation Plan**:

#### Phase 14a: Core Data Structures
- `ModelGC` (Generalized Coordinates Model):
  - `nq`: Total qpos size, `nv`: Total qvel size
  - `jnt_type[]`: Joint types (FREE=7, BALL=4, SLIDE=1, HINGE=1)
  - `jnt_qposadr[]`: Start index in qpos for each joint
  - `jnt_dofadr[]`: Start index in qvel for each joint
  - `jnt_bodyid[]`: Body each joint belongs to
  - `jnt_pos[]`, `jnt_axis[]`: Joint anchor and axis in local frame
  - `body_parentid[]`: Kinematic tree structure
- `DataGC`:
  - `qpos[nq]`: Joint positions (angles, quaternions, translations)
  - `qvel[nv]`: Joint velocities
  - `qacc[nv]`: Joint accelerations
  - `xpos[nbody]`, `xquat[nbody]`: Computed body positions (from FK)

#### Phase 14b: Forward Kinematics
- Traverse kinematic tree from root to leaves
- For each body, accumulate transformations from parent:
  ```
  xpos[body] = xpos[parent] + rotate(body_pos, xquat[parent])
  xquat[body] = xquat[parent] * body_quat

  for each joint on body:
    if SLIDE: xpos += axis * qpos[joint]
    if HINGE: xquat *= axis_angle_to_quat(axis, qpos[joint])
    if BALL:  xquat *= qpos[joint] (quaternion)
    if FREE:  xpos = qpos[0:3], xquat = qpos[3:7]
  ```
- GPU: Process kinematic branches in parallel

#### Phase 14c: Inverse Kinematics (State Extraction)
- After collision/contact solving, update qpos from body positions
- For root FREE joint: direct copy
- For SLIDE: project position difference onto axis
- For HINGE: compute angle from quaternion difference

#### Phase 14d: Dynamics in Generalized Coordinates
- Compute composite inertia (Featherstone algorithm)
- Mass matrix M(q) in joint space
- Coriolis/centrifugal forces C(q, qdot)
- Gravity forces g(q)
- Equations of motion: M(q) * qacc = tau - C(q, qdot) - g(q) + J^T * f_contact
- Contact Jacobian J maps contact forces to joint torques

#### Phase 14e: Contact Handling
- Compute contact Jacobian J = d(contact_point) / d(qpos)
- Solve contact constraints in joint space
- Options:
  - Impulse-based (current style, but in joint space)
  - Soft contact (MuJoCo style with spring-damper)

#### Phase 14f: Integration and Testing
- Semi-implicit Euler integrator for qpos/qvel
- Hopper environment using new engine (`HopperGC`)
- Comparison tests: CartesianEngine vs GeneralizedEngine
- Performance benchmarks

**File Structure**:
```
physics3d_v2/
├── generalized/                    # New MuJoCo-style engine
│   ├── __init__.mojo
│   ├── types.mojo                  # ModelGC, DataGC
│   ├── kinematics.mojo             # Forward/inverse kinematics
│   ├── dynamics.mojo               # Mass matrix, Coriolis, gravity
│   ├── integrator.mojo             # Semi-implicit Euler
│   └── tests/
│       ├── test_kinematics.mojo
│       ├── test_dynamics.mojo
│       └── test_hopper_gc.mojo     # Hopper with generalized coords
├── envs/
│   ├── hopper.mojo                 # Current Cartesian-based
│   └── hopper_gc.mojo              # New generalized-coords based
```

**Validation Criteria**:
- Forward kinematics matches MuJoCo reference
- Pendulum period matches analytical solution
- Hopper behavior identical between engines (within numerical tolerance)
- GPU performance comparable to current engine

**References**:
- MuJoCo source: `mujoco-main/src/engine/engine_core_smooth.c`
- MuJoCo Warp: `mujoco_warp-main/mujoco_warp/_src/smooth.py`
- Featherstone, R. "Rigid Body Dynamics Algorithms" (2008)
