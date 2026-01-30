# Physics Engine Rebuild Plan: Phase 1-2

## Goal
Build a minimal, mathematically correct 3D physics engine following MuJoCo's computational foundations, designed for incremental extension.

## Design Principles

1. **MuJoCo-style separation**: Model (static) vs Data (runtime state)
2. **Mojo trait-based**: Compile-time dimensions, stateless computation graphs
3. **Incremental validation**: Each phase has analytical tests
4. **GPU-ready from start**: But CPU-first for debugging
5. **Parameterized dtype**: `float32` for GPU (Metal), configurable for debugging

---

## Phase 1: Single Free-Falling Body

### Data Structures

```
mojo-rl/physics3d_v2/
├── __init__.mojo
├── constants.mojo      # dtype, TILE, TPB, physics constants
├── types.mojo          # Model, Data, Body structs
├── kinematics.mojo     # Forward kinematics (trivial for free body)
├── dynamics.mojo       # Compute accelerations (just gravity)
├── integrator.mojo     # Semi-implicit Euler
└── tests/
    └── test_freefall.mojo  # Validation script
```

#### constants.mojo - Configuration

```mojo
"""Physics3D v2 constants - minimal rebuild."""

# Parameterized dtype - float32 for Metal GPU, can change to float64 for CPU debugging
comptime dtype = DType.float32

# GPU kernel configuration (same as deep_rl)
comptime TILE: Int = 16   # Optimal for Apple Silicon
comptime TPB: Int = 256   # Threads per block

# Physics defaults
comptime DEFAULT_GRAVITY_Z: Float64 = -9.81
comptime DEFAULT_TIMESTEP: Float64 = 0.01
```

#### types.mojo - Minimal Structures

```mojo
from .constants import dtype

# Following MuJoCo: Model is static config, Data is mutable state

struct Body:
    """Static body properties."""
    var mass: Scalar[dtype]
    var inertia: SIMD[dtype, 4]  # Diagonal inertia [Ixx, Iyy, Izz, 0] (padded)

struct Model:
    """Static simulation configuration."""
    var gravity: SIMD[dtype, 4]   # [gx, gy, gz, 0] (padded for SIMD)
    var timestep: Scalar[dtype]
    var body: Body

struct Data:
    """Mutable simulation state."""
    # Generalized coordinates (FREE joint: 3 pos + 4 quat = 7)
    var qpos: InlineArray[Scalar[dtype], 7]  # [x, y, z, qx, qy, qz, qw]
    var qvel: InlineArray[Scalar[dtype], 6]  # [vx, vy, vz, wx, wy, wz]
    var qacc: InlineArray[Scalar[dtype], 6]  # Computed accelerations

    # World-frame quantities (computed from qpos)
    var xpos: SIMD[dtype, 4]   # [x, y, z, 0] padded
    var xquat: SIMD[dtype, 4]  # [qx, qy, qz, qw]

    # Forces
    var qfrc_applied: InlineArray[Scalar[dtype], 6]  # External forces/torques
```

#### dynamics.mojo - Acceleration Computation

```mojo
from .constants import dtype

fn compute_acceleration(model: Model, mut data: Data):
    """Compute qacc from forces (Newton's 2nd law)."""
    var inv_mass = Scalar[dtype](1.0) / model.body.mass

    # Linear acceleration: a = F/m + g
    data.qacc[0] = data.qfrc_applied[0] * inv_mass
    data.qacc[1] = data.qfrc_applied[1] * inv_mass
    data.qacc[2] = data.qfrc_applied[2] * inv_mass + model.gravity[2]

    # Angular acceleration: α = I⁻¹·τ (diagonal inertia)
    data.qacc[3] = data.qfrc_applied[3] / model.body.inertia[0]
    data.qacc[4] = data.qfrc_applied[4] / model.body.inertia[1]
    data.qacc[5] = data.qfrc_applied[5] / model.body.inertia[2]
```

#### integrator.mojo - Semi-Implicit Euler

```mojo
from .constants import dtype

fn integrate(model: Model, mut data: Data):
    """Semi-implicit Euler: update vel then pos."""
    var dt = model.timestep

    # 1. Update velocities (using current accelerations)
    @parameter
    for i in range(6):
        data.qvel[i] += dt * data.qacc[i]

    # 2. Update positions (using NEW velocities - semi-implicit)
    data.qpos[0] += dt * data.qvel[0]
    data.qpos[1] += dt * data.qvel[1]
    data.qpos[2] += dt * data.qvel[2]

    # 3. Quaternion integration: q' = q + 0.5*dt*ω⊗q
    var half_dt = Scalar[dtype](0.5) * dt
    var wx = data.qvel[3]
    var wy = data.qvel[4]
    var wz = data.qvel[5]
    var qx = data.qpos[3]
    var qy = data.qpos[4]
    var qz = data.qpos[5]
    var qw = data.qpos[6]

    # Quaternion derivative: q_dot = 0.5 * [ω, 0] ⊗ q (Hamilton convention)
    data.qpos[3] += half_dt * ( wx*qw + wy*qz - wz*qy)
    data.qpos[4] += half_dt * (-wx*qz + wy*qw + wz*qx)
    data.qpos[5] += half_dt * ( wx*qy - wy*qx + wz*qw)
    data.qpos[6] += half_dt * (-wx*qx - wy*qy - wz*qz)

    # 4. Normalize quaternion to prevent drift
    normalize_quat(data.qpos)

fn normalize_quat(mut qpos: InlineArray[Scalar[dtype], 7]):
    """Normalize quaternion stored at qpos[3:7]."""
    var qx = qpos[3]
    var qy = qpos[4]
    var qz = qpos[5]
    var qw = qpos[6]
    var norm_sq = qx*qx + qy*qy + qz*qz + qw*qw
    if norm_sq > Scalar[dtype](1e-10):
        var inv_norm = Scalar[dtype](1.0) / sqrt(norm_sq)
        qpos[3] *= inv_norm
        qpos[4] *= inv_norm
        qpos[5] *= inv_norm
        qpos[6] *= inv_norm
```

#### kinematics.mojo - Update World Frame

```mojo
from .constants import dtype

fn update_kinematics(mut data: Data):
    """Update world-frame quantities from generalized coordinates."""
    # For FREE joint, xpos = qpos[0:3], xquat = qpos[3:7]
    data.xpos = SIMD[dtype, 4](data.qpos[0], data.qpos[1], data.qpos[2], 0)
    data.xquat = SIMD[dtype, 4](data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6])
```

### Validation (Phase 1)

**Analytical test**: Free fall from height h=10m
- Expected: `z(t) = 10 - 0.5*g*t²`, `vz(t) = -g*t`
- At t=1s: z ≈ 5.095m, vz ≈ -9.81 m/s
- Acceptance: < 1% error vs analytical

---

## Phase 2: Ground Contact (Sphere-Plane)

### Additional Structures

```
mojo-rl/physics3d_v2/
├── ... (Phase 1 files)
├── collision.mojo      # Sphere-plane detection
├── constraint.mojo     # Contact constraint formulation
├── solver.mojo         # Simple impulse solver
└── test_contact.mojo   # Validation script
```

#### types.mojo - Add Contact

```mojo
from .constants import dtype

# Geometry types
comptime GEOM_PLANE: Int = 0
comptime GEOM_SPHERE: Int = 1

struct Geom:
    """Collision geometry."""
    var type: Int                  # GEOM_PLANE or GEOM_SPHERE
    var size: Scalar[dtype]        # radius for sphere, unused for plane
    var pos: SIMD[dtype, 4]        # Local position offset [x, y, z, 0]

struct Contact:
    """Contact information."""
    var active: Bool
    var pos: SIMD[dtype, 4]        # Contact point (world) [x, y, z, 0]
    var normal: SIMD[dtype, 4]     # Contact normal [nx, ny, nz, 0]
    var depth: Scalar[dtype]       # Penetration depth (positive = penetrating)
    var impulse: Scalar[dtype]     # Normal impulse (for warm-start)

# Extended Model (Phase 2)
struct Model:
    var gravity: SIMD[dtype, 4]    # [gx, gy, gz, 0]
    var timestep: Scalar[dtype]
    var body: Body
    var geom: Geom                 # Body's collision shape
    var ground_z: Scalar[dtype]    # Ground plane height (default: 0)

# Extended Data (Phase 2)
struct Data:
    # Phase 1 fields...
    var qpos: InlineArray[Scalar[dtype], 7]
    var qvel: InlineArray[Scalar[dtype], 6]
    var qacc: InlineArray[Scalar[dtype], 6]
    var xpos: SIMD[dtype, 4]
    var xquat: SIMD[dtype, 4]
    var qfrc_applied: InlineArray[Scalar[dtype], 6]
    # Phase 2 addition
    var contact: Contact
```

#### collision.mojo - Sphere-Plane Detection

```mojo
from .constants import dtype

fn detect_sphere_plane(model: Model, mut data: Data):
    """Detect sphere-ground collision."""
    var sphere_z = data.xpos[2]
    var radius = model.geom.size
    var ground_z = model.ground_z

    # Penetration depth (positive = penetrating)
    var depth = radius - (sphere_z - ground_z)

    if depth > Scalar[dtype](0):
        data.contact.active = True
        data.contact.depth = depth
        data.contact.normal = SIMD[dtype, 4](0, 0, 1, 0)  # Up
        data.contact.pos = SIMD[dtype, 4](
            data.xpos[0], data.xpos[1], ground_z, 0
        )
    else:
        data.contact.active = False
        data.contact.depth = Scalar[dtype](0)
```

#### solver.mojo - Impulse-Based Contact

```mojo
from .constants import dtype

fn solve_contact(model: Model, mut data: Data):
    """Apply contact impulse to prevent penetration."""
    if not data.contact.active:
        return

    var m = model.body.mass

    # Velocity toward ground (negative = approaching)
    # Contact normal is [0, 0, 1], so vn = qvel[2]
    var vn = data.qvel[2]

    if vn < Scalar[dtype](0):  # Only if approaching ground
        # Impulse magnitude: j = -(1+e)*m*vn
        # For inelastic collision (e=0): j = -m*vn
        var restitution = Scalar[dtype](0.0)
        var j = -(Scalar[dtype](1) + restitution) * m * vn

        # Apply impulse: Δv = j/m (only in z)
        data.qvel[2] += j / m

    # Position correction (Baumgarte stabilization)
    var beta = Scalar[dtype](0.2)
    var slop = Scalar[dtype](0.001)
    var correction = max(data.contact.depth - slop, Scalar[dtype](0)) * beta
    data.qpos[2] += correction
```

#### Main Simulation Loop (__init__.mojo)

```mojo
from .constants import dtype
from .types import Model, Data
from .kinematics import update_kinematics
from .dynamics import compute_acceleration
from .integrator import integrate
from .collision import detect_sphere_plane
from .solver import solve_contact

fn step(model: Model, mut data: Data):
    """One simulation step (Phase 2 version)."""
    # 1. Update world-frame positions from qpos
    update_kinematics(data)

    # 2. Collision detection
    detect_sphere_plane(model, data)

    # 3. Compute accelerations (gravity + applied forces)
    compute_acceleration(model, data)

    # 4. Integrate velocities and positions
    integrate(model, data)

    # 5. Solve contact constraints (post-integration)
    solve_contact(model, data)

fn step_phase1(model: Model, mut data: Data):
    """One simulation step (Phase 1 - no collision)."""
    update_kinematics(data)
    compute_acceleration(model, data)
    integrate(model, data)
```

### Validation (Phase 2)

**Test 1**: Ball drop onto ground
- Drop from h=1m, radius=0.1m
- Expected: Ball stops at z=0.1m (radius above ground)
- Acceptance: Final z within 1mm of expected, velocity < 0.01 m/s

**Test 2**: Ball at rest
- Start at z=0.1m (touching ground), zero velocity
- Expected: Stays at z=0.1m for 1000 steps
- Acceptance: No drift (< 1mm), no oscillation

**Test 3**: Ball bounce (optional, e=0.5)
- Drop from h=1m
- Expected: Bounces to h≈0.25m (energy = e² × initial)

---

## File Structure

```
mojo-rl/physics3d_v2/
├── __init__.mojo           # Module exports
├── constants.mojo          # dtype, TILE, TPB, physics constants
├── types.mojo              # Model, Data, Body, Geom, Contact
├── kinematics.mojo         # Forward kinematics (xpos/xquat from qpos)
├── dynamics.mojo           # Acceleration computation
├── integrator.mojo         # Semi-implicit Euler
├── collision.mojo          # Sphere-plane collision
├── solver.mojo             # Contact impulse solver
└── tests/
    ├── test_freefall.mojo  # Phase 1 validation
    └── test_contact.mojo   # Phase 2 validation
```

---

## Implementation Order

### Phase 1 (Free Fall)
1. **constants.mojo** - Define dtype (float32), TILE, TPB, physics constants
2. **types.mojo** - Define Body, Model, Data structs (minimal)
3. **dynamics.mojo** - `compute_acceleration()` (gravity only)
4. **integrator.mojo** - Semi-implicit Euler with quaternion integration
5. **kinematics.mojo** - `update_kinematics()` (copy qpos to xpos/xquat)
6. **__init__.mojo** - Export `step()` function combining above
7. **tests/test_freefall.mojo** - Validate against analytical solution

### Phase 2 (Ground Contact)
8. **types.mojo** - Add Geom, Contact structs
9. **collision.mojo** - Sphere-plane detection
10. **solver.mojo** - Contact impulse solver with Baumgarte
11. **__init__.mojo** - Update `step()` to include collision/solve
12. **tests/test_contact.mojo** - Validate ball drop and rest

---

## Verification Plan

### Phase 1 Verification
```bash
cd mojo-rl
pixi run mojo run physics3d_v2/tests/test_freefall.mojo
```
Expected output:
```
Free fall test:
  t=0.0s: z=10.000, vz=0.000 (expected: 10.000, 0.000)
  t=0.5s: z=8.774, vz=-4.905 (expected: 8.774, -4.905)
  t=1.0s: z=5.095, vz=-9.810 (expected: 5.095, -9.810)
  Error: < 0.1%
  PASSED
```

### Phase 2 Verification
```bash
cd mojo-rl
pixi run mojo run physics3d_v2/tests/test_contact.mojo
```
Expected output:
```
Contact test 1 (ball drop):
  Initial: z=1.0, vz=0.0
  Final: z=0.100, vz=0.000
  Expected: z=0.100
  PASSED

Contact test 2 (ball at rest):
  1000 steps, max drift: 0.0001m
  PASSED
```

---

## Key Differences from Previous Attempt

| Previous physics3d | New physics3d_v2 |
|-------------------|------------------|
| 26-float flat buffer per body | Structured qpos/qvel arrays |
| Diagonal mass approximation | Proper mass matrix (simple for single body) |
| Complex joint system from start | FREE joint only initially |
| GPU kernels immediately | CPU-first, GPU later |
| No analytical validation | Analytical tests at each phase |

---

## Next Phases (Future)

- **Phase 3**: Multiple bodies + sphere-sphere collision
- **Phase 4**: Single hinge joint (pendulum)
- **Phase 5**: Two-link chain with proper constraint solver
- **Phase 6**: Gauss-Seidel/PGS constraint solver
- **Phase 7**: Friction model
- **Phase 8**: Simple walker environment
