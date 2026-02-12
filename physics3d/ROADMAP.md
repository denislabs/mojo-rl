# Physics3D Engine Roadmap: Closing the Gap with MuJoCo

This document details every difference between our physics engine and MuJoCo,
with exact algorithms, formulas, data structure changes, and file-level implementation
plans so that each item can be picked up and implemented directly.

Reference material:
- MuJoCo docs: https://mujoco.readthedocs.io/en/stable/computation/index.html
- MuJoCo C source: `/mujoco-main/src/engine/`
- MuJoCo Warp source: `/mujoco_warp-main/mujoco_warp/_src/`

---

## Table of Contents

- [Current State Summary](#current-state-summary)
- [Phase 0: Model Definition & Scene Description](#phase-0-model-definition--scene-description)
  - [0.1 Compile-Time Model Definition (BodySpec/JointSpec/GeomSpec)](#01-compile-time-model-definition-bodyspecjointspecgeomspec)
  - [0.2 Unified Geom-Based Contact Detection](#02-unified-geom-based-contact-detection)
  - [0.3 Contype/Conaffinity Collision Filtering](#03-contypeconaffinity-collision-filtering)
  - [0.4 ModelRenderer (Geom-Based Visualization)](#04-modelrenderer-geom-based-visualization)
  - [0.5 MJCF XML Parser](#05-mjcf-xml-parser)
- [Phase 1: Core Physics Correctness](#phase-1-core-physics-correctness)
  - [1.1 Full Mass Matrix (off-diagonal terms)](#11-full-mass-matrix-off-diagonal-terms)
  - [1.2 Full RNE Bias Forces (Coriolis + centrifugal)](#12-full-rne-bias-forces-coriolis--centrifugal)
  - [1.3 Unified Collision Detection (Geom-Based)](#13-unified-collision-detection-geom-based)
  - [1.4 Joint Limits as Constraints](#14-joint-limits-as-constraints)
- [Phase 2: Integrator Improvements](#phase-2-integrator-improvements)
  - [2.1 Implicit-fast Integrator](#21-implicit-fast-integrator)
  - [2.2 Implicit Integrator (full)](#22-implicit-integrator-full)
  - [2.3 RK4 Integrator](#23-rk4-integrator)
- [Phase 3: Constraint System](#phase-3-constraint-system)
  - [3.1 Unified Constraint Representation](#31-unified-constraint-representation)
  - [3.2 Friction Cone Models (pyramidal + elliptic)](#32-friction-cone-models-pyramidal--elliptic)
  - [3.3 Equality Constraints (weld + connect)](#33-equality-constraints-weld--connect)
  - [3.4 Per-Contact Solver Parameters (solref/solimp)](#34-per-contact-solver-parameters-solrefsolimp)
- [Phase 4: Collision Pipeline](#phase-4-collision-pipeline)
  - [4.1 Broadphase Collision (bounding sphere)](#41-broadphase-collision-bounding-sphere)
  - [4.2 Broadphase Collision (AABB/SAP)](#42-broadphase-collision-aabbsap)
- [Phase 5: Advanced Features](#phase-5-advanced-features)
  - [5.1 Solver Warmstart](#51-solver-warmstart)
  - [5.2 Solver Islands](#52-solver-islands)
  - [5.3 Passive Forces (spring/damper per joint)](#53-passive-forces-springdamper-per-joint)
  - [5.4 Actuator Dynamics](#54-actuator-dynamics)
  - [5.5 Tendon System](#55-tendon-system)
  - [5.6 No-Slip Friction Post-Solver](#56-no-slip-friction-post-solver)

---

## Current State Summary

### What We Have (working well)

**Model Definition (MuJoCo-aligned, Phase 0 DONE)**:
- **Compile-time trait-based model definition**: `BodySpec`, `JointSpec`, `GeomSpec` traits with variadic containers (`Bodies`, `Joints`, `Geoms`) and `ModelDef` compositor
- **Body specs**: `CapsuleBody`, `SphereBody`, `BoxBody` with auto-computed inertia from geometry + mass
- **Joint specs**: `HingeJoint`, `SlideJoint` with all MuJoCo parameters (armature, damping, stiffness, springref, frictionloss, limits, axis, init_qpos)
- **Geom specs**: Separate from bodies, mirroring MuJoCo's `<geom>` elements:
  - Static (worldbody): `PlaneGeom`, `SphereGeom`, `BoxGeom`, `CapsuleGeom` (BODY_IDX=-1)
  - Body-attached: `BodyCapsuleGeom`, `BodySphereGeom`, `BodyBoxGeom` (BODY_IDX>=0)
  - Per-geom: friction, contype, conaffinity, local pos/quat offset
- **Unified geom-based contact detection**: Single `detect_contacts` / `detect_contacts_gpu` function handles all geom pairs (plane-sphere, plane-capsule, sphere-sphere, capsule-sphere, capsule-capsule, box-sphere, box-capsule, box-box) with automatic world-frame transform, parent-child filtering, and contype/conaffinity filtering
- **ModelRenderer**: Generic renderer parameterized by `GeomSpec` types, auto-draws all body-attached geoms with camera tracking

**Physics Engine (Phases 1-3 DONE)**:
- **Engine core**: qpos/qvel state, forward kinematics, body velocities
- **Full mass matrix (CRBA)** with LDL factorization, armature, implicit damping, stiffness (Phase 1.1 DONE)
- **Full RNE bias forces** with gravity + Coriolis + centrifugal terms, CPU + GPU (Phase 1.2 DONE)
- **Three constraint solvers**: PGS (30 iter), CG (projected), Newton (active-set + line search)
- **Contact Jacobians**: bilateral spatial algebra (body_a - body_b), CPU + GPU
- **Joint types**: FREE, BALL, HINGE, SLIDE with correct cdof computation
- **Collision primitives**: sphere, capsule, box (all pair combinations + ground-plane) (Phase 1.3 DONE)
- **GPU support**: all solvers, kinematics, dynamics have GPU kernels
- **Passive forces**: armature, implicit damping, stiffness+springref, frictionloss (Phase 5.3 DONE)
- **Joint limits as constraints**: unilateral inequality constraints in all 3 solvers, CPU + GPU (Phase 1.4 DONE)
- **Two integrators**: EulerIntegrator + ImplicitFastIntegrator, both with correct damping, CPU + GPU (Phase 2.1 DONE)
- **Physics stability**: MuJoCo solref/solimp impedance model, velocity clamping (MAX_QVEL=10)
- **Constraint solver parameters**: per-model solref/solimp for contacts and joint limits (Phase 3.4 DONE)
- **Unified constraint representation**: ConstraintData/ConstraintRow structs with single builder (Phase 3.1 DONE)

### What We're Missing (by impact)

| Gap | Impact | Phase |
|-----|--------|-------|
| ~~Hardcoded model definitions~~ | ~~High~~ | ~~0~~ DONE |
| ~~Separate ground/body-body detection~~ | ~~High~~ | ~~0~~ DONE |
| ~~No collision filtering~~ | ~~Medium~~ | ~~0~~ DONE |
| No MJCF XML parser | Medium - manual model translation | 0 |
| ~~Diagonal-only mass matrix~~ | ~~High~~ | ~~1~~ DONE |
| ~~No Coriolis/centrifugal in bias forces~~ | ~~High~~ | ~~1~~ DONE |
| ~~Ground contacts only~~ | ~~High~~ | ~~1~~ DONE |
| ~~Joint limits via post-clamping~~ | ~~Medium~~ | ~~1~~ DONE |
| ~~No implicit integrators~~ | ~~Medium~~ | ~~2~~ DONE |
| No RK4 integrator | Medium - energy conservation | 2 |
| ~~No unified constraint rows~~ | ~~Medium~~ | ~~3~~ DONE |
| Simple Coulomb friction only | Medium - no torsional/rolling | 3 |
| No equality constraints | Medium - can't model welds/connects | 3 |
| ~~No per-contact solref/solimp~~ | ~~Low~~ | ~~3~~ DONE |
| No broadphase | Low - performance for many bodies | 4 |
| No warmstart | Low - more solver iterations | 5 |
| No solver islands | Low - parallelism optimization | 5 |
| No actuators / tendons | Low - feature completeness | 5 |

---

## Phase 0: Model Definition & Scene Description

### 0.1 Compile-Time Model Definition (BodySpec/JointSpec/GeomSpec) — DONE

**Status**: COMPLETE. Trait-based compile-time model definition system that mirrors
MuJoCo's XML scene description. Bodies, joints, and geoms are defined as separate
compile-time specs, composed into variadic containers, and used to populate runtime
Model/Data structs.

**Files created**:
- `model/body_spec.mojo` — `BodySpec` trait + `CapsuleBody`, `SphereBody`, `BoxBody` structs
- `model/geom_spec.mojo` — `GeomSpec` trait + static geoms (`PlaneGeom`, `SphereGeom`, `BoxGeom`, `CapsuleGeom`) + body-attached geoms (`BodyCapsuleGeom`, `BodySphereGeom`, `BodyBoxGeom`)
- `model/joint_spec.mojo` — `JointSpec` trait + `HingeJoint`, `SlideJoint` structs
- `model/model_def.mojo` — `Bodies`, `Joints`, `Geoms` variadic containers + `ModelDef` compositor
- `model/model_renderer.mojo` — Generic renderer parameterized by `GeomSpec` types
- `model/__init__.mojo` — Module exports

#### Architecture

The system separates model description into three orthogonal concepts, matching MuJoCo XML:

| MuJoCo XML | Mojo Trait | Concrete Types |
|-----------|-----------|----------------|
| `<body>` | `BodySpec` | `CapsuleBody`, `SphereBody`, `BoxBody` |
| `<geom>` in worldbody | `GeomSpec` (BODY_IDX=-1) | `PlaneGeom`, `SphereGeom`, `BoxGeom`, `CapsuleGeom` |
| `<geom>` in body | `GeomSpec` (BODY_IDX>=0) | `BodyCapsuleGeom`, `BodySphereGeom`, `BodyBoxGeom` |
| `<joint>` | `JointSpec` | `HingeJoint`, `SlideJoint` |
| `<compiler>`, `<option>` | `ModelDef` | Global params (gravity, timestep, solref/solimp) |

**Key design decisions**:
- **Inertia auto-computed from geometry**: `BodySpec` computes `ixx()`, `iyy()`, `izz()` from shape + mass, matching MuJoCo's `inertiafromgeom="true"` default
- **Geoms separate from bodies**: A body can have zero, one, or multiple geoms. Static obstacles (worldbody geoms) are first-class citizens.
- **Variadic containers**: `Bodies[*B: BodySpec]`, `Joints[*J: JointSpec]`, `Geoms[*G: GeomSpec]` use Mojo variadic generics for compile-time iteration
- **`ModelDef` takes concrete ints**: Avoids Mojo's "variadic nesting causes unbound parameter" issue by computing `NQ`, `NV`, `NGEOM` etc. from containers before passing to `ModelDef`

**Key methods on containers**:
- `Bodies.setup_model()` — populates Model body arrays (mass, inertia, pos, quat, parent)
- `Joints.reset_data()` — initializes Data with INIT_QPOS values
- `Joints.extract_obs()` / `extract_obs_gpu()` — builds observation vector respecting EXCLUDE flags
- `Joints.apply_actions()` / `apply_actions_gpu()` — maps normalized [-1,1] actions to qfrc via tau_limit
- `Joints.enforce_limits()` / `enforce_limits_gpu()` — post-step position clamping
- `Geoms.setup_model()` — populates Model geom arrays (type, body, pos, quat, radius, half-extents, friction, contype, conaffinity)

#### Usage Example (HalfCheetah)

```mojo
# Bodies (kinematic tree)
comptime Torso = CapsuleBody[parent=-1, mass=4.5, radius=0.046, half_length=0.09, conaffinity=0]
comptime BThigh = CapsuleBody[parent=0, mass=0.9, radius=0.046, half_length=0.09, ...]
# ... 7 more bodies ...

# Joints (degrees of freedom)
comptime RootX = SlideJoint[body_idx=0, axis_x=1.0, is_actuated=False]
comptime RootZ = SlideJoint[body_idx=0, axis_z=1.0, is_actuated=False]
comptime BThighJoint = HingeJoint[body_idx=1, range_min=-2.36, range_max=0.785, stiffness=240.0, ...]
# ... 7 more joints ...

# Geoms (collision geometry)
comptime Ground = PlaneGeom[z=0.0, friction=0.4]
comptime TorsoGeom = BodyCapsuleGeom[body_idx=0, radius=0.046, half_length=0.09, conaffinity=0]
# ... 8 more body geoms ...

# Compose
comptime HalfCheetahBodies = Bodies[Torso, BThigh, BShin, FThigh, FShin, ...]
comptime HalfCheetahJoints = Joints[RootX, RootZ, BThighJoint, BShinJoint, ...]
comptime HalfCheetahGeoms = Geoms[Ground, TorsoGeom, BThighGeom, ...]

comptime HalfCheetahModel = ModelDef[
    HalfCheetahBodies.N, HalfCheetahJoints.N,
    HalfCheetahJoints._sum_nq(), HalfCheetahJoints._sum_nv(),
    HalfCheetahGeoms.N,
]
```

---

### 0.2 Unified Geom-Based Contact Detection — DONE

**Status**: COMPLETE. Replaced the old two-pass approach (separate `detect_ground_contacts` +
`detect_body_body_contacts`) with a single unified `detect_contacts` function that iterates
over all geom pairs, handling both ground-plane contacts and body-body contacts uniformly.

**Files modified**:
- `collision/contact_detection.mojo` — Unified `detect_contacts` (CPU) and `detect_contacts_gpu` (GPU)
- `collision/collision_primitives.mojo` — Narrowphase primitives (sphere-sphere, capsule-sphere, capsule-capsule, box-sphere, box-capsule, box-box)
- `integrator/euler_integrator.mojo` — Single `detect_contacts()` call replaces two separate calls
- `integrator/implicit_fast_integrator.mojo` — Same

#### How It Works

The unified detection iterates over all NGEOM*(NGEOM-1)/2 geom pairs:

```
for gi in range(NGEOM):
    for gj in range(gi + 1, NGEOM):
        # Skip static-static pairs
        if gi is plane and gj has no body: continue
        # Skip same-body geoms
        if gi_body == gj_body: continue
        # Skip parent-child body pairs (share a joint)
        if parent(gj_body) == gi_body or parent(gi_body) == gj_body: continue
        # Contype/conaffinity filter
        if (contype_a & conaffinity_b) == 0 and (contype_b & conaffinity_a) == 0: continue

        # Transform geom to world frame (body pos/quat + local offset)
        world_pos_i, world_quat_i = geom_world_pos(gi)
        world_pos_j, world_quat_j = geom_world_pos(gj)

        # Dispatch to narrowphase based on geom types
        if plane + capsule: two endpoint contacts
        elif plane + sphere: single contact
        elif sphere + sphere: sphere_sphere()
        elif capsule + sphere: capsule_sphere()
        # ... all 10 type combinations ...

        # Store contact with body_a, body_b, normal, position, distance, friction
```

**Key improvements over the old approach**:
- **Single code path** for ground and body-body contacts (no duplication)
- **Geom local offsets** properly handled — geoms can be offset from body center via local pos/quat
- **Multiple geoms per body** — a body can have separate collision and visual geometry
- **Static obstacles** — worldbody geoms (boxes, spheres, capsules) participate in detection naturally
- **Per-geom friction** — friction comes from the geom, not the body

#### World Frame Transform

For body-attached geoms with local offset:
```
world_pos = body_pos + quat_rotate(body_quat, local_pos)
world_quat = quat_mul(body_quat, local_quat)
```

Identity optimization: if local pos/quat is zero/identity, skip the transform.

---

### 0.3 Contype/Conaffinity Collision Filtering — DONE

**Status**: COMPLETE. MuJoCo-style collision filtering using per-geom bitmasks.

**Implementation**:
- Two geoms collide only if `(contype_a & conaffinity_b) || (contype_b & conaffinity_a) != 0`
- Stored per-geom in `GeomSpec` traits and `Model.geom_contype[]` / `Model.geom_conaffinity[]`
- GPU: `GEOM_IDX_CONTYPE` and `GEOM_IDX_CONAFFINITY` in the model geom buffer
- Default: contype=1, conaffinity=1 (everything collides)

**Usage examples**:
- **HalfCheetah**: All body geoms have `conaffinity=0` (disables body-body self-collision, matching MuJoCo XML where all geoms have `conaffinity="0"`)
- **Hopper**: Uses defaults (conaffinity=1, self-collision enabled, matches MuJoCo XML)

**Files modified**:
- `model/geom_spec.mojo` — `CONTYPE` and `CONAFFINITY` fields on all GeomSpec types
- `types.mojo` — `Model.geom_contype[]`, `Model.geom_conaffinity[]` arrays
- `collision/contact_detection.mojo` — filtering logic in both CPU and GPU paths
- `gpu/constants.mojo` — `GEOM_IDX_CONTYPE`, `GEOM_IDX_CONAFFINITY`
- `gpu/buffer_utils.mojo` — packs contype/conaffinity into GPU model buffer

---

### 0.4 ModelRenderer (Geom-Based Visualization) — DONE

**Status**: COMPLETE. Generic renderer that draws all body-attached geoms using
compile-time iteration over `GeomSpec` types.

**Files created**:
- `model/model_renderer.mojo` — `ModelRenderer` struct parameterized by variadic `GeomSpec` types

**Features**:
- Automatically renders capsule geoms at body world positions with local offset + rotation
- Camera follows torso (body 0) with configurable offsets
- Draws velocity indicator arrow
- Skips static geoms (planes) — only renders body-attached geometry
- Compile-time `@parameter` iteration over geom types (zero runtime overhead)

---

### 0.5 MJCF XML Parser

**Status**: NOT STARTED. Currently, MuJoCo XML models are manually translated to
compile-time trait definitions. An MJCF parser would automate this.

**Problem**: Each new MuJoCo environment requires manually translating the XML
into `BodySpec`, `JointSpec`, and `GeomSpec` definitions. This is tedious and
error-prone (wrong inertia values, missed parameters, etc.).

**Approach options**:

**Option A (compile-time, recommended)**: Python script that reads MJCF XML and
generates Mojo source code with the appropriate trait instantiations:
```bash
python mjcf_to_mojo.py half_cheetah.xml > envs/half_cheetah/model_def.mojo
```
- Pro: Zero runtime overhead, all parameters are compile-time constants
- Pro: Generated code is readable and can be hand-tuned
- Con: Requires regeneration when XML changes

**Option B (runtime)**: Mojo MJCF parser that builds `Model` at runtime:
- Pro: Direct XML loading, no code generation step
- Con: All parameters become runtime values (can't use compile-time features)
- Con: Requires XML parsing library in Mojo (doesn't exist yet)

**Scope**: Parse `<worldbody>`, `<body>`, `<joint>`, `<geom>`, `<option>`, `<compiler>`,
`<default>` elements. Handle `<default>` class inheritance. Extract solref/solimp,
gravity, timestep, and all joint/body/geom parameters.

**Files to create**:
- `tools/mjcf_to_mojo.py` (Python code generator)
- Or `model/mjcf_parser.mojo` (runtime parser, if Mojo XML parsing becomes available)

---

## Phase 1: Core Physics Correctness

### 1.1 Full Mass Matrix (off-diagonal terms) — DONE

**Status**: COMPLETE. Implemented full CRBA mass matrix with LDL factorization,
armature, implicit damping (`M[i,i] += dt * damping`), and stiffness. Both CPU and GPU.
Dense storage (Option A) for NV <= 9. See `dynamics/mass_matrix.mojo` and `dynamics/jacobian.mojo`.

**Problem** (original): We only compute diagonal `M[i,i]`. Off-diagonal coupling terms `M[i,j]`
are zero, meaning each DOF is treated as independent. For articulated bodies (robot arms,
legs), moving one joint affects forces at other joints through inertial coupling. Without
off-diagonal terms, dynamics are incorrect for multi-joint systems.

**Files to modify**:
- `dynamics/mass_matrix.mojo` (main implementation)
- `types.mojo` (add sparse storage fields to Data if needed)
- `integrator/euler_integrator.mojo` (use full M solve instead of diagonal inverse)
- `gpu/constants.mojo` (state buffer layout if M is stored)

#### Algorithm: Composite Rigid Body Algorithm (CRBA)

MuJoCo reference: `engine_core_smooth.c` lines 1888-2015.

```
1. Initialize: crb[i] = body_inertia[i] for each body i

2. Backward pass (leaves to root):
   for i = NBODY-1 down to 1:
     parent = body_parent[i]
     crb[parent] += transform(crb[i], body_pos[i], body_quat[i])

3. Compute M entries:
   for each DOF i:
     body_i = dof_body[i]
     // Diagonal
     buf = crb[body_i] @ cdof[i]     // inertia * motion vector
     M[i,i] = cdof[i] . buf + armature[i]

     // Off-diagonal: walk up ancestors
     j = dof_parent[i]
     while j >= 0:
       M[i,j] = cdof[j] . buf
       M[j,i] = M[i,j]              // symmetry
       j = dof_parent[j]
```

Where `crb @ cdof` is the spatial inertia times spatial motion vector (6D):
```
Given crb as (mass, I_3x3, com_offset):
  buf_angular = I @ cdof_angular + mass * (com × cdof_linear)
  buf_linear  = mass * cdof_linear - mass * (com × cdof_angular)
```

MuJoCo stores the composite inertia as a 10-element vector `cinert`:
`[mass, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, cx, cy, cz]`

For our engine, we can use a simpler representation since we already have
`body_inertia[3]` (diagonal) and `body_mass`.

#### Storage Format

**Option A (dense, simpler)**: Store full `M[NV][NV]` array.
- Pro: simple indexing, works for small NV (< 30 DOFs)
- Con: O(NV^2) memory and solve time
- Good for: HalfCheetah (NV=9), Hopper (NV=6), Walker (NV=9)

**Option B (sparse LDL, MuJoCo-style)**: Store lower triangle in CSR format.
- Pro: O(NV * tree_depth) memory and solve
- Con: more complex implementation
- Good for: Humanoid (NV=27+), complex robots

**Recommendation**: Start with Option A (dense). Our environments have NV <= 9.
Add sparse later if needed for humanoids.

#### Data Structure Changes

In `Data`, add:
```mojo
# Full mass matrix (dense, NV x NV)
var M_full: InlineArray[Scalar[DTYPE], NV * NV]

# LDL factorization storage
var M_L: InlineArray[Scalar[DTYPE], NV * NV]   # lower triangle L
var M_D: InlineArray[Scalar[DTYPE], NV]         # diagonal D
```

Or pass as local variables in the integrator (avoids state bloat).

#### LDL Factorization

MuJoCo reference: `engine_core_smooth.c` lines 1991-2015 (`mj_factorI`).

```
Input: Symmetric positive-definite M (NV x NV, stored lower triangle)
Output: L (unit lower triangular), D (diagonal) such that M = L * D * L^T

Algorithm (backward elimination):
  for k = NV-1 down to 0:
    D[k] = M[k,k]
    for j = 0 to k-1:
      L[k,j] = M[k,j] / D[k]
    for j = 0 to k-1:
      for i = 0 to j:
        M[j,i] -= L[k,j] * D[k] * L[k,i]
```

#### LDL Solve (M x = b)

MuJoCo reference: `engine_core_smooth.c` lines 2131-2215 (`mj_solveLD`).

```
Given M = L * D * L^T, solve M * x = b:

Step 1: Forward substitution (solve L^T * y1 = b):
  for k = NV-1 down to 0:
    x[k] = b[k]
    for j = k+1 to NV-1:
      x[j] -= L[j,k] * x[k]

Step 2: Diagonal solve (y2 = D^-1 * y1):
  for k = 0 to NV-1:
    x[k] /= D[k]

Step 3: Backward substitution (solve L * x = y2):
  for k = 0 to NV-1:
    for j = 0 to k-1:
      x[k] -= L[k,j] * x[j]
```

#### Changes to Integrator

In `euler_integrator.mojo`, replace:
```mojo
# OLD: diagonal solve
if M_diag[i] > 1e-10:
    qacc[i] = f_net[i] / M_diag[i]
```

With:
```mojo
# NEW: full M solve via LDL
compute_mass_matrix_full(model, data, M_full)
ldl_factor(M_full, L, D, NV)
ldl_solve(L, D, f_net, qacc, NV)
```

#### GPU Considerations

For GPU, the full M can be stored in registers/local memory for small NV.
For NV <= 9 (HalfCheetah), a 9x9 matrix fits in registers easily.
The LDL factorization and solve are sequential per environment but
parallelized across the BATCH dimension.

#### Testing

- Compare M_full vs M_diag for double pendulum (2 DOF) - should see coupling
- Compare dynamics with MuJoCo for HalfCheetah: apply same qfrc, compare qacc
- Energy conservation test: free-falling articulated body should conserve energy

---

### 1.2 Full RNE Bias Forces (Coriolis + centrifugal) — DONE

**Status**: COMPLETE. Implemented full Recursive Newton-Euler Algorithm computing
b(q,qvel) = C(q,qvel)*qvel + g(q) including gravity, Coriolis, and centrifugal forces.
Both CPU (`compute_bias_forces_rne`) and GPU (`compute_bias_forces_rne_gpu`) versions.
See `dynamics/bias_forces.mojo`.

Algorithm: 5-step RNE in world frame:
1. World-frame inertia tensors (R @ diag(I) @ R^T)
2. Forward pass: spatial accelerations with gravity + cdof_dot*qvel
3. Body forces: I*cacc + cvel x* (I*cvel) — gyroscopic + centripetal
4. Backward pass: force accumulation with moment transfer (r x f)
5. Projection to joint space: bias[d] = cdof[d] . cfrc[body]

**Problem** (original): We only compute gravitational torques. At any non-trivial velocity,
Coriolis and centrifugal forces are significant. A spinning body or fast-moving
robot will have wrong dynamics without them.

**Files to modify**:
- `dynamics/bias_forces.mojo` (main implementation)

#### Algorithm: Recursive Newton-Euler (RNE)

MuJoCo reference: `engine_core_smooth.c` lines 2425-2486 (`mj_rne`).

The RNE computes `bias = M * 0 + C(q, qvel)` by setting qacc=0:

```
FORWARD PASS (root to leaves):
  // World body "acceleration" = -gravity (handles gravity uniformly)
  cacc[0] = [0, 0, 0, -gx, -gy, -gz]  // 6D spatial acceleration

  for each body i (in topological order, skip world):
    parent = body_parent[i]

    // Spatial acceleration of body i:
    // cacc[i] = cacc[parent] + cdof_dot[i] * qvel[i]
    // (cdof_dot captures velocity-dependent acceleration = Coriolis/centrifugal)
    cacc[i] = cacc[parent]
    for each DOF d of body i:
      cacc[i] += cdof_dot[d] * qvel[d]

    // Body force = inertia * acceleration + velocity x (inertia * velocity)
    // The cross product term is the Coriolis/centrifugal contribution
    cfrc[i] = I[i] @ cacc[i] + cvel[i] x* (I[i] @ cvel[i])

BACKWARD PASS (leaves to root):
  for each body i (in reverse topological order):
    parent = body_parent[i]
    if parent > 0:
      cfrc[parent] += cfrc[i]    // accumulate child forces to parent

PROJECTION (body forces to joint torques):
  for each DOF d:
    bias[d] = cdof[d] . cfrc[body_of_dof[d]]    // dot product of motion axis with force
```

Where:
- `cacc` = 6D spatial acceleration per body (angular[3], linear[3])
- `cvel` = 6D spatial velocity per body (already computed in `compute_body_velocities`)
- `cfrc` = 6D spatial force per body (torque[3], force[3])
- `cdof` = 6D spatial motion axis per DOF (already computed)
- `cdof_dot` = time derivative of cdof (velocity-dependent, NEW)
- `x*` = spatial cross-force product

#### New: cdof_dot Computation

`cdof_dot` captures how the motion axis changes with velocity. For each joint type:

```
HINGE:
  // Axis rotates with parent body angular velocity
  cdof_dot_angular = parent_angvel x axis_world
  cdof_dot_linear  = parent_angvel x (axis_world x offset) + axis_world x vel_at_joint

SLIDE:
  cdof_dot_angular = [0, 0, 0]
  cdof_dot_linear  = parent_angvel x axis_world

FREE:
  cdof_dot = [0, 0, 0, 0, 0, 0]  // world-fixed axes don't change

BALL:
  cdof_dot_angular = parent_angvel x axis_world  (for each of the 3 axes)
  cdof_dot_linear  = [0, 0, 0]
```

#### Spatial Cross-Force Product

The `x*` operation (spatial cross-force, also called `crossForce`):
```
Given v = [w, v_lin] (spatial velocity) and f = [tau, f_lin] (spatial force):
  v x* f = [w x tau + v_lin x f_lin, w x f_lin]

In components:
  result_angular = cross(w, tau) + cross(v_lin, f_lin)
  result_linear  = cross(w, f_lin)
```

#### Spatial Inertia-Vector Product

The `I @ v` operation (inertia times spatial vector):
```
Given I = (mass, I_3x3, com_offset) and v = [w, v_lin]:
  result_angular = I_3x3 @ w + mass * (com x v_lin)
  result_linear  = mass * v_lin - mass * (com x w)
```

For diagonal inertia `I = diag(Ixx, Iyy, Izz)`:
```
  result_angular = [Ixx*wx, Iyy*wy, Izz*wz] + mass * cross(com, v_lin)
  result_linear  = mass * v_lin - mass * cross(com, w)
```

#### Implementation Plan

1. Add `cdof_dot` computation to `jacobian.mojo`:
   ```mojo
   fn compute_cdof_dot[...](model, data, cdof, mut cdof_dot):
   ```

2. Rewrite `compute_bias_forces` in `bias_forces.mojo`:
   - Forward pass: compute `cacc[i]` and `cfrc[i]` per body
   - Backward pass: accumulate `cfrc` up the tree
   - Project: `bias[d] = cdof[d] . cfrc[body[d]]`

3. Keep the old gravity-only version as a fast path option (compile-time flag)

#### Testing

- Zero-velocity test: RNE with qvel=0 should match current gravity-only bias
- Spinning body test: single body rotating fast, check centrifugal force
- Compare with MuJoCo: same qpos/qvel, compare bias force output
- Double pendulum at speed: verify energy drift is reduced

---

### 1.3 Body-Body Collision in GC Engine — DONE

**Status**: COMPLETE. Implemented body-body collision detection and bilateral contact
Jacobians for the GC engine. Both CPU and GPU. All three solvers (PGS, CG, Newton)
updated to pass `body_b` through all Jacobian calls. Detection dispatches to existing
collision primitives (sphere-sphere, capsule-sphere, capsule-capsule, box-sphere,
box-capsule) with parent-child pair filtering. Backward-compatible: ground contacts
have `body_b = -1`, reducing bilateral Jacobian to original unilateral form.

**Files modified**:
- `constants.mojo` (added `GEOM_CAPSULE`, `GEOM_BOX` constants)
- `dynamics/jacobian.mojo` (bilateral Jacobian: `J_row[d] += J_a - J_b`, CPU + GPU)
- `solver/pgs_solver.mojo`, `solver/cg_solver.mojo`, `solver/newton_solver.mojo`
  (pass `body_b` through all Jacobian calls, CPU + GPU)
- `collision/contact_detection.mojo` (`detect_body_body_contacts` and `detect_body_body_contacts_gpu`)
- `integrator/euler_integrator.mojo` (calls body-body detection after ground detection)

#### What Needs to Happen

1. **Detection**: For each pair of bodies (i, j) where i != j:
   - Get world positions and geometry from `data.xpos`, `data.xquat`, `model.body_geom_type`
   - Dispatch to appropriate primitive: `sphere_sphere`, `capsule_capsule`, `box_sphere`, etc.
   - Output: `ContactInfo` with `body_a = i`, `body_b = j`, contact point, normal, distance

2. **Jacobian**: For body-body contacts, the contact Jacobian has contributions from
   both bodies (unlike ground contacts where body_b = -1):
   ```
   J_contact[d] = J_body_a[d] . normal - J_body_b[d] . normal
   ```
   Where `J_body_x` is the velocity Jacobian mapping DOF d to the contact point on body x.

3. **Solver Integration**: The solvers already handle the Jacobian correctly if it's
   computed correctly. The key change is in `compute_contact_jacobian_row`:
   - Currently: `J_row[d] = J_trans_a . direction` (ground contacts only)
   - New: `J_row[d] = J_trans_a . direction - J_trans_b . direction`

#### Implementation Steps

**Step 1**: Add `detect_body_body_contacts` function:
```mojo
fn detect_body_body_contacts[...](model: Model, mut data: Data):
    """Detect contacts between all body pairs using world-space geometry."""
    for i in range(NBODY):
        for j in range(i + 1, NBODY):
            # Skip parent-child pairs (they share a joint)
            if model.body_parent[j] == i or model.body_parent[i] == j:
                continue

            # Get world positions and orientations from FK results
            pos_i = (data.xpos[3*i], data.xpos[3*i+1], data.xpos[3*i+2])
            pos_j = (data.xpos[3*j], data.xpos[3*j+1], data.xpos[3*j+2])
            # ... dispatch to collision primitives based on geom_type ...

            if dist < margin:
                var contact = ContactInfo[DTYPE]()
                contact.body_a = i
                contact.body_b = j
                contact.normal_x = normal.x
                # ... fill contact ...
                data.contacts[data.num_contacts] = contact
                data.num_contacts += 1
```

**Step 2**: Modify `compute_contact_jacobian_row` in `jacobian.mojo`:
```mojo
# Add body_b parameter (currently not used for ground contacts)
fn compute_contact_jacobian_row[...](
    model, data, cdof,
    body_a: Int, body_b: Int,  # body_b = -1 for ground
    contact_pos, direction,
    mut J_row
):
    # Contribution from body_a (existing code)
    for d in range(NV):
        if _joint_affects_body(model, d, body_a):
            J_row[d] += compute_J_at_point(cdof, d, contact_pos, body_a) . direction

    # NEW: Contribution from body_b (subtract because relative velocity)
    if body_b >= 0:
        for d in range(NV):
            if _joint_affects_body(model, d, body_b):
                J_row[d] -= compute_J_at_point(cdof, d, contact_pos, body_b) . direction
```

**Step 3**: Call both detection functions in the integrator:
```mojo
# In euler_integrator.mojo step():
detect_ground_contacts(model, data)
detect_body_body_contacts(model, data)  # NEW
```

#### Testing

- Two spheres dropping onto each other (no ground): should bounce
- Robot touching a box on the ground: contact forces on both
- Self-collision: two links of a chain hitting each other
- Compare contact forces with Cartesian engine (ImpulseIntegrator) for same scenario

---

### 1.4 Joint Limits as Constraints — DONE

**Status**: COMPLETE. Joint limits are now enforced as unilateral inequality constraints
inside all three constraint solvers (PGS, CG, Newton) on both CPU and GPU. Post-step
clamping removed from the integrator and GPU kernel.

Each solver detects active limits (within 0.01 margin) for HINGE/SLIDE joints with
finite ranges, then solves them via PGS iterations using the 1D Jacobian (J[dof] = ±1)
and effective mass K = M_inv[dof,dof]. Baumgarte correction with penetration cap (0.01)
prevents energy injection. CPU solvers use full M_inv column for velocity correction;
GPU solvers use diagonal M_inv approximation (consistent with GPU contact solving).

**Files modified**:
- `solver/pgs_solver.mojo` (limit detection + PGS in CPU `solve` and GPU `solve_gpu`)
- `solver/cg_solver.mojo` (same pattern, CPU + GPU)
- `solver/newton_solver.mojo` (same pattern, CPU + GPU)
- `integrator/euler_integrator.mojo` (removed `enforce_joint_limits` call)

#### MuJoCo's Approach

Reference: `engine_core_constraint.c` lines 756-903 (`mj_instantiateLimit`).

For each HINGE/SLIDE joint with limits `[q_min, q_max]`:
```
For side in {lower, upper}:
  q_limit = q_min if lower, q_max if upper
  dist = side * (q_limit - q_current)    // positive = inside limits

  if dist < margin:  // approaching or past limit
    // Create constraint row:
    J[dof_adr] = -side   // Jacobian is +1 or -1 at the joint's DOF
    // All other J entries = 0

    // Constraint position: dist (negative = violated)
    // Reference acceleration: from impedance model (solref/solimp)
    aref = -k * imp * dist - b * vel[dof_adr]
```

For BALL joints:
```
angle = ||axis_angle(quat)||
dist = max_angle - angle

if dist < margin:
  J[dof_adr:dof_adr+3] = -axis  // 3D Jacobian along rotation axis
  aref = -k * imp * dist - b * (axis . angvel)
```

#### Implementation

**Step 1**: Add limit detection before the solver call:
```mojo
fn detect_joint_limits[...](model, data, mut limit_contacts, mut num_limits):
    for j in range(NJOINT):
        var jnt = model.joints[j]
        if jnt.type == JNT_HINGE or jnt.type == JNT_SLIDE:
            var q = data.qpos[jnt.qpos_adr]
            var margin = Scalar[DTYPE](0.01)  # activation margin

            # Lower limit
            var dist_lo = q - jnt.range_min
            if dist_lo < margin:
                # Add constraint: J[dof_adr] = +1, pos = dist_lo
                limit_contacts[num_limits] = LimitConstraint(
                    dof=jnt.dof_adr, sign=1, dist=dist_lo
                )
                num_limits += 1

            # Upper limit
            var dist_hi = jnt.range_max - q
            if dist_hi < margin:
                limit_contacts[num_limits] = LimitConstraint(
                    dof=jnt.dof_adr, sign=-1, dist=dist_hi
                )
                num_limits += 1
```

**Step 2**: In the solver, handle limit constraints alongside contact constraints.
Limit constraints are simpler than contacts (1D Jacobian, single DOF):
```
For each limit constraint:
  velocity_error = sign * qvel[dof]
  K = 1.0 / M_diag[dof]  // or M_inv[dof,dof] with full M
  bias = BAUMGARTE * min(dist, 0) / dt
  delta_lambda = -(velocity_error + bias) / K
  lambda = max(lambda + delta_lambda, 0)  // inequality: push away from limit
  qvel[dof] += sign * delta_lambda * M_inv[dof]
```

**Step 3**: Remove post-clamping from the integrator pipeline (or make it optional).

#### Testing

- Pendulum at joint limit: should bounce smoothly, not stick
- Energy test: repeated bouncing off limits should not gain energy
- Compare with MuJoCo: same model, hit limit, compare qvel trajectory

---

## Phase 2: Integrator Improvements

### 2.1 Implicit-fast Integrator — DONE

**Status**: COMPLETE. Two MuJoCo-matching integrators implemented with correct damping
treatment (both implicit mass matrix modification AND explicit damping force).

**Files created/modified**:
- `integrator/implicit_fast_integrator.mojo` (NEW — qDeriv-based mass matrix modification)
- `integrator/euler_integrator.mojo` (fixed damping bug: added explicit `f_net -= D*v`)
- `integrator/__init__.mojo` (exports both, DefaultIntegrator = ImplicitFastIntegrator[PGSSolver])
- `envs/half_cheetah/half_cheetah.mojo` (switched to ImplicitFastIntegrator[NewtonSolver])

#### What Was Implemented

**Critical damping bug fixed**: Both integrators were only modifying the mass matrix
diagonal (`M[i,i] += dt * damping`), but the explicit damping force `f_net -= damping * qvel`
was missing. MuJoCo requires BOTH:
1. Mass matrix: `M_hat = M + arm + dt*damping` (implicit part — damps the NEW velocity)
2. Force: `f_net -= damping * qvel` (explicit part — applies force from CURRENT velocity)

Without (2), there was essentially zero velocity decay from damping when no external
forces act. This is mathematically wrong and means physics under-damps significantly.

**Two integrators**:
- **EulerIntegrator[SOLVER]**: `M_hat = M + arm + dt*diag(damping)`, simple diagonal treatment
- **ImplicitFastIntegrator[SOLVER]**: `M_hat = M + arm - dt*qDeriv` where `qDeriv[i,i] = -damping[i]`
  - Currently identical results (no actuators), but extensible for actuator velocity derivatives
  - `qDeriv` will later include `gainprm[2]`, `biasprm[2]`, tendon damping

**DefaultIntegrator** = `ImplicitFastIntegrator[PGSSolver]` (was `EulerIntegrator[PGSSolver]`).
HalfCheetah uses `ImplicitFastIntegrator[NewtonSolver]`. Hopper uses DefaultIntegrator.

Both CPU and GPU paths updated in all integrators, all joint types (FREE, BALL, HINGE, SLIDE).

#### Algorithm

MuJoCo reference: `engine_forward.c` lines 1140-1163 (implicitfast path).

```
Solve: (M + armature - dt * qDeriv) * qacc = f_net

Where:
  qDeriv[i,i] = d(forces)/d(qvel_i) = -damping[i]  (passive only)
  f_net = qfrc - bias - damping * qvel - stiffness * (qpos - springref) - frictionloss * sign(qvel)
```

Future extension for actuators:
```
qDeriv[i,i] += d(actuator_force)/d(qvel_i)  (gainprm[2], biasprm[2])
```

---

### 2.2 Implicit Integrator (full)

**Problem**: `implicitfast` skips the Coriolis derivative. For systems with significant
gyroscopic effects (rapidly spinning objects, robot arms at high speed), the full
implicit integrator includes `d(bias)/d(qvel)` for better accuracy.

**Files to create/modify**:
- `integrator/implicit_integrator.mojo` (NEW)
- `dynamics/velocity_derivatives.mojo` (add RNE velocity derivative)

#### Additional Computation: RNE Velocity Derivative

MuJoCo reference: `engine_derivative.c` lines 596-700 (`mjd_rne_vel`).

This computes `d(C(q,v))/dv` where C is the Coriolis/centrifugal term from RNE.
The derivative is generally non-symmetric, requiring LU factorization instead of
Cholesky/LDL:

```
dF/dv_full = dF_passive/dv + dF_actuator/dv - d(bias)/dv

Modified system: (M - dt * dF/dv_full) * qacc = f_total
Factorize via LU (not LDL, because dF/dv_full is not symmetric)
```

**Implementation**: Same as implicitfast but:
1. Compute full `dF/dv` matrix (NV x NV), including `d(bias)/dv`
2. Use LU factorization instead of LDL
3. More expensive but more stable for gyroscopic systems

**Recommendation**: Implement after implicitfast. Most robotics tasks don't need this.

---

### 2.3 RK4 Integrator

**Problem**: For systems that should conserve energy (pendulums, mechanical systems
without damping), semi-implicit Euler drifts energy over time. RK4 provides 4th-order
accuracy and much better energy conservation.

**Files to create**:
- `integrator/rk4_integrator.mojo` (NEW)

#### Algorithm

MuJoCo reference: `engine_forward.c` lines 1005-1090 (`mj_RungeKutta`).

```
RK4 Butcher tableau:
  c = [0, 1/2, 1/2, 1]
  A = [[0,   0,   0,   0],
       [1/2, 0,   0,   0],
       [0,   1/2, 0,   0],
       [0,   0,   1,   0]]
  b = [1/6, 1/3, 1/3, 1/6]

Algorithm:
  Save initial state: q0, v0

  // Stage 1: evaluate at (q0, v0)
  a1 = forward_dynamics(q0, v0)
  k1_v = a1, k1_q = v0

  // Stage 2: evaluate at (q0 + dt/2 * k1_q, v0 + dt/2 * k1_v)
  q2 = integrate_pos(q0, k1_q, dt/2)
  v2 = v0 + dt/2 * k1_v
  a2 = forward_dynamics(q2, v2)
  k2_v = a2, k2_q = v2

  // Stage 3: evaluate at (q0 + dt/2 * k2_q, v0 + dt/2 * k2_v)
  q3 = integrate_pos(q0, k2_q, dt/2)
  v3 = v0 + dt/2 * k2_v
  a3 = forward_dynamics(q3, v3)
  k3_v = a3, k3_q = v3

  // Stage 4: evaluate at (q0 + dt * k3_q, v0 + dt * k3_v)
  q4 = integrate_pos(q0, k3_q, dt)
  v4 = v0 + dt * k3_v
  a4 = forward_dynamics(q4, v4)
  k4_v = a4, k4_q = v4

  // Combine
  v_new = v0 + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
  q_new = integrate_pos(q0, (k1_q + 2*k2_q + 2*k3_q + k4_q)/6, dt)
```

**Key points**:
- Requires 4 forward dynamics evaluations per step (4x cost)
- `integrate_pos` must handle quaternions correctly (not simple addition)
- Constraint solving should happen at each stage (expensive) or only at the final
  stage (cheaper but less accurate)

**Recommendation**: Implement as a quality option. Not needed for RL training
(where fast iteration matters more than accuracy), but useful for validation
and high-fidelity simulation.

#### Position Integration with Quaternions

For FREE and BALL joints, position integration is not `q += v*dt` because
quaternion space is not Euclidean:
```
For HINGE/SLIDE: q_new = q + v * dt  (standard)

For BALL (quaternion):
  dq = axis_angle_to_quat(angvel * dt)
  q_new = quat_mul(q, dq)
  q_new = quat_normalize(q_new)

For FREE:
  pos_new = pos + vel * dt
  dq = axis_angle_to_quat(angvel * dt)
  quat_new = quat_mul(quat, dq)
  quat_new = quat_normalize(quat_new)
```

---

## Phase 3: Constraint System

### 3.1 Unified Constraint Representation — DONE

**Status**: COMPLETE. Extracted duplicated constraint setup code from all 3 CPU solvers
(PGS, CG, Newton) into a single constraint builder. Solvers now consume pre-built
`ConstraintData` instead of computing Jacobians, impedance, and limits inline.
Adding a new constraint type is now a 1-file change (constraint_builder.mojo).
GPU paths unchanged (deferred to follow-up).

**Files created**:
- `constraints/constraint_data.mojo` — `ConstraintRow[DTYPE]` and `ConstraintData[DTYPE, MAX_ROWS, NV]` structs
- `constraints/constraint_builder.mojo` — `build_constraints()` (contacts + limits + friction) and `writeback_impulses()`

**Files modified**:
- `traits/solver.mojo` — `solve()` signature: removed `CDOF_SIZE`/`cdof`, added `MAX_ROWS`/`ConstraintData`
- `solver/pgs_solver.mojo` — CPU `solve()` iterates ConstraintData rows (normals, limits, friction)
- `solver/cg_solver.mojo` — CPU `solve()` builds Delassus matrix from precomputed J/MinvJT
- `solver/newton_solver.mojo` — CPU `solve()` uses ConstraintData for projected Newton
- `solver/friction_solver.mojo` — Removed `_solve_friction_pgs_cpu` (dead code), GPU unchanged
- `solver/__init__.mojo` — Exports ConstraintData, ConstraintRow, build_constraints, writeback_impulses
- `integrator/euler_integrator.mojo` — Calls `build_constraints()` before solver, `writeback_impulses()` after
- `integrator/implicit_fast_integrator.mojo` — Same as euler_integrator

**Key design**:
- `MAX_ROWS = 3 * MAX_CONTACTS + 2 * NJOINT` (normal + 2 tangents per contact + 2 limits per joint)
- Constraint ordering: normals [0..num_normals), friction [num_normals..num_normals+num_friction), limits [num_normals+num_friction..num_rows)
- Builder precomputes J, MinvJT, K, impedance bias per row — solvers just iterate
- Friction rows have `friction_parent` pointing to their normal row for Coulomb cone clamping

**TODO — GPU refactor**:
- GPU `solve_gpu()` methods in all 3 solvers still inline contact/limit/friction setup using workspace arrays
- Need GPU-side `build_constraints_gpu()` writing constraint rows to workspace, and GPU solvers consuming them
- Complex due to workspace layout (LayoutTensor offsets, no InlineArrays on GPU) — separate follow-up task

#### Data Structure

MuJoCo reference: constraint arrays in `mjdata.h`.

```mojo
struct ConstraintRow[DTYPE: DType, NV: Int]:
    """Single row of the constraint system."""
    var type: Int           # CONTACT_NORMAL, CONTACT_FRICTION, LIMIT, EQUALITY
    var id: Int             # source object (contact index, joint index, etc.)
    var J: InlineArray[Scalar[DTYPE], NV]   # Jacobian row (1 x NV)
    var pos: Scalar[DTYPE]  # constraint position error
    var vel: Scalar[DTYPE]  # constraint velocity (J @ qvel)
    var aref: Scalar[DTYPE] # reference acceleration
    var D: Scalar[DTYPE]    # effective mass (1 / (J @ M_inv @ J^T))
    var R: Scalar[DTYPE]    # regularization
    var lo: Scalar[DTYPE]   # force lower bound (0 for contacts, -inf for equality)
    var hi: Scalar[DTYPE]   # force upper bound (+inf for contacts)
    var force: Scalar[DTYPE] # computed constraint force (output)

# Constraint types
alias CNSTR_CONTACT_NORMAL = 0
alias CNSTR_CONTACT_FRICTION = 1
alias CNSTR_LIMIT_JOINT = 2
alias CNSTR_EQUALITY_CONNECT = 3
alias CNSTR_EQUALITY_WELD = 4

struct ConstraintData[DTYPE: DType, NV: Int, MAX_ROWS: Int]:
    """All constraint rows for one simulation step."""
    var rows: InlineArray[ConstraintRow[DTYPE, NV], MAX_ROWS]
    var num_rows: Int
    var ne: Int   # number of equality constraints (come first)
    var nf: Int   # number of friction constraints
    var ni: Int   # number of inequality constraints (contacts + limits)
```

#### Builder Pattern

```mojo
fn build_constraints[...](
    model, data, M_inv, cdof,
    mut cdata: ConstraintData
):
    cdata.num_rows = 0

    # 1. Equality constraints (come first in MuJoCo)
    build_equality_constraints(model, data, cdof, cdata)
    cdata.ne = cdata.num_rows

    # 2. Joint limit constraints
    build_limit_constraints(model, data, cdof, cdata)

    # 3. Contact constraints (normal)
    build_contact_normal_constraints(model, data, M_inv, cdof, cdata)

    # 4. Contact friction constraints (after normal)
    build_contact_friction_constraints(model, data, M_inv, cdof, cdata)
    cdata.nf = cdata.num_rows - cdata.ne - ni_before_friction
```

#### Solver Changes

All three solvers would consume `ConstraintData` uniformly:
```mojo
trait ConstraintSolver:
    @staticmethod
    fn solve[...](
        model, data, M_diag,  # or M_full + L, D
        mut cdata: ConstraintData,
        mut qvel: InlineArray[...],
        dt: Scalar[DTYPE],
    ):
        ...
```

---

### 3.2 Friction Cone Models (pyramidal + elliptic)

**Problem**: We use simple Coulomb clamping: `|f_tangent| <= mu * f_normal`.
MuJoCo supports two friction cone models that handle multi-dimensional friction
(tangent1, tangent2, torsional, rolling).

**Files to create/modify**:
- `constraint/friction_cone.mojo` (NEW - cone projection functions)
- `constraint/constraint_builder.mojo` (friction row generation)
- Solvers (friction projection step)

#### Pyramidal Friction Cone

MuJoCo reference: `engine_core_constraint.c` lines 1050-1068.

Instead of one friction constraint, create pairs of pyramid edges:
```
For each tangent direction k (k = 1 to condim-1):
  Edge+: J_edge = J_normal + mu_k * J_tangent[k]
  Edge-: J_edge = J_normal - mu_k * J_tangent[k]

  Both edges have: force >= 0 (inequality constraint)
```

The pyramid approximates the friction cone with linear constraints.
For 3D contact (condim=3): 1 normal + 2 pairs = 5 constraint rows.

**Advantages**: All constraints are simple inequalities (lambda >= 0).
**Disadvantages**: Approximation of the true cone, can allow sliding at pyramid edges.

#### Elliptic Friction Cone

MuJoCo reference: `engine_core_constraint.c` lines 1070-1075, solver lines 268-307.

Uses the true elliptic cone constraint:
```
sum((f_tangent[k] / (mu_k * f_normal))^2) <= 1
```

This requires the QCQP solver for the friction subproblem:
```
minimize:   0.5 * f_t^T * A * f_t + b^T * f_t
subject to: sum((f_t[k] / mu_k)^2) <= f_normal^2
```

The QCQP solver uses Newton's method on the Lagrangian (augmented with the
elliptic constraint). For 2D friction (the common case), this is the QCQP2
solver with closed-form Newton steps.

#### QCQP Solver

MuJoCo reference: `engine_util_solve.c` lines 986-1052 (`mju_QCQP2`).

```
Input: A (2x2 Hessian), b (2D gradient), d (2D scaling), r (radius = f_normal)
Output: x (2D optimal friction force)

Algorithm:
1. Scale A and b so constraint becomes ||x||^2 <= r^2
2. Newton iteration on dual variable lambda:
   - Solve (A + lambda*I) * x = -b
   - Check ||x||^2 <= r^2
   - If violated: lambda += -(||x||^2 - r^2) / deriv
3. Unscale result
```

#### Implementation Plan

1. Start with pyramidal (simpler, compatible with existing PGS solver)
2. Add elliptic later with QCQP solver
3. Add `condim` parameter to contacts (1=frictionless, 3=standard, 4=torsional, 6=rolling)

---

### 3.3 Equality Constraints (weld + connect)

**Problem**: We cannot model fixed attachments between bodies or ball-joint connections
that aren't part of the kinematic tree. MuJoCo's equality constraints allow welding
two bodies together or connecting them at a point.

**Files to create**:
- `constraint/equality_constraints.mojo` (NEW)

#### Connect Constraint (ball joint)

MuJoCo reference: `engine_core_constraint.c` lines 428-457.

```
Position error (3D): e = world_pos(body_a, anchor_a) - world_pos(body_b, anchor_b)
Jacobian (3 x NV): J = J_pos(body_a, anchor_a) - J_pos(body_b, anchor_b)
Constraint: e = 0 (equality, bounds = [-inf, +inf])
```

This creates 3 constraint rows.

#### Weld Constraint (fixed attachment)

MuJoCo reference: `engine_core_constraint.c` lines 459-533.

```
Position error (3D): e_pos = world_pos(body_a, anchor_a) - world_pos(body_b, anchor_b)
Orientation error (3D): e_rot = 0.5 * imag(inv(quat_b) * quat_a * relpose)

Jacobian position (3 x NV): J_pos = J_pos(body_a) - J_pos(body_b)
Jacobian rotation (3 x NV): J_rot = 0.5 * corrected_quaternion_jacobian

Constraint: [e_pos; e_rot] = 0 (6D equality)
```

This creates 6 constraint rows.

#### Data Structure

Add to `Model`:
```mojo
struct EqualityConstraint[DTYPE: DType]:
    var type: Int          # EQ_CONNECT or EQ_WELD
    var body_a: Int
    var body_b: Int
    var anchor_a: SIMD[DTYPE, 4]  # local anchor on body_a
    var anchor_b: SIMD[DTYPE, 4]  # local anchor on body_b
    var relpose: SIMD[DTYPE, 4]   # relative quaternion (for weld)
    var solref: SIMD[DTYPE, 2]    # solver reference parameters
    var solimp: SIMD[DTYPE, 4]    # solver impedance parameters
```

---

### 3.4 Per-Contact Solver Parameters (solref/solimp) — DONE

**Implemented**: MuJoCo-style impedance model replacing Baumgarte stabilization.
Global `solref` (time constant + damping ratio) and `solimp` (impedance curve)
parameters stored per model, used by all three solvers (PGS, CG, Newton) on CPU and GPU.

**Files modified**:
- `types.mojo` — added `solref_contact/solimp_contact/solref_limit/solimp_limit` to Model
- `gpu/constants.mojo` — added `MODEL_META_IDX_SOLREF_*` / `MODEL_META_IDX_SOLIMP_*` (10 indices)
- `gpu/buffer_utils.mojo` — writes solref/solimp to GPU model buffer
- `solver/pgs_solver.mojo`, `solver/cg_solver.mojo`, `solver/newton_solver.mojo` — impedance bias in all 12 sections (3 solvers × 2 platforms × 2 constraint types)
- `envs/half_cheetah/` and `envs/hopper/` — environment-specific solref/solimp from XML

#### Implementation Details

**Impedance model** (simplified Hermite smoothstep, MuJoCo-compatible):
```
x = min(penetration / width, 1.0)
imp = dmin + (3x² - 2x³) * (dmax - dmin)
imp = max(imp, 0.2)  # floor prevents zero-force contacts
```

**Velocity-level bias** (adapted from MuJoCo's acceleration-level formulation):
```
inv_tc_dr = 1 / (timeconst * dampratio)
b_vel_coef = 2 * dampratio * dt / (dmax * timeconst)
bias = -imp * penetration * inv_tc_dr - b_vel_coef * v_n
delta = -(v_n + bias) / (K / imp)
```

Key insight: MuJoCo's `aref = -k*imp*pen - b*v_n` is acceleration-level. Our PGS/CG/Newton
solvers work at velocity level, so naively multiplying by `dt` gives O(dt²) position recovery
(~50x too weak at dt=0.002). The velocity-level formulation derives Baumgarte-equivalent
coefficients directly from solref parameters.

**Parameters** (from half_cheetah.xml / hopper.xml):
- `solref = [0.02, 1.0]` — 20ms time constant, critical damping
- `solimp_contact = [0.0, 0.8, 0.01]` — soft at surface, firm at depth
- `solimp_limit = [0.0, 0.8, 0.03]` — wider transition for joint limits
- Penetration capped at 0.05m for safety

---

## Phase 4: Collision Pipeline

### 4.1 Broadphase Collision (bounding sphere)

**Problem**: O(N^2) geom-pair collision is expensive for many geoms. A bounding
sphere pre-filter eliminates most pairs cheaply.

**Files to modify**:
- `collision/contact_detection.mojo` (add broadphase filter before narrowphase)
- `model/geom_spec.mojo` (add bounding radius computation per geom)

#### Algorithm

MuJoCo Warp reference: `collision_driver.py` lines 271-318.

```
For each geom pair (gi, gj):
  bound = rbound[gi] + rbound[gj]
  dist_sq = |world_pos[gi] - world_pos[gj]|^2
  if dist_sq > bound^2:
    skip pair  // bounding spheres don't overlap
```

Where `rbound` is per-geom (not per-body):
- Sphere: `rbound = radius`
- Capsule: `rbound = half_length + radius`
- Box: `rbound = sqrt(hx^2 + hy^2 + hz^2)`

**Note**: With the unified geom-based detection, broadphase operates on geom pairs
rather than body pairs, which is more natural and handles multiple geoms per body correctly.

---

### 4.2 Broadphase Collision (AABB/SAP)

**Problem**: Bounding spheres are O(N^2) in pair count. For scenes with many geoms,
sweep-and-prune (SAP) on axis-aligned bounding boxes reduces this to O(N log N).

#### Algorithm: Sweep-and-Prune

```
1. Project each body's AABB onto the sweep axis (e.g., X axis)
2. Sort intervals by their lower bound: O(N log N)
3. Sweep through sorted list:
   - Maintain active set of overlapping intervals
   - For each new interval, check overlap with all active intervals
   - Add overlapping pairs to candidate list
   - Remove intervals whose upper bound is passed
4. For candidate pairs, verify overlap on Y and Z axes
```

MuJoCo uses the principal eigenvector of the geom covariance matrix as the
sweep axis (adapts to the scene geometry).

**Recommendation**: Implement after bounding sphere filter. Only needed for
scenes with 50+ geoms.

---

## Phase 5: Advanced Features

### 5.1 Solver Warmstart

**Problem**: Each step, solvers start from zero. MuJoCo warmstarts from the
previous step's constraint forces, reducing iteration count significantly.

**Files to modify**:
- `types.mojo` (add warmstart storage to Data)
- All three solvers (initialize from warmstart)
- Both integrators (save result for next step)

#### Implementation

```mojo
# In Data:
var qacc_warmstart: InlineArray[Scalar[DTYPE], NV]
var lambda_warmstart: InlineArray[Scalar[DTYPE], MAX_CONSTRAINTS]

# In solver, at start:
if use_warmstart:
    for i in range(num_constraints):
        lambda[i] = data.lambda_warmstart[i]
else:
    for i in range(num_constraints):
        lambda[i] = 0

# After solver converges:
for i in range(num_constraints):
    data.lambda_warmstart[i] = lambda[i]
data.qacc_warmstart = qacc
```

MuJoCo also compares the warmstart cost with the cold start cost and picks
the better one (reference: `engine_forward.c` line 630 `warmstart()` function).

---

### 5.2 Solver Islands

**Problem**: In scenes with multiple disconnected contact groups, solving all
constraints together wastes computation. Islands identify independent subproblems.

#### Algorithm

```
1. Build constraint graph:
   - Nodes = bodies
   - Edges = contacts and joints connecting bodies
2. Find connected components (BFS/DFS or union-find)
3. Each component = one island
4. Solve each island independently (potentially in parallel)
```

**Benefits**:
- Unconstrained bodies (flying in air) cost zero solver time
- Multiple contact groups solve in parallel
- Smaller systems converge faster

**Recommendation**: Implement after the unified constraint system (Phase 3.1).
Most useful for multi-agent scenarios.

---

### 5.3 Passive Forces (spring/damper per joint) — DONE

**Status**: COMPLETE. All four MuJoCo passive force types implemented per joint:
- **Armature**: `M[i,i] += armature` (rotor inertia regularization)
- **Implicit damping**: `M[i,i] += dt * damping` (unconditionally stable)
- **Stiffness with springref**: `f -= stiffness * (qpos - springref)` (restoring spring)
- **Frictionloss**: `f -= frictionloss * sign(qvel)` (dry friction)

All passive forces are defined per-joint via `JointSpec` traits and work on CPU + GPU.

**Implementation details**:
- Damping uses BOTH implicit (`M[i,i] += dt*damping`) AND explicit (`f_net -= damping * qvel`) treatment, matching MuJoCo
- Stiffness and frictionloss are applied as explicit forces in `f_net` before the LDL solve
- Frictionloss uses `sign(qvel)` with a 1e-4 velocity dead zone to avoid chatter at zero velocity

---

### 5.4 Actuator Dynamics

**Problem**: MuJoCo has a full actuator system with activation dynamics, gain/bias
computation, force limits, and multiple transmission types. Our engine applies
torques directly via `qfrc` (mapped through `JointSpec.TAU_LIMIT`).

**Current state**: `Joints.apply_actions()` maps normalized [-1, 1] actions to
`qfrc` via `tau_limit * action`, filtering by `IS_ACTUATED` flag. This is equivalent
to MuJoCo's simplest actuator (motor with fixed gain = tau_limit).

**Missing**:
- Activation dynamics (INTEGRATOR, FILTER, MUSCLE)
- Gain/bias functions (AFFINE with gainprm/biasprm)
- Force clamping (forcerange)
- Position/velocity actuators (PD control)

#### MuJoCo Actuator Pipeline

```
1. Activation dynamics: act_dot = f(act, ctrl)
   - INTEGRATOR: act_dot = ctrl
   - FILTER: act_dot = (ctrl - act) / tau
   - MUSCLE: Hill muscle model

2. Gain: g = gain(act, vel, ...)
   - FIXED: g = gainprm[0]
   - AFFINE: g = gainprm[0] + gainprm[1] * act

3. Bias: b = bias(act, vel, ...)
   - NONE: b = 0
   - AFFINE: b = biasprm[0] + biasprm[1] * act + biasprm[2] * vel

4. Force: f = g * ctrl + b
5. Clamping: f = clamp(f, forcerange)
6. Transmission: qfrc += J_actuator^T * f
```

**Recommendation**: Start with simple position/velocity actuators (PD control).
Add muscle dynamics later if needed for biomechanics.

---

### 5.5 Tendon System

**Problem**: MuJoCo supports tendons (cables that span multiple joints).

**Recommendation**: Low priority. Only needed for biomechanics/musculoskeletal models.

---

### 5.6 No-Slip Friction Post-Solver

**Problem**: After the main constraint solver converges, friction forces may
allow small slip.

**Recommendation**: Low priority. Current friction handling is sufficient for
locomotion and manipulation tasks.

---

## Implementation Priority Order

For RL training (locomotion, manipulation):

```
Sprint 0 (Model definition):
  0.1 Compile-time model def    DONE (BodySpec/JointSpec/GeomSpec traits, Bodies/Joints/Geoms containers)
  0.2 Unified contact detection DONE (single detect_contacts for all geom pairs, CPU + GPU)
  0.3 Collision filtering       DONE (contype/conaffinity per geom, MuJoCo-compatible)
  0.4 ModelRenderer             DONE (geom-based visualization with camera tracking)
  0.5 MJCF XML parser           <- automate model translation from MuJoCo XML

Sprint 1 (Core correctness):
  1.1 Full mass matrix          DONE (CRBA + LDL + armature + implicit damping + stiffness)
  1.2 Full RNE bias forces      DONE (gravity + Coriolis + centrifugal, CPU + GPU)
  1.3 Unified collision         DONE (geom-based, all primitives, bilateral Jacobians, CPU + GPU)
  1.4 Joint limits              DONE (unilateral constraints in all 3 solvers, CPU + GPU)

Sprint 2 (Stability):
  2.1 Implicit-fast integrator  DONE (EulerIntegrator + ImplicitFastIntegrator, CPU + GPU)
  5.3 Passive forces            DONE (armature, damping, stiffness+springref, frictionloss)

Sprint 3 (Constraint system):
  3.1 Unified constraint rows   DONE (ConstraintData/ConstraintRow, constraint_builder)
  3.4 Per-contact solref/solimp DONE (impedance model, all 3 solvers, CPU + GPU)

Sprint 4 (Polish):
  5.1 Solver warmstart          <- performance (fewer iterations)
  3.2 Friction cone models      <- better friction physics
  4.1 Broadphase (spheres)      <- performance for many geoms

Sprint 5 (Advanced):
  3.3 Equality constraints      <- weld/connect
  2.3 RK4 integrator            <- energy conservation option
  2.2 Implicit integrator       <- gyroscopic stability
  5.2 Solver islands            <- multi-agent parallelism
  5.4 Actuator dynamics         <- MuJoCo model compatibility
```

---

## Validation Strategy

For each feature, validate against MuJoCo:

1. **Unit test**: Isolated component test (e.g., mass matrix values match MuJoCo)
2. **Integration test**: Full step comparison (same initial state, compare next state)
3. **Trajectory test**: Multi-step rollout, compare trajectory divergence
4. **RL test**: Train agent, compare learning curves and final performance

Tools:
- Use `mujoco` Python package to generate reference values
- Export qpos/qvel/qacc from MuJoCo, import into our engine, compare
- Use HalfCheetah as the standard benchmark (NV=9, well-studied)

```python
# Example: generate reference mass matrix from MuJoCo
import mujoco
model = mujoco.MjModel.from_xml_path("half_cheetah.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data.qM)
print("Mass matrix:\n", M)
print("Bias forces:", data.qfrc_bias)
```
