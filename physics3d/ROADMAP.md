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
  - [2.2 Implicit Integrator (full) — DONE](#22-implicit-integrator-full--done)
  - [2.3 RK4 Integrator — DONE](#23-rk4-integrator--done)
- [Phase 3: Constraint System](#phase-3-constraint-system)
  - [3.1 Unified Constraint Representation](#31-unified-constraint-representation)
  - [3.2 Friction Cone Models (pyramidal + elliptic) — DONE](#32-friction-cone-models-pyramidal--elliptic--done)
  - [3.3 Equality Constraints (weld + connect) — DONE](#33-equality-constraints-weld--connect--done)
  - [3.4 Per-Contact Solver Parameters (solref/solimp)](#34-per-contact-solver-parameters-solrefsolimp)
- [Phase 4: Collision Pipeline](#phase-4-collision-pipeline)
  - [4.1 Broadphase Collision (bounding sphere) — DONE](#41-broadphase-collision-bounding-sphere--done)
  - [4.2 Broadphase Collision (AABB/SAP)](#42-broadphase-collision-aabbsap)
- [Phase 5: Advanced Features](#phase-5-advanced-features)
  - [5.1 Solver Warmstart](#51-solver-warmstart)
  - [5.2 Solver Islands](#52-solver-islands)
  - [5.3 Passive Forces (spring/damper per joint)](#53-passive-forces-springdamper-per-joint)
  - [5.4 Actuator Dynamics — DONE](#54-actuator-dynamics--done)
  - [5.5 Tendon System (fixed tendons)](#55-tendon-system-fixed-tendons)
  - [5.6 No-Slip Friction Post-Solver](#56-no-slip-friction-post-solver)
- [Render Sprint: Visual Fidelity](#render-sprint-visual-fidelity)
  - [R.1 Multi-Geom-Type Rendering](#r1-multi-geom-type-rendering)
  - [R.2 Ground Plane Rendering](#r2-ground-plane-rendering)
  - [R.3 RGBA Alpha Support](#r3-rgba-alpha-support)
  - [R.4 Camera Spec from Model](#r4-camera-spec-from-model)
  - [R.5 Lighting Model](#r5-lighting-model)
  - [R.6 Materials & Textures](#r6-materials--textures)
  - [R.7 Site Markers (visual)](#r7-site-markers-visual)
- [Phase 6: MuJoCo XML Compatibility Gaps](#phase-6-mujoco-xml-compatibility-gaps)
  - [6.1 settotalmass Compiler Directive — DONE](#61-settotalmass-compiler-directive--done)
  - [6.2 inertiafromgeom from Child Geoms — DONE](#62-inertiafromgeom-from-child-geoms--done)
  - [6.3 fromto Capsule Specification — DONE](#63-fromto-capsule-specification--done)
  - [6.4 Contact Margin](#64-contact-margin)
  - [6.5 Full solimp (5 params)](#65-full-solimp-5-params)
  - [6.6 Cylinder Geom Collision](#66-cylinder-geom-collision)
  - [6.7 Geom Density (mass from density) — DONE](#67-geom-density-mass-from-density--done)
  - [6.8 Site Elements](#68-site-elements)
  - [6.9 Fluid Dynamics (density/viscosity)](#69-fluid-dynamics-densityviscosity)
  - [6.10 cfrc_ext (Contact Forces per Body)](#610-cfrc_ext-contact-forces-per-body)
  - [6.11 Runtime Solver Selection from XML](#611-runtime-solver-selection-from-xml)

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
  - Per-geom: friction, contype, conaffinity, density/mass, local pos/quat offset
- **inertiafromgeom**: Automatic body mass/inertia/ipos/iquat computation from child geoms via parallel axis theorem + eigendecomposition (Phase 6.2 DONE)
- **Geom density**: Per-geom density/mass with model-level default (1000 kg/m³), volume computation for sphere/capsule/box (Phase 6.7 DONE)
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
- **Four integrators**: EulerIntegrator + ImplicitFastIntegrator + ImplicitIntegrator (full) + RK4Integrator, all with correct damping, CPU + GPU (Phase 2.1 + 2.2 + 2.3 DONE)
- **Physics stability**: MuJoCo solref/solimp impedance model, velocity clamping (MAX_QVEL=10)
- **Constraint solver parameters**: per-model solref/solimp for contacts and joint limits (Phase 3.4 DONE)
- **Unified constraint representation**: ConstraintData/ConstraintRow structs with single builder (Phase 3.1 DONE)
- **Friction cone models**: Pyramidal + elliptic cones with condim 1/3/4/6, QCQP solver, torsional/rolling friction, CPU + GPU (Phase 3.2 DONE)
- **Equality constraints**: Connect (3-row ball joint) and weld (6-row rigid attachment) constraints with bilateral PGS, MuJoCo impedance, CPU + GPU (Phase 3.3 DONE)

### What We're Missing (by impact)

| Gap | Impact | Phase |
|-----|--------|-------|
| ~~Hardcoded model definitions~~ | ~~High~~ | ~~0~~ DONE |
| ~~Separate ground/body-body detection~~ | ~~High~~ | ~~0~~ DONE |
| ~~No collision filtering~~ | ~~Medium~~ | ~~0~~ DONE |
| No MJCF XML parser | Medium — manual model translation | 0 |
| ~~No `settotalmass` compiler directive~~ | ~~Low~~ | ~~6~~ DONE |
| ~~No `inertiafromgeom` from child geoms~~ | ~~Medium~~ | ~~6~~ DONE |
| No `fromto` capsule specification | Low — requires endpoint→center conversion | 6 |
| ~~No `margin` on geom/joint contacts~~ | ~~Medium~~ | ~~6~~ DONE |
| `solimp` only 3 params (not 5) | Low — midpoint/power hardcoded to 0.5/2.0 | 6 |
| No `cylinder` geom collision | Medium — Reacher, Pusher need it | 6 |
| ~~No `density` on geoms (mass from density)~~ | ~~Low~~ | ~~6~~ DONE |
| No `<site>` elements | Low — massless reference points for observations | 6 |
| No `<tendon><fixed>` joint coupling | Medium — Humanoid hip-knee coupling | 5 |
| No fluid dynamics (`density`/`viscosity` option) | Low — Swimmer only | 6 |
| No `cfrc_ext` (contact forces per body) | Medium — Humanoid observations | 6 |
| No runtime solver/iterations selection from XML | Low — Humanoid requests PGS+50 iter | 6 |
| ~~Diagonal-only mass matrix~~ | ~~High~~ | ~~1~~ DONE |
| ~~No Coriolis/centrifugal in bias forces~~ | ~~High~~ | ~~1~~ DONE |
| ~~Ground contacts only~~ | ~~High~~ | ~~1~~ DONE |
| ~~Joint limits via post-clamping~~ | ~~Medium~~ | ~~1~~ DONE |
| ~~No implicit integrators~~ | ~~Medium~~ | ~~2~~ DONE |
| ~~No full implicit integrator~~ | ~~Medium - gyroscopic stability~~ | ~~2~~ DONE |
| ~~No RK4 integrator~~ | ~~Medium - energy conservation~~ | ~~2~~ DONE |
| ~~No unified constraint rows~~ | ~~Medium~~ | ~~3~~ DONE |
| ~~Simple Coulomb friction only~~ | ~~Medium~~ | ~~3~~ DONE |
| ~~No equality constraints~~ | ~~Medium - can't model welds/connects~~ | ~~3~~ DONE |
| ~~No per-contact solref/solimp~~ | ~~Low~~ | ~~3~~ DONE |
| ~~No broadphase~~ | ~~Low - performance for many bodies~~ | ~~4~~ DONE |
| ~~No warmstart~~ | ~~Low~~ | ~~5~~ DONE |
| No solver islands | Low - parallelism optimization | 5 |
| ~~No actuators~~ / No tendons | ~~Low~~ | ~~5~~ DONE (actuators) |

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

### 2.2 Implicit Integrator (full) — DONE

**Status**: COMPLETE. Full implicit integrator with RNE velocity derivatives
(`d(bias)/d(qvel)`) and LU factorization for the non-symmetric modified mass matrix.
Both CPU and GPU implementations. More accurate than `implicitfast` for systems with
significant gyroscopic effects (rapidly spinning objects, robot arms at high speed).

**Files created**:
- `integrator/implicit_integrator.mojo` (NEW — full implicit integrator with RNE velocity derivatives)
- `dynamics/velocity_derivatives.mojo` (NEW — `compute_rne_vel_derivative()`, dense NV×NV Jacobian)
- `dynamics/lu_factorization.mojo` (NEW — LU factorization with partial pivoting for non-symmetric matrices)
- `tests/test_implicit_integrator.mojo` (NEW — LU tests, zero-velocity qDeriv validation)

**Files modified**:
- `integrator/__init__.mojo` — exports `ImplicitIntegrator`
- `gpu/constants.mojo` — `implicit_extra_workspace_size`, offset functions for derivative workspace arrays

#### What Was Implemented

**Full qDeriv computation** (vs implicitfast which only uses passive damping diagonal):
1. Initialize `qDeriv[i,i] = -damping[i]` (passive forces, same as implicitfast)
2. Add RNE velocity derivative: `qDeriv -= d(qfrc_bias)/d(qvel)` (Coriolis/centrifugal derivative)
3. Form `M_hat = M + armature - dt * qDeriv` (full NV×NV matrix, generally non-symmetric)
4. **LU factorization** of M_hat (not LDL, because qDeriv is non-symmetric)
5. LU solve: `qacc = M_hat^{-1} * f_net`

**RNE velocity derivative algorithm** (matching MuJoCo's `mjd_rne_vel_dense`):
1. Precompute body-origin quantities (cdof, cvel, cinert at body origin)
2. Compute `Dcvel` and `Dcdofdot` (velocity Jacobians)
3. Forward pass: compute `Dcacc` and `Dcfrcbody` (force Jacobians)
4. Backward pass: accumulate `Dcfrcbody` to parents
5. Project to joint space: subtract from qDeriv

**LU factorization** (`lu_factorization.mojo`):
- `lu_factor()`: In-place LU with partial pivoting
- `lu_solve()`: Solve Ax=b using LU factors
- `compute_M_inv_from_lu()`: Compute M_inv column-by-column for constraint solver

**GPU implementation**:
- Three-kernel approach to avoid Metal register pressure
- Extra workspace section for large derivative matrices (Dcvel, Dcdofdot, Dcacc, Dcfrcbody)
- Offset functions: `ws_implicit_qderiv_offset`, `ws_implicit_cdof_origin_offset`, etc.

MuJoCo reference: `engine_derivative.c` lines 596-700 (`mjd_rne_vel`),
`engine_forward.c` lines 1117-1137 (implicit integration path).

---

### 2.3 RK4 Integrator — DONE

**Status**: COMPLETE. 4th-order Runge-Kutta integrator with 4 force evaluations
per step. Provides better energy conservation for systems without damping.

**Files**:
- `integrator/rk4_integrator.mojo`

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

### 3.2 Friction Cone Models (pyramidal + elliptic) — DONE

**Status**: COMPLETE. Both pyramidal and elliptic friction cone models implemented
with variable contact dimensionality (condim 1/3/4/6) on CPU and GPU. Default is
elliptic cone with condim=3 (matching previous behavior).

**Two cone types** (`model.cone_type`):
- **Pyramidal** (0): Edge constraints `J_edge± = J_normal ± μ_k * J_tangent_k`, all `λ ≥ 0`. Simpler solver (no Coulomb projection needed).
- **Elliptic** (1, default): Separate normal + friction rows with QCQP projection onto the true elliptic cone.

**Variable condim per contact** (combined as `max(geom_a, geom_b)` at contact detection):
- **condim=1**: Frictionless (no friction rows)
- **condim=3**: 2 tangent friction directions (slide), default
- **condim=4**: +1 torsional friction (angular, along contact normal)
- **condim=6**: +2 rolling friction (angular, along tangent directions)

**Per-geom friction arrays**: `GeomSpec` trait provides `FRICTION` (slide), `FRICTION_SPIN` (torsional, default 0.005), `FRICTION_ROLL` (rolling, default 0.0001). Combined via `max()` per-element at contact detection.

**QCQP solver** (`solver/qcqp.mojo`): Newton on dual variable for elliptic cone projection.
- `qcqp2[DTYPE]()`: 2D (condim=3) — radial projection
- `qcqp3[DTYPE]()`: 3D (condim=4) — 3×3 Newton dual
- `qcqp5[DTYPE]()`: 5D (condim=6) — 5×5 Newton dual

**Angular-only Jacobian** (`dynamics/jacobian.mojo`): `compute_angular_jacobian_row_gpu` for torsional/rolling friction — uses only angular cdof components (no cross product with position offset).

**CPU implementation**:
- `constraint_builder.mojo`: Builds variable friction rows (2/3/5 per contact) for elliptic, or edge rows for pyramidal. Pyramidal uses `CNSTR_PYRAMID_EDGE` type with `source_dof` encoding tangent direction and sign.
- All 3 CPU solvers (PGS, CG, Newton): Group-based friction iteration with QCQP dispatch by group size (elliptic) or simple `λ ≥ 0` clamping (pyramidal).
- `writeback_forces`: Handles all constraint types including pyramid edge force decode.

**GPU implementation**:
- `friction_solver.mojo`: Rewritten for variable condim. Workspace: `31*MC + 10*MC*NV` (5 friction directions × Jacobians, MinvJ, K, coefficients, directions).
- `pgs_solver.mojo`: Extended precompute phase and coupled PGS with condim-aware friction and QCQP projection.
- Solver workspace sizes: PGS `44*MC+12*MC*NV`, Newton `49*MC+12*MC*NV+MC*MC`, CG `48*MC+12*MC*NV+MC*MC`.
- `MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT` (worst case: pyramidal condim=6).

**Constraint type constants** (`constraint_data.mojo`):
- `CNSTR_FRICTION_TORSION=4`, `CNSTR_FRICTION_ROLL1=5`, `CNSTR_FRICTION_ROLL2=6`, `CNSTR_PYRAMID_EDGE=7`

**Files created**:
- `solver/qcqp.mojo` — QCQP2/3/5 elliptic cone projection solvers

**Files modified**:
- `model/geom_spec.mojo` — `CONDIM`, `FRICTION_SPIN`, `FRICTION_ROLL` on trait + 7 geom structs
- `types.mojo` — ContactInfo: +7 fields; Model: +5 fields (3 geom arrays, cone_type, impratio)
- `gpu/constants.mojo` — CONTACT_SIZE 13→20, MODEL_GEOM_SIZE 17→20, new indices, workspace sizes
- `gpu/buffer_utils.mojo` — Pack/unpack new geom + contact fields
- `constraints/constraint_data.mojo` — +4 constraint type constants
- `constraints/constraint_builder.mojo` — condim-aware friction rows, pyramidal branch, writeback
- `constraints/constraint_builder_gpu.mojo` — Extended warmstart/writeback
- `solver/friction_solver.mojo` — Variable condim, both cone types, GPU
- `solver/pgs_solver.mojo` — CPU: group-based friction; GPU: extended workspace + condim
- `solver/cg_solver.mojo` — Same pattern
- `solver/newton_solver.mojo` — Same pattern
- `dynamics/jacobian.mojo` — `compute_angular_jacobian_row_gpu` for torsion/rolling
- `collision/contact_detection.mojo` — Friction combination (max), condim propagation, CPU + GPU
- `model/model_def.mojo` — Copy new geom fields to Model
- `integrator/euler_integrator.mojo` — MAX_ROWS formula
- `integrator/implicit_fast_integrator.mojo` — MAX_ROWS formula

---

### 3.3 Equality Constraints (weld + connect) — DONE

**Status**: COMPLETE. Both connect (ball joint, 3 rows) and weld (rigid attachment, 6 rows)
equality constraints implemented with bilateral PGS solving, MuJoCo impedance model,
and full CPU + GPU support. Backward compatible: `MAX_EQUALITY` defaults to 0.

**Two constraint types**:
- **Connect** (3 rows): Position-only ball joint. Error = `world_anchor_a - world_anchor_b`. Bilateral (`lo=-inf, hi=+inf`).
- **Weld** (6 rows): Rigid attachment. 3 position rows + 3 orientation rows. Orientation error = `0.5 * imag(conj(quat_b) * quat_a * relpose)`. Bilateral.

**Implementation**:
- `EqualityConstraintDef` struct on Model with per-constraint `solref`/`solimp`
- `EqualitySpec` trait + `ConnectConstraint`/`WeldConstraint` compile-time specs
- `Equalities` variadic container in `ModelDef` (mirrors `Bodies`/`Joints`/`Geoms` pattern)
- `MAX_EQUALITY` parameter threaded through Model, integrators, solvers, and GPU paths
- `MAX_ROWS` formula updated: `11 * MAX_CONTACTS + 2 * NJOINT + 6 * MAX_EQUALITY`

**CPU implementation** (`constraint_builder.mojo`):
- Phase 4 in builder: computes world anchors, position Jacobians (reuses `compute_contact_jacobian_row`),
  angular Jacobians (reuses torsional friction pattern), MuJoCo impedance with smoothstep
- Bilateral sign handling: `bias = -K*imp*pen + B*v_n`, flipped when error is negative
- All 3 solvers (PGS, CG, Newton): bilateral PGS iteration block after limits (no `lambda >= 0` clamping)

**GPU implementation** (`constraint_builder_gpu.mojo`):
- `build_and_solve_equality_gpu()`: self-contained function (like `detect_and_solve_limits_gpu`)
- Reads equality defs from GPU model buffer, computes world anchors, builds J + MinvJ, runs bilateral PGS
- InlineArray-based (no workspace allocation needed — equality count is small)
- Called from all 3 GPU solvers after limits, before friction

**GPU buffer layout**:
- `MODEL_EQ_SIZE = 18` floats per constraint (type, body_a, body_b, anchor_a xyz, anchor_b xyz, relpose xyzw, solref 2, solimp 3)
- `MODEL_META_IDX_NEQUALITY = 20` in model metadata
- `model_size` includes `+ NEQUALITY * MODEL_EQ_SIZE`

**Files created**:
- `model/equality_spec.mojo` — `EqualitySpec` trait, `ConnectConstraint`, `WeldConstraint` structs

**Files modified**:
- `types.mojo` — `EqualityConstraintDef`, `EQ_CONNECT`/`EQ_WELD` constants, `MAX_EQUALITY` param on Model
- `model/model_def.mojo` — `Equalities` variadic container, `ModelDef` update
- `constraints/constraint_data.mojo` — `CNSTR_EQUALITY_CONNECT=8`, `CNSTR_EQUALITY_WELD=9`, `num_equality`
- `constraints/constraint_builder.mojo` — Phase 4: build equality rows (CPU)
- `constraints/constraint_builder_gpu.mojo` — `build_and_solve_equality_gpu()` (GPU)
- `solver/pgs_solver.mojo` — CPU: bilateral PGS block; GPU: call `build_and_solve_equality_gpu`
- `solver/cg_solver.mojo` — Same pattern
- `solver/newton_solver.mojo` — Same pattern
- `integrator/euler_integrator.mojo` — `MAX_ROWS` formula, `MAX_EQUALITY` in `step`/`simulate`/`step_gpu`
- `integrator/implicit_fast_integrator.mojo` — Same
- `traits/solver.mojo` — `NGEOM`/`MAX_EQUALITY` on `solve_gpu`
- `traits/integrator.mojo` — `MAX_EQUALITY` on `step_gpu`
- `gpu/constants.mojo` — `MODEL_EQ_SIZE`, `EQ_IDX_*` constants, `MODEL_META_IDX_NEQUALITY`, `model_equality_offset`, `model_size` update
- `gpu/buffer_utils.mojo` — `copy_equality_to_buffer()`, `NEQUALITY` param on `create_model_buffer`

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

### 4.1 Broadphase Collision (bounding sphere) — DONE

**Status**: COMPLETE. Pre-computed bounding sphere radius per geom, used as a
broadphase filter before narrowphase dispatch. Skips expensive narrowphase
primitives (SAT for boxes, closest-point for capsules) when bounding spheres
don't overlap. Plane geoms skip the broadphase check (they're infinite).

**Bounding radius formulas** (matching MuJoCo Warp `collision_driver.py`):
- Sphere: `rbound = radius`
- Capsule: `rbound = half_length + radius`
- Box: `rbound = sqrt(hx^2 + hy^2 + hz^2)`
- Plane: `rbound = 1e10` (sentinel, broadphase skipped for planes)

**Implementation**:
- `geom_rbound[]` array on Model struct, computed at model setup time
- Broadphase check inserted after world pos computation, before narrowphase dispatch
- Both CPU and GPU paths updated
- GPU: `GEOM_IDX_RBOUND = 20`, `MODEL_GEOM_SIZE = 21`

**Files modified**:
- `types.mojo` — `Model.geom_rbound[]` array
- `model/model_def.mojo` — compute `geom_rbound` from geom type at setup
- `collision/contact_detection.mojo` — broadphase check in CPU + GPU paths
- `gpu/constants.mojo` — `MODEL_GEOM_SIZE 20→21`, `GEOM_IDX_RBOUND = 20`
- `gpu/buffer_utils.mojo` — pack `geom_rbound` into GPU model buffer

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

### 5.1 Solver Warmstart — DONE

**Status**: COMPLETE. All three solvers (PGS, CG, Newton) warm-start from the
previous step's constraint forces on both CPU and GPU. Solved forces are written
back to `data.contacts` via `writeback_forces()` (CPU) and stored in the state
buffer (GPU) for use in the next timestep.

**Implementation**:
- `constraint_builder.mojo`: `writeback_forces()` saves solved normal/friction forces back to contacts
- `constraint_builder_gpu.mojo`: `warmstart_normals_gpu()` applies warm-start impulses to predicted velocity
- `constraint_data.mojo`: `ConstraintRow.force` field initialized from previous step's solved forces
- All 3 solvers: Load warm-start at solve start, apply to qacc, then write back final forces
- GPU path: `warmstart_normals_gpu` + friction warm-start in each solver's `solve_gpu`

**Files modified**:
- `constraints/constraint_builder.mojo`, `constraints/constraint_builder_gpu.mojo`
- `constraints/constraint_data.mojo`
- `solver/pgs_solver.mojo`, `solver/cg_solver.mojo`, `solver/newton_solver.mojo`
- `solver/friction_solver.mojo`

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

### 5.4 Actuator Dynamics — DONE

**Status**: COMPLETE. Trait-based compile-time actuator system following the
BodySpec/JointSpec/GeomSpec pattern. Supports 4 actuator types (Motor, Position,
Velocity, General) with full MuJoCo gain/bias pipeline, force clamping, and
qDeriv contributions for implicit integration.

**Architecture**:
- `ActuatorSpec` trait with 19 compile-time fields (joint_idx, dof_adr, qpos_adr, gear, dyntype, gaintype, biastype, gain/bias params, ctrl/force ranges)
- `Actuators[*A: ActuatorSpec]` variadic container with `apply_actions` (CPU), `apply_actions_gpu` (GPU), `compute_qderiv_contribution` (CPU/GPU)
- All actuator params are compile-time via `@parameter for` — no GPU buffer storage needed

**Actuator types**:
- `MotorActuator[joint_idx, dof_adr, gear]`: `force = gear * clamp(ctrl, ctrl_min, ctrl_max)`
- `PositionActuator[joint_idx, dof_adr, kp, kd]`: `force = kp*(ctrl - qpos) - kd*qvel` (PD servo)
- `VelocityActuator[joint_idx, dof_adr, kv]`: `force = kv*(ctrl - qvel)` (velocity servo)
- `GeneralActuator[...]`: Full control over dyntype, gaintype, biastype, all params

**MuJoCo force pipeline** (implemented):
```
1. Clamp ctrl to [ctrl_min, ctrl_max]
2. Compute gain: FIXED → gainprm_0, AFFINE → gainprm_0 + gainprm_1*qpos + gainprm_2*qvel
3. Compute bias: NONE → 0, AFFINE → biasprm_0 + biasprm_1*qpos + biasprm_2*qvel
4. Force = gain * ctrl + bias
5. Clamp force to [force_min, force_max]
6. qfrc[dof_adr] = force
```

**Environments migrated**: HalfCheetah (6 MotorActuators, gear=120/90/60/120/60/30), Hopper (3 MotorActuators, gear=200)

**Not yet implemented**: Activation dynamics (DYN_INTEGRATOR, DYN_FILTER) — constants defined, runtime pipeline ready, but no env uses them yet.

**Files**: `actuator_spec.mojo` (NEW), `model_def.mojo`, `types.mojo`, `half_cheetah_def.mojo`, `hopper_def.mojo`, `half_cheetah.mojo`, `hopper.mojo`

---

### 5.5 Tendon System (fixed tendons)

**Problem**: MuJoCo supports tendons (cables that span multiple joints). The Humanoid
environment uses `<fixed>` tendons to couple hip and knee joints.

**Used by**: Humanoid, HumanoidStandup (hip-knee coupling).

**Fixed tendon** = linear constraint on joint positions/velocities:
```xml
<fixed name="left_hipknee">
    <joint coef="-1" joint="left_hip_y"/>
    <joint coef="1" joint="left_knee"/>
</fixed>
```

This enforces `Σ coef_i * q_i = const` as a bilateral equality constraint.
The tendon Jacobian is trivial: `J[dof_i] = coef_i` for each participating joint.

**Implementation plan**:
1. Add `TendonSpec` trait with `COEFS` and `JOINT_INDICES`
2. Add `Tendons` variadic container (like `Equalities`)
3. In constraint builder, add tendon rows as bilateral equality constraints
4. Tendon Jacobian: `J[dof_adr_i] = coef_i` — simplest possible Jacobian

**Recommendation**: Medium priority. Required for Humanoid. Spatial tendons (routing
through bodies) are low priority — only fixed tendons are needed for Gymnasium envs.

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
  2.2 Implicit integrator       DONE (full RNE velocity derivative, LU factorization, CPU + GPU)
  2.3 RK4 integrator            DONE (4th-order Runge-Kutta, energy conservation)
  5.3 Passive forces            DONE (armature, damping, stiffness+springref, frictionloss)

Sprint 3 (Constraint system):
  3.1 Unified constraint rows   DONE (ConstraintData/ConstraintRow, constraint_builder)
  3.2 Friction cone models      DONE (pyramidal + elliptic, condim 1/3/4/6, QCQP solver, CPU + GPU)
  3.3 Equality constraints      DONE (connect + weld, bilateral PGS, MuJoCo impedance, CPU + GPU)
  3.4 Per-contact solref/solimp DONE (impedance model, all 3 solvers, CPU + GPU)

Sprint 4 (Polish):
  4.1 Broadphase (spheres)      DONE (bounding sphere pre-filter, CPU + GPU)
  5.1 Solver warmstart          DONE (all 3 solvers, CPU + GPU, writeback_forces)
  5.4 Actuator dynamics         DONE (ActuatorSpec trait, Motor/Position/Velocity/General, CPU + GPU)

Sprint 5 (Next environments — Walker2d, Ant):
  6.4 Contact margin            DONE (per-geom margin, adjusted dist, CPU + GPU)
  6.1 settotalmass              DONE (ModelDefaultsLike trait, ModelDef.finalize rescaling)
  6.3 fromto capsule spec       DONE (FromToCapsule struct, compile-time conversion)
  6.7 Geom density              DONE (subsumed by 6.2 inertiafromgeom)

Sprint 6 (Humanoid):
  5.5 Fixed tendons             <- Humanoid hip-knee coupling
  6.10 cfrc_ext                 <- Humanoid observations
  6.6 Cylinder geom             <- Reacher, Pusher
  6.2 inertiafromgeom           <- DONE

Sprint 7 (Specialized):
  6.9 Fluid dynamics            <- Swimmer only
  6.5 Full solimp (5 params)    <- non-default impedance curves
  6.8 Site elements             <- InvertedDoublePendulum
  5.2 Solver islands            <- multi-agent parallelism
  4.2 Broadphase (AABB/SAP)     <- large scenes
  0.5 MJCF XML parser           <- automate model translation

Render Sprint (independent, can be done in parallel with any physics sprint):
  R.1 Multi-geom-type render    <- DONE (dispatch on GEOM_TYPE + renderer factorization)
  R.2 Ground plane render       <- DONE (plane geom drives ground at POS_Z)
  R.3 RGBA alpha                <- DONE (transparency support on GeomSpec + Renderer3D)
  R.4 Camera spec from model    <- DONE (CameraSpec trait + TrackCamera, env renderers use camera specs)
  R.5 Lighting model            <- DONE (LightSpec trait + DirectionalLight, configurable light params in Renderer3D/ModelRenderer)
  R.6 Materials & textures      <- MaterialSpec, built-in textures (checker, gradient), skybox
  R.7 Site markers (visual)     <- render sites as small spheres/crosses (depends on 6.8)
```

---

## Render Sprint: Visual Fidelity

This sprint is **independent from the physics pipeline** and can be done in parallel
with any physics sprint. The goal is to close the gap between our wireframe/solid-color
rendering and MuJoCo's visual output. Items are ordered by impact-to-effort ratio.

**Current state**: `ModelRenderer[*G: GeomSpec]` draws all body-attached geoms as capsules
using compile-time `COLOR`, `RADIUS`, `HALF_LENGTH` from `GeomSpec`. Camera follows torso.
Ground is a wireframe grid. No lighting, materials, or textures.

---

### R.1 Multi-Geom-Type Rendering

**Status**: DONE.
**Effort**: ~1 day.
**Priority**: High — currently ALL geoms render as capsules regardless of type.

**Implemented**: `ModelRenderer.render()` now dispatches on `GG.GEOM_TYPE` inside the
`@parameter for` loop, calling `draw_capsule()`, `draw_sphere()`, or `draw_box()`
as appropriate. World position and orientation are computed once per geom (with local
pos/quat offsets applied), then the correct draw primitive is called.

Also factored out per-environment renderer boilerplate:
- `HalfCheetahRenderer` and `HopperRenderer` wrapper structs replaced with `comptime`
  type aliases for `ModelRenderer[...geoms...]` (~160 lines removed)
- Added `render_from_body_state()` method on `ModelRenderer` that takes raw `xpos`/`xquat`
  `InlineArray`s directly from `Data`, eliminating identical body-extraction loops in each env
- Camera defaults moved inline to each env's `init_renderer()`

**Files modified**:
- `model/model_renderer.mojo` — dispatch on `GEOM_TYPE`, added `render_from_body_state()`
- `envs/half_cheetah/renderer.mojo` — struct replaced with type alias
- `envs/hopper/renderer.mojo` — struct replaced with type alias
- `envs/half_cheetah/half_cheetah.mojo` — simplified `render_frame()`
- `envs/hopper/hopper.mojo` — simplified `render_frame()`

---

### R.2 Ground Plane Rendering

**Status**: DONE.
**Effort**: ~0.5 day.
**Priority**: Medium — replaces wireframe grid with a solid ground.

**Implemented**: `GEOM_PLANE` geoms are now rendered via `draw_ground_grid()` at their
`POS_Z` height (from GeomSpec). The plane is driven by the geom definition, not hardcoded.
A fallback grid is drawn if no plane geom is defined. The existing `Renderer3D.draw_ground_grid()`
already renders a procedural checkerboard ground with reflections.

**Files modified**:
- `model/model_renderer.mojo` — plane geoms rendered via `draw_ground_grid()` at `GG.POS_Z`

---

### R.3 Color Unification + RGBA Alpha Support

**Status**: DONE.

**What was done**:
1. Renamed `SDL_Color` → `Color` in `render/types.mojo` (RGBA struct, the single color type)
2. Added `comptime SDL_Color = Color` alias for backward compatibility with SDL FFI code
3. Deleted `Color3D` struct from `render/renderer3d.mojo` — all 3D code now uses `Color`
4. Added `color_to_vec4(color: Color)` overload in `render/gpu_types.mojo` — passes alpha through
5. Updated `GeomSpec.COLOR` and `BodySpec.COLOR` from `Color3D` to `Color` (alpha in `Color.a`)
6. Updated all environment defs, examples, and tests

**Result**: One unified `Color` type (RGBA) for both 2D and 3D rendering. Transparency support
is now plumbed through to the GPU shader via `color_to_vec4`. Back-to-front sorting for
transparent geoms is not yet implemented (only needed when alpha < 255).

---

### R.4 Camera Spec from Model

**Status**: **DONE**.

**What was done**:
1. Created `model/camera_spec.mojo` — `CameraSpec` trait, `TrackCamera` struct, `CAM_TRACKCOM`/`CAM_FIXED` constants
2. Added `Cameras` variadic container to `model/model_def.mojo`
3. Added `HalfCheetahCamera` (pos_y=-3.0, pos_z=0.3) and `HopperCamera` (pos_y=-3.0, pos_z=-0.25) to env defs
4. Updated renderer construction in both environments to use camera spec values
5. Exported camera specs from renderer modules and `__init__.mojo`

---

### R.5 Lighting Model

**Status**: **DONE**.
**Effort**: ~0.5 day (Phase 1: configurable light params from model spec).

**What was done**:
1. Created `model/light_spec.mojo` — `LightSpec` trait + `DirectionalLight` struct with
   compile-time fields: `DIR_X/Y/Z`, `COLOR_R/G/B`, `AMBIENT`, `SPECULAR_INTENSITY`,
   `SPECULAR_EXPONENT`, `CAST_SHADOW`, `MODE` (directional/point)
2. Added `Lights` variadic container to `model/model_def.mojo`
3. Made `Renderer3D._build_scene_uniforms()` use configurable `light_dir`, `light_color`,
   `light_ambient` fields instead of hardcoded literals
4. Added light params to `ModelRenderer.__init__()` (pass through to `Renderer3D`)
5. Both env definitions (`HalfCheetahLight`, `HopperLight`) use `DirectionalLight[]` defaults
6. Env renderers pass light spec values to `ModelRenderer` constructor

**Note**: Blinn-Phong shading + shadow mapping already exist in the MSL shaders.
This task made the light configuration data-driven from model specs. Specular intensity
and exponent are defined on `LightSpec` for future shader parameterization (R.6).

**Files created**:
- `model/light_spec.mojo` — `LightSpec` trait, `DirectionalLight` struct, mode constants

**Files modified**:
- `model/__init__.mojo` — exports
- `model/model_def.mojo` — `Lights` container + `LightSpec` import
- `render/renderer3d.mojo` — configurable light fields in struct, init, moveinit, uniforms
- `model/model_renderer.mojo` — light params in struct, init, moveinit, passthrough
- `envs/half_cheetah/half_cheetah_def.mojo` — `HalfCheetahLight` alias
- `envs/hopper/hopper_def.mojo` — `HopperLight` alias
- `envs/half_cheetah/renderer.mojo` — export `HalfCheetahLight`
- `envs/hopper/renderer.mojo` — export `HopperLight`
- `envs/half_cheetah/half_cheetah.mojo` — pass light params to renderer
- `envs/hopper/hopper.mojo` — pass light params to renderer

---

### R.6 Materials & Textures

**Status**: NOT STARTED.
**Effort**: ~3-5 days.
**Priority**: Low for RL — cosmetic improvement only.

**Problem**: MuJoCo XML defines textures and materials:
```xml
<texture builtin="checker" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d"/>
<material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texture="texplane"/>
<texture builtin="gradient" type="skybox" rgb1="1 1 1" rgb2="0 0 0"/>
```
Our renderer has zero texture/material support.

**What to do**:
1. Add `TextureSpec` for built-in textures (checker, gradient, flat) — procedurally generated,
   no file I/O needed
2. Add `MaterialSpec` with `SHININESS`, `SPECULAR`, `REFLECTANCE`, `TEXTURE_NAME`
3. Per-geom material reference: `MATERIAL` field on `GeomSpec`
4. Skybox: render a gradient background behind the scene
5. Texture mapping for ground plane (checker) and geom surfaces (cube-mapped)

**Files to create**:
- `model/texture_spec.mojo` — `TextureSpec` trait, `CheckerTexture`, `GradientTexture`
- `model/material_spec.mojo` — `MaterialSpec` trait

**Files to modify**:
- `model/geom_spec.mojo` — `MATERIAL` reference field
- `model/model_def.mojo` — `Textures`, `Materials` containers
- `render/renderer3d.mojo` — texture sampling, material application

---

### R.7 Site Markers (visual)

**Status**: NOT STARTED. Depends on Phase 6.8 (site elements).
**Effort**: ~0.5 day (after 6.8 is done).
**Priority**: Low — massless reference points, optional visualization.

**Problem**: MuJoCo sites (e.g., `<site name='tip' pos='.15 0 .11'/>`) are useful debugging
markers. Once `SiteSpec` exists (Phase 6.8), they should be optionally renderable.

**What to do**:
1. In `ModelRenderer`, optionally iterate over sites and draw small spheres or crosses
   at their world positions
2. Use a distinct color (e.g., bright green) and small radius (e.g., 0.01m)
3. Gate behind a `show_sites: Bool` flag (default `False`)

**Files to modify**:
- `model/model_renderer.mojo` — optional site rendering loop

---

### Render Sprint Summary

| Item | Effort | Priority | Dependencies |
|------|--------|----------|--------------|
| R.1 Multi-geom-type render | ~1 day | **DONE** | None |
| R.2 Ground plane render | ~0.5 day | **DONE** | None |
| R.3 Color Unification + RGBA Alpha Support | ~0.5 day | **DONE** | None |
| R.4 Camera spec | ~0.5 day | **DONE** | None |
| R.5 Lighting model | ~0.5 day | **DONE** | None |
| R.6 Materials & textures | ~3-5 days | Low | R.5 (for material shading) |
| R.7 Site markers | ~0.5 day | Low | Phase 6.8 (site elements) |
| **Total** | **~8-11 days** | | |

**Recommended order**: ~~R.1~~ → ~~R.2~~ → R.4 → R.3 → R.5 → R.6 → R.7

R.1 + R.2 are done. Next up: R.4 (camera spec) for the biggest remaining improvement
correctly with a solid ground plane. The rest is cosmetic polish.

---

## Phase 6: MuJoCo XML Compatibility Gaps

Gaps identified by comparing our model definition against all Gymnasium MuJoCo XML
files (HalfCheetah, Hopper, Walker2d, Ant, Humanoid, Swimmer, Reacher, Pusher,
InvertedPendulum, InvertedDoublePendulum).

### Gymnasium MuJoCo Environment Feature Matrix

| Feature | HalfCheetah | Hopper | Walker2d | Ant | Humanoid | Swimmer | Reacher | Pusher |
|---|---|---|---|---|---|---|---|---|
| **Joints** | | | | | | | | |
| hinge | YES | YES | YES | YES | YES | YES | YES | YES |
| slide | YES | YES | YES | - | - | YES | YES | YES |
| free | - | - | - | YES | YES | - | - | - |
| **Geoms** | | | | | | | | |
| capsule | YES | YES | YES | YES | YES | YES | YES | YES |
| plane | YES | YES | YES | YES | YES | YES | YES | YES |
| sphere | - | - | - | YES | YES | - | YES | YES |
| cylinder | - | - | - | - | - | - | YES | YES |
| `fromto` spec | YES | - | - | YES | YES | YES | YES | YES |
| **Root DOF** | | | | | | | | |
| Planar (2 slide+hinge) | YES | YES | YES | - | - | YES | - | - |
| Free (6DOF) | - | - | - | YES | YES | - | - | - |
| **Integrator** | Euler | RK4 | RK4 | RK4 | RK4 | RK4 | RK4 | Euler |
| **Special** | | | | | | | | |
| settotalmass | YES | - | - | - | - | - | - | - |
| margin on geom | - | YES | - | YES | YES | - | - | YES |
| condim=1 default | - | YES | - | - | YES | YES | - | YES |
| Fluid dynamics | - | - | - | - | - | YES | - | - |
| Tendons (fixed) | - | - | - | - | YES | - | - | - |
| geom density | - | - | YES | YES | - | YES | - | YES |

---

### 6.1 settotalmass Compiler Directive — DONE

**Status**: DONE.
**Used by**: HalfCheetah only.

MuJoCo's `<compiler settotalmass="14"/>` rescales all body masses and inertias after
model setup so the total equals the target.

**Implementation**:
- Added `SETTOTALMASS: Float64` to `ModelDefaultsLike` trait and `ModelDefaults` struct (sentinel -1.0 = disabled)
- `ModelDef.finalize()` takes a `Defaults: ModelDefaultsLike` type parameter; when `SETTOTALMASS > 0`, it sums masses of bodies 1..NBODY (skipping worldbody), computes `scale = target / total`, and scales `body_mass`, `body_inv_mass`, `body_inertia`, `body_inv_inertia`
- HalfCheetahDefaults sets `settotalmass=14.0`; finalize call sites pass `[Defaults=HalfCheetahDefaults]`
- Since HalfCheetah body masses are already pre-scaled to sum to 14, the scaling is a no-op (scale ~= 1.0)
- Hopper and other envs use the default `ModelDefaults[]` (settotalmass=-1.0, disabled)

**Files changed**: `model_def.mojo`, `half_cheetah_def.mojo`, `half_cheetah.mojo`

---

### 6.2 inertiafromgeom from Child Geoms — DONE

**Status**: DONE. Implements MuJoCo's `inertiafromgeom="true"` compiler directive.
Computes body mass, inertia, ipos (CoM offset), and iquat (inertia frame) from
child geoms using density, volume, parallel axis theorem, and Jacobi eigendecomposition.
Also subsumes 6.7 (geom density).

**Used by**: ALL Gymnasium environments.

**Implementation**:
- `DENSITY` and `GEOM_MASS` fields on `GeomSpec` trait (sentinel -1.0 = use default)
- `INERTIAFROMGEOM` and `GEOM_DENSITY` on `ModelDefaultsLike` (defaults: `True`, `1000.0`)
- `geom_mass` array on `Model` struct, computed in `Geoms.setup_model` (priority: explicit mass > explicit density > default density)
- `compute_inertia_from_geoms()` called in `finalize()` before `settotalmass`
- Single-geom bodies: direct copy of pos/quat/mass/inertia
- Multi-geom bodies: mass-weighted CoM, full 6-element inertia tensor via `globalinertia` + `offcenter`, then Jacobi eigendecomposition for principal axes

Key functions in `inertia_from_geom.mojo`:
- `geom_volume()` — sphere, capsule, box volume formulas
- `geom_inertia()` — MuJoCo-matching diagonal inertia (capsule uses hemisphere parallel axis theorem)
- `globalinertia()` — R * diag(I) * R^T (matches `mjuu_globalinertia`)
- `offcenter()` — parallel axis theorem (matches `mjuu_offcenter`)
- `eig3_symmetric()` — Jacobi eigendecomposition (matches `mjuu_eig3`)

**Validation**: `test_inertiafromgeom_vs_mujoco.mojo` — HalfCheetah + Hopper pass (mass err ~1e-16, inertia err ~2e-8)

**Files**: `model/inertia_from_geom.mojo` (NEW), `model/geom_spec.mojo`, `model/model_def.mojo`, `types.mojo`, `model/__init__.mojo`

---

### 6.3 fromto Capsule Specification — DONE

**Status**: DONE.
**Used by**: HalfCheetah, Ant, Humanoid, Swimmer, Reacher, Pusher.
**Impact**: Low (conversion is straightforward, currently done manually).

MuJoCo capsules can be defined by two endpoints: `fromto="x1 y1 z1 x2 y2 z2"`.
Conversion to center + axis-angle + half_length:
```
center = (from + to) / 2
axis = normalize(to - from)
half_length = length(to - from) / 2
quat = axis_to_quat(axis)  # rotation from default Z-axis to axis
```

**Implementation**: `FromToCapsule` struct in `geom_spec.mojo` that conforms to
`GeomSpec`. Takes `from_x/y/z` and `to_x/y/z` parameters and computes center,
half_length, and orientation quaternion at compile time.

Compile-time helpers:
- `_comptime_sqrt()`: Newton's method sqrt (10 iterations, machine precision)
- `_fromto_center_x/y/z()`: Midpoint calculation
- `_fromto_half_length()`: Distance / 2
- `_fromto_quat_component()`: Quaternion from Z-to-axis rotation using half-angle
  formula `q = normalize(cross(Z, d), 1 + dot(Z, d))` — avoids trig functions.

Usage:
```mojo
# MuJoCo XML: <geom fromto="-.5 0 0 .5 0 0" size="0.046" type="capsule"/>
comptime TorsoGeom = FromToCapsule[
    body_idx=1, radius=0.046,
    from_x=-0.5, to_x=0.5,
]
```

**Files modified**: `geom_spec.mojo`, `model/__init__.mojo`.

---

### 6.4 Contact Margin

**Status**: DONE.
**Used by**: Hopper (`margin=0.001`), Ant (`margin=0.01`), Humanoid (`margin=0.001`), Pusher (`margin=0.002`).

MuJoCo generates contacts when `dist < margin` instead of `dist < 0`. This creates
a "soft zone" where contacts activate before actual penetration, enabling smoother
force onset.

**Design**: "Adjusted dist" — store `dist_adjusted = raw_dist - margin` in the contact.
All downstream solver checks (`dist >= 0`, `penetration = -dist`) automatically work
correctly with zero solver file changes. Contact position is computed from raw dist
before adjustment. Margin defaults to 0.0 (fully backward compatible).

**Implementation**:
1. `MARGIN: Float64` on `GeomSpec` trait + all concrete geom structs (Plane, Sphere, Capsule, Box)
2. `GEOM_MARGIN` on `ModelDefaultsLike` trait + `ModelDefaults` (default 0.0)
3. `geom_margin[NGEOM]` array on `Model`, resolved in `Geoms.setup_model()`
4. GPU: `GEOM_IDX_MARGIN = 26`, `MODEL_GEOM_SIZE = 27`, packed in `buffer_utils.mojo`
5. `contact_detection.mojo`: CPU + GPU paths — `contact_margin = max(margin_gi, margin_gj)`,
   activation check `dist < contact_margin`, stored dist adjusted `dist - contact_margin`
6. Hopper: `geom_margin=0.001` set in `HopperDefaults`

**Files modified**: `geom_spec.mojo`, `model_def.mojo`, `types.mojo`, `constants.mojo`,
`buffer_utils.mojo`, `contact_detection.mojo`, `hopper_def.mojo`.

---

### 6.5 Full solimp (5 params)

**Status**: PARTIAL. Currently storing `solimp[0..2]` = `[dmin, dmax, width]`.
MuJoCo uses `solimp[5]` = `[dmin, dmax, width, midpoint, power]`.

**Used by**: All environments (but midpoint=0.5, power=2.0 are the defaults that
99% of models use).
**Impact**: Low — only matters for non-default impedance curves.

The smoothstep currently uses hardcoded midpoint=0.5, power=2.0:
```
x = clamp(penetration / width, 0, 1)
imp = dmin + (3x² - 2x³) * (dmax - dmin)  // hardcoded cubic Hermite
```

MuJoCo generalizes this with configurable midpoint `m` and power `p`:
```
y = 0.5 - cos(pi * clamp(x, 0, 1)) / 2  // sinusoidal smoothstep
imp = dmin + y^(1/p) * (dmax - dmin)       // power curve
```

Wait — MuJoCo actually uses a different formula. From `engine_core_constraint.c`:
```
t = penetration / width
imp = dmin + (dmax - dmin) * smoothstep(t, midpoint, power)
```

**Implementation**: Add `solimp[3]` (midpoint) and `solimp[4]` (power) throughout
the impedance calculation in constraint builder + GPU. Low effort but touches many files.

---

### 6.6 Cylinder Geom Collision

**Status**: NOT STARTED.
**Used by**: Reacher (root post), Pusher (object, goal).
**Impact**: Medium — needed for Reacher and Pusher environments.

Required collision primitives:
- `cylinder-plane`: Similar to capsule-plane but with flat ends
- `cylinder-sphere`: Closest point on cylinder surface to sphere center
- `cylinder-capsule`: SAT or GJK
- `cylinder-cylinder`: Most complex, SAT-based

MuJoCo reference: `engine_collision_primitive.c` (`mjc_PlaneCylinder`, `mjc_SphereCylinder`).

**Implementation**:
1. Add `GEOM_CYLINDER = 4` constant
2. Add `CylinderGeom` / `BodyCylinderGeom` to `geom_spec.mojo`
3. Implement narrowphase primitives in `collision_primitives.mojo`
4. Register in dispatch table in `contact_detection.mojo`

---

### 6.7 Geom Density (mass from density) — DONE

**Status**: DONE. Subsumed by 6.2 (inertiafromgeom).

Per-geom `DENSITY` and `GEOM_MASS` fields on `GeomSpec` trait with sentinel -1.0.
Model-level default density via `GEOM_DENSITY` on `ModelDefaultsLike` (default 1000.0).
Mass resolution priority: explicit `GEOM_MASS` > explicit `DENSITY * volume` > `default_density * volume`.
Computed in `Geoms.setup_model` and stored in `model.geom_mass[NGEOM]`.

**Files**: `model/geom_spec.mojo`, `model/model_def.mojo`, `types.mojo`

---

### 6.8 Site Elements

**Status**: NOT STARTED.
**Used by**: InvertedDoublePendulum (tip position for reward).
**Impact**: Low — massless reference points.

Sites are body-attached reference points with zero mass/inertia. They participate
in FK (get world position/orientation) but not dynamics. Used for:
- Observation reference points (tip of pendulum)
- Sensor attachment points
- Reward computation

**Implementation**:
1. Add `SiteSpec` trait with `BODY_IDX`, `POS_X/Y/Z`
2. Add `site_xpos[NSITE*3]` to `Data`, computed during FK
3. FK: `site_xpos = body_xpos + rotate(site_pos, body_xquat)`

---

### 6.9 Fluid Dynamics (density/viscosity)

**Status**: NOT STARTED.
**Used by**: Swimmer only (`density=4000`, `viscosity=0.1`).
**Impact**: Low — specialized feature.

MuJoCo applies drag and buoyancy forces when `option.density > 0`:
```
For each geom:
    F_drag = -0.5 * density * Cd * A * |v| * v  (quadratic drag)
    F_viscous = -viscosity * 6π * r * v          (Stokes drag)
    F_buoyancy = density * volume * g             (Archimedes)
```

MuJoCo reference: `engine_passive.c` (`mj_passive`, fluid model).

**Recommendation**: Implement only when adding Swimmer environment. Self-contained
addition to the force pipeline.

---

### 6.10 cfrc_ext (Contact Forces per Body)

**Status**: NOT STARTED.
**Used by**: Humanoid (external contact forces as part of observation).
**Impact**: Medium — needed for Humanoid observations.

`cfrc_ext` accumulates contact constraint forces per body in Cartesian (6D) space.
After constraint solving:
```
For each contact:
    F_contact = lambda * J_contact  (in joint space)
    For body_a and body_b:
        cfrc_ext[body] += contact_force_at_point (6D: torque + force)
```

**Implementation**:
1. Add `cfrc_ext[NBODY*6]` to `Data`
2. After solver, accumulate contact forces per body using contact Jacobians
3. Transform to body-local frame if needed (MuJoCo stores in subtree CoM frame)

---

### 6.11 Runtime Solver Selection from XML

**Status**: NOT STARTED.
**Used by**: Humanoid (`solver="PGS"`, `iterations="50"`).
**Impact**: Low — currently solver type is a compile-time parameter.

MuJoCo allows selecting solver type and iteration count in `<option>`:
```xml
<option solver="PGS" iterations="50"/>
```

Our solver type is a compile-time generic parameter (`EulerIntegrator[PGSSolver]`).
Making this runtime would require either:
- A solver dispatch enum (runtime overhead)
- Or generating separate compiled paths per solver type

**Recommendation**: Keep compile-time solver selection. Document which solver each
environment should use. The MJCF parser can select the correct compile-time path.

---

## Implementation Priority — Per Environment

What's needed to support each Gymnasium MuJoCo environment:

### HalfCheetah — SUPPORTED
Currently working. Minor gaps: `fromto` capsules (manually converted).
`settotalmass` now implemented (DONE).

### Hopper — SUPPORTED
Currently working. Minor gaps: `ref` on rootz joint
(verify `springref` handles this), RK4 integrator (now DONE).
`margin` on geoms — DONE (0.001).

### Walker2d — NEEDS: geom density
Similar to Hopper. Needs `density` on geoms. RK4 done. `margin` — DONE.

### Ant — NEEDS: geom density, sphere-sphere collision improvements
Uses free joint (already supported). Needs `density` on geoms.
Sphere geom for torso. Otherwise structurally similar to existing envs. `margin` — DONE.

### Humanoid — NEEDS: fixed tendons, cfrc_ext, sphere collision
Most complex environment. Key blockers:
1. Fixed tendons (hip-knee coupling) — Phase 5.5
2. `cfrc_ext` for observations — Phase 6.10
3. Sphere geoms (head, feet, hands)
`margin` on geoms — DONE.

### Swimmer — NEEDS: fluid dynamics
Unique requirement for drag/buoyancy forces. Otherwise simple (2 bodies, slide joints).

### Reacher — NEEDS: cylinder geom
Simple arm task. Needs cylinder geom for the root post. Zero gravity variant exists (Pusher).

### Pusher — NEEDS: cylinder geom, zero gravity
Table-top arm task. Needs cylinder geom and `gravity="0 0 0"` support
(gravity is already configurable — just set to zero).

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
