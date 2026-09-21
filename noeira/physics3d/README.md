# physics3d/ - 3D Generalized Coordinates Physics Engine

MuJoCo-inspired constraint-based physics engine using generalized (joint-space) coordinates. Supports CPU and GPU simulation with configurable constraint solvers.

Checked against live MuJoCo 3.12.0 — see [Validation](https://noeira.ai/docs/physics/validation/) and the `tests/physics3d/*_vs_mujoco*` gates.

## Architecture

The engine follows MuJoCo's design: models are defined at compile time with bodies, joints, geoms, and actuators. Simulation state (qpos, qvel, xpos, xquat) is stored in a `Data` struct. The pipeline is: collision detection -> constraint building -> constraint solving -> integration.

## Module Structure

```
physics3d/
├── types.mojo · constants.mojo · joint_types.mojo   Core types, defaults, JNT_FREE/BALL/SLIDE/HINGE
├── model/        Compile-time model specification (body, joint, geom, actuator, ModelDef, renderer hook)
├── parser/       MJCF: xml_parser, full_parser, expander (<include>/<attach>/defaults), flat_model,
│                 fields_build, runtime_load, model_def_from_xml, mesh_bvh_build, hfield_loader
├── fields/       Model / Data storage, dims, scratch pools (contact, dynamics, implicit, RK4)
├── kinematics/   Forward kinematics, quaternion math, sites (CPU + GPU)
├── dynamics/     CRBA mass matrix, RNE bias forces, Jacobians, LDL/LU, tendons + wrap,
│                 actuation, gravity compensation, OSC pose, velocity derivatives
├── collision/    SAP broadphase, primitives, GJK/EPA, multi-CCD, native multi-contact,
│                 convex hulls (qhull shim + hull cache), heightfields, robust predicates
├── constraints/  Constraint rows: contacts, limits, equality, tendons, friction (CPU + GPU)
├── solver/       PGS, island PGS, Newton (incl. blocked and elliptic-cooperative), CG,
│                 elliptic cones, noslip, warm start
├── integrator/   Euler, ImplicitFast, Implicit, RK4
├── sensors/      Touch, rangefinder, frame, subtree, site acceleration
├── ray/          Ray casts against geoms, meshes and heightfields
├── raytrace/     Batched GPU ray-traced camera renderer, appearance, host renderer
├── gpu/          Shared GPU kernels and constants
└── studio/       Physics studio: pick, gizmo, edit, history, validate, MJCF writer
```

## Supported Joint Types

| Type | DOF | Description |
|------|-----|-------------|
| `JNT_FREE` | 7 (3 pos + 4 quat) | Free-floating root body |
| `JNT_BALL` | 4 (quaternion) | Ball-and-socket joint |
| `JNT_SLIDE` | 1 | Prismatic (linear) joint |
| `JNT_HINGE` | 1 | Revolute (rotational) joint |

## Constraint Solvers

| Solver | Space | Description |
|--------|-------|-------------|
| **PGS** | Dual (lambda) | Projected Gauss-Seidel, good general-purpose |
| **Newton** | Primal (qacc) | Quadratic convergence for stiff contacts |
| **CG** | Primal (qacc) | Conjugate Gradient for well-conditioned systems |
| **IslandPGS** | Dual | PGS with per-island early termination |
| **Newton, blocked** | Primal | GPU Newton with per-block factorisation (NVIDIA; not routed on Metal) |
| **Newton, elliptic-coop** | Primal | Newton over elliptic friction cones |
| **Noslip** | Post-pass | MuJoCo's noslip iterations on the friction rows |

## Integrators

| Integrator | Description |
|------------|-------------|
| **Euler** | Semi-implicit Euler (simplest) |
| **ImplicitFast** | Default: M_hat = M + arm - dt*qDeriv (fast approximation) |
| **Implicit** | Full implicit with RNE velocity derivative (most stable) |
| **RK4** | 4th-order Runge-Kutta (explicit, high accuracy) |

## Key Design Patterns

- **Compile-time model definition**: Bodies, joints, geoms defined as type parameters via trait-based specs
- **Variadic iteration**: `ModelDef` uses `Variadic.types + comptime for` for N-body composition
- **CPU/GPU dual paths**: Most functions have both CPU (LayoutTensor) and GPU (DeviceBuffer) versions
- **Island detection**: Constraints partitioned into independent islands for faster solving
