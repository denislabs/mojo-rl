# physics3d/ - 3D Generalized Coordinates Physics Engine

MuJoCo-inspired constraint-based physics engine using generalized (joint-space) coordinates. Supports CPU and GPU simulation with configurable constraint solvers.

## Architecture

The engine follows MuJoCo's design: models are defined at compile time with bodies, joints, geoms, and actuators. Simulation state (qpos, qvel, xpos, xquat) is stored in a `Data` struct. The pipeline is: collision detection -> constraint building -> constraint solving -> integration.

## Module Structure

```
physics3d/
├── types.mojo              # Model[NQ,NV,NBODY,NJOINT], Data, ContactInfo
├── constants.mojo          # Geometry types, physics defaults, GPU config
├── joint_types.mojo        # JointDef, JNT_FREE/BALL/SLIDE/HINGE
├── model/                  # Compile-time model specification (17 files)
│   ├── body_spec.mojo      # BodySpec trait + CapsuleBody, SphereBody, BoxBody
│   ├── joint_spec.mojo     # JointSpec trait + HingeJoint, SlideJoint
│   ├── geom_spec.mojo      # GeomSpec trait + Plane, Sphere, Box, Capsule
│   ├── actuator_spec.mojo  # ActuatorSpec (motor/position/velocity control)
│   ├── equality_spec.mojo  # EqualitySpec (ball-joint/weld constraints)
│   ├── tendon_spec.mojo    # TendonSpec (fixed-distance constraints)
│   ├── site_spec.mojo      # SiteSpec (named body points)
│   ├── model_def.mojo      # ModelDef compositor (variadic iteration)
│   ├── model_renderer.mojo # 3D rendering integration
│   ├── inertia_from_geom.mojo # Auto-compute inertia from geometry
│   └── defaults_spec.mojo  # Default model parameters
├── kinematics/             # Forward kinematics + quaternion math
│   ├── forward_kinematics.mojo # qpos -> xpos/xquat (CPU + GPU)
│   └── quat_math.mojo     # Quaternion ops (CPU + GPU)
├── dynamics/               # Equation of motion computation (8 files)
│   ├── mass_matrix.mojo    # CRBA: M(q), LDL/LU decomposition, sparse variants
│   ├── bias_forces.mojo    # RNE: C(q,qdot) + g(q)
│   ├── jacobian.mojo       # Contact/analytical Jacobians, composite inertia
│   ├── velocity_derivatives.mojo # d(bias)/d(qvel) for implicit integration
│   ├── lu_factorization.mojo # Non-symmetric factorization
│   └── cfrc_ext.mojo       # External/actuator forces
├── integrator/             # Time-stepping algorithms (5 files)
│   ├── euler_integrator.mojo         # MuJoCo-style Euler
│   ├── implicit_fast_integrator.mojo # Default: M + arm - dt*qDeriv (fast)
│   ├── implicit_integrator.mojo      # Full implicit with RNE velocity derivative
│   └── rk4_integrator.mojo           # 4th-order Runge-Kutta
├── solver/                 # Constraint solvers (14 files)
│   ├── pgs_solver.mojo     # Projected Gauss-Seidel (dual, lambda space)
│   ├── newton_solver.mojo  # Newton (primal, qacc space)
│   ├── cg_solver.mojo      # Conjugate Gradient (primal)
│   ├── island_detection.mojo    # Connected component analysis
│   ├── island_solver.mojo       # Per-island early termination
│   ├── island_pgs_solver.mojo   # Island-aware PGS
│   ├── qcqp.mojo           # Quadratic constraint QP (2/3/5-dim)
│   └── friction_solver.mojo # Friction-specific solving
├── collision/              # Contact detection (4 files)
│   ├── collision_primitives.mojo # Sphere/capsule/box narrow-phase
│   ├── contact_detection.mojo    # Contact manifold generation (CPU + GPU)
│   └── broadphase_sap.mojo      # Sweep-and-Prune broadphase
├── constraints/            # Constraint representation (3 files)
│   ├── constraint_data.mojo      # ConstraintRow, ConstraintData
│   ├── constraint_builder.mojo   # CPU constraint building
│   └── constraint_builder_gpu.mojo # GPU constraint building
├── gpu/                    # GPU buffer management (4 files)
├── traits/                 # Integrator + ConstraintSolver traits
├── parser/                 # MJCF XML model loading (4 files)
│   ├── xml_parser.mojo     # DOM parser
│   ├── flat_model.mojo     # Flattened model representation
│   └── full_parser.mojo    # Complete XML -> Model/Data pipeline
└── tests/                  # 75 validation tests (CPU/GPU, MuJoCo comparison)
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
