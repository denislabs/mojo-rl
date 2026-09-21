# physics2d/ - GPU-Accelerated 2D Physics Engine

Batched 2D rigid body physics for environments like LunarLander, BipedalWalker, and CarRacing. Uses strided `[BATCH, STATE_SIZE]` layout for efficient GPU access.

## Architecture

Two specialized physics models:
1. **Impulse-based collision** (LunarLander, BipedalWalker) - standard contact resolution with Coulomb friction
2. **Slip-based tire friction** (CarRacing) - top-down vehicle dynamics with tire slip model

## Module Structure

```
physics2d/
├── constants.mojo          # BODY_STATE_SIZE=13, shape types, contact/joint layout
├── layout.mojo             # PhysicsLayout[NUM_BODIES, MAX_CONTACTS, ...]: compile-time offsets
├── state.mojo              # PhysicsState: strided buffer accessor
├── kernel.mojo             # PhysicsKernel.step_gpu(): unified one-call physics step
├── env_helpers.mojo        # init_body(), apply_force(), set_joint_*(), extract_observation()
├── integrators/            # Semi-implicit Euler (CPU + GPU)
│   └── euler.mojo          # SemiImplicitEuler: velocity + position integration
├── collision/              # Terrain collision detection
│   ├── flat_terrain.mojo   # FlatTerrainCollision: body vs ground plane
│   └── edge_terrain.mojo   # EdgeTerrainCollision: body vs edge segments
├── solvers/                # Constraint resolution
│   ├── impulse.mojo        # ImpulseSolver: velocity + position impulse solving
│   └── unified.mojo        # UnifiedConstraintSolver: contacts + joints
├── joints/                 # Joint constraints
│   └── revolute.mojo       # RevoluteJointSolver: motor, spring, angle limits
├── articulated/            # Multi-body chain support
│   ├── chain.mojo          # ArticulatedChain, LinkDef
│   └── constants.mojo      # Hopper/Walker/Cheetah body/joint counts
├── car/                    # CarRacing slip-based tire physics (6 files)
│   ├── constants.mojo      # Engine power, friction coefficients, wheel positions
│   ├── layout.mojo         # CarRacingLayout[BATCH]: state offsets
│   ├── car_dynamics.mojo   # Full car physics step
│   ├── wheel_friction.mojo # Tire slip model
│   ├── tile_collision.mojo # Track tile friction zone lookup
│   └── car_kernel.mojo     # Fused GPU kernel for complete step
├── lidar/                  # Distance sensing
│   └── lidar.mojo          # LidarSensor: ray-cast collision
├── traits/                 # Integrator, CollisionSystem, ConstraintSolver traits
└── kernels/                # Unified physics step orchestration
    └── physics_step.mojo
```

## Key Design Patterns

- **Strided 2D layout**: Flat `[BATCH, STATE_SIZE]` buffers for efficient batched GPU access
- **Compile-time layout**: `PhysicsLayout` computes all offsets at compile time
- **Trait-based extensibility**: Integrator, CollisionSystem, ConstraintSolver traits
- **Warm-starting**: Impulse caching across frames for faster convergence
