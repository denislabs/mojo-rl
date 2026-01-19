"""GPU-accelerated physics engine for Mojo-RL.

This module provides a trait-based, modular physics engine that works
on both CPU and GPU with the same interface. Perfect for RL environments
that need fast, batched physics simulation.

Architecture:
- PhysicsWorld: Central orchestrator for physics simulation
- Integrator trait: Velocity and position integration (SemiImplicitEuler)
- CollisionSystem trait: Collision detection (FlatTerrainCollision)
- ConstraintSolver trait: Contact resolution (ImpulseSolver)

Example:
    ```mojo
    from physics_gpu import PhysicsWorld
    from gpu.host import DeviceContext

    # Create world with 1024 parallel environments, 3 bodies each
    var world = PhysicsWorld[1024, 3, 2](
        ground_y=0.0,
        gravity_y=-10.0,
    )

    # Setup bodies and shapes (define vertices first)
    var vx = List[Float64]()
    var vy = List[Float64]()
    vx.append(-0.5); vx.append(0.5); vx.append(0.5); vx.append(-0.5)
    vy.append(-0.5); vy.append(-0.5); vy.append(0.5); vy.append(0.5)
    world.define_polygon_shape(0, vx, vy)
    world.set_body_mass(env=0, body=0, mass=1.0, inertia=0.1)

    # Run physics on CPU
    world.step()

    # Or run on GPU
    var ctx = DeviceContext()
    world.step_gpu(ctx)
    ```
"""

# Core constants and types
from .constants import (
    dtype,
    TPB,
    TILE,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_EDGE,
    MAX_POLYGON_VERTS,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_DEPTH,
    CONTACT_NORMAL_IMPULSE,
    CONTACT_TANGENT_IMPULSE,
    # Joint constants
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_MAX_MOTOR_TORQUE,
    JOINT_MOTOR_SPEED,
    JOINT_STIFFNESS,
    JOINT_DAMPING,
    JOINT_FLAGS,
    JOINT_IMPULSE,
    JOINT_MOTOR_IMPULSE,
    JOINT_REVOLUTE,
    JOINT_DISTANCE,
    JOINT_PRISMATIC,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    JOINT_FLAG_SPRING_ENABLED,
    MAX_JOINTS_PER_ENV,
    DEFAULT_GRAVITY_X,
    DEFAULT_GRAVITY_Y,
    DEFAULT_DT,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_BAUMGARTE,
    DEFAULT_SLOP,
    PI,
    TWO_PI,
)

# Core traits
from .traits import Integrator, CollisionSystem, ConstraintSolver

# Implementations
from .integrators import SemiImplicitEuler
from .collision import FlatTerrainCollision, EdgeTerrainCollision, MAX_TERRAIN_EDGES
from .solvers import ImpulseSolver
from .joints import RevoluteJointSolver

# Main orchestrator (original API)
from .world import PhysicsWorld

# New GPU-compatible architecture (mirrors deep_rl patterns)
from .layout import PhysicsLayout, LunarLanderLayout, CartPoleLayout
from .kernel import PhysicsKernel, PhysicsConfig
from .state import PhysicsState

# Strided state helper for GPUDiscreteEnv (flat layout)
from .state_strided import PhysicsStateStrided

# Strided layout and helpers for modular physics (reduces env boilerplate)
from .layout_strided import (
    PhysicsLayoutStrided,
    LunarLanderLayoutStrided,
    CartPoleLayoutStrided,
    AcrobotLayoutStrided,
)
from .kernel_strided import PhysicsKernelStrided, PhysicsConfigStrided, CollisionType
from .env_helpers import PhysicsEnvHelpers
