"""3D Physics Engine for MuJoCo-style Environments.

This module provides the core infrastructure for 3D rigid body physics
simulation, supporting articulated body chains for locomotion environments.

Modules:
    - constants: Body state layout, joint types, default parameters
    - state: PhysicsState3D accessor for reading/writing body data
    - layout: PhysicsLayout3D compile-time layout calculator
    - body: Inertia tensor computation utilities
    - traits: Integrator3D, ConstraintSolver3D interfaces

Example:
    ```mojo
    from physics3d import PhysicsLayout3D, PhysicsState3D
    from physics3d.body import compute_box_inertia

    # Define layout for Hopper environment
    comptime Layout = PhysicsLayout3D[
        NUM_BODIES=4,
        MAX_JOINTS=3,
        OBS_DIM=11,
    ]

    # Initialize a body
    var inertia = compute_box_inertia(mass=2.0, half_extents=Vec3(0.1, 0.05, 0.2))
    PhysicsState3D[1, 4, Layout.STATE_SIZE, Layout.BODIES_OFFSET].init_body(
        state, env=0, body=0,
        position=Vec3(0, 0, 1.25),
        orientation=Quat.identity(),
        mass=2.0,
        inertia=inertia,
    )
    ```
"""

from .constants import (
    dtype,
    TILE,
    TPB,
    BODY_STATE_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    IDX_SHAPE_3D,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    BODY_KINEMATIC,
    BODY_STATIC,
    SHAPE_BOX,
    SHAPE_SPHERE,
    SHAPE_CAPSULE,
    SHAPE_PLANE,
    CONTACT_DATA_SIZE_3D,
    JOINT_DATA_SIZE_3D,
    JOINT_HINGE,
    JOINT_BALL,
    JOINT_FREE,
    JOINT_FIXED,
    DEFAULT_GRAVITY_Z_3D,
    DEFAULT_DT_3D,
    DEFAULT_VELOCITY_ITERATIONS_3D,
    DEFAULT_POSITION_ITERATIONS_3D,
    DEFAULT_FRICTION_3D,
    DEFAULT_BAUMGARTE_3D,
    DEFAULT_SLOP_3D,
    PI,
    TWO_PI,
    DEG_TO_RAD,
    RAD_TO_DEG,
)

from .layout import (
    PhysicsLayout3D,
    HopperLayout3D,
    Walker2dLayout3D,
    HalfCheetahLayout3D,
    AntLayout3D,
    HumanoidLayout3D,
)

from .state import PhysicsState3D

from .body import (
    compute_box_inertia,
    compute_sphere_inertia,
    compute_capsule_inertia,
    compute_cylinder_inertia,
    compute_ellipsoid_inertia,
    parallel_axis_offset,
    combine_inertias,
    get_torso_inertia,
    get_limb_inertia,
    get_foot_inertia,
)

# Joints
from .joints import Hinge3D, Ball3D, Motor3D, PDController

# Collision
from .collision import (
    Contact3D,
    ContactManifold,
    SpherePlaneCollision,
    SpherePlaneCollisionGPU,
    CapsulePlaneCollision,
    CapsulePlaneCollisionGPU,
)

# Integrators
from .integrators import (
    SemiImplicitEuler3D,
    SemiImplicitEuler3DGPU,
    integrate_velocities_3d,
    integrate_positions_3d,
    integrate_quaternion,
)

# Solvers
from .solvers import (
    ContactSolver3D,
    ImpulseSolver3DGPU,
    solve_contact_velocity,
    solve_contact_position,
    JointSolver3D,
    solve_hinge_velocity,
    solve_hinge_position,
)

# Kernels / Physics World
from .kernels import PhysicsWorld3D, Physics3DStepKernel
