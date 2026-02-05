"""Physics3D v2 - Minimal physics engine rebuild.

A MuJoCo-inspired physics engine with Model/Data separation.
- Model: Static simulation configuration (parameterized by NUM_BODIES, MAX_JOINTS)
- Data: Mutable simulation state

Example usage:
    from physics3d_v2 import Model, Data, ImpulseIntegrator

    # Create a 2-body system with max 10 contacts
    var model = Model[DType.float64, 2, 10](
        gravity_z=-9.81, restitution=0.6
    )
    model.set_body(0, mass=1.0, radius=0.1)
    model.set_body(1, mass=1.0, radius=0.1)

    var data = Data[DType.float64, 2, 10]()
    data.set_body_position(0, 0, 0, 1.0)  # Body 0 at height 1m
    data.set_body_position(1, 0, 0, 0.3)  # Body 1 at height 0.3m

    # Simulate using ImpulseIntegrator (Bullet/Box2D style)
    for i in range(100):
        ImpulseIntegrator.step(model, data)
        print("body0 z =", data.get_body_z(0))

    # Or use PGSIntegrator (MuJoCo style)
    from physics3d_v2 import PGSIntegrator
    PGSIntegrator.step(model, data)

Single body is just Model[DTYPE, 1, MAX_CONTACTS]:
    var model = Model[DType.float64, 1, 5](gravity_z=-9.81)
    model.set_body(0, mass=1.0, radius=0.1)

With joints (pendulum example):
    from physics3d_v2 import Model, Data, ImpulseIntegrator, HingeJoint

    # MAX_JOINTS=1 as 4th parameter
    var model = Model[DType.float64, 1, 5, 1](gravity_z=-9.81)
    model.set_body(0, mass=1.0, radius=0.1)
    model.add_hinge_joint(
        parent=-1, child=0,  # -1 = world anchor
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),  # Y-axis rotation
    )
"""

from .constants import TILE, TPB, PhysicsConstants
from .constants import GEOM_PLANE, GEOM_SPHERE

# Primary types (unified Model/Data with compile-time NUM_BODIES, MAX_JOINTS)
from .types import Model, Data, ContactInfo

# Joint types
from .joints import HingeJoint

# Traits
from .traits import CollisionSystem, Integrator

# Collision detection
from .collision import CollisionDetector, sphere_sphere, sphere_plane

# Constraint solvers
from .solver import (
    # Impulse solver (Bullet/Box2D style)
    solve_velocity_constraints,
    solve_position_constraints,
    solve_resting_contacts,
    # PGS solver (MuJoCo style)
    solve_constraints_pgs,
    correct_positions,
)

# Joint solvers
from .joints import (
    solve_joint_velocity_constraints,
    solve_joint_position_constraints,
)

# Integrators (primary API)
from .integrator import (
    ImpulseIntegrator,
    PGSIntegrator,
    SemiImplicitEulerIntegrator,
)

# Generalized Coordinates (GC) engine
from .types import ModelGC, DataGC, ContactInfoGC, compute_capsule_inertia
from .joint_types import JointDef, JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
from .traits import GcIntegrator
from .kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from .kinematics.quat_math import (
    quat_mul,
    quat_conjugate,
    quat_rotate,
    quat_normalize,
    axis_angle_to_quat,
    quat_integrate,
)
from .dynamics.mass_matrix import compute_mass_matrix, solve_linear_diagonal
from .dynamics.bias_forces import compute_bias_forces

# Note: render modules are imported separately to avoid SDL2 dependency:
#   from physics3d_v2.render import Physics3DRenderer
