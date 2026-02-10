"""Physics3D - MuJoCo-inspired generalized coordinates physics engine.

Constraint-based contact solving with joint-space dynamics.
- Model: Static simulation configuration (kinematic tree, masses, etc.)
- Data: Mutable simulation state (qpos, qvel, computed xpos/xquat)
- EulerIntegrator[SOLVER]: Configurable constraint solver integrator
"""

from .constants import TILE, TPB, PhysicsConstants
from .constants import GEOM_PLANE, GEOM_SPHERE

# Types
from .types import Model, Data, ContactInfo, compute_capsule_inertia, _max_one
from .joint_types import JointDef, JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
from .traits import Integrator, ConstraintSolver

# Kinematics
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

# Dynamics
from .dynamics.mass_matrix import compute_mass_matrix, solve_linear_diagonal
from .dynamics.bias_forces import compute_bias_forces

# Integrator
from .integrator import DefaultIntegrator, EulerIntegrator, ImplicitFastIntegrator

# Solvers
from .solver import PGSSolver, CGSolver, NewtonSolver

# Collision
from .collision import (
    sphere_sphere,
    sphere_plane,
    capsule_plane,
    capsule_sphere,
    capsule_capsule,
    box_plane,
    box_sphere,
    box_capsule,
    box_box,
    detect_ground_contacts,
    detect_body_body_contacts,
    normalize_qpos_quaternions,
)
