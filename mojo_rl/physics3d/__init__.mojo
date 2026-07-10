"""Physics3D - MuJoCo-inspired generalized coordinates physics engine.

Constraint-based contact solving with joint-space dynamics.
- Model: Static simulation configuration (kinematic tree, masses, etc.)
- Data: Mutable simulation state (qpos, qvel, computed xpos/xquat)
- EulerIntegrator[SOLVER]: Configurable constraint solver integrator
"""

from .constants import TILE, TPB, PhysicsConstants
from .constants import GEOM_PLANE, GEOM_SPHERE

# Types
from .types import (
    Model,
    Data,
    ContactInfo,
    compute_capsule_inertia,
    _max_one,
    ConeType,
)
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
from .dynamics.bias_forces import compute_bias_forces
from .dynamics.cfrc_ext import compute_cfrc_ext

# Legacy slab integrators + solvers were deleted at the P6 fields sunset; the
# fields integrators/solvers are imported directly by module.

# Collision primitives (shared leaves)
from .types import ConeType
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
)
