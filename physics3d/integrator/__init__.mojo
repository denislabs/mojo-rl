"""Physics3D v2 Integrators.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

Integrator implementations:

1. ImpulseIntegrator (Bullet/Box2D style):
   - Split Impulse method
   - Separate velocity and position solving
   - Good for stable stacking

2. PGSIntegrator (MuJoCo style):
   - Projected Gauss-Seidel with spring-damper constraints
   - Soft contact dynamics
   - Configurable timeconst/dampratio/impedance

3. SemiImplicitEulerIntegrator (Generalized Coordinates engine):
   - Joint-space dynamics
   - Forward kinematics + mass matrix + bias forces
   - Symplectic integration for energy conservation

4. ConstraintGcIntegratorWith[SOLVER] (GC with configurable solver):
   - Joint-space dynamics with constraint-based contact solving
   - Three solver choices (mirroring MuJoCo):
     * GcPGSSolver: Projected Gauss-Seidel (default)
     * GcCGSolver: Conjugate Gradient
     * GcNewtonSolver: Projected Newton with line search
   - ConstraintGcIntegrator is an alias for ConstraintGcIntegratorWith[GcPGSSolver]
"""

from .impulse_integrator import ImpulseIntegrator
from .pgs_integrator import PGSIntegrator
from .semi_implicit_euler_integrator import SemiImplicitEulerIntegrator
from .constraint_gc_integrator import ConstraintGcIntegrator, ConstraintGcIntegratorWith
