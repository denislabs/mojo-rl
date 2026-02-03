"""Physics3D v2 Integrators.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

Three integrator implementations:

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
"""

from .impulse_integrator import ImpulseIntegrator
from .pgs_integrator import PGSIntegrator
from .semi_implicit_euler_integrator import SemiImplicitEulerIntegrator
