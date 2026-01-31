"""Physics3D v2 Integrators.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

Two integrator implementations:

1. ImpulseIntegrator (Bullet/Box2D style):
   - Split Impulse method
   - Separate velocity and position solving
   - Good for stable stacking

2. PGSIntegrator (MuJoCo style):
   - Projected Gauss-Seidel with spring-damper constraints
   - Soft contact dynamics
   - Configurable timeconst/dampratio/impedance
"""

from .impulse_integrator import ImpulseIntegrator
from .pgs_integrator import PGSIntegrator
