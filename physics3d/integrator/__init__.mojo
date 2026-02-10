"""Physics3D Integrators.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

EulerIntegrator[SOLVER]:
  - MuJoCo Euler integration: M_hat = M + arm + dt*diag(damping)
  - Three solver choices: PGSSolver, CGSolver, NewtonSolver

ImplicitFastIntegrator[SOLVER]:
  - MuJoCo implicit-fast integration: M_hat = M + arm - dt*qDeriv
  - Same result as Euler for passive systems (no actuators)
  - Extensible for actuator velocity derivatives

DefaultIntegrator is an alias for ImplicitFastIntegrator[PGSSolver].
"""

from .euler_integrator import EulerDefaultIntegrator, EulerIntegrator
from .implicit_fast_integrator import ImplicitFastIntegrator
from ..solver.pgs_solver import PGSSolver

# Default integrator uses implicit-fast (matches MuJoCo default)
comptime DefaultIntegrator = ImplicitFastIntegrator[PGSSolver]
