"""Physics3D Integrators.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

EulerIntegrator[SOLVER] (GC with configurable solver):
  - Joint-space dynamics with constraint-based contact solving
  - Three solver choices (mirroring MuJoCo):
    * PGSSolver: Projected Gauss-Seidel (default)
    * CGSolver: Conjugate Gradient
    * NewtonSolver: Projected Newton with line search
  - DefaultIntegrator is an alias for EulerIntegrator[PGSSolver]
"""

from .euler_integrator import DefaultIntegrator, EulerIntegrator
