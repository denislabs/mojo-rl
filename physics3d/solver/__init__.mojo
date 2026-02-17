"""Physics3D Constraint Solvers.

GC constraint solvers for constraint-based contacts:
- PGSSolver: Projected Gauss-Seidel (default, dual)
- CGSolver: Conjugate Gradient (dual)
- NewtonSolver: Projected Newton with line search (dual)
- PrimalNewtonSolver: MuJoCo-style primal Newton in qacc space
- PrimalCGSolver: MuJoCo-style primal CG in qacc space

Dual solvers operate on constraint forces (lambda space).
Primal solvers operate on accelerations (qacc space), matching MuJoCo.

Unified constraint representation:
- ConstraintData: Pre-built constraint rows consumed by solvers
- build_constraints: Builds ConstraintData from contacts and joint limits
- writeback_forces: Writes solved forces back for warm-starting
"""

from .pgs_solver import PGSSolver
from .cg_solver import CGSolver
from .newton_solver import NewtonSolver
from .primal_newton_solver import PrimalNewtonSolver
from .primal_cg_solver import PrimalCGSolver
