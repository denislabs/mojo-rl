"""Physics3D Constraint Solvers.

GC constraint solvers for constraint-based contacts:
- PGSSolver: Projected Gauss-Seidel (default)
- CGSolver: Conjugate Gradient
- NewtonSolver: Projected Newton with line search
"""

from .pgs_solver import PGSSolver
from .cg_solver import CGSolver
from .newton_solver import NewtonSolver
