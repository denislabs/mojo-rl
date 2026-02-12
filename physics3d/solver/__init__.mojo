"""Physics3D Constraint Solvers.

GC constraint solvers for constraint-based contacts:
- PGSSolver: Projected Gauss-Seidel (default)
- CGSolver: Conjugate Gradient
- NewtonSolver: Projected Newton with line search

Unified constraint representation:
- ConstraintData: Pre-built constraint rows consumed by solvers
- build_constraints: Builds ConstraintData from contacts and joint limits
- writeback_forces: Writes solved forces back for warm-starting
"""

from .pgs_solver import PGSSolver
from .cg_solver import CGSolver
from .newton_solver import NewtonSolver
