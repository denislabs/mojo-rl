"""Physics3D Constraint Solvers.

GC constraint solvers for constraint-based contacts:
- PGSSolver: Projected Gauss-Seidel (default)
- CGSolver: Conjugate Gradient
- NewtonSolver: Projected Newton with line search

Unified constraint representation:
- ConstraintData: Pre-built constraint rows consumed by solvers
- build_constraints: Builds ConstraintData from contacts and joint limits
- writeback_impulses: Writes solved impulses back for warm-starting
"""

from .pgs_solver import PGSSolver
from .cg_solver import CGSolver
from .newton_solver import NewtonSolver
from .constraint_data import ConstraintData, ConstraintRow
from .constraint_builder import build_constraints, writeback_impulses
