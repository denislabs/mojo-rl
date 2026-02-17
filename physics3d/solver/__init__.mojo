"""Physics3D Constraint Solvers.

GC constraint solvers for constraint-based contacts:
- PGSSolver: Projected Gauss-Seidel (dual, operates in lambda space)
- NewtonSolver: MuJoCo-style Newton in qacc space (primal)
- CGSolver: MuJoCo-style CG in qacc space (primal)

Unified constraint representation:
- ConstraintData: Pre-built constraint rows consumed by solvers
- build_constraints: Builds ConstraintData from contacts and joint limits
- writeback_forces: Writes solved forces back for warm-starting
"""

from .pgs_solver import PGSSolver
from .newton_solver import NewtonSolver
from .cg_solver import CGSolver
