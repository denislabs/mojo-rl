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
from .constraint_builder_gpu import (
    common_normal_size,
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
    warmstart_normals_gpu,
    apply_solved_normals_gpu,
    detect_and_solve_limits_gpu,
)
