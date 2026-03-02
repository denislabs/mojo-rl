"""Physics3D Constraint Solvers.

GC constraint solvers for constraint-based contacts:
- PGSSolver: Projected Gauss-Seidel (dual, operates in lambda space)
- NewtonSolver: MuJoCo-style Newton in qacc space (primal)
- CGSolver: MuJoCo-style CG in qacc space (primal)
- IslandPGSSolver: PGSSolver with per-island early termination (drop-in replacement)

Unified constraint representation:
- ConstraintData: Pre-built constraint rows consumed by solvers
- build_constraints: Builds ConstraintData from contacts and joint limits
- writeback_forces: Writes solved forces back for warm-starting
"""

from .pgs_solver import PGSSolver
from .newton_solver import NewtonSolver
from .old_newton_solver import OldNewtonSolver
from .cg_solver import CGSolver
from .island_detection import (
    detect_islands,
    IslandData,
    MAX_ISLANDS,
    ISLAND_J_THRESH,
)
from .island_solver import solve_with_islands, ISLAND_CONVERGE_EPS
from .island_pgs_solver import IslandPGSSolver
