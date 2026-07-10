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

# Legacy slab solvers (PGS/Newton/CG/IslandPGS + island detection) were deleted
# at the P6 fields sunset. The fields solvers live in `newton_solve_fields` /
# `cg_solve_fields` / `contact_solve_fields` / `island_pgs_solve_fields` and are
# imported directly. `cholesky` and `qcqp` (leaf helpers) remain — the fields
# solvers import them by submodule.
