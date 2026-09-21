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
# at the P6 fields sunset. The fields solvers live in `newton_solve` /
# `cg_solve` / `contact_solve` / `island_pgs_solve` and are
# imported directly. `cholesky` (leaf helper) remains — the fields solvers
# import it by submodule.
#
# ⚠ `qcqp` AND `elliptic_layout` MOVED TO `constraints/` (phase 2.0). Both are
# leaf math — `elliptic_layout` imports nothing at all — and living here made
# `constraints` import `solver`, which with `solver -> constraints` formed a
# cycle. That cycle put {constraints, dynamics, solver} in one 22.5k-line SCC
# with no intermediate green state, which is what made §5.4's
# package-at-a-time gating protocol unexecutable. See §11.3.
