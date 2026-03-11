"""Island-aware PGS constraint solver for the physics3d engine.

Wraps the standard PGSSolver with per-island early termination.  The key
optimisation: once an island's impulse changes fall below the convergence
threshold (ISLAND_CONVERGE_EPS) that island is frozen — its rows are skipped
in all subsequent iterations.  For systems with many independent sub-problems
(e.g. multiple non-interacting robots, objects falling in separate regions)
this can dramatically reduce the total iteration count.

Single-island fallback
----------------------
When detect_islands() finds ≤ 1 island (the common single-robot case) the
function delegates directly to PGSSolver.solve() with zero overhead.

Multi-island path
-----------------
The solver runs up to PGS_ITERATIONS outer iterations.  Inside each iteration
it processes every non-converged row in constraint order (preserving the
normal→friction→limit→equality ordering that PGS requires), accumulates the
maximum |Δλ| per island, then marks any island whose max change fell below
ISLAND_CONVERGE_EPS as converged.  Iteration stops when all islands converge
or the iteration cap is reached.

Note: only the standard PGS update (R-regularised, λ ≥ lo clamping) is
applied here; the full QCQP block update for elliptic contacts is expensive
to replicate.  For high-accuracy elliptic-cone friction a caller should use
PGSSolver.solve() directly.  This solver is optimised for the multi-island
performance case where convergence speed matters more than per-iteration
accuracy.

Exports
-------
    solve_with_islands  — island-aware PGS solve (main entry point)
"""

from std.math import abs
from ..types import Model, Data, _max_one, ConeType
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_LIMIT,
    CNSTR_EQUALITY_CONNECT,
    CNSTR_EQUALITY_WELD,
    CNSTR_EQUALITY_TENDON,
)
from .pgs_solver import PGSSolver
from .island_detection import detect_islands, IslandData, MAX_ISLANDS

# Per-island convergence threshold: if max |Δλ| < eps the island is frozen.
comptime ISLAND_CONVERGE_EPS: Float64 = 1e-6


fn solve_with_islands[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    M_inv: List[Scalar[DTYPE]],
    mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut qacc: List[Scalar[DTYPE]],
    dt: Scalar[DTYPE],
):
    """Solve constraints using island-aware PGS on CPU.

    Detects constraint islands and applies per-island early termination:
    once an island converges its rows are skipped, saving iterations for
    the remaining active islands.

    For single-island systems this falls back to PGSSolver.solve() directly
    with no overhead beyond the O(MAX_ROWS*NV) island-detection scan.

    Args:
        model:       Static physics model.
        data:        Mutable simulation state (contacts, qvel, etc.).
        M_inv:       Inverse mass matrix (NV × NV, row-major).
        constraints: Pre-built ConstraintData (J, MinvJT, rows).
        qacc:        Predicted (unconstrained) acceleration — modified in place.
        dt:          Timestep (used for limit/equality aref computation).
    """
    if constraints.num_rows == 0:
        return

    # ---- Detect islands ----
    var islands = detect_islands[DTYPE, MAX_ROWS, NV](constraints)

    # ---- Single-island fast path: delegate to standard PGS ----
    if islands.num_islands <= 1:
        PGSSolver.solve[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ](model, data, M_inv, constraints, qacc, dt)
        return

    # ---- Multi-island path: per-island early termination ----
    var num_normals = constraints.num_normals
    var num_friction = constraints.num_friction
    var num_limits = constraints.num_limits
    var num_equality = constraints.num_equality

    var friction_start = num_normals
    var limits_start = num_normals + num_friction
    var equality_start = limits_start + num_limits

    # Convergence state: 0 = active, 1 = converged
    var island_converged = InlineArray[Int, MAX_ISLANDS](fill=0)
    var num_converged = 0
    var num_islands = islands.num_islands
    var eps = Scalar[DTYPE](ISLAND_CONVERGE_EPS)

    # =====================================================================
    # Phase 1: Warm-start — apply stored impulses for normals and friction
    # =====================================================================
    for r in range(num_normals):
        if constraints.rows[r].lambda_val > Scalar[DTYPE](0):
            for i in range(NV):
                qacc[i] += (
                    constraints.MinvJT[r * NV + i]
                    * constraints.rows[r].lambda_val
                )

    for r_off in range(num_friction):
        var r = friction_start + r_off
        if constraints.rows[r].lambda_val != Scalar[DTYPE](0):
            for i in range(NV):
                qacc[i] += (
                    constraints.MinvJT[r * NV + i]
                    * constraints.rows[r].lambda_val
                )

    # =====================================================================
    # Phase 2: Island-aware PGS iterations
    # =====================================================================
    comptime PGS_ITERS: Int = 100

    for _ in range(PGS_ITERS):
        if num_converged >= num_islands:
            break

        # Per-island max |Δλ| accumulators
        var island_max_delta = InlineArray[Scalar[DTYPE], MAX_ISLANDS](
            fill=Scalar[DTYPE](0)
        )

        # --- Normal constraints ---
        for r in range(num_normals):
            var iid = islands.row_island[r]
            if iid >= 0 and island_converged[iid] == 1:
                continue

            var a_n = Scalar[DTYPE](0)
            for i in range(NV):
                a_n += constraints.J[r * NV + i] * qacc[i]
            var R_n = (
                Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp
                - constraints.rows[r].K
            )
            var residual = (
                a_n
                + constraints.rows[r].bias
                + R_n * constraints.rows[r].lambda_val
            )
            var delta = -residual * constraints.rows[r].inv_K_imp
            var old_lambda = constraints.rows[r].lambda_val
            constraints.rows[r].lambda_val = old_lambda + delta
            if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                constraints.rows[r].lambda_val = Scalar[DTYPE](0)
            var actual = constraints.rows[r].lambda_val - old_lambda
            if actual != Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual
            # Track convergence
            if iid >= 0:
                var d_abs = abs(actual)
                if d_abs > island_max_delta[iid]:
                    island_max_delta[iid] = d_abs

        # --- Friction constraints (basic R-regularised update) ---
        for r_off in range(num_friction):
            var r = friction_start + r_off
            var iid = islands.row_island[r]
            if iid >= 0 and island_converged[iid] == 1:
                continue

            var parent_r = constraints.rows[r].friction_parent
            var mu = constraints.rows[r].friction_coef
            var fn_val = Scalar[DTYPE](0)
            if parent_r >= 0:
                fn_val = constraints.rows[parent_r].lambda_val

            var a_f = Scalar[DTYPE](0)
            for i in range(NV):
                a_f += constraints.J[r * NV + i] * qacc[i]
            var R_f = (
                Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp
                - constraints.rows[r].K
            )
            var residual = (
                a_f
                + constraints.rows[r].bias
                + R_f * constraints.rows[r].lambda_val
            )
            var delta = -residual * constraints.rows[r].inv_K_imp
            var old_lambda = constraints.rows[r].lambda_val
            var new_lambda = old_lambda + delta
            # Coulomb cone clamp
            var cone_hi = mu * fn_val
            var cone_lo = -cone_hi
            if new_lambda > cone_hi:
                new_lambda = cone_hi
            if new_lambda < cone_lo:
                new_lambda = cone_lo
            constraints.rows[r].lambda_val = new_lambda
            var actual = new_lambda - old_lambda
            if actual != Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual
            if iid >= 0:
                var d_abs = abs(actual)
                if d_abs > island_max_delta[iid]:
                    island_max_delta[iid] = d_abs

        # --- Joint limit constraints ---
        for r_off in range(num_limits):
            var r = limits_start + r_off
            var iid = islands.row_island[r]
            if iid >= 0 and island_converged[iid] == 1:
                continue

            var dof = constraints.rows[r].source_dof
            var sign = constraints.rows[r].limit_sign
            var a_limit = sign * qacc[dof]
            var R_lim = (
                Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp
                - constraints.rows[r].K
            )
            var residual = (
                a_limit
                + constraints.rows[r].bias
                + R_lim * constraints.rows[r].lambda_val
            )
            var delta = -residual * constraints.rows[r].inv_K_imp
            var old_lambda = constraints.rows[r].lambda_val
            constraints.rows[r].lambda_val = old_lambda + delta
            if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                constraints.rows[r].lambda_val = Scalar[DTYPE](0)
            var actual = constraints.rows[r].lambda_val - old_lambda
            for i in range(NV):
                qacc[i] += constraints.MinvJT[r * NV + i] * actual
            if iid >= 0:
                var d_abs = abs(actual)
                if d_abs > island_max_delta[iid]:
                    island_max_delta[iid] = d_abs

        # --- Equality constraints (bilateral, no clamping) ---
        for r_off in range(num_equality):
            var r = equality_start + r_off
            var iid = islands.row_island[r]
            if iid >= 0 and island_converged[iid] == 1:
                continue

            var a_eq = Scalar[DTYPE](0)
            for i in range(NV):
                a_eq += constraints.J[r * NV + i] * qacc[i]
            var R_eq = (
                Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp
                - constraints.rows[r].K
            )
            var residual = (
                a_eq
                + constraints.rows[r].bias
                + R_eq * constraints.rows[r].lambda_val
            )
            var delta = -residual * constraints.rows[r].inv_K_imp
            var old_lambda = constraints.rows[r].lambda_val
            constraints.rows[r].lambda_val = old_lambda + delta
            var actual = constraints.rows[r].lambda_val - old_lambda
            for i in range(NV):
                qacc[i] += constraints.MinvJT[r * NV + i] * actual
            if iid >= 0:
                var d_abs = abs(actual)
                if d_abs > island_max_delta[iid]:
                    island_max_delta[iid] = d_abs

        # ---- Check per-island convergence ----
        for iid in range(num_islands):
            if island_converged[iid] == 0:
                if island_max_delta[iid] < eps:
                    island_converged[iid] = 1
                    num_converged += 1
