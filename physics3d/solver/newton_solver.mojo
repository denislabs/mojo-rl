"""Newton constraint solver (MuJoCo-matching).

Operates in qacc (acceleration) space, minimizing:
  cost = 0.5*(qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)  [Gauss term]
       + sum_i penalty_i(J*qacc - aref)                         [constraint costs]

Forces are derived from qacc: force[i] = -D[i] * (J*qacc - aref)[i].
The Hessian is H = M + J^T*D_active*J (naturally positive definite).

This is fundamentally different from a dual Newton solver which builds
the Delassus matrix A = J*M^{-1}*J^T and solves a QP over forces.

The constraint_update uses MuJoCo's 3-zone cone logic for elliptic contacts:
- Top zone (N >= 0, N >= mu*T): satisfied, no forces
- Bottom zone (N + mu*T <= 0): full quadratic on all rows
- Middle zone: cone boundary projection with Dm-weighted cost

This eliminates the need for a separate PGS friction phase — the
Newton optimization handles ALL constraints (normals, friction, limits,
equality) in a unified framework.

Algorithm:
1. qacc = qacc_smooth (unconstrained acceleration)
2. Ma = M * qacc, jar = J*qacc - aref
3. constraint_update → force, state, cost (cone-aware)
4. Build H = M + J^T * D_active * J, Cholesky factorize
5. Main loop:
   a. grad = Ma - qfrc_smooth - J^T * force
   b. search = -chol_solve(H, grad)
   c. Armijo linesearch → alpha
   d. qacc += alpha * search, Ma += alpha * M*search
   e. constraint_update → force, state, cost
   f. Incremental Hessian update if state changed
   g. Check convergence

Reference: mujoco-main/src/engine/engine_solver.c (mj_solPrimal)
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from ..types import Model, Data, _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..traits.solver import ConstraintSolver
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_LIMIT,
    CNSTR_PYRAMID_EDGE,
    CNSTR_EQUALITY_CONNECT,
    CNSTR_EQUALITY_WELD,
)
from .primal_common import (
    constraint_update,
    constraint_update_with_D,
    compute_jar,
    compute_qfrc_constraint,
    compute_gauss_cost,
    primal_linesearch,
    primal_linesearch_with_D,
    primal_D,
    PRIMAL_SATISFIED,
    PRIMAL_QUADRATIC,
    PRIMAL_CONE,
    PRIMAL_MINVAL,
)
from .cholesky import chol_factor, chol_solve, chol_rank1_update

# Import shared friction solver for GPU
from .friction_solver import _solve_friction_pgs_gpu

from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_M_offset,
    ws_fnet_offset,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_FRICTION,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
)

from ..constraints.constraint_builder_gpu import (
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
    warmstart_normals_gpu,
    apply_solved_normals_gpu,
    detect_and_solve_limits_gpu,
    build_and_solve_equality_gpu,
)

# Newton solver parameters
comptime NEWTON_CPU_ITERATIONS: Int = 200
comptime NEWTON_CPU_TOLERANCE: Float64 = 1e-12
# Debug flag
comptime NEWTON_CPU_DEBUG: Bool = False
comptime MINVAL: Float64 = 1e-10


@always_inline
fn _build_hessian[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
    V_SIZE: Int,
    M_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    D_vals: InlineArray[Scalar[DTYPE], _max_one[MAX_ROWS]()],
    jar: InlineArray[Scalar[DTYPE], _max_one[MAX_ROWS]()],
    cstate: InlineArray[Int, _max_one[MAX_ROWS]()],
    mut H: InlineArray[Scalar[DTYPE], M_SIZE],
):
    """Build Hessian contributions from active constraints.

    For QUADRATIC state: add D_r * J_r * J_r^T (standard per-row).
    For CONE state: add J_group^T * H_cone * J_group (coupled dim x dim).

    The cone Hessian H_cone is d^2(cost)/d(jar)^2 for the contact group:
      H[0,0] = Dm
      H[0,j] = -Dm * mu * jar_fj / T
      H[j,k] = Dm * mu^2 * jar_fj*jar_fk/T^2 + Dm*mu*(N-mu*T)/T * (jar_fj*jar_fk/T^2 - delta_jk)
      where N = jar_n, T = ||jar_f||, Dm = D_n/(1+mu^2)

    Reference: mujoco-main/src/engine/engine_core_constraint.c:2530-2569
    """
    comptime MR = _max_one[MAX_ROWS]()
    var num_normals = constraints.num_normals
    var num_friction = constraints.num_friction
    var friction_start = num_normals
    var limits_start = num_normals + num_friction

    # Track which rows are handled as part of cone groups
    var handled = InlineArray[Bool, MR](fill=False)

    # Process cone contact groups (normal + friction children together)
    var fric_idx = 0
    for n in range(num_normals):
        # Find friction children
        var group_size = 0
        while fric_idx + group_size < num_friction:
            if (
                constraints.rows[
                    friction_start + fric_idx + group_size
                ].friction_parent
                != n
            ):
                break
            group_size += 1

        if cstate[n] == PRIMAL_CONE and group_size > 0:
            # Cone state: build full dim x dim Hessian
            var D_n = D_vals[n]
            var mu = constraints.rows[
                friction_start + fric_idx
            ].friction_coef
            # Dm = D_n / (1 + mu^2) in jar-space (no group_size factor)
            var Dm = D_n / (Scalar[DTYPE](1) + mu * mu)

            var N = jar[n]
            var T_sq: Scalar[DTYPE] = 0
            for g in range(group_size):
                T_sq += jar[friction_start + fric_idx + g] * jar[friction_start + fric_idx + g]
            var T = sqrt(T_sq)
            var T_safe = T
            if T_safe < Scalar[DTYPE](MINVAL):
                T_safe = Scalar[DTYPE](MINVAL)
            var s = N - mu * T

            # Build cone Hessian as J^T * H_cone * J
            # Rather than form H_cone explicitly, compute J_cone^T * H_cone * J_cone
            # row indices: n (normal), friction_start + fric_idx + g (friction children)

            # H_cone[0,0] = Dm → add Dm * J_n * J_n^T
            for i in range(NV):
                for j in range(NV):
                    H[i * NV + j] += (
                        Dm
                        * constraints.J[n * NV + i]
                        * constraints.J[n * NV + j]
                    )

            # H_cone[0,g+1] = -Dm * mu * jar_fg / T_safe  (cross-terms)
            for g in range(group_size):
                var fr = friction_start + fric_idx + g
                var h_cross = -Dm * mu * jar[fr] / T_safe
                for i in range(NV):
                    for j in range(NV):
                        # J_n^T * h_cross * J_fr + J_fr^T * h_cross * J_n (symmetric)
                        H[i * NV + j] += h_cross * (
                            constraints.J[n * NV + i] * constraints.J[fr * NV + j]
                            + constraints.J[fr * NV + i] * constraints.J[n * NV + j]
                        )

            # H_cone[g1+1,g2+1]: friction-friction block
            for g1 in range(group_size):
                var fr1 = friction_start + fric_idx + g1
                for g2 in range(group_size):
                    var fr2 = friction_start + fric_idx + g2
                    # Outer product term: Dm * mu^2 * jar_f1*jar_f2 / T^2
                    var h_ff = Dm * mu * mu * jar[fr1] * jar[fr2] / (T_safe * T_safe)
                    # Rank-correction: Dm * mu * s / T * (jar_f1*jar_f2/T^2 - delta)
                    h_ff += Dm * mu * s / T_safe * (
                        jar[fr1] * jar[fr2] / (T_safe * T_safe)
                    )
                    if g1 == g2:
                        h_ff -= Dm * mu * s / T_safe
                    for i in range(NV):
                        for j in range(NV):
                            H[i * NV + j] += (
                                h_ff
                                * constraints.J[fr1 * NV + i]
                                * constraints.J[fr2 * NV + j]
                            )

            # Mark these rows as handled
            handled[n] = True
            for g in range(group_size):
                handled[friction_start + fric_idx + g] = True

        fric_idx += group_size

    # Add standard per-row contributions for non-cone active rows
    for r in range(constraints.num_rows):
        if handled[r] or cstate[r] == PRIMAL_SATISFIED:
            continue
        # QUADRATIC state: standard D * J * J^T
        var D_r = D_vals[r]
        for i in range(NV):
            for j in range(NV):
                H[i * NV + j] += (
                    D_r
                    * constraints.J[r * NV + i]
                    * constraints.J[r * NV + j]
                )


struct NewtonSolver(ConstraintSolver):
    """MuJoCo-style Newton constraint solver.

    Operates in qacc space, minimizing the cost function
    (Gauss + constraint penalties) using Newton's method with
    Cholesky-factored Hessian and Armijo linesearch.

    The Hessian H = M + J^T*D*J is always positive definite (since M is PD
    and J^T*D*J is PSD), so Newton direction is always a descent direction.

    Handles ALL constraints (including friction cone) in a unified
    optimization via cone-aware constraint_update.
    No separate PGS friction phase needed on CPU.
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Newton solver workspace size."""
        comptime MC = _max_one[MAX_CONTACTS]()
        return 84 * MC + 12 * MC * NV + MC * MC

    @staticmethod
    fn solve[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        MAX_ROWS: Int,
        V_SIZE: Int,
        M_SIZE: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
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
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qacc: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints using Newton on CPU.

        Unified optimization over all constraints (normals + friction cone +
        limits + equality) using cone-aware constraint_update.
        """
        if constraints.num_rows == 0:
            return

        comptime MR = _max_one[MAX_ROWS]()

        var num_rows = constraints.num_rows
        var num_normals = constraints.num_normals
        var num_friction = constraints.num_friction
        var friction_start = num_normals

        # Compute D values from stored diagApprox and inv_K_imp
        # D = 1/R where R = 1/inv_K_imp - K = (1-imp)/imp * diagApprox
        # All constraint types now store diagApprox, so primal_D works correctly.
        var D_vals = InlineArray[Scalar[DTYPE], MR](fill=Scalar[DTYPE](0))
        for r in range(num_rows):
            D_vals[r] = primal_D(
                constraints.rows[r].inv_K_imp,
                constraints.rows[r].K,
            )
        # Each row keeps its own D = 1/R computed from its own diagApprox.
        # In MuJoCo's BOTTOM zone, force = -D[j]*jar[j] uses per-row D.
        # In CONE zone, Dm = D_n/(1+mu²) handles normal-friction coupling.

        # Save qacc_smooth (unconstrained acceleration)
        var qacc_smooth = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qacc_smooth[i] = qacc[i]

        # qfrc_smooth from constraints (filled by integrator)
        var qfrc_smooth = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qfrc_smooth[i] = constraints.qfrc_smooth[i]

        # Compute Ma = M * qacc
        var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            Ma[i] = Scalar[DTYPE](0)
            for j in range(NV):
                Ma[i] += constraints.M_hat[i * NV + j] * qacc[j]

        # Compute jar = J * qacc - aref (aref = -bias)
        var jar = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        for i in range(MR):
            jar[i] = Scalar[DTYPE](0)
        compute_jar[DTYPE, MAX_ROWS, NV, V_SIZE, MR](constraints, qacc, jar)

        # Compute initial force, state, cost (cone-aware with MuJoCo D)
        var force = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        var cstate = InlineArray[Int, MR](uninitialized=True)
        for i in range(MR):
            force[i] = Scalar[DTYPE](0)
            cstate[i] = PRIMAL_SATISFIED
        var constraint_cost: Scalar[DTYPE] = 0
        constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
            constraints, jar, D_vals, force, cstate, constraint_cost
        )

        @parameter
        if NEWTON_CPU_DEBUG:
            print("  [PRIMAL] num_rows=", num_rows, " normals=", constraints.num_normals, " friction=", constraints.num_friction, " limits=", constraints.num_limits)
            for r in range(num_rows):
                var state_str = "SAT"
                if cstate[r] == PRIMAL_QUADRATIC:
                    state_str = "QUAD"
                elif cstate[r] == PRIMAL_CONE:
                    state_str = "CONE"
                print("  row", r, " type=", constraints.rows[r].constraint_type, " D_mj=", Float64(D_vals[r]), " jar=", Float64(jar[r]), " force=", Float64(force[r]), " state=", state_str, " bias=", Float64(constraints.rows[r].bias), " mu=", Float64(constraints.rows[r].friction_coef), " parent=", constraints.rows[r].friction_parent)

        # Compute qfrc_constraint = J^T * force
        var qfrc_constraint = InlineArray[Scalar[DTYPE], V_SIZE](
            uninitialized=True
        )
        compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
            constraints, force, qfrc_constraint
        )

        # Build Hessian H = M + J^T * D_active * J + cone Hessians
        var H = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(NV * NV):
            H[i] = constraints.M_hat[i]

        _build_hessian[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                       MAX_ROWS, V_SIZE, M_SIZE](
            constraints, D_vals, jar, cstate, H
        )

        # Cholesky factorize H
        chol_factor[DTYPE, NV, M_SIZE](H, L)

        # Compute scale for convergence check (MuJoCo: 1/sum(M diagonal))
        var scale: Scalar[DTYPE] = 0
        for i in range(NV):
            scale += constraints.M_hat[i * NV + i]
        if scale > Scalar[DTYPE](MINVAL):
            scale = Scalar[DTYPE](1.0) / scale
        else:
            scale = Scalar[DTYPE](1.0)

        # Main Newton iteration loop
        var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        var total_iter = 0

        for iter in range(NEWTON_CPU_ITERATIONS):
            total_iter += 1
            # Compute gradient: grad = Ma - qfrc_smooth - qfrc_constraint
            var grad_norm: Scalar[DTYPE] = 0
            for i in range(NV):
                grad[i] = Ma[i] - qfrc_smooth[i] - qfrc_constraint[i]
                grad_norm += grad[i] * grad[i]

            @parameter
            if NEWTON_CPU_DEBUG:
                print("    [PRIMAL_NEWTON] iter_start", total_iter, " grad_norm=", Float64(sqrt(grad_norm)), " scaled=", Float64(scale * sqrt(grad_norm)))

            # Check gradient convergence
            if scale * sqrt(grad_norm) < Scalar[DTYPE](NEWTON_CPU_TOLERANCE):
                @parameter
                if NEWTON_CPU_DEBUG:
                    print("    [PRIMAL_NEWTON] CONVERGED at iter", total_iter, " (gradient)")
                break

            # Newton direction: search = -H^{-1} * grad via Cholesky solve
            chol_solve[DTYPE, NV, M_SIZE, V_SIZE](L, grad, search)
            for i in range(NV):
                search[i] = -search[i]

            # Compute Mv = M * search (needed for line search)
            for i in range(NV):
                Mv[i] = Scalar[DTYPE](0)
                for j in range(NV):
                    Mv[i] += constraints.M_hat[i * NV + j] * search[j]

            # Forward-exploring linesearch with MuJoCo D
            var alpha = primal_linesearch_with_D[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                constraints,
                D_vals,
                qacc,
                qacc_smooth,
                qfrc_smooth,
                Ma,
                Mv,
                search,
                jar,
                force,
                Scalar[DTYPE](NEWTON_CPU_TOLERANCE),
            )

            if alpha == Scalar[DTYPE](0):
                @parameter
                if NEWTON_CPU_DEBUG:
                    print("    [PRIMAL_NEWTON] STOPPED at iter", total_iter, " (alpha=0)")
                break

            # Save old state, cost, qacc, Ma
            var old_cost = constraint_cost + compute_gauss_cost[DTYPE, NV, V_SIZE](
                Ma, qfrc_smooth, qacc, qacc_smooth
            )
            var old_state = InlineArray[Int, MR](uninitialized=True)
            for i in range(num_rows):
                old_state[i] = cstate[i]
            var old_qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            var old_Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            for i in range(NV):
                old_qacc[i] = qacc[i]
                old_Ma[i] = Ma[i]

            # Update qacc, Ma
            for i in range(NV):
                qacc[i] += alpha * search[i]
                Ma[i] += alpha * Mv[i]

            # Recompute jar
            compute_jar[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                constraints, qacc, jar
            )

            # Recompute force, state, cost (cone-aware with MuJoCo D)
            constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
                constraints, jar, D_vals, force, cstate, constraint_cost
            )

            # Recompute qfrc_constraint
            compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                constraints, force, qfrc_constraint
            )

            # Check improvement
            var new_cost = constraint_cost + compute_gauss_cost[DTYPE, NV, V_SIZE](
                Ma, qfrc_smooth, qacc, qacc_smooth
            )
            var improvement = scale * (old_cost - new_cost)

            @parameter
            if NEWTON_CPU_DEBUG:
                print(
                    "    [PRIMAL_NEWTON] iter",
                    total_iter,
                    " alpha=",
                    Float64(alpha),
                    " cost=",
                    Float64(new_cost),
                    " improvement=",
                    Float64(improvement),
                    " grad=",
                    Float64(sqrt(grad_norm)),
                )

            if improvement < Scalar[DTYPE](NEWTON_CPU_TOLERANCE) and iter > 0:
                # Restore qacc/Ma if cost increased
                if improvement < Scalar[DTYPE](0):
                    for i in range(NV):
                        qacc[i] = old_qacc[i]
                        Ma[i] = old_Ma[i]
                    # Recompute jar/force at restored point
                    compute_jar[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                        constraints, qacc, jar
                    )
                    constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
                        constraints, jar, D_vals, force, cstate, constraint_cost
                    )
                    compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                        constraints, force, qfrc_constraint
                    )
                break

            # Check if any states changed — if so, rebuild Hessian
            var state_changed = False
            for r in range(num_rows):
                if old_state[r] != cstate[r]:
                    state_changed = True
                    break

            if state_changed:
                for i in range(NV * NV):
                    H[i] = constraints.M_hat[i]
                _build_hessian[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                               MAX_ROWS, V_SIZE, M_SIZE](
                    constraints, D_vals, jar, cstate, H
                )
                chol_factor[DTYPE, NV, M_SIZE](H, L)

        @parameter
        if NEWTON_CPU_DEBUG:
            print("  [PRIMAL] Final states:")
            for r in range(num_rows):
                var state_str = "SAT"
                if cstate[r] == PRIMAL_QUADRATIC:
                    state_str = "QUAD"
                elif cstate[r] == PRIMAL_CONE:
                    state_str = "CONE"
                print("    row", r, " state=", state_str, " jar=", Float64(jar[r]), " force=", Float64(force[r]))

        # Write forces back to constraint lambda_val for warm-starting
        for r in range(num_rows):
            constraints.rows[r].lambda_val = force[r]

    @staticmethod
    fn solver_threads[
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ]() -> Int:
        return _max_one[MAX_CONTACTS]()

    @staticmethod
    @always_inline
    fn solve_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        V_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """GPU solve — uses PGS-based approach on GPU.

        The GPU path uses a PGS-based dual approach while the CPU path
        uses the full Newton optimization matching MuJoCo exactly.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime solver_ws_idx = ws_solver_offset[NV, NBODY]()
        comptime MC = _max_one[MAX_CONTACTS]()

        # Common normal block offsets
        comptime ws_lambda_n_idx = solver_ws_idx + 0 * MC
        comptime ws_K_n_idx = solver_ws_idx + 1 * MC
        comptime ws_c_dist_idx = solver_ws_idx + 2 * MC
        comptime ws_J_n_idx = solver_ws_idx + 13 * MC
        comptime ws_MinvJn_idx = solver_ws_idx + 13 * MC + MC * NV

        # Newton-specific offsets (after common normal block)
        comptime NW_START = solver_ws_idx + 13 * MC + 2 * MC * NV
        comptime ws_rhs_idx = NW_START + 0 * MC
        comptime ws_A_idx = NW_START + 1 * MC
        comptime ws_grad_idx = NW_START + MC + MC * MC
        comptime ws_d_idx = NW_START + 2 * MC + MC * MC
        comptime ws_ltrial_idx = NW_START + 3 * MC + MC * MC
        comptime ws_fmap_idx = NW_START + 4 * MC + MC * MC

        # === PARALLEL: Initialize workspace ===
        if valid_env:
            init_common_normal_workspace_gpu[
                DTYPE,
                NV,
                NBODY,
                MAX_CONTACTS,
                WS_SIZE,
                BATCH,
            ](env, contact_tid, workspace)
            # Init Newton-specific
            workspace[env, ws_rhs_idx + contact_tid] = 0
            workspace[env, ws_grad_idx + contact_tid] = 0
            workspace[env, ws_d_idx + contact_tid] = 0
            workspace[env, ws_ltrial_idx + contact_tid] = 0
            workspace[env, ws_fmap_idx + contact_tid] = -1
            for c2 in range(MC):
                workspace[
                    env, ws_A_idx + contact_tid * MAX_CONTACTS + c2
                ] = 0

        # Read metadata
        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var nc = 0
        var dt: Scalar[DTYPE] = 0
        var friction_coef: Scalar[DTYPE] = 0
        var K_spring: Scalar[DTYPE] = 0
        var B_damp: Scalar[DTYPE] = 0
        var si_dmin: Scalar[DTYPE] = 0
        var si_dmax: Scalar[DTYPE] = 0
        var si_width: Scalar[DTYPE] = 1

        if valid_env:
            dt = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
            )
            nc = Int(
                rebind[Scalar[DTYPE]](
                    state[env, meta_off + META_IDX_NUM_CONTACTS]
                )
            )
            friction_coef = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_FRICTION]
            )
            if nc > MAX_CONTACTS:
                nc = MAX_CONTACTS
            var sr_tc = rebind[Scalar[DTYPE]](
                model[
                    0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0
                ]
            )
            var sr_dr = rebind[Scalar[DTYPE]](
                model[
                    0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1
                ]
            )
            si_dmin = rebind[Scalar[DTYPE]](
                model[
                    0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_0
                ]
            )
            si_dmax = rebind[Scalar[DTYPE]](
                model[
                    0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1
                ]
            )
            si_width = rebind[Scalar[DTYPE]](
                model[
                    0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_2
                ]
            )
            if si_width < Scalar[DTYPE](1e-6):
                si_width = Scalar[DTYPE](1e-6)
            if si_dmax < Scalar[DTYPE](1e-4):
                si_dmax = Scalar[DTYPE](1e-4)
            K_spring = Scalar[DTYPE](1.0) / (
                sr_tc * sr_tc * si_dmax * si_dmax
            )
            B_damp = Scalar[DTYPE](2.0) * sr_dr / (sr_tc * si_dmax)

        # === PARALLEL PHASE 1: Each thread precomputes one contact ===
        if valid_env:
            precompute_contact_normal_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                V_SIZE,
                BATCH,
                WS_SIZE,
                NGEOM,
                MAX_EQUALITY,
                COMPUTE_RHS=True,
                RHS_IDX=ws_rhs_idx,
            ](
                env,
                contact_tid,
                nc,
                state,
                model,
                workspace,
                K_spring,
                B_damp,
                si_dmin,
                si_dmax,
                si_width,
            )

        barrier()

        # === PARALLEL DELASSUS BUILD ===
        if valid_env and contact_tid < nc:
            if workspace[env, ws_c_dist_idx + contact_tid] < Scalar[DTYPE](
                0
            ):
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](
                        0
                    ):
                        continue
                    var a_val: workspace.element_type = 0
                    for i in range(NV):
                        a_val += (
                            workspace[
                                env, ws_J_n_idx + contact_tid * NV + i
                            ]
                            * workspace[env, ws_MinvJn_idx + c2 * NV + i]
                        )
                    workspace[
                        env, ws_A_idx + contact_tid * MAX_CONTACTS + c2
                    ] = a_val
                # Add regularizer
                comptime ws_inv_K_imp_idx = solver_ws_idx + 12 * MC
                var R_c = (
                    Scalar[DTYPE](1.0)
                    / workspace[env, ws_inv_K_imp_idx + contact_tid]
                    - workspace[env, ws_K_n_idx + contact_tid]
                )
                workspace[
                    env,
                    ws_A_idx
                    + contact_tid * MAX_CONTACTS
                    + contact_tid,
                ] += R_c

        barrier()

        # === SEQUENTIAL: Thread 0 handles Newton iterations ===
        if not valid_env or contact_tid != 0:
            return

        warmstart_normals_gpu[
            DTYPE,
            NV,
            NBODY,
            MAX_CONTACTS,
            WS_SIZE,
            BATCH,
        ](env, nc, workspace)

        # Newton iterations for GPU
        comptime NEWTON_ITERATIONS = 100
        comptime NEWTON_TOLERANCE: Float64 = 1e-8
        comptime LINESEARCH_ITERATIONS = 10
        comptime LINESEARCH_BETA: Float64 = 0.5
        comptime LINESEARCH_ARMIJO: Float64 = 1e-4

        for _ in range(NEWTON_ITERATIONS):
            var grad_norm: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    workspace[env, ws_grad_idx + c] = Scalar[DTYPE](0)
                    continue
                var g: workspace.element_type = workspace[
                    env, ws_rhs_idx + c
                ]
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](
                        0
                    ):
                        continue
                    g += (
                        workspace[
                            env, ws_A_idx + c * MAX_CONTACTS + c2
                        ]
                        * workspace[env, ws_lambda_n_idx + c2]
                    )
                workspace[env, ws_grad_idx + c] = g

            grad_norm = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](
                    0
                ) or (workspace[env, ws_grad_idx + c]) < Scalar[DTYPE](0):
                    grad_norm += (
                        workspace[env, ws_grad_idx + c]
                        * workspace[env, ws_grad_idx + c]
                    )

            if grad_norm < Scalar[DTYPE](NEWTON_TOLERANCE):
                break

            var free_count = 0
            for c in range(nc):
                workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](-1)
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](
                    0
                ) or (workspace[env, ws_grad_idx + c]) < Scalar[DTYPE](0):
                    workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](
                        free_count
                    )
                    free_count += 1

            if free_count == 0:
                break

            for c in range(nc):
                workspace[env, ws_d_idx + c] = Scalar[DTYPE](0)

            for c in range(nc):
                if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                    continue
                var AR_diag = workspace[
                    env, ws_A_idx + c * MAX_CONTACTS + c
                ]
                if AR_diag > Scalar[DTYPE](1e-14):
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c]) / AR_diag
                    )

            for _ in range(5):
                for c in range(nc):
                    if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                        continue
                    var sum_off_diag: workspace.element_type = 0
                    for c2 in range(nc):
                        if c2 == c:
                            continue
                        if workspace[env, ws_fmap_idx + c2] < Scalar[
                            DTYPE
                        ](0):
                            continue
                        sum_off_diag += workspace[
                            env, ws_A_idx + c * MAX_CONTACTS + c2
                        ] * (workspace[env, ws_d_idx + c2])
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c]) - sum_off_diag
                    ) / (
                        workspace[env, ws_A_idx + c * MAX_CONTACTS + c]
                    )

            var f_current: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                f_current += (
                    workspace[env, ws_rhs_idx + c]
                    * workspace[env, ws_lambda_n_idx + c]
                )
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](
                        0
                    ):
                        continue
                    f_current += (
                        Scalar[DTYPE](0.5)
                        * workspace[env, ws_lambda_n_idx + c]
                        * workspace[
                            env, ws_A_idx + c * MAX_CONTACTS + c2
                        ]
                        * workspace[env, ws_lambda_n_idx + c2]
                    )

            var gtd: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                    continue
                gtd += (workspace[env, ws_grad_idx + c]) * (
                    workspace[env, ws_d_idx + c]
                )

            var step = Scalar[DTYPE](1.0)
            var armijo = Scalar[DTYPE](LINESEARCH_ARMIJO)
            var beta = Scalar[DTYPE](LINESEARCH_BETA)

            for _ in range(LINESEARCH_ITERATIONS):
                for c in range(nc):
                    workspace[env, ws_ltrial_idx + c] = workspace[
                        env, ws_lambda_n_idx + c
                    ]
                    if workspace[env, ws_fmap_idx + c] >= Scalar[DTYPE](0):
                        workspace[env, ws_ltrial_idx + c] = workspace[
                            env, ws_lambda_n_idx + c
                        ] + step * (workspace[env, ws_d_idx + c])
                    if workspace[env, ws_ltrial_idx + c] < Scalar[DTYPE](
                        0
                    ):
                        workspace[env, ws_ltrial_idx + c] = Scalar[DTYPE](
                            0
                        )

                var f_trial: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](
                        0
                    ):
                        continue
                    f_trial += (
                        workspace[env, ws_rhs_idx + c]
                        * workspace[env, ws_ltrial_idx + c]
                    )
                    for c2 in range(nc):
                        if workspace[
                            env, ws_c_dist_idx + c2
                        ] >= Scalar[DTYPE](0):
                            continue
                        f_trial += (
                            Scalar[DTYPE](0.5)
                            * workspace[env, ws_ltrial_idx + c]
                            * workspace[
                                env, ws_A_idx + c * MAX_CONTACTS + c2
                            ]
                            * workspace[env, ws_ltrial_idx + c2]
                        )

                if f_trial <= f_current + armijo * step * gtd:
                    break

                step = step * beta

            for c in range(nc):
                workspace[env, ws_lambda_n_idx + c] = workspace[
                    env, ws_ltrial_idx + c
                ]

        apply_solved_normals_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            MAX_CONTACTS,
            STATE_SIZE,
            WS_SIZE,
            BATCH,
        ](env, nc, state, workspace)

        detect_and_solve_limits_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            WS_SIZE,
            BATCH,
            NEWTON_ITERATIONS,
            NGEOM,
            MAX_EQUALITY,
        ](env, dt, state, model, workspace)

        build_and_solve_equality_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_EQUALITY,
            NGEOM,
            STATE_SIZE,
            MODEL_SIZE,
            V_SIZE,
            WS_SIZE,
            BATCH,
            NEWTON_ITERATIONS,
        ](env, state, model, workspace)

        comptime FRICTION_WS_OFFSET = 18 * MC + 2 * MC * NV + MC * MC
        _solve_friction_pgs_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            V_SIZE,
            BATCH,
            WS_SIZE,
            FRICTION_WS_OFFSET,
            CONE_TYPE,
        ](
            env,
            state,
            model,
            workspace,
            nc,
            friction_coef,
            contacts_off,
        )
