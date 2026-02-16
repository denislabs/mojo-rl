"""Primal Newton constraint solver (MuJoCo-matching).

Operates in qacc (acceleration) space, minimizing:
  cost = 0.5*(qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)  [Gauss term]
       + sum_i penalty_i(J*qacc - aref)                         [constraint costs]

Forces are derived from qacc: force[i] = -D[i] * (J*qacc - aref)[i].
The Hessian is H = M + J^T*D_active*J (naturally positive definite).

This is fundamentally different from the dual Newton solver which builds
the Delassus matrix A = J*M^{-1}*J^T and solves a QP over forces.

The constraint_update uses MuJoCo's 3-zone cone logic for elliptic contacts:
- Top zone (N >= 0, N >= mu*T): satisfied, no forces
- Bottom zone (N + mu*T <= 0): full quadratic on all rows
- Middle zone: cone boundary projection with Dm-weighted cost

This eliminates the need for a separate PGS friction phase — the primal
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

# Import shared friction solver for GPU (still uses dual approach until primal GPU is done)
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

# Primal Newton solver parameters
comptime PRIMAL_NEWTON_ITERATIONS: Int = 200
comptime PRIMAL_NEWTON_TOLERANCE: Float64 = 1e-12
# Debug flag
comptime PRIMAL_NEWTON_DEBUG: Bool = False
comptime MINVAL: Float64 = 1e-10


struct PrimalNewtonSolver(ConstraintSolver):
    """MuJoCo-style primal Newton constraint solver.

    Operates in qacc space, minimizing the primal cost function
    (Gauss + constraint penalties) using Newton's method with
    Cholesky-factored Hessian and Armijo linesearch.

    The Hessian H = M + J^T*D*J is always positive definite (since M is PD
    and J^T*D*J is PSD), so Newton direction is always a descent direction.

    Unlike the dual solvers, this handles ALL constraints (including friction
    cone) in a unified optimization via cone-aware constraint_update.
    No separate PGS friction phase needed on CPU.
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Primal Newton workspace: same as dual Newton for GPU compatibility."""
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
        """Solve constraints using primal Newton on CPU.

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

        # Compute D values using body_invweight0 (MuJoCo-matching)
        # For each contact: diagApprox = body_invweight0[2*body_a] + body_invweight0[2*body_b]
        # For ground contacts (body_b = -1/0): diagApprox = body_invweight0[2*body_a]
        # D = imp / ((1-imp) * diagApprox)
        var D_vals = InlineArray[Scalar[DTYPE], MR](fill=Scalar[DTYPE](0))
        var fric_idx = 0
        for n in range(num_normals):
            var ci = constraints.rows[n].source_contact_idx
            var body_a = data.contacts[ci].body_a
            var body_b = data.contacts[ci].body_b

            # diagApprox from body inverse weights
            var diag_n: Scalar[DTYPE] = 0
            if body_a >= 0 and body_a < NBODY:
                diag_n += model.body_invweight0[body_a * 2]
            if body_b >= 0 and body_b < NBODY:
                diag_n += model.body_invweight0[body_b * 2]

            # Extract impedance from stored values
            var imp = constraints.rows[n].inv_K_imp * constraints.rows[n].K
            if imp < Scalar[DTYPE](1e-12):
                imp = Scalar[DTYPE](1e-12)
            if imp > Scalar[DTYPE](1) - Scalar[DTYPE](1e-12):
                imp = Scalar[DTYPE](1) - Scalar[DTYPE](1e-12)

            # D = imp / ((1-imp) * diagApprox)
            if diag_n > Scalar[DTYPE](1e-12):
                D_vals[n] = imp / (
                    (Scalar[DTYPE](1) - imp) * diag_n
                )
            else:
                # Fallback to primal_D if no body_invweight0
                D_vals[n] = primal_D(
                    constraints.rows[n].inv_K_imp,
                    constraints.rows[n].K,
                )

            # Friction children get same D as parent normal
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
            for g in range(group_size):
                D_vals[friction_start + fric_idx + g] = D_vals[n]
            fric_idx += group_size

        # Limits and equality: use primal_D (exact Delassus diagonal)
        var limits_start = num_normals + num_friction
        for r_off in range(constraints.num_limits):
            var r = limits_start + r_off
            D_vals[r] = primal_D(
                constraints.rows[r].inv_K_imp,
                constraints.rows[r].K,
            )
        var eq_start = limits_start + constraints.num_limits
        for r_off in range(constraints.num_equality):
            var r = eq_start + r_off
            D_vals[r] = primal_D(
                constraints.rows[r].inv_K_imp,
                constraints.rows[r].K,
            )

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
        if PRIMAL_NEWTON_DEBUG:
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

        # Build Hessian H = M + J^T * D_active * J
        var H = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(NV * NV):
            H[i] = constraints.M_hat[i]

        # Add contributions from active constraints using MuJoCo D
        for r in range(num_rows):
            if cstate[r] == PRIMAL_SATISFIED:
                continue
            var D_r = D_vals[r]
            for i in range(NV):
                for j in range(NV):
                    H[i * NV + j] += (
                        D_r
                        * constraints.J[r * NV + i]
                        * constraints.J[r * NV + j]
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

        for iter in range(PRIMAL_NEWTON_ITERATIONS):
            # Compute gradient: grad = Ma - qfrc_smooth - qfrc_constraint
            var grad_norm: Scalar[DTYPE] = 0
            for i in range(NV):
                grad[i] = Ma[i] - qfrc_smooth[i] - qfrc_constraint[i]
                grad_norm += grad[i] * grad[i]

            # Check gradient convergence
            if scale * sqrt(grad_norm) < Scalar[DTYPE](PRIMAL_NEWTON_TOLERANCE):
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

            # Cone-aware Armijo linesearch with MuJoCo D
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
                Scalar[DTYPE](PRIMAL_NEWTON_TOLERANCE),
            )

            if alpha == Scalar[DTYPE](0):
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
            if PRIMAL_NEWTON_DEBUG:
                print(
                    "    [PRIMAL_NEWTON] iter",
                    iter,
                    " alpha=",
                    Float64(alpha),
                    " cost=",
                    Float64(new_cost),
                    " improvement=",
                    Float64(improvement),
                    " grad=",
                    Float64(sqrt(grad_norm)),
                )

            if improvement < Scalar[DTYPE](PRIMAL_NEWTON_TOLERANCE) and iter > 0:
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

            # Incremental Hessian update: rank-1 Cholesky update/downdate
            # for constraints that changed state
            for r in range(num_rows):
                if old_state[r] == cstate[r]:
                    continue

                var D_r = D_vals[r]
                var sqrt_D = sqrt(D_r)

                # Build vector v = sqrt(D) * J[r,:]
                var v = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
                for i in range(NV):
                    v[i] = sqrt_D * constraints.J[r * NV + i]

                if (
                    old_state[r] == PRIMAL_SATISFIED
                    and cstate[r] != PRIMAL_SATISFIED
                ):
                    # Became active: add D*J*J^T to H
                    chol_rank1_update[DTYPE, NV, M_SIZE, V_SIZE](
                        L, v, Scalar[DTYPE](1)
                    )
                elif (
                    old_state[r] != PRIMAL_SATISFIED
                    and cstate[r] == PRIMAL_SATISFIED
                ):
                    # Became inactive: remove D*J*J^T from H
                    chol_rank1_update[DTYPE, NV, M_SIZE, V_SIZE](
                        L, v, Scalar[DTYPE](-1)
                    )

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
        """GPU solve — delegates to dual Newton GPU for now.

        The GPU primal Newton is Phase 2 work. For now, use the same
        GPU path as the dual Newton solver (which already works well).
        The CPU path is where the primal approach matters most for
        matching MuJoCo's results exactly.
        """
        # For GPU, reuse the existing dual approach (Phase 2 will add primal GPU)
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
                si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr
            )
            B_damp = Scalar[DTYPE](2.0) / (si_dmax * sr_tc)

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

        # Dual Newton iterations (same as NewtonSolver for GPU)
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
