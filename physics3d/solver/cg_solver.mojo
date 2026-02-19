"""CG constraint solver (MuJoCo-matching).

Operates in qacc (acceleration) space, minimizing the same cost as Newton:
  cost = 0.5*(qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)  [Gauss term]
       + sum_i penalty_i(J*qacc - aref)                         [constraint costs]

The difference from Newton:
- Newton: search = -H^{-1} * grad  (Cholesky solve of full Hessian H = M + J^T*D*J)
- CG:     search = -M^{-1} * grad + beta * search_old  (LDL solve of M only, CG direction)

Beta is computed via Polak-Ribiere formula for conjugate direction updates.
Both share: constraint_update, linesearch, convergence, cost.

The preconditioner is M (mass matrix), not the full Hessian.

Reference: mujoco-main/src/engine/engine_solver.c (mj_solPrimal, mjSOL_CG branch)
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
from ..dynamics.mass_matrix import ldl_factor, ldl_solve

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
    build_and_solve_tendon_gpu,
)

# CG solver parameters
comptime CG_CPU_ITERATIONS: Int = 200
comptime CG_CPU_TOLERANCE: Float64 = 1e-12
# Debug flag
comptime CG_CPU_DEBUG: Bool = False
comptime MINVAL: Float64 = 1e-10


struct CGSolver(ConstraintSolver):
    """MuJoCo-style CG constraint solver.

    Operates in qacc space, minimizing the cost function
    (Gauss + constraint penalties) using nonlinear CG with M as
    preconditioner and Newton-based linesearch.

    The CG direction uses Polak-Ribiere formula with M-preconditioning:
      Mgrad = M^{-1} * grad
      beta = dot(grad, Mgrad - Mgradold) / dot(gradold, Mgradold)
      search = -Mgrad + beta * search_old

    Unlike Newton, no Hessian build/factorize is needed — only M^{-1}
    via LDL solve. This makes CG cheaper per iteration but may need
    more iterations to converge.
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """CG solver workspace size."""
        comptime MC = _max_one[MAX_CONTACTS]()
        return 79 * MC + 12 * MC * NV

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
        MAX_TENDON: Int = 0,
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
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qacc: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints using CG on CPU.

        Unified optimization over all constraints (normals + friction cone +
        limits + equality) using cone-aware constraint_update with M-preconditioned
        nonlinear conjugate gradient.
        """
        if constraints.num_rows == 0:
            return

        comptime MR = _max_one[MAX_ROWS]()

        var num_rows = constraints.num_rows

        # Compute D values from stored diagApprox and inv_K_imp
        var D_vals = InlineArray[Scalar[DTYPE], MR](fill=Scalar[DTYPE](0))
        for r in range(num_rows):
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

        # Copy M_hat into local M_SIZE array for LDL factorization
        var M_local = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(NV * NV):
            M_local[i] = constraints.M_hat[i]

        # LDL factorize M_hat (preconditioner for CG)
        var L_ldl = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var D_ldl = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M_local, L_ldl, D_ldl)

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
        if CG_CPU_DEBUG:
            print("  [PRIMAL_CG] num_rows=", num_rows, " normals=", constraints.num_normals, " friction=", constraints.num_friction, " limits=", constraints.num_limits)

        # Compute qfrc_constraint = J^T * force
        var qfrc_constraint = InlineArray[Scalar[DTYPE], V_SIZE](
            uninitialized=True
        )
        compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
            constraints, force, qfrc_constraint
        )

        # Compute scale for convergence check (MuJoCo: 1/sum(M diagonal))
        var scale: Scalar[DTYPE] = 0
        for i in range(NV):
            scale += constraints.M_hat[i * NV + i]
        if scale > Scalar[DTYPE](MINVAL):
            scale = Scalar[DTYPE](1.0) / scale
        else:
            scale = Scalar[DTYPE](1.0)

        # Compute initial gradient: grad = Ma - qfrc_smooth - qfrc_constraint
        var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mgrad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # CG-specific: arrays to track previous gradient for Polak-Ribiere
        var gradold = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mgradold = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        var grad_norm: Scalar[DTYPE] = 0
        for i in range(NV):
            grad[i] = Ma[i] - qfrc_smooth[i] - qfrc_constraint[i]
            grad_norm += grad[i] * grad[i]

        # Check initial convergence
        if scale * sqrt(grad_norm) < Scalar[DTYPE](CG_CPU_TOLERANCE):
            # Write forces back for warm-starting
            for r in range(num_rows):
                constraints.rows[r].lambda_val = force[r]
            return

        # Initial preconditioned gradient: Mgrad = M^{-1} * grad
        ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L_ldl, D_ldl, grad, Mgrad)

        # Initial search direction: search = -Mgrad
        for i in range(NV):
            search[i] = -Mgrad[i]

        # Main CG iteration loop
        var total_iter = 0

        for iter in range(CG_CPU_ITERATIONS):
            total_iter += 1

            @parameter
            if CG_CPU_DEBUG:
                print("    [PRIMAL_CG] iter_start", total_iter, " grad_norm=", Float64(sqrt(grad_norm)), " scaled=", Float64(scale * sqrt(grad_norm)))

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
                Scalar[DTYPE](CG_CPU_TOLERANCE),
            )

            if alpha == Scalar[DTYPE](0):
                @parameter
                if CG_CPU_DEBUG:
                    print("    [PRIMAL_CG] STOPPED at iter", total_iter, " (alpha=0)")
                break

            # Save old cost, qacc, Ma
            var old_cost = constraint_cost + compute_gauss_cost[DTYPE, NV, V_SIZE](
                Ma, qfrc_smooth, qacc, qacc_smooth
            )
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
            if CG_CPU_DEBUG:
                print(
                    "    [PRIMAL_CG] iter",
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

            if improvement < Scalar[DTYPE](CG_CPU_TOLERANCE) and iter > 0:
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

            # Save old gradient and preconditioned gradient for Polak-Ribiere
            for i in range(NV):
                gradold[i] = grad[i]
                Mgradold[i] = Mgrad[i]

            # Compute new gradient
            grad_norm = 0
            for i in range(NV):
                grad[i] = Ma[i] - qfrc_smooth[i] - qfrc_constraint[i]
                grad_norm += grad[i] * grad[i]

            # Check gradient convergence
            if scale * sqrt(grad_norm) < Scalar[DTYPE](CG_CPU_TOLERANCE):
                @parameter
                if CG_CPU_DEBUG:
                    print("    [PRIMAL_CG] CONVERGED at iter", total_iter, " (gradient)")
                break

            # Compute new preconditioned gradient: Mgrad = M^{-1} * grad
            ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L_ldl, D_ldl, grad, Mgrad)

            # Polak-Ribiere beta
            # beta = dot(grad, Mgrad - Mgradold) / max(MINVAL, dot(gradold, Mgradold))
            var num: Scalar[DTYPE] = 0
            var den: Scalar[DTYPE] = 0
            for i in range(NV):
                num += grad[i] * (Mgrad[i] - Mgradold[i])
                den += gradold[i] * Mgradold[i]
            if den < Scalar[DTYPE](MINVAL):
                den = Scalar[DTYPE](MINVAL)
            var beta = num / den
            # Reset conjugacy if beta < 0
            if beta < Scalar[DTYPE](0):
                beta = Scalar[DTYPE](0)

            # Update search direction: search = -Mgrad + beta * search
            for i in range(NV):
                search[i] = -Mgrad[i] + beta * search[i]

        @parameter
        if CG_CPU_DEBUG:
            print("  [PRIMAL_CG] Final iteration count:", total_iter)

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
        MAX_TENDON: Int = 0,
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

        The GPU path uses a PGS-based approach while the CPU path
        uses the full CG optimization matching MuJoCo exactly.
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

        # PGS-specific offsets (after common normal block)
        comptime PGS_START = solver_ws_idx + 13 * MC + 2 * MC * NV
        comptime ws_rhs_idx = PGS_START + 0 * MC

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

        # Read metadata
        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var nc = 0
        var dt: Scalar[DTYPE] = 0
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

        # === SEQUENTIAL: Thread 0 handles PGS iterations ===
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

        # PGS normal iterations for GPU
        comptime PGS_ITERATIONS = 100
        comptime inv_K_imp_idx = solver_ws_idx + 12 * MC

        for _ in range(PGS_ITERATIONS):
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                var a: workspace.element_type = workspace[env, ws_rhs_idx + c]
                for k in range(NV):
                    a += workspace[env, ws_J_n_idx + c * NV + k] * workspace[env, qacc_idx + k]
                var R_c = Scalar[DTYPE](1.0) / workspace[env, inv_K_imp_idx + c] - workspace[env, ws_K_n_idx + c]
                var residual = a + R_c * workspace[env, ws_lambda_n_idx + c]
                var delta = -residual * workspace[env, inv_K_imp_idx + c]
                var old_lambda = workspace[env, ws_lambda_n_idx + c]
                workspace[env, ws_lambda_n_idx + c] = workspace[env, ws_lambda_n_idx + c] + delta
                if workspace[env, ws_lambda_n_idx + c] < Scalar[DTYPE](0):
                    workspace[env, ws_lambda_n_idx + c] = Scalar[DTYPE](0)
                var actual = rebind[Scalar[DTYPE]](workspace[env, ws_lambda_n_idx + c]) - rebind[Scalar[DTYPE]](old_lambda)
                if actual != Scalar[DTYPE](0):
                    for k in range(NV):
                        workspace[env, qacc_idx + k] = rebind[Scalar[DTYPE]](workspace[env, qacc_idx + k]) + rebind[Scalar[DTYPE]](workspace[env, ws_MinvJn_idx + c * NV + k]) * actual

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
            PGS_ITERATIONS,
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
            PGS_ITERATIONS,
        ](env, state, model, workspace)

        # Tendon equality constraints
        @parameter
        if MAX_TENDON > 0:
            build_and_solve_tendon_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                MAX_EQUALITY,
                NGEOM,
                MAX_TENDON,
                STATE_SIZE,
                MODEL_SIZE,
                V_SIZE,
                WS_SIZE,
                BATCH,
                PGS_ITERATIONS,
            ](env, state, model, workspace)

        comptime FRICTION_WS_OFFSET = 13 * MC + 2 * MC * NV
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
        MAX_TENDON,
        ](
            env,
            state,
            model,
            workspace,
            nc,
            contacts_off,
        )
