"""CG constraint solver (MuJoCo-matching).

Operates in qacc (acceleration) space, minimizing the same cost as Newton:
  cost = 0.5*(qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)  [Gauss term]
       + sum_i penalty_i(J*qacc - aref)                         [constraint costs]

The difference from Newton:
- Newton: search = -H^{-1} * grad  (Cholesky solve of full Hessian H = M + J^T*D*J)
- CG:     search = -M^{-1} * grad + beta * search_old  (Cholesky solve of M only, CG direction)

Beta is computed via Polak-Ribiere formula for conjugate direction updates.
Both share: constraint_update, linesearch, convergence, cost.

The preconditioner is M (mass matrix), not the full Hessian.

Reference: mujoco-main/src/engine/engine_solver.c (mj_solPrimal, mjSOL_CG branch)
"""

from std.math import sqrt
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
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
from .cholesky import (
    chol_factor,
    chol_solve,
    chol_factor_inline,
    chol_solve_inline,
)

from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_M_offset,
    ws_fnet_offset,
    qvel_offset,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_IMPRATIO,
)

from ..dynamics.jacobian import compute_contact_jacobian_row_gpu

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

    comptime NEEDS_M_INV: Bool = True

    @staticmethod
    def solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """CG solver workspace size (primal, qacc-space).

        Layout identical to Newton primal layout:
          Common normal block: 15*MC + 2*MC*NV
          Primal-specific block: 12*MC + 4*MC*NV
            [J_t1: MC*NV | J_t2: MC*NV | MinvJt1: MC*NV | MinvJt2: MC*NV |
             mu: MC | D_n: MC | D_f: MC | bt1: MC | bt2: MC |
             jar_n: MC | jar_t1: MC | jar_t2: MC | fn: MC | ft1: MC | ft2: MC | cstate: MC]
        Total = 27*MC + 6*MC*NV
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 27 * MC + 6 * MC * NV

    @staticmethod
    def solve[
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
        """Solve constraints using CG on CPU.

        Unified optimization over all constraints (normals + friction cone +
        limits + equality) using cone-aware constraint_update with M-preconditioned
        nonlinear conjugate gradient.
        """
        if constraints.num_rows == 0:
            return

        comptime V_SIZE = _max_one[NV]()
        comptime M_SIZE = _max_one[NV * NV]()
        comptime MR = _max_one[MAX_ROWS]()

        var num_rows = constraints.num_rows

        # Compute D values from stored diagApprox and inv_K_imp
        var D_vals = List[Scalar[DTYPE]](capacity=MR)
        for _ in range(MR):
            D_vals.append(Scalar[DTYPE](0))
        for r in range(num_rows):
            D_vals[r] = primal_D(
                constraints.rows[r].inv_K_imp,
                constraints.rows[r].K,
            )

        # Save qacc_smooth (unconstrained acceleration)
        var qacc_smooth = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qacc_smooth.append(Scalar[DTYPE](0))
        for i in range(V_SIZE):
            qacc_smooth[i] = qacc[i]

        # qfrc_smooth from constraints (filled by integrator)
        var qfrc_smooth = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qfrc_smooth.append(Scalar[DTYPE](0))
        for i in range(V_SIZE):
            qfrc_smooth[i] = constraints.qfrc_smooth[i]

        # Copy M_hat into local M_SIZE array for LDL factorization
        var M_local = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            M_local.append(Scalar[DTYPE](0))
        for i in range(NV * NV):
            M_local[i] = constraints.M_hat[i]

        # LDL factorize M_hat (preconditioner for CG)
        var L_ldl = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            L_ldl.append(Scalar[DTYPE](0))
        var D_ldl = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            D_ldl.append(Scalar[DTYPE](0))
        ldl_factor[DTYPE, NV](M_local, L_ldl, D_ldl)

        # Compute Ma = M * qacc
        var Ma = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            Ma.append(Scalar[DTYPE](0))
        for i in range(NV):
            Ma[i] = Scalar[DTYPE](0)
            for j in range(NV):
                Ma[i] += constraints.M_hat[i * NV + j] * qacc[j]

        # Compute jar = J * qacc - aref (aref = -bias)
        var jar = List[Scalar[DTYPE]](capacity=MR)
        for _ in range(MR):
            jar.append(Scalar[DTYPE](0))
        compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)

        # Compute initial force, state, cost (cone-aware with MuJoCo D)
        var force = List[Scalar[DTYPE]](capacity=MR)
        for _ in range(MR):
            force.append(Scalar[DTYPE](0))
        var cstate = List[Int](capacity=MR)
        for _ in range(MR):
            cstate.append(PRIMAL_SATISFIED)
        var constraint_cost: Scalar[DTYPE] = 0
        constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
            constraints, jar, D_vals, force, cstate, constraint_cost
        )

        comptime if CG_CPU_DEBUG:
            print(
                "  [PRIMAL_CG] num_rows=",
                num_rows,
                " normals=",
                constraints.num_normals,
                " friction=",
                constraints.num_friction,
                " limits=",
                constraints.num_limits,
            )

        # Compute qfrc_constraint = J^T * force
        var qfrc_constraint = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qfrc_constraint.append(Scalar[DTYPE](0))

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
        var grad = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            grad.append(Scalar[DTYPE](0))
        var Mgrad = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            Mgrad.append(Scalar[DTYPE](0))
        var search = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            search.append(Scalar[DTYPE](0))
        var Mv = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            Mv.append(Scalar[DTYPE](0))

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
        ldl_solve[DTYPE, NV](L_ldl, D_ldl, grad, Mgrad)

        # Initial search direction: search = -Mgrad
        for i in range(NV):
            search[i] = -Mgrad[i]

        # Main CG iteration loop
        var total_iter = 0

        for iter in range(CG_CPU_ITERATIONS):
            total_iter += 1

            comptime if CG_CPU_DEBUG:
                print(
                    "    [PRIMAL_CG] iter_start",
                    total_iter,
                    " grad_norm=",
                    Float64(sqrt(grad_norm)),
                    " scaled=",
                    Float64(scale * sqrt(grad_norm)),
                )

            # Compute Mv = M * search (needed for line search)
            for i in range(NV):
                Mv[i] = Scalar[DTYPE](0)
                for j in range(NV):
                    Mv[i] += constraints.M_hat[i * NV + j] * search[j]

            # Forward-exploring linesearch with MuJoCo D
            var alpha = primal_linesearch_with_D[
                DTYPE, MAX_ROWS, NV, V_SIZE, MR
            ](
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
                comptime if CG_CPU_DEBUG:
                    print(
                        "    [PRIMAL_CG] STOPPED at iter",
                        total_iter,
                        " (alpha=0)",
                    )
                break

            # Save old cost, qacc, Ma
            var old_cost = constraint_cost + compute_gauss_cost[
                DTYPE, NV, V_SIZE
            ](Ma, qfrc_smooth, qacc, qacc_smooth)
            var old_qacc = InlineArray[Scalar[DTYPE], V_SIZE](
                uninitialized=True
            )
            var old_Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            for i in range(NV):
                old_qacc[i] = qacc[i]
                old_Ma[i] = Ma[i]

            # Update qacc, Ma
            for i in range(NV):
                qacc[i] += alpha * search[i]
                Ma[i] += alpha * Mv[i]

            # Recompute jar
            compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)

            # Recompute force, state, cost (cone-aware with MuJoCo D)
            constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
                constraints, jar, D_vals, force, cstate, constraint_cost
            )

            # Recompute qfrc_constraint
            compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                constraints, force, qfrc_constraint
            )

            # Check improvement
            var new_cost = constraint_cost + compute_gauss_cost[
                DTYPE, NV, V_SIZE
            ](Ma, qfrc_smooth, qacc, qacc_smooth)
            var improvement = scale * (old_cost - new_cost)

            comptime if CG_CPU_DEBUG:
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
                    compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)
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
                comptime if CG_CPU_DEBUG:
                    print(
                        "    [PRIMAL_CG] CONVERGED at iter",
                        total_iter,
                        " (gradient)",
                    )
                break

            # Compute new preconditioned gradient: Mgrad = M^{-1} * grad
            ldl_solve[DTYPE, NV](L_ldl, D_ldl, grad, Mgrad)

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

        comptime if CG_CPU_DEBUG:
            print("  [PRIMAL_CG] Final iteration count:", total_iter)

        # Write forces back to constraint lambda_val for warm-starting
        for r in range(num_rows):
            constraints.rows[r].lambda_val = force[r]

    @staticmethod
    def solver_threads[
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ]() -> Int:
        return _max_one[MAX_CONTACTS]()

    @staticmethod
    @always_inline
    def solve_gpu[
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
        NSITE: Int = 0,
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
        """GPU primal CG solver — matches CPU CG exactly.

        Operates in qacc (NV-dimensional) space with unified friction cone.
        Uses M-preconditioned nonlinear CG with Polak-Ribiere beta.
        Handles normal + friction (T1+T2) contacts in a single optimization.
        No separate PGS friction phase needed.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime solver_ws_idx = ws_solver_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime MC = _max_one[MAX_CONTACTS]()
        comptime M_SIZE = _max_one[NV * NV]()

        # Common normal block offsets
        comptime ws_lambda_n_idx = solver_ws_idx + 0 * MC
        comptime ws_K_n_idx = solver_ws_idx + 1 * MC
        comptime ws_c_dist_idx = solver_ws_idx + 2 * MC
        comptime ws_c_body_idx = solver_ws_idx + 3 * MC
        comptime ws_c_body_b_idx = solver_ws_idx + 4 * MC
        comptime ws_c_px_idx = solver_ws_idx + 5 * MC
        comptime ws_c_py_idx = solver_ws_idx + 6 * MC
        comptime ws_c_pz_idx = solver_ws_idx + 7 * MC
        comptime ws_c_nx_idx = solver_ws_idx + 8 * MC
        comptime ws_c_ny_idx = solver_ws_idx + 9 * MC
        comptime ws_c_nz_idx = solver_ws_idx + 10 * MC
        comptime ws_pos_bias_idx = solver_ws_idx + 11 * MC
        comptime ws_inv_K_imp_idx = solver_ws_idx + 12 * MC
        comptime ws_J_n_idx = solver_ws_idx + 15 * MC
        comptime ws_MinvJn_idx = solver_ws_idx + 15 * MC + MC * NV

        # Primal-specific offsets (after common normal block)
        comptime PRIMAL_START = solver_ws_idx + 15 * MC + 2 * MC * NV
        comptime ws_Jt1_idx = PRIMAL_START + 0 * MC * NV
        comptime ws_Jt2_idx = PRIMAL_START + 1 * MC * NV
        comptime ws_MinvJt1_idx = PRIMAL_START + 2 * MC * NV
        comptime ws_MinvJt2_idx = PRIMAL_START + 3 * MC * NV
        comptime SC = PRIMAL_START + 4 * MC * NV
        comptime ws_mu_idx = SC + 0 * MC
        comptime ws_D_n_idx = SC + 1 * MC
        comptime ws_D_f_idx = SC + 2 * MC
        comptime ws_bt1_idx = SC + 3 * MC
        comptime ws_bt2_idx = SC + 4 * MC
        comptime CVS = SC + 5 * MC
        comptime ws_jar_n_idx = CVS + 0 * MC
        comptime ws_jar_t1_idx = CVS + 1 * MC
        comptime ws_jar_t2_idx = CVS + 2 * MC
        comptime ws_fn_idx = CVS + 3 * MC
        comptime ws_ft1_idx = CVS + 4 * MC
        comptime ws_ft2_idx = CVS + 5 * MC
        comptime ws_cstate_idx = CVS + 6 * MC

        # === PARALLEL: Initialize common normal workspace ===
        if valid_env:
            init_common_normal_workspace_gpu[
                DTYPE,
                NV,
                NBODY,
                MAX_CONTACTS,
                WS_SIZE,
                BATCH,
            ](env, contact_tid, workspace)
            # Zero primal workspace for this contact slot
            if contact_tid < MC:
                for d in range(NV):
                    workspace[env, ws_Jt1_idx + contact_tid * NV + d] = 0
                    workspace[env, ws_Jt2_idx + contact_tid * NV + d] = 0
                    workspace[env, ws_MinvJt1_idx + contact_tid * NV + d] = 0
                    workspace[env, ws_MinvJt2_idx + contact_tid * NV + d] = 0
                workspace[env, ws_mu_idx + contact_tid] = 0
                workspace[env, ws_D_n_idx + contact_tid] = 0
                workspace[env, ws_D_f_idx + contact_tid] = 0
                workspace[env, ws_bt1_idx + contact_tid] = 0
                workspace[env, ws_bt2_idx + contact_tid] = 0
                workspace[env, ws_jar_n_idx + contact_tid] = 0
                workspace[env, ws_jar_t1_idx + contact_tid] = 0
                workspace[env, ws_jar_t2_idx + contact_tid] = 0
                workspace[env, ws_fn_idx + contact_tid] = 0
                workspace[env, ws_ft1_idx + contact_tid] = 0
                workspace[env, ws_ft2_idx + contact_tid] = 0
                workspace[env, ws_cstate_idx + contact_tid] = 0

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
        var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
        var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)
        var impratio: Scalar[DTYPE] = Scalar[DTYPE](1.0)

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
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0]
            )
            var sr_dr = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1]
            )
            si_dmin = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_0]
            )
            si_dmax = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1]
            )
            si_width = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_2]
            )
            si_midpoint = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_3]
            )
            si_power = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_4]
            )
            if si_width < Scalar[DTYPE](1e-6):
                si_width = Scalar[DTYPE](1e-6)
            if si_dmax < Scalar[DTYPE](1e-4):
                si_dmax = Scalar[DTYPE](1e-4)
            K_spring = Scalar[DTYPE](1.0) / (si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr)
            B_damp = Scalar[DTYPE](2.0) / (si_dmax * sr_tc)
            impratio = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_IMPRATIO]
            )
            if impratio < Scalar[DTYPE](1e-6):
                impratio = Scalar[DTYPE](1.0)

        # === PARALLEL PHASE 1: Each thread precomputes one contact's normal data ===
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
                COMPUTE_RHS=False,
                RHS_IDX=0,
                MAX_TENDON=MAX_TENDON,
                NSITE=NSITE,
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
                si_midpoint,
                si_power,
            )

        barrier()

        # === SEQUENTIAL: Thread 0 handles primal CG ===
        if not valid_env or contact_tid != 0:
            return

        comptime qvel_off = qvel_offset[NQ, NV]()
        comptime CG_ITER_GPU: Int = 100
        comptime CG_TOL_GPU: Float64 = 1e-6
        comptime LINESEARCH_ITER: Int = 10
        comptime ARMIJO: Float64 = 1e-4
        comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

        # === Step 1: Precompute friction tangent frames, D values, and bias ===
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for c in range(nc):
            if rebind[Scalar[DTYPE]](
                workspace[env, ws_c_dist_idx + c]
            ) >= Scalar[DTYPE](0):
                continue

            var nx = rebind[Scalar[DTYPE]](workspace[env, ws_c_nx_idx + c])
            var ny = rebind[Scalar[DTYPE]](workspace[env, ws_c_ny_idx + c])
            var nz = rebind[Scalar[DTYPE]](workspace[env, ws_c_nz_idx + c])

            var c_off = contacts_off + c * CONTACT_SIZE
            var hint_x = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_FRAME_T1_X]
            )
            var hint_y = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_FRAME_T1_Y]
            )
            var hint_z = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_FRAME_T1_Z]
            )
            if hint_x * hint_x + hint_y * hint_y + hint_z * hint_z < Scalar[
                DTYPE
            ](0.25):
                var abs_nx = abs(nx)
                var abs_ny = abs(ny)
                var abs_nz = abs(nz)
                if abs_nx <= abs_ny and abs_nx <= abs_nz:
                    hint_x = Scalar[DTYPE](1); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](0)
                elif abs_ny <= abs_nz:
                    hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](1); hint_z = Scalar[DTYPE](0)
                else:
                    hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](1)

            # Gram-Schmidt: orthogonalize hint against normal → T1
            var dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
            var t1x = hint_x - dot_nh * nx
            var t1y = hint_y - dot_nh * ny
            var t1z = hint_z - dot_nh * nz
            var t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
            if t1_mag < Scalar[DTYPE](1e-10):
                # Hint parallel to normal — fall back to least-aligned axis
                var abs_nx = abs(nx)
                var abs_ny = abs(ny)
                var abs_nz = abs(nz)
                if abs_nx <= abs_ny and abs_nx <= abs_nz:
                    hint_x = Scalar[DTYPE](1); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](0)
                elif abs_ny <= abs_nz:
                    hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](1); hint_z = Scalar[DTYPE](0)
                else:
                    hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](1)
                dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
                t1x = hint_x - dot_nh * nx
                t1y = hint_y - dot_nh * ny
                t1z = hint_z - dot_nh * nz
                t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
            if t1_mag > Scalar[DTYPE](1e-10):
                t1x = t1x / t1_mag
                t1y = t1y / t1_mag
                t1z = t1z / t1_mag

            # T2 = cross(normal, T1)
            var t2x = ny * t1z - nz * t1y
            var t2y = nz * t1x - nx * t1z
            var t2z = nx * t1y - ny * t1x

            var body_a = Int(
                rebind[Scalar[DTYPE]](workspace[env, ws_c_body_idx + c])
            )
            var body_b = Int(
                rebind[Scalar[DTYPE]](workspace[env, ws_c_body_b_idx + c])
            )
            var px = rebind[Scalar[DTYPE]](workspace[env, ws_c_px_idx + c])
            var py = rebind[Scalar[DTYPE]](workspace[env, ws_c_py_idx + c])
            var pz = rebind[Scalar[DTYPE]](workspace[env, ws_c_pz_idx + c])

            # Compute J_t1 and MinvJ_t1
            compute_contact_jacobian_row_gpu[
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
            ](
                env,
                state,
                model,
                workspace,
                body_a,
                body_b,
                px,
                py,
                pz,
                t1x,
                t1y,
                t1z,
                J_row,
            )
            for i in range(NV):
                workspace[env, ws_Jt1_idx + c * NV + i] = J_row[i]
                var mij: Scalar[DTYPE] = 0
                for j in range(NV):
                    mij += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, M_inv_idx + i * NV + j]
                        )
                        * J_row[j]
                    )
                workspace[env, ws_MinvJt1_idx + c * NV + i] = mij

            # Compute J_t2 and MinvJ_t2
            compute_contact_jacobian_row_gpu[
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
            ](
                env,
                state,
                model,
                workspace,
                body_a,
                body_b,
                px,
                py,
                pz,
                t2x,
                t2y,
                t2z,
                J_row,
            )
            for i in range(NV):
                workspace[env, ws_Jt2_idx + c * NV + i] = J_row[i]
                var mij: Scalar[DTYPE] = 0
                for j in range(NV):
                    mij += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, M_inv_idx + i * NV + j]
                        )
                        * J_row[j]
                    )
                workspace[env, ws_MinvJt2_idx + c * NV + i] = mij

            # D_n = 1/R_n, computed directly from stored imp and diag_n
            comptime ws_imp_n_cg = solver_ws_idx + 13 * MC
            comptime ws_diag_n_cg = solver_ws_idx + 14 * MC
            var imp_cg = rebind[Scalar[DTYPE]](
                workspace[env, ws_imp_n_cg + c]
            )
            var diag_cg = rebind[Scalar[DTYPE]](
                workspace[env, ws_diag_n_cg + c]
            )
            var R_n_c = (Scalar[DTYPE](1.0) - imp_cg) / imp_cg * diag_cg
            if R_n_c < Scalar[DTYPE](1e-14):
                R_n_c = Scalar[DTYPE](1e-14)
            var D_n_c = Scalar[DTYPE](1.0) / R_n_c
            workspace[env, ws_D_n_idx + c] = D_n_c
            workspace[env, ws_D_f_idx + c] = D_n_c / impratio

            # Friction coefficient
            var mu_c = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_FRICTION]
            )
            if mu_c <= Scalar[DTYPE](0):
                mu_c = Scalar[DTYPE](0.5)
            workspace[env, ws_mu_idx + c] = mu_c

            # Friction velocity-damping bias: bt = B_damp * J_t * qvel
            var bt1_c: Scalar[DTYPE] = 0
            var bt2_c: Scalar[DTYPE] = 0
            for i in range(NV):
                var qv_i = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
                bt1_c += (
                    rebind[Scalar[DTYPE]](
                        workspace[env, ws_Jt1_idx + c * NV + i]
                    )
                    * qv_i
                )
                bt2_c += (
                    rebind[Scalar[DTYPE]](
                        workspace[env, ws_Jt2_idx + c * NV + i]
                    )
                    * qv_i
                )
            workspace[env, ws_bt1_idx + c] = B_damp * bt1_c
            workspace[env, ws_bt2_idx + c] = B_damp * bt2_c

        # === Step 2: Cholesky factorize M (preconditioner) ===
        var M_chol = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var L_M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for k in range(NV * NV):
            M_chol[k] = rebind[Scalar[DTYPE]](workspace[env, M_idx + k])
        _ = chol_factor_inline[DTYPE, NV, M_SIZE](M_chol, L_M)

        # === Step 3: Initialize qacc, Ma, qfrc_sm ===
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var qacc_sm = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var qfrc_sm = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mgrad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var gradold = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mgradold = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        for i in range(NV):
            var q_i = rebind[Scalar[DTYPE]](workspace[env, qacc_idx + i])
            qacc[i] = q_i
            qacc_sm[i] = q_i
            qfrc_sm[i] = rebind[Scalar[DTYPE]](workspace[env, fnet_idx + i])

        # Ma = M * qacc
        for i in range(NV):
            var s: Scalar[DTYPE] = 0
            for j in range(NV):
                s += (
                    rebind[Scalar[DTYPE]](workspace[env, M_idx + i * NV + j])
                    * qacc[j]
                )
            Ma[i] = s

        # Scale = 1/trace(M) for convergence check
        var scale: Scalar[DTYPE] = 0
        for i in range(NV):
            scale += rebind[Scalar[DTYPE]](workspace[env, M_idx + i * NV + i])
        if scale > Scalar[DTYPE](1e-10):
            scale = Scalar[DTYPE](1.0) / scale
        else:
            scale = Scalar[DTYPE](1.0)

        # === Step 4: Compute initial jar and forces via 3-zone cone logic ===
        for c in range(nc):
            if rebind[Scalar[DTYPE]](
                workspace[env, ws_c_dist_idx + c]
            ) >= Scalar[DTYPE](0):
                workspace[env, ws_fn_idx + c] = 0
                workspace[env, ws_ft1_idx + c] = 0
                workspace[env, ws_ft2_idx + c] = 0
                workspace[env, ws_cstate_idx + c] = 0
                continue

            var jar_n_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                workspace[env, ws_pos_bias_idx + c]
            )
            var jar_t1_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                workspace[env, ws_bt1_idx + c]
            )
            var jar_t2_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                workspace[env, ws_bt2_idx + c]
            )
            for i in range(NV):
                var qa_i = qacc[i]
                jar_n_c += (
                    rebind[Scalar[DTYPE]](
                        workspace[env, ws_J_n_idx + c * NV + i]
                    )
                    * qa_i
                )
                jar_t1_c += (
                    rebind[Scalar[DTYPE]](
                        workspace[env, ws_Jt1_idx + c * NV + i]
                    )
                    * qa_i
                )
                jar_t2_c += (
                    rebind[Scalar[DTYPE]](
                        workspace[env, ws_Jt2_idx + c * NV + i]
                    )
                    * qa_i
                )
            workspace[env, ws_jar_n_idx + c] = jar_n_c
            workspace[env, ws_jar_t1_idx + c] = jar_t1_c
            workspace[env, ws_jar_t2_idx + c] = jar_t2_c

            var mu_c = rebind[Scalar[DTYPE]](workspace[env, ws_mu_idx + c])
            var D_n_c = rebind[Scalar[DTYPE]](workspace[env, ws_D_n_idx + c])
            var D_f_c = rebind[Scalar[DTYPE]](workspace[env, ws_D_f_idx + c])
            var T = sqrt(jar_t1_c * jar_t1_c + jar_t2_c * jar_t2_c)
            var T_safe = T
            if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

            if jar_n_c >= mu_c * T_safe:
                workspace[env, ws_fn_idx + c] = 0
                workspace[env, ws_ft1_idx + c] = 0
                workspace[env, ws_ft2_idx + c] = 0
                workspace[env, ws_cstate_idx + c] = 0
            elif mu_c * jar_n_c + T <= Scalar[DTYPE](0):
                workspace[env, ws_fn_idx + c] = -D_n_c * jar_n_c
                workspace[env, ws_ft1_idx + c] = -D_f_c * jar_t1_c
                workspace[env, ws_ft2_idx + c] = -D_f_c * jar_t2_c
                workspace[env, ws_cstate_idx + c] = 1
            else:
                var s = jar_n_c - mu_c * T_safe
                var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
                workspace[env, ws_fn_idx + c] = -Dm * s
                workspace[env, ws_ft1_idx + c] = (
                    Dm * mu_c * s * jar_t1_c / T_safe
                )
                workspace[env, ws_ft2_idx + c] = (
                    Dm * mu_c * s * jar_t2_c / T_safe
                )
                workspace[env, ws_cstate_idx + c] = 2

        # === Step 5: Compute initial gradient and preconditioned gradient ===
        var grad_norm_sq: Scalar[DTYPE] = 0
        for i in range(NV):
            var g: Scalar[DTYPE] = Ma[i] - qfrc_sm[i]
            for c in range(nc):
                var cs = Int(
                    rebind[Scalar[DTYPE]](workspace[env, ws_cstate_idx + c])
                )
                if cs == 0:
                    continue
                g -= (
                    rebind[Scalar[DTYPE]](
                        workspace[env, ws_J_n_idx + c * NV + i]
                    )
                    * rebind[Scalar[DTYPE]](workspace[env, ws_fn_idx + c])
                    + rebind[Scalar[DTYPE]](
                        workspace[env, ws_Jt1_idx + c * NV + i]
                    )
                    * rebind[Scalar[DTYPE]](workspace[env, ws_ft1_idx + c])
                    + rebind[Scalar[DTYPE]](
                        workspace[env, ws_Jt2_idx + c * NV + i]
                    )
                    * rebind[Scalar[DTYPE]](workspace[env, ws_ft2_idx + c])
                )
            grad[i] = g
            grad_norm_sq += g * g

        # Initial preconditioned gradient: Mgrad = M^{-1} * grad (Cholesky solve)
        chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](L_M, grad, Mgrad)

        # Initial search direction: search = -Mgrad
        for i in range(NV):
            search[i] = -Mgrad[i]

        # === Step 6: CG iteration loop ===
        for _iter in range(CG_ITER_GPU):
            # Convergence check
            if scale * sqrt(grad_norm_sq) < Scalar[DTYPE](CG_TOL_GPU):
                break

            # Mv = M * search (for linesearch Gauss cost)
            for i in range(NV):
                var s: Scalar[DTYPE] = 0
                for j in range(NV):
                    s += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, M_idx + i * NV + j]
                        )
                        * search[j]
                    )
                Mv[i] = s

            # Precompute J * search per contact direction (for linesearch)
            var Js_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
            var Js_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
            var Js_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
            for c in range(nc):
                var js_n_c: Scalar[DTYPE] = 0
                var js_t1_c: Scalar[DTYPE] = 0
                var js_t2_c: Scalar[DTYPE] = 0
                if rebind[Scalar[DTYPE]](
                    workspace[env, ws_c_dist_idx + c]
                ) < Scalar[DTYPE](0):
                    for i in range(NV):
                        var s_i = search[i]
                        js_n_c += (
                            rebind[Scalar[DTYPE]](
                                workspace[env, ws_J_n_idx + c * NV + i]
                            )
                            * s_i
                        )
                        js_t1_c += (
                            rebind[Scalar[DTYPE]](
                                workspace[env, ws_Jt1_idx + c * NV + i]
                            )
                            * s_i
                        )
                        js_t2_c += (
                            rebind[Scalar[DTYPE]](
                                workspace[env, ws_Jt2_idx + c * NV + i]
                            )
                            * s_i
                        )
                Js_n[c] = js_n_c
                Js_t1[c] = js_t1_c
                Js_t2[c] = js_t2_c

            # Compute current total cost and gradient-direction dot product
            var gauss_0: Scalar[DTYPE] = 0
            var g1: Scalar[DTYPE] = 0
            var g2: Scalar[DTYPE] = 0
            var gtd: Scalar[DTYPE] = 0
            for i in range(NV):
                var Ma_diff_i = Ma[i] - qfrc_sm[i]
                var qa_diff_i = qacc[i] - qacc_sm[i]
                gauss_0 += Ma_diff_i * qa_diff_i
                g1 += Ma_diff_i * search[i] + Mv[i] * qa_diff_i
                g2 += Mv[i] * search[i]
                gtd += grad[i] * search[i]
            gauss_0 = Scalar[DTYPE](0.5) * gauss_0
            g1 = Scalar[DTYPE](0.5) * g1
            g2 = Scalar[DTYPE](0.5) * g2

            # Current constraint cost
            var c_cost_0: Scalar[DTYPE] = 0
            for c in range(nc):
                if rebind[Scalar[DTYPE]](
                    workspace[env, ws_c_dist_idx + c]
                ) >= Scalar[DTYPE](0):
                    continue
                var cs = Int(
                    rebind[Scalar[DTYPE]](workspace[env, ws_cstate_idx + c])
                )
                var N = rebind[Scalar[DTYPE]](workspace[env, ws_jar_n_idx + c])
                var T1 = rebind[Scalar[DTYPE]](
                    workspace[env, ws_jar_t1_idx + c]
                )
                var T2 = rebind[Scalar[DTYPE]](
                    workspace[env, ws_jar_t2_idx + c]
                )
                var mu_c = rebind[Scalar[DTYPE]](workspace[env, ws_mu_idx + c])
                var D_n_c = rebind[Scalar[DTYPE]](
                    workspace[env, ws_D_n_idx + c]
                )
                var D_f_c = rebind[Scalar[DTYPE]](
                    workspace[env, ws_D_f_idx + c]
                )
                if cs == 1:
                    c_cost_0 += Scalar[DTYPE](0.5) * (
                        D_n_c * N * N + D_f_c * (T1 * T1 + T2 * T2)
                    )
                elif cs == 2:
                    var T_s = sqrt(T1 * T1 + T2 * T2)
                    if T_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        T_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    var s = N - mu_c * T_s
                    var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
                    c_cost_0 += Scalar[DTYPE](0.5) * Dm * s * s

            var current_cost = gauss_0 + c_cost_0

            # Armijo linesearch
            var alpha = Scalar[DTYPE](1.0)
            var armijo_c = Scalar[DTYPE](ARMIJO)
            for _ in range(LINESEARCH_ITER):
                var trial_gauss = gauss_0 + alpha * g1 + alpha * alpha * g2
                var trial_c_cost: Scalar[DTYPE] = 0
                for c in range(nc):
                    if rebind[Scalar[DTYPE]](
                        workspace[env, ws_c_dist_idx + c]
                    ) >= Scalar[DTYPE](0):
                        continue
                    var trial_N = (
                        rebind[Scalar[DTYPE]](workspace[env, ws_jar_n_idx + c])
                        + alpha * Js_n[c]
                    )
                    var trial_T1 = (
                        rebind[Scalar[DTYPE]](workspace[env, ws_jar_t1_idx + c])
                        + alpha * Js_t1[c]
                    )
                    var trial_T2 = (
                        rebind[Scalar[DTYPE]](workspace[env, ws_jar_t2_idx + c])
                        + alpha * Js_t2[c]
                    )
                    var mu_c = rebind[Scalar[DTYPE]](
                        workspace[env, ws_mu_idx + c]
                    )
                    var D_n_c = rebind[Scalar[DTYPE]](
                        workspace[env, ws_D_n_idx + c]
                    )
                    var D_f_c = rebind[Scalar[DTYPE]](
                        workspace[env, ws_D_f_idx + c]
                    )
                    var trial_T = sqrt(
                        trial_T1 * trial_T1 + trial_T2 * trial_T2
                    )
                    var trial_T_safe = trial_T
                    if trial_T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        trial_T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    if trial_N >= mu_c * trial_T_safe:
                        pass
                    elif mu_c * trial_N + trial_T <= Scalar[DTYPE](0):
                        trial_c_cost += Scalar[DTYPE](0.5) * (
                            D_n_c * trial_N * trial_N
                            + D_f_c
                            * (trial_T1 * trial_T1 + trial_T2 * trial_T2)
                        )
                    else:
                        var trial_s = trial_N - mu_c * trial_T_safe
                        var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
                        trial_c_cost += (
                            Scalar[DTYPE](0.5) * Dm * trial_s * trial_s
                        )
                var trial_cost = trial_gauss + trial_c_cost
                if trial_cost <= current_cost + armijo_c * alpha * gtd:
                    break
                alpha = alpha * Scalar[DTYPE](0.5)

            if alpha < Scalar[DTYPE](1e-12):
                break

            # Update qacc and Ma
            for i in range(NV):
                qacc[i] = qacc[i] + alpha * search[i]
                Ma[i] = Ma[i] + alpha * Mv[i]

            # Recompute jar and forces (3-zone cone logic)
            for c in range(nc):
                if rebind[Scalar[DTYPE]](
                    workspace[env, ws_c_dist_idx + c]
                ) >= Scalar[DTYPE](0):
                    continue
                var jar_n_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                    workspace[env, ws_pos_bias_idx + c]
                )
                var jar_t1_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                    workspace[env, ws_bt1_idx + c]
                )
                var jar_t2_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                    workspace[env, ws_bt2_idx + c]
                )
                for i in range(NV):
                    var qa_i = qacc[i]
                    jar_n_c += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_J_n_idx + c * NV + i]
                        )
                        * qa_i
                    )
                    jar_t1_c += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_Jt1_idx + c * NV + i]
                        )
                        * qa_i
                    )
                    jar_t2_c += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_Jt2_idx + c * NV + i]
                        )
                        * qa_i
                    )
                workspace[env, ws_jar_n_idx + c] = jar_n_c
                workspace[env, ws_jar_t1_idx + c] = jar_t1_c
                workspace[env, ws_jar_t2_idx + c] = jar_t2_c

                var mu_c = rebind[Scalar[DTYPE]](workspace[env, ws_mu_idx + c])
                var D_n_c = rebind[Scalar[DTYPE]](
                    workspace[env, ws_D_n_idx + c]
                )
                var D_f_c = rebind[Scalar[DTYPE]](
                    workspace[env, ws_D_f_idx + c]
                )
                var T = sqrt(jar_t1_c * jar_t1_c + jar_t2_c * jar_t2_c)
                var T_safe = T
                if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                if jar_n_c >= mu_c * T_safe:
                    workspace[env, ws_fn_idx + c] = 0
                    workspace[env, ws_ft1_idx + c] = 0
                    workspace[env, ws_ft2_idx + c] = 0
                    workspace[env, ws_cstate_idx + c] = 0
                elif mu_c * jar_n_c + T <= Scalar[DTYPE](0):
                    workspace[env, ws_fn_idx + c] = -D_n_c * jar_n_c
                    workspace[env, ws_ft1_idx + c] = -D_f_c * jar_t1_c
                    workspace[env, ws_ft2_idx + c] = -D_f_c * jar_t2_c
                    workspace[env, ws_cstate_idx + c] = 1
                else:
                    var s = jar_n_c - mu_c * T_safe
                    var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
                    workspace[env, ws_fn_idx + c] = -Dm * s
                    workspace[env, ws_ft1_idx + c] = (
                        Dm * mu_c * s * jar_t1_c / T_safe
                    )
                    workspace[env, ws_ft2_idx + c] = (
                        Dm * mu_c * s * jar_t2_c / T_safe
                    )
                    workspace[env, ws_cstate_idx + c] = Scalar[DTYPE](2)

            # Save old gradient for Polak-Ribiere
            for i in range(NV):
                gradold[i] = grad[i]
                Mgradold[i] = Mgrad[i]

            # Compute new gradient
            grad_norm_sq = 0
            for i in range(NV):
                var g: Scalar[DTYPE] = Ma[i] - qfrc_sm[i]
                for c in range(nc):
                    var cs = Int(
                        rebind[Scalar[DTYPE]](workspace[env, ws_cstate_idx + c])
                    )
                    if cs == 0:
                        continue
                    g -= (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_J_n_idx + c * NV + i]
                        )
                        * rebind[Scalar[DTYPE]](workspace[env, ws_fn_idx + c])
                        + rebind[Scalar[DTYPE]](
                            workspace[env, ws_Jt1_idx + c * NV + i]
                        )
                        * rebind[Scalar[DTYPE]](workspace[env, ws_ft1_idx + c])
                        + rebind[Scalar[DTYPE]](
                            workspace[env, ws_Jt2_idx + c * NV + i]
                        )
                        * rebind[Scalar[DTYPE]](workspace[env, ws_ft2_idx + c])
                    )
                grad[i] = g
                grad_norm_sq += g * g

            # Compute new preconditioned gradient: Mgrad = M^{-1} * grad
            chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](L_M, grad, Mgrad)

            # Polak-Ribiere beta
            var num: Scalar[DTYPE] = 0
            var den: Scalar[DTYPE] = 0
            for i in range(NV):
                num += grad[i] * (Mgrad[i] - Mgradold[i])
                den += gradold[i] * Mgradold[i]
            if den < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                den = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
            var beta = num / den
            if beta < Scalar[DTYPE](0):
                beta = Scalar[DTYPE](0)

            # Update search direction: search = -Mgrad + beta * search
            for i in range(NV):
                search[i] = -Mgrad[i] + beta * search[i]

        # Write solved qacc back to workspace
        for i in range(NV):
            workspace[env, qacc_idx + i] = qacc[i]

        # Write forces to state buffer
        for c in range(nc):
            var c_off = contacts_off + c * CONTACT_SIZE
            state[env, c_off + CONTACT_IDX_FORCE_N] = workspace[
                env, ws_fn_idx + c
            ]
            state[env, c_off + CONTACT_IDX_FORCE_T1] = workspace[
                env, ws_ft1_idx + c
            ]
            state[env, c_off + CONTACT_IDX_FORCE_T2] = workspace[
                env, ws_ft2_idx + c
            ]

        comptime SOLVER_ITER_GPU: Int = 50
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
            SOLVER_ITER_GPU,
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
            SOLVER_ITER_GPU,
        ](env, state, model, workspace)

        comptime if MAX_TENDON > 0:
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
                SOLVER_ITER_GPU,
            ](env, state, model, workspace)
