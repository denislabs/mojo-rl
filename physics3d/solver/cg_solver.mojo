"""Conjugate Gradient (CG) constraint solver for Generalized Coordinates engine.

Implements MuJoCo-style CG constraint solving in joint space:
1. Form the Delassus matrix: A[c1,c2] = J[c1] * M^-1 * J[c2]^T
2. Solve A * lambda = -b using Projected Conjugate Gradient
3. Apply impulses: qvel += M^-1 * J^T * lambda

The CG solver converges faster than PGS for well-conditioned problems
because it uses global search directions instead of sweeping one constraint
at a time. The projection step handles the unilateral constraint (lambda >= 0).

Friction is solved with PGS iterations (same as MuJoCo's approach).

Reference: MuJoCo Technical Notes, Section on CG solver.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..traits.solver import ConstraintSolver
from ..dynamics.jacobian import (
    compute_contact_jacobian_row,
    compute_contact_jacobian_row_gpu,
)
from .constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    model_joint_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_qvel_pred_offset,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_IMPULSE_N,
    CONTACT_IDX_IMPULSE_T1,
    CONTACT_IDX_IMPULSE_T2,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_FRICTION,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MAX_POS_CORRECTION_VEL,
)
from ..joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
)

from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_IMPULSE_N,
    CONTACT_IDX_IMPULSE_T1,
    CONTACT_IDX_IMPULSE_T2,
)

# Import shared friction solver (GPU only now — CPU friction uses ConstraintData)
from .friction_solver import _solve_friction_pgs_gpu

# CG solver parameters
comptime CG_ITERATIONS: Int = 30
comptime CG_TOLERANCE: Float64 = 1e-8


struct CGSolver(ConstraintSolver):
    """Projected Conjugate Gradient constraint solver for GC engine.

    Solves the normal constraint MLCP using CG with projection onto
    the feasible set (lambda >= 0). Friction is handled via PGS.

    Advantages over PGS:
    - Faster convergence for well-conditioned systems
    - Uses global conjugate directions instead of per-constraint sweeps

    Disadvantages:
    - Projection breaks CG conjugacy, requiring residual resets
    - Slightly more memory per iteration (search direction, A*p product)
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """CG solver workspace: 25*MC + 6*MC*NV + MC*MC floats.

        Layout (offsets relative to solver workspace start):
          [0..11*MC)                                    Common contact block
          [11*MC..12*MC)                                rhs
          [12*MC..12*MC+MC*NV)                          J_n (normal Jacobian)
          [12*MC+MC*NV..12*MC+2*MC*NV)                  MinvJn (M_inv @ J_n^T)
          [12*MC+2*MC*NV..12*MC+2*MC*NV+MC*MC)          A (Delassus matrix)
          [12*MC+2*MC*NV+MC*MC..13*MC+2*MC*NV+MC*MC)    r (residual)
          [13*MC+2*MC*NV+MC*MC..14*MC+2*MC*NV+MC*MC)    p (search direction)
          [14*MC+2*MC*NV+MC*MC..15*MC+2*MC*NV+MC*MC)    Ap (A*p product)
          [15*MC+2*MC*NV+MC*MC..25*MC+6*MC*NV+MC*MC)    Friction (10*MC + 4*MC*NV)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 25 * MC + 6 * MC * NV + MC * MC

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
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints using Projected CG on CPU.

        Iterates over pre-built ConstraintData:
        1. Build Delassus matrix from normal constraint rows
        2. Projected CG for normal constraints
        3. PGS for joint limit constraints
        4. PGS for friction (with Coulomb cone clamping)
        """
        if constraints.num_rows == 0:
            return

        var num_normals = constraints.num_normals
        var num_friction = constraints.num_friction
        var num_limits = constraints.num_limits
        var friction_start = num_normals
        var limits_start = num_normals + num_friction

        comptime MR = _max_one[MAX_ROWS]()
        comptime A_SIZE = _max_one[MAX_ROWS * MAX_ROWS]()

        # RHS for CG: rhs[r] = v_n + bias (computed from constraints)
        var rhs = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        for i in range(MR):
            rhs[i] = Scalar[DTYPE](0)

        # Compute RHS for normal rows and apply warm-start
        for r in range(num_normals):
            var v_n: Scalar[DTYPE] = 0
            for i in range(NV):
                v_n += constraints.J[r * NV + i] * qvel[i]
            rhs[r] = v_n + constraints.rows[r].bias

            # Apply warm-start
            if constraints.rows[r].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qvel[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # Build Delassus matrix A[c1,c2] = J[c1] . MinvJT[c2]
        var A = InlineArray[Scalar[DTYPE], A_SIZE](uninitialized=True)
        for i in range(A_SIZE):
            A[i] = Scalar[DTYPE](0)
        for c1 in range(num_normals):
            for c2 in range(num_normals):
                var a_val: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_val += constraints.J[c1 * NV + i] * constraints.MinvJT[c2 * NV + i]
                A[c1 * num_normals + c2] = a_val

        # =====================================================================
        # Phase 2: Projected CG for normal constraints
        # Solve: A * lambda = -rhs, subject to lambda >= 0
        # =====================================================================
        var r_vec = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        var p = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        var Ap = InlineArray[Scalar[DTYPE], MR](uninitialized=True)

        for i in range(MR):
            r_vec[i] = Scalar[DTYPE](0)
            p[i] = Scalar[DTYPE](0)
            Ap[i] = Scalar[DTYPE](0)

        # Compute initial residual: r = -rhs - A*lambda
        for c in range(num_normals):
            var ax: Scalar[DTYPE] = 0
            for c2 in range(num_normals):
                ax += A[c * num_normals + c2] * constraints.rows[c2].lambda_val
            r_vec[c] = -rhs[c] - ax

        # Project residual
        for c in range(num_normals):
            if constraints.rows[c].lambda_val <= Scalar[DTYPE](0) and r_vec[c] < Scalar[DTYPE](0):
                r_vec[c] = Scalar[DTYPE](0)

        for c in range(num_normals):
            p[c] = r_vec[c]

        var rr: Scalar[DTYPE] = 0
        for c in range(num_normals):
            rr += r_vec[c] * r_vec[c]

        # CG iterations
        for _ in range(CG_ITERATIONS):
            if rr < Scalar[DTYPE](CG_TOLERANCE):
                break

            for c in range(num_normals):
                Ap[c] = Scalar[DTYPE](0)
                for c2 in range(num_normals):
                    Ap[c] += A[c * num_normals + c2] * p[c2]

            var pAp: Scalar[DTYPE] = 0
            for c in range(num_normals):
                pAp += p[c] * Ap[c]

            if pAp < Scalar[DTYPE](1e-14):
                break

            var alpha = rr / pAp

            var projected = False
            for c in range(num_normals):
                constraints.rows[c].lambda_val = constraints.rows[c].lambda_val + alpha * p[c]
                if constraints.rows[c].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[c].lambda_val = Scalar[DTYPE](0)
                    projected = True

            # Full residual recompute
            for c in range(num_normals):
                var ax: Scalar[DTYPE] = 0
                for c2 in range(num_normals):
                    ax += A[c * num_normals + c2] * constraints.rows[c2].lambda_val
                r_vec[c] = -rhs[c] - ax

            for c in range(num_normals):
                if constraints.rows[c].lambda_val <= Scalar[DTYPE](0) and r_vec[c] < Scalar[DTYPE](0):
                    r_vec[c] = Scalar[DTYPE](0)

            var rr_new: Scalar[DTYPE] = 0
            for c in range(num_normals):
                rr_new += r_vec[c] * r_vec[c]

            if projected or rr < Scalar[DTYPE](1e-14):
                for c in range(num_normals):
                    p[c] = r_vec[c]
            else:
                var beta = rr_new / rr
                for c in range(num_normals):
                    p[c] = r_vec[c] + beta * p[c]

            rr = rr_new

        # Apply solved impulses: remove warm-start, apply final
        for c in range(num_normals):
            var warm = data.contacts[constraints.rows[c].source_contact_idx].impulse_n
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    qvel[i] -= constraints.MinvJT[c * NV + i] * warm

        for c in range(num_normals):
            if constraints.rows[c].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qvel[i] += constraints.MinvJT[c * NV + i] * constraints.rows[c].lambda_val

        # =====================================================================
        # Phase 2b: PGS joint limit iterations
        # =====================================================================
        if num_limits > 0:
            for _ in range(CG_ITERATIONS):
                for r_off in range(num_limits):
                    var r = limits_start + r_off
                    var dof = constraints.rows[r].source_dof
                    var sign = constraints.rows[r].limit_sign
                    var v_limit = sign * qvel[dof]

                    var delta = -(v_limit + constraints.rows[r].bias) * constraints.rows[r].inv_K_imp
                    var old_lambda = constraints.rows[r].lambda_val
                    constraints.rows[r].lambda_val = constraints.rows[r].lambda_val + delta

                    if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                        constraints.rows[r].lambda_val = Scalar[DTYPE](0)

                    var actual = constraints.rows[r].lambda_val - old_lambda

                    for i in range(NV):
                        qvel[i] += constraints.MinvJT[r * NV + i] * actual

        # =====================================================================
        # Phase 3: Friction PGS with Coulomb cone
        # =====================================================================
        if num_friction == 0:
            return

        # Apply friction warm-start
        for r_off in range(num_friction):
            var r = friction_start + r_off
            if constraints.rows[r].lambda_val != Scalar[DTYPE](0):
                for i in range(NV):
                    qvel[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # Friction PGS iterations (t1 and t2 are consecutive pairs)
        for _ in range(CG_ITERATIONS):
            var pair_idx = 0
            while pair_idx < num_friction:
                var r_t1 = friction_start + pair_idx
                var r_t2 = friction_start + pair_idx + 1
                var normal_row = constraints.rows[r_t1].friction_parent
                var mu = constraints.rows[r_t1].friction_coef
                var lambda_n = constraints.rows[normal_row].lambda_val

                if lambda_n <= Scalar[DTYPE](0):
                    pair_idx += 2
                    continue

                var max_friction = mu * lambda_n

                # Tangent 1
                var v_t1: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_t1 += constraints.J[r_t1 * NV + i] * qvel[i]
                var delta_t1 = -v_t1 / constraints.rows[r_t1].K
                var old_t1 = constraints.rows[r_t1].lambda_val
                constraints.rows[r_t1].lambda_val = constraints.rows[r_t1].lambda_val + delta_t1

                # Tangent 2
                var v_t2: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_t2 += constraints.J[r_t2 * NV + i] * qvel[i]
                var delta_t2 = -v_t2 / constraints.rows[r_t2].K
                var old_t2 = constraints.rows[r_t2].lambda_val
                constraints.rows[r_t2].lambda_val = constraints.rows[r_t2].lambda_val + delta_t2

                # Coulomb cone clamping
                var t_mag = sqrt(
                    constraints.rows[r_t1].lambda_val * constraints.rows[r_t1].lambda_val
                    + constraints.rows[r_t2].lambda_val * constraints.rows[r_t2].lambda_val
                )
                if t_mag > max_friction:
                    var scale = max_friction / t_mag
                    constraints.rows[r_t1].lambda_val = constraints.rows[r_t1].lambda_val * scale
                    constraints.rows[r_t2].lambda_val = constraints.rows[r_t2].lambda_val * scale

                var actual_t1 = constraints.rows[r_t1].lambda_val - old_t1
                var actual_t2 = constraints.rows[r_t2].lambda_val - old_t2

                for i in range(NV):
                    qvel[i] += (
                        constraints.MinvJT[r_t1 * NV + i] * actual_t1
                        + constraints.MinvJT[r_t2 * NV + i] * actual_t2
                    )

                pair_idx += 2

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
        """Solve contact constraints using Projected CG on GPU.

        Uses thread_x for environment index, thread_y for contact index.
        Phase 1 and Delassus build are parallelized across contacts.
        CG iterations are sequential on thread_y==0.
        All threads must hit all barriers (no early returns between them).
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        comptime qvel_idx = ws_qvel_pred_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime solver_ws_idx = ws_solver_offset[NV, NBODY]()

        comptime MC = _max_one[MAX_CONTACTS]()

        # Common contact block (11*MC)
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
        # CG-specific
        comptime ws_rhs_idx = solver_ws_idx + 11 * MC
        comptime ws_J_n_idx = solver_ws_idx + 12 * MC  # MC*NV floats
        comptime ws_MinvJn_idx = solver_ws_idx + 12 * MC + MC * NV  # MC*NV floats
        comptime ws_A_idx = solver_ws_idx + 12 * MC + 2 * MC * NV  # MC*MC floats (Delassus)
        comptime ws_r_idx = solver_ws_idx + 12 * MC + 2 * MC * NV + MC * MC  # residual
        comptime ws_p_idx = solver_ws_idx + 13 * MC + 2 * MC * NV + MC * MC  # search direction
        comptime ws_Ap_idx = solver_ws_idx + 14 * MC + 2 * MC * NV + MC * MC  # A*p

        # === PARALLEL: Initialize workspace (each thread handles one slot) ===
        if valid_env:
            workspace[env, ws_lambda_n_idx + contact_tid] = 0
            workspace[env, ws_K_n_idx + contact_tid] = 1
            workspace[env, ws_c_dist_idx + contact_tid] = 0
            workspace[env, ws_c_body_idx + contact_tid] = 0
            workspace[env, ws_c_body_b_idx + contact_tid] = -1
            workspace[env, ws_c_px_idx + contact_tid] = 0
            workspace[env, ws_c_py_idx + contact_tid] = 0
            workspace[env, ws_c_pz_idx + contact_tid] = 0
            workspace[env, ws_c_nx_idx + contact_tid] = 0
            workspace[env, ws_c_ny_idx + contact_tid] = 0
            workspace[env, ws_c_nz_idx + contact_tid] = 1
            workspace[env, ws_rhs_idx + contact_tid] = 0
            workspace[env, ws_r_idx + contact_tid] = 0
            workspace[env, ws_p_idx + contact_tid] = 0
            workspace[env, ws_Ap_idx + contact_tid] = 0
            for i in range(NV):
                workspace[env, ws_J_n_idx + contact_tid * NV + i] = 0
                workspace[env, ws_MinvJn_idx + contact_tid * NV + i] = 0
            for c2 in range(MC):
                workspace[env, ws_A_idx + contact_tid * MAX_CONTACTS + c2] = 0

        # All threads read metadata
        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var nc = 0
        var dt: Scalar[DTYPE] = 0
        var friction_coef: Scalar[DTYPE] = 0
        var inv_tc_dr: Scalar[DTYPE] = 0
        var b_vel_coef: Scalar[DTYPE] = 0
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
            if si_width < Scalar[DTYPE](1e-6):
                si_width = Scalar[DTYPE](1e-6)
            if si_dmax < Scalar[DTYPE](1e-4):
                si_dmax = Scalar[DTYPE](1e-4)
            inv_tc_dr = Scalar[DTYPE](1.0) / (sr_tc * sr_dr)
            b_vel_coef = Scalar[DTYPE](2.0) * sr_dr * dt / (si_dmax * sr_tc)

        # === PARALLEL PHASE 1: Each thread precomputes one contact ===
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            J_row[i] = 0

        if valid_env and contact_tid < nc:
            var c = contact_tid
            var c_off = contacts_off + c * CONTACT_SIZE
            var body = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_A])
            )
            var body_b = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_B])
            )
            var dist = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_DIST]
            )

            workspace[env, ws_c_dist_idx + c] = dist
            workspace[env, ws_c_body_idx + c] = Scalar[DTYPE](body)
            workspace[env, ws_c_body_b_idx + c] = Scalar[DTYPE](body_b)

            if dist < Scalar[DTYPE](0):
                workspace[env, ws_c_px_idx + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_POS_X]
                )
                workspace[env, ws_c_py_idx + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_POS_Y]
                )
                workspace[env, ws_c_pz_idx + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_POS_Z]
                )
                workspace[env, ws_c_nx_idx + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_NX]
                )
                workspace[env, ws_c_ny_idx + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_NY]
                )
                workspace[env, ws_c_nz_idx + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_NZ]
                )

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
                    body,
                    body_b,
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_px_idx + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_py_idx + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_pz_idx + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_nx_idx + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_ny_idx + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_nz_idx + c]),
                    J_row,
                )

                var k: workspace.element_type = 0
                var v_n: workspace.element_type = 0
                for i in range(NV):
                    workspace[env, ws_J_n_idx + c * NV + i] = J_row[i]
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, ws_MinvJn_idx + c * NV + i] = mi_j_sum
                    k += J_row[i] * mi_j_sum
                    v_n += J_row[i] * workspace[env, qvel_idx + i]
                if k < Scalar[DTYPE](1e-10):
                    k = Scalar[DTYPE](1e-10)
                workspace[env, ws_K_n_idx + c] = k

                # MuJoCo impedance model for RHS
                var penetration = -dist
                var x = penetration / si_width
                if x > Scalar[DTYPE](1.0):
                    x = Scalar[DTYPE](1.0)
                var imp = si_dmin + (
                    Scalar[DTYPE](3.0) * x * x - Scalar[DTYPE](2.0) * x * x * x
                ) * (si_dmax - si_dmin)
                if imp < Scalar[DTYPE](0.2):
                    imp = Scalar[DTYPE](0.2)
                var pos_correction = imp * penetration * inv_tc_dr
                if pos_correction > Scalar[DTYPE](MAX_POS_CORRECTION_VEL):
                    pos_correction = Scalar[DTYPE](MAX_POS_CORRECTION_VEL)
                var bias = -pos_correction - b_vel_coef * v_n
                workspace[env, ws_rhs_idx + c] = v_n + bias

                workspace[env, ws_lambda_n_idx + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_IMPULSE_N]
                )

        # All threads must hit this barrier
        barrier()

        # === PARALLEL DELASSUS BUILD: Each thread computes one row of A ===
        if valid_env and contact_tid < nc:
            if workspace[env, ws_c_dist_idx + contact_tid] < Scalar[DTYPE](0):
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    var a_val: workspace.element_type = 0
                    for i in range(NV):
                        a_val += (
                            workspace[env, ws_J_n_idx + contact_tid * NV + i]
                            * workspace[env, ws_MinvJn_idx + c2 * NV + i]
                        )
                    workspace[
                        env, ws_A_idx + contact_tid * MAX_CONTACTS + c2
                    ] = a_val

        barrier()

        # === SEQUENTIAL: Thread 0 handles warm-start, CG iterations, limits, friction ===
        if not valid_env or contact_tid != 0:
            return

        # Apply warm start impulses to velocity
        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0):
                for i in range(NV):
                    workspace[env, qvel_idx + i] += (
                        workspace[env, ws_MinvJn_idx + c * NV + i]
                        * workspace[env, ws_lambda_n_idx + c]
                    )

        # Phase 2: Projected CG for normal constraints
        # Compute initial residual: r = -rhs - A*lambda
        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            var ax: workspace.element_type = 0
            for c2 in range(nc):
                if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                    continue
                ax += (
                    workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                    * workspace[env, ws_lambda_n_idx + c2]
                )
            workspace[env, ws_r_idx + c] = -workspace[env, ws_rhs_idx + c] - ax

        # Project residual
        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                workspace[env, ws_r_idx + c] = 0
                continue
            if workspace[env, ws_lambda_n_idx + c] <= Scalar[DTYPE](
                0
            ) and workspace[env, ws_r_idx + c] < Scalar[DTYPE](0):
                workspace[env, ws_r_idx + c] = 0

        for c in range(nc):
            workspace[env, ws_p_idx + c] = workspace[env, ws_r_idx + c]

        var rr: workspace.element_type = 0
        for c in range(nc):
            rr += workspace[env, ws_r_idx + c] * workspace[env, ws_r_idx + c]

        # CG iterations
        for _ in range(CG_ITERATIONS):
            if rr < Scalar[DTYPE](CG_TOLERANCE):
                break

            # Compute A*p using precomputed Delassus matrix
            for c in range(nc):
                workspace[env, ws_Ap_idx + c] = 0
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    workspace[env, ws_Ap_idx + c] = (
                        workspace[env, ws_Ap_idx + c]
                        + workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                        * workspace[env, ws_p_idx + c2]
                    )

            var pAp: workspace.element_type = 0
            for c in range(nc):
                pAp += (
                    workspace[env, ws_p_idx + c] * workspace[env, ws_Ap_idx + c]
                )
            if pAp < Scalar[DTYPE](1e-14):
                break

            var alpha = rr / pAp

            var projected = False
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                workspace[env, ws_lambda_n_idx + c] = (
                    workspace[env, ws_lambda_n_idx + c]
                    + alpha * workspace[env, ws_p_idx + c]
                )
                if workspace[env, ws_lambda_n_idx + c] < Scalar[DTYPE](0):
                    workspace[env, ws_lambda_n_idx + c] = 0
                    projected = True

            # Full residual recompute: r = -rhs - A*lambda
            # Always recompute to avoid Float32 drift from cheap r -= alpha*Ap
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    workspace[env, ws_r_idx + c] = 0
                    continue
                var ax: workspace.element_type = 0
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    ax += (
                        workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                        * workspace[env, ws_lambda_n_idx + c2]
                    )
                workspace[env, ws_r_idx + c] = (
                    -workspace[env, ws_rhs_idx + c] - ax
                )

            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] <= Scalar[DTYPE](
                    0
                ) and workspace[env, ws_r_idx + c] < Scalar[DTYPE](0):
                    workspace[env, ws_r_idx + c] = 0

            var rr_new: workspace.element_type = 0
            for c in range(nc):
                rr_new += (
                    workspace[env, ws_r_idx + c] * workspace[env, ws_r_idx + c]
                )

            if projected or rr < Scalar[DTYPE](1e-14):
                for c in range(nc):
                    workspace[env, ws_p_idx + c] = workspace[env, ws_r_idx + c]
            else:
                var beta = rr_new / rr
                for c in range(nc):
                    workspace[env, ws_p_idx + c] = (
                        workspace[env, ws_r_idx + c]
                        + beta * workspace[env, ws_p_idx + c]
                    )

            rr = rr_new

        # Remove warm-start and apply final solved impulses using precomputed MinvJn
        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            var c_off = contacts_off + c * CONTACT_SIZE
            var warm = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    workspace[env, qvel_idx + i] -= rebind[Scalar[DTYPE]](
                        workspace[env, ws_MinvJn_idx + c * NV + i] * warm
                    )

        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0):
                for i in range(NV):
                    workspace[env, qvel_idx + i] += rebind[Scalar[DTYPE]](
                        workspace[env, ws_MinvJn_idx + c * NV + i]
                        * workspace[env, ws_lambda_n_idx + c]
                    )

        # Phase 2b: Joint limit constraints (PGS) with impedance precompute + early exit
        comptime MAX_LIMITS = _max_one[2 * NJOINT]()
        var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
        var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var limit_dist_arr = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var K_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
        var lambda_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        for i in range(MAX_LIMITS):
            limit_dof[i] = 0
            limit_sign[i] = Scalar[DTYPE](0)
            limit_dist_arr[i] = Scalar[DTYPE](0)
            K_limit[i] = Scalar[DTYPE](1)
            lambda_limit[i] = Scalar[DTYPE](0)

        var num_limits = 0
        var qpos_off_lim = 0
        for j in range(NJOINT):
            var j_off = model_joint_offset[NBODY](j)
            var jtype = Int(
                rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_TYPE])
            )
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(
                rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_DOF_ADR])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_QPOS_ADR])
            )
            var rmin = rebind[Scalar[DTYPE]](
                model[0, j_off + JOINT_IDX_RANGE_MIN]
            )
            var rmax = rebind[Scalar[DTYPE]](
                model[0, j_off + JOINT_IDX_RANGE_MAX]
            )
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            var pos = rebind[Scalar[DTYPE]](state[env, qpos_off_lim + qpos_adr])
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](1)
                limit_dist_arr[num_limits] = dist_lo
                K_limit[num_limits] = rebind[Scalar[DTYPE]](
                    workspace[env, M_inv_idx + dof * NV + dof]
                )
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](-1)
                limit_dist_arr[num_limits] = dist_hi
                K_limit[num_limits] = rebind[Scalar[DTYPE]](
                    workspace[env, M_inv_idx + dof * NV + dof]
                )
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1

        if num_limits > 0:
            var lr_tc = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_0]
            )
            var lr_dr = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_1]
            )
            var li_dmin = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_0]
            )
            var li_dmax = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_1]
            )
            var li_width = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_2]
            )
            if li_width < Scalar[DTYPE](1e-6):
                li_width = Scalar[DTYPE](1e-6)
            if li_dmax < Scalar[DTYPE](1e-4):
                li_dmax = Scalar[DTYPE](1e-4)
            var l_inv_tc_dr = Scalar[DTYPE](1.0) / (lr_tc * lr_dr)
            var l_b_vel_coef = (
                Scalar[DTYPE](2.0) * lr_dr * dt / (li_dmax * lr_tc)
            )
            var l_vel_factor = Scalar[DTYPE](1.0) - l_b_vel_coef

            # Precompute impedance and MinvJ for limits
            var lim_pos_bias = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            var lim_inv_K_imp = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            comptime MINVJ_LIM_SIZE = _max_one[2 * NJOINT * NV]()
            var lim_MinvJ = InlineArray[Scalar[DTYPE], MINVJ_LIM_SIZE](
                uninitialized=True
            )
            for l in range(num_limits):
                var penetration = -limit_dist_arr[l]
                if penetration < Scalar[DTYPE](0):
                    penetration = Scalar[DTYPE](0)
                var x_lim = penetration / li_width
                if x_lim > Scalar[DTYPE](1.0):
                    x_lim = Scalar[DTYPE](1.0)
                var imp_lim = li_dmin + (
                    Scalar[DTYPE](3.0) * x_lim * x_lim
                    - Scalar[DTYPE](2.0) * x_lim * x_lim * x_lim
                ) * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](0.2):
                    imp_lim = Scalar[DTYPE](0.2)
                var lim_pos_corr = imp_lim * penetration * l_inv_tc_dr
                if lim_pos_corr > Scalar[DTYPE](MAX_POS_CORRECTION_VEL):
                    lim_pos_corr = Scalar[DTYPE](MAX_POS_CORRECTION_VEL)
                lim_pos_bias[l] = lim_pos_corr
                lim_inv_K_imp[l] = imp_lim / K_limit[l]
                var ldof = limit_dof[l]
                var lsign = limit_sign[l]
                for i in range(NV):
                    lim_MinvJ[l * NV + i] = (
                        rebind[Scalar[DTYPE]](
                            workspace[env, M_inv_idx + i * NV + ldof]
                        )
                        * lsign
                    )

            for _ in range(CG_ITERATIONS):
                var max_lim_delta: Scalar[DTYPE] = 0
                for l in range(num_limits):
                    var v_limit = (
                        limit_sign[l] * workspace[env, qvel_idx + limit_dof[l]]
                    )
                    var delta_l = (
                        -(v_limit * l_vel_factor - lim_pos_bias[l])
                        * lim_inv_K_imp[l]
                    )
                    var old_lam = lambda_limit[l]
                    lambda_limit[l] = lambda_limit[l] + rebind[Scalar[DTYPE]](
                        delta_l
                    )
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual_l = lambda_limit[l] - old_lam
                    var abs_l = abs(actual_l)
                    if abs_l > max_lim_delta:
                        max_lim_delta = abs_l
                    for i in range(NV):
                        workspace[env, qvel_idx + i] += (
                            lim_MinvJ[l * NV + i] * actual_l
                        )
                if max_lim_delta < Scalar[DTYPE](1e-4):
                    break

        # Phase 3: Friction via PGS
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
            FRICTION_WS_OFFSET = 15 * MC + 2 * MC * NV + MC * MC,
        ](
            env,
            state,
            model,
            workspace,
            nc,
            friction_coef,
            contacts_off,
        )
