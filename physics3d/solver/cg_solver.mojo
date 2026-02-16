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
from ..constraints import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_LIMIT,
)
from .qcqp import qcqp2, qcqp3, qcqp5, mj_qcqp2, mj_qcqp3, mj_qcqp5, cost_change
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_qacc_constrained_offset,
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

# Import shared friction solver (GPU only now — CPU friction uses ConstraintData)
from .friction_solver import _solve_friction_pgs_gpu

# CG solver parameters
comptime CG_ITERATIONS: Int = 30
comptime CG_TOLERANCE: Float64 = 1e-8
# Minimum K for friction tangent rows — below this, direction is degenerate
comptime FRICTION_K_MIN: Float64 = 1e-6


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
        """CG solver workspace: 83*MC + 12*MC*NV + MC*MC floats.

        Layout (offsets relative to solver workspace start):
          [0..13*MC+2*MC*NV)                            Common normal block
          [13*MC+2*MC*NV..14*MC+2*MC*NV)                rhs
          [14*MC+2*MC*NV..14*MC+2*MC*NV+MC*MC)          A (Delassus matrix)
          [14*MC+2*MC*NV+MC*MC..15*MC+2*MC*NV+MC*MC)    r (residual)
          [15*MC+2*MC*NV+MC*MC..16*MC+2*MC*NV+MC*MC)    p (search direction)
          [16*MC+2*MC*NV+MC*MC..17*MC+2*MC*NV+MC*MC)    Ap (A*p product)
          [17*MC+2*MC*NV+MC*MC..83*MC+12*MC*NV+MC*MC)   Friction (66*MC + 10*MC*NV)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 83 * MC + 12 * MC * NV + MC * MC

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
        """Solve constraints using Projected CG on CPU (acceleration-level).

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
        var num_equality = constraints.num_equality
        var friction_start = num_normals
        var limits_start = num_normals + num_friction
        var equality_start = limits_start + num_limits

        comptime MR = _max_one[MAX_ROWS]()
        comptime A_SIZE = _max_one[MAX_ROWS * MAX_ROWS]()

        # RHS for CG: rhs[r] = J[r] · qacc_unconstrained + bias[r]
        # IMPORTANT: Compute ALL rhs from unconstrained qacc BEFORE warm-start
        var rhs = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        for i in range(MR):
            rhs[i] = Scalar[DTYPE](0)

        for r in range(num_normals):
            var a_n: Scalar[DTYPE] = 0
            for i in range(NV):
                a_n += constraints.J[r * NV + i] * qacc[i]
            rhs[r] = a_n + constraints.rows[r].bias

        # Apply warm-start (after rhs is fully computed)
        for r in range(num_normals):
            if constraints.rows[r].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # Build Delassus matrix A[c1,c2] = J[c1] . MinvJT[c2]
        # Then add regularizer R to diagonal: AR[c,c] = K + R where R = K/imp - K
        var A = InlineArray[Scalar[DTYPE], A_SIZE](uninitialized=True)
        for i in range(A_SIZE):
            A[i] = Scalar[DTYPE](0)
        for c1 in range(num_normals):
            for c2 in range(num_normals):
                var a_val: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_val += (
                        constraints.J[c1 * NV + i]
                        * constraints.MinvJT[c2 * NV + i]
                    )
                A[c1 * num_normals + c2] = a_val

        # Add MuJoCo regularizer R to diagonal: AR[c,c] = K/imp
        for c in range(num_normals):
            var R = (
                Scalar[DTYPE](1.0) / constraints.rows[c].inv_K_imp
                - constraints.rows[c].K
            )
            A[c * num_normals + c] += R

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
            if constraints.rows[c].lambda_val <= Scalar[DTYPE](0) and r_vec[
                c
            ] < Scalar[DTYPE](0):
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
                constraints.rows[c].lambda_val = (
                    constraints.rows[c].lambda_val + alpha * p[c]
                )
                if constraints.rows[c].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[c].lambda_val = Scalar[DTYPE](0)
                    projected = True

            # Full residual recompute
            for c in range(num_normals):
                var ax: Scalar[DTYPE] = 0
                for c2 in range(num_normals):
                    ax += (
                        A[c * num_normals + c2]
                        * constraints.rows[c2].lambda_val
                    )
                r_vec[c] = -rhs[c] - ax

            for c in range(num_normals):
                if constraints.rows[c].lambda_val <= Scalar[DTYPE](0) and r_vec[
                    c
                ] < Scalar[DTYPE](0):
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

        # Apply solved forces: remove warm-start, apply final
        for c in range(num_normals):
            var warm = data.contacts[
                constraints.rows[c].source_contact_idx
            ].force_n
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] -= constraints.MinvJT[c * NV + i] * warm

        for c in range(num_normals):
            if constraints.rows[c].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[c * NV + i]
                        * constraints.rows[c].lambda_val
                    )

        # =====================================================================
        # Phase 3: Coupled PGS (normals + friction + limits together)
        # MuJoCo-style: iterate over ALL constraints in each pass.
        # =====================================================================
        if num_friction == 0 and num_limits == 0 and num_equality == 0:
            return

        # Apply friction warm-start before coupled iterations
        for r_off in range(num_friction):
            var r = friction_start + r_off
            if constraints.rows[r].K < Scalar[DTYPE](FRICTION_K_MIN):
                constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                continue
            if constraints.rows[r].lambda_val != Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # Coupled PGS iterations
        comptime MINVAL: Float64 = 1e-10
        for _ in range(CG_ITERATIONS):
            @parameter
            if CONE_TYPE == ConeType.PYRAMIDAL:
                # === PYRAMIDAL CONE: Independent PGS on normals + edge constraints ===
                # Normal constraints: simple λ≥0 PGS
                for normal_r in range(num_normals):
                    var a_n: Scalar[DTYPE] = 0
                    for i in range(NV):
                        a_n += constraints.J[normal_r * NV + i] * qacc[i]
                    var R_n = Scalar[DTYPE](1.0) / constraints.rows[normal_r].inv_K_imp - constraints.rows[normal_r].K
                    var residual_n = (
                        a_n
                        + constraints.rows[normal_r].bias
                        + R_n * constraints.rows[normal_r].lambda_val
                    )
                    var delta_n = -residual_n * constraints.rows[normal_r].inv_K_imp
                    var old_lambda_n = constraints.rows[normal_r].lambda_val
                    constraints.rows[normal_r].lambda_val = (
                        constraints.rows[normal_r].lambda_val + delta_n
                    )
                    if constraints.rows[normal_r].lambda_val < Scalar[DTYPE](0):
                        constraints.rows[normal_r].lambda_val = Scalar[DTYPE](0)
                    var actual_n = constraints.rows[normal_r].lambda_val - old_lambda_n
                    if actual_n != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc[i] += constraints.MinvJT[normal_r * NV + i] * actual_n

                # Pyramid edge constraints: each edge is λ≥0
                for r_off in range(num_friction):
                    var r = friction_start + r_off
                    var a_edge: Scalar[DTYPE] = 0
                    for i in range(NV):
                        a_edge += constraints.J[r * NV + i] * qacc[i]
                    var R_edge = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                    var residual_edge = (
                        a_edge
                        + constraints.rows[r].bias
                        + R_edge * constraints.rows[r].lambda_val
                    )
                    var delta_edge = -residual_edge * constraints.rows[r].inv_K_imp
                    var old_lambda_edge = constraints.rows[r].lambda_val
                    constraints.rows[r].lambda_val = (
                        constraints.rows[r].lambda_val + delta_edge
                    )
                    if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                        constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                    var actual_edge = constraints.rows[r].lambda_val - old_lambda_edge
                    if actual_edge != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc[i] += constraints.MinvJT[r * NV + i] * actual_edge
            else:
                # === ELLIPTIC CONE: MuJoCo-style block QCQP updates ===
                var fric_idx = 0
                for normal_r in range(num_normals):
                    var group_size = 0
                    while fric_idx + group_size < num_friction:
                        if constraints.rows[friction_start + fric_idx + group_size].friction_parent != normal_r:
                            break
                        group_size += 1

                    var dim = 1 + group_size
                    var row_idx = InlineArray[Int, 6](fill=0)
                    row_idx[0] = normal_r
                    for g in range(group_size):
                        row_idx[1 + g] = friction_start + fric_idx + g

                    # Build block AR matrix
                    var AR = InlineArray[Scalar[DTYPE], 36](fill=Scalar[DTYPE](0))
                    for bi in range(dim):
                        for bj in range(dim):
                            var a_val: Scalar[DTYPE] = 0
                            for k in range(NV):
                                a_val += constraints.J[row_idx[bi] * NV + k] * constraints.MinvJT[row_idx[bj] * NV + k]
                            if bi == bj:
                                a_val += Scalar[DTYPE](1.0) / constraints.rows[row_idx[bi]].inv_K_imp - constraints.rows[row_idx[bi]].K
                            AR[bi * dim + bj] = a_val

                    var block_res = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
                    for bj in range(dim):
                        var a: Scalar[DTYPE] = 0
                        for k in range(NV):
                            a += constraints.J[row_idx[bj] * NV + k] * qacc[k]
                        var R_row = Scalar[DTYPE](1.0) / constraints.rows[row_idx[bj]].inv_K_imp - constraints.rows[row_idx[bj]].K
                        block_res[bj] = a + constraints.rows[row_idx[bj]].bias + R_row * constraints.rows[row_idx[bj]].lambda_val

                    var oldforce = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
                    oldforce[0] = constraints.rows[normal_r].lambda_val
                    for g in range(group_size):
                        oldforce[1 + g] = constraints.rows[row_idx[1 + g]].lambda_val

                    var ARinv0: Scalar[DTYPE] = 0
                    if AR[0] > Scalar[DTYPE](MINVAL):
                        ARinv0 = Scalar[DTYPE](1.0) / AR[0]

                    if dim == 1:
                        constraints.rows[normal_r].lambda_val -= block_res[0] * ARinv0
                        if constraints.rows[normal_r].lambda_val < Scalar[DTYPE](0):
                            constraints.rows[normal_r].lambda_val = Scalar[DTYPE](0)
                    else:
                        if constraints.rows[normal_r].lambda_val < Scalar[DTYPE](MINVAL):
                            constraints.rows[normal_r].lambda_val -= block_res[0] * ARinv0
                            if constraints.rows[normal_r].lambda_val < Scalar[DTYPE](0):
                                constraints.rows[normal_r].lambda_val = Scalar[DTYPE](0)
                            for g in range(group_size):
                                constraints.rows[row_idx[1 + g]].lambda_val = Scalar[DTYPE](0)
                        else:
                            var v = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
                            v[0] = constraints.rows[normal_r].lambda_val
                            for g in range(group_size):
                                v[1 + g] = constraints.rows[row_idx[1 + g]].lambda_val
                            var denom: Scalar[DTYPE] = 0
                            for bi in range(dim):
                                for bj in range(dim):
                                    denom += v[bi] * AR[bi * dim + bj] * v[bj]
                            if denom >= Scalar[DTYPE](MINVAL):
                                var vdotr: Scalar[DTYPE] = 0
                                for bi in range(dim):
                                    vdotr += v[bi] * block_res[bi]
                                var x = -vdotr / denom
                                if constraints.rows[normal_r].lambda_val + x * v[0] < Scalar[DTYPE](0):
                                    x = -constraints.rows[normal_r].lambda_val / v[0]
                                constraints.rows[normal_r].lambda_val += x * v[0]
                                for g in range(group_size):
                                    constraints.rows[row_idx[1 + g]].lambda_val += x * v[1 + g]

                        if constraints.rows[normal_r].lambda_val >= Scalar[DTYPE](MINVAL) and group_size > 0:
                            var Ac = InlineArray[Scalar[DTYPE], 25](fill=Scalar[DTYPE](0))
                            var bc = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
                            for j in range(group_size):
                                for j2 in range(group_size):
                                    Ac[j * group_size + j2] = AR[(1 + j) * dim + (1 + j2)]
                                bc[j] = block_res[1 + j]
                                for j2 in range(group_size):
                                    bc[j] -= Ac[j * group_size + j2] * oldforce[1 + j2]
                                bc[j] += AR[(1 + j) * dim + 0] * (constraints.rows[normal_r].lambda_val - oldforce[0])

                            var mu_arr = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
                            for g in range(group_size):
                                mu_arr[g] = constraints.rows[row_idx[1 + g]].friction_coef

                            var fn_val = constraints.rows[normal_r].lambda_val
                            var flg_active = False

                            if group_size == 2:
                                var A2 = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
                                var b2 = InlineArray[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))
                                var d2 = InlineArray[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))
                                for ii in range(2):
                                    b2[ii] = bc[ii]
                                    d2[ii] = mu_arr[ii]
                                    for jj in range(2):
                                        A2[ii * 2 + jj] = Ac[ii * group_size + jj]
                                var r0: Scalar[DTYPE] = 0
                                var r1: Scalar[DTYPE] = 0
                                flg_active = mj_qcqp2[DTYPE](r0, r1, A2, b2, d2, fn_val)
                                constraints.rows[row_idx[1]].lambda_val = r0
                                constraints.rows[row_idx[2]].lambda_val = r1
                            elif group_size == 3:
                                var A3 = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
                                var b3 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
                                var d3 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
                                for ii in range(3):
                                    b3[ii] = bc[ii]
                                    d3[ii] = mu_arr[ii]
                                    for jj in range(3):
                                        A3[ii * 3 + jj] = Ac[ii * group_size + jj]
                                var r0: Scalar[DTYPE] = 0
                                var r1: Scalar[DTYPE] = 0
                                var r2: Scalar[DTYPE] = 0
                                flg_active = mj_qcqp3[DTYPE](r0, r1, r2, A3, b3, d3, fn_val)
                                constraints.rows[row_idx[1]].lambda_val = r0
                                constraints.rows[row_idx[2]].lambda_val = r1
                                constraints.rows[row_idx[3]].lambda_val = r2
                            elif group_size == 5:
                                var A5 = InlineArray[Scalar[DTYPE], 25](fill=Scalar[DTYPE](0))
                                var b5 = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
                                var d5 = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
                                for ii in range(5):
                                    b5[ii] = bc[ii]
                                    d5[ii] = mu_arr[ii]
                                    for jj in range(5):
                                        A5[ii * 5 + jj] = Ac[ii * group_size + jj]
                                var res5 = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
                                flg_active = mj_qcqp5[DTYPE](res5, A5, b5, d5, fn_val)
                                for g in range(5):
                                    constraints.rows[row_idx[1 + g]].lambda_val = res5[g]

                            if flg_active:
                                var s: Scalar[DTYPE] = 0
                                for g in range(group_size):
                                    var fv = constraints.rows[row_idx[1 + g]].lambda_val
                                    var mu_g = mu_arr[g]
                                    if mu_g > Scalar[DTYPE](MINVAL):
                                        s += fv * fv / (mu_g * mu_g)
                                if s > Scalar[DTYPE](MINVAL):
                                    var scale = sqrt(fn_val * fn_val / s)
                                    for g in range(group_size):
                                        constraints.rows[row_idx[1 + g]].lambda_val *= scale

                    var newforce = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
                    newforce[0] = constraints.rows[normal_r].lambda_val
                    for g in range(group_size):
                        newforce[1 + g] = constraints.rows[row_idx[1 + g]].lambda_val
                    var change = cost_change[DTYPE, 6, 36](newforce, oldforce, AR, block_res, dim)
                    if change > Scalar[DTYPE](MINVAL):
                        constraints.rows[normal_r].lambda_val = oldforce[0]
                        for g in range(group_size):
                            constraints.rows[row_idx[1 + g]].lambda_val = oldforce[1 + g]

                    for bi in range(dim):
                        var actual = constraints.rows[row_idx[bi]].lambda_val - oldforce[bi]
                        if actual != Scalar[DTYPE](0):
                            for k in range(NV):
                                qacc[k] += constraints.MinvJT[row_idx[bi] * NV + k] * actual

                    fric_idx += group_size

            # --- Joint limit constraints ---
            for r_off in range(num_limits):
                var r = limits_start + r_off
                var dof = constraints.rows[r].source_dof
                var sign = constraints.rows[r].limit_sign
                var a_limit = sign * qacc[dof]
                var R_lim = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                var residual = (
                    a_limit
                    + constraints.rows[r].bias
                    + R_lim * constraints.rows[r].lambda_val
                )
                var delta = -residual * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = (
                    constraints.rows[r].lambda_val + delta
                )
                if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                var actual = constraints.rows[r].lambda_val - old_lambda
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual

            # --- Equality constraints (bilateral, NO clamping) ---
            for r_off in range(num_equality):
                var r = equality_start + r_off
                var a_eq: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_eq += constraints.J[r * NV + i] * qacc[i]
                var R_eq = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                var residual = (
                    a_eq
                    + constraints.rows[r].bias
                    + R_eq * constraints.rows[r].lambda_val
                )
                var delta = -residual * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = (
                    constraints.rows[r].lambda_val + delta
                )
                var actual = constraints.rows[r].lambda_val - old_lambda
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual

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
        """Solve contact constraints using Projected CG on GPU.

        Uses thread_x for environment index, thread_y for contact index.
        Phase 1 and Delassus build are parallelized across contacts.
        CG iterations are sequential on thread_y==0.
        All threads must hit all barriers (no early returns between them).
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

        # CG-specific offsets (after common normal block)
        comptime CG_START = solver_ws_idx + 13 * MC + 2 * MC * NV
        comptime ws_rhs_idx = CG_START + 0 * MC
        comptime ws_A_idx = CG_START + 1 * MC
        comptime ws_r_idx = CG_START + MC + MC * MC
        comptime ws_p_idx = CG_START + 2 * MC + MC * MC
        comptime ws_Ap_idx = CG_START + 3 * MC + MC * MC

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
            # Init CG-specific
            workspace[env, ws_rhs_idx + contact_tid] = 0
            workspace[env, ws_r_idx + contact_tid] = 0
            workspace[env, ws_p_idx + contact_tid] = 0
            workspace[env, ws_Ap_idx + contact_tid] = 0
            for c2 in range(MC):
                workspace[env, ws_A_idx + contact_tid * MAX_CONTACTS + c2] = 0

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

        # === PARALLEL DELASSUS BUILD: Each thread computes one row of A ===
        # Add regularizer R to diagonal: AR[c,c] = K + R = K/imp
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
                # Add MuJoCo regularizer R to diagonal
                comptime ws_inv_K_imp_cg = solver_ws_idx + 12 * MC
                var R_c = (
                    Scalar[DTYPE](1.0)
                    / workspace[env, ws_inv_K_imp_cg + contact_tid]
                    - workspace[env, ws_K_n_idx + contact_tid]
                )
                workspace[
                    env, ws_A_idx + contact_tid * MAX_CONTACTS + contact_tid
                ] += R_c

        barrier()

        # === SEQUENTIAL: Thread 0 handles warm-start, CG iterations, limits, friction ===
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

            # Full residual recompute
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

        # Apply solved normals (remove warm-start, apply final)
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

        # Joint limits
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
            CG_ITERATIONS,
        ](env, dt, state, model, workspace)

        # Equality constraints
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
            CG_ITERATIONS,
        ](env, state, model, workspace)

        # Friction via PGS
        comptime FRICTION_WS_OFFSET = 17 * MC + 2 * MC * NV + MC * MC
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
        ](env, state, model, workspace, nc, friction_coef, contacts_off)
