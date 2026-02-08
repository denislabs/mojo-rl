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

from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..traits.solver import ConstraintSolver
from ..dynamics.jacobian import (
    compute_contact_jacobian_row,
    compute_contact_jacobian_row_gpu,
)
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    model_joint_offset,
    ws_m_inv_offset,
    ws_solver_offset,
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

# CG solver parameters
comptime CG_ITERATIONS: Int = 30
comptime CG_TOLERANCE: Float64 = 1e-8
# Friction uses PGS iterations
comptime FRICTION_PGS_ITERATIONS: Int = 30


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
        """CG solver workspace: 25*MC + MC*NV floats.

        Layout (offsets relative to solver workspace start):
          [0..11*MC)                    Common contact block
          [11*MC..12*MC)                rhs
          [12*MC..12*MC+MC*NV)          J_n (normal Jacobian)
          [12*MC+MC*NV..13*MC+MC*NV)    r (residual)
          [13*MC+MC*NV..14*MC+MC*NV)    p (search direction)
          [14*MC+MC*NV..15*MC+MC*NV)    Ap (A*p product)
          [15*MC+MC*NV..25*MC+MC*NV)    Friction block (10 arrays)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 25 * MC + MC * NV

    @staticmethod
    fn solve[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        V_SIZE: Int,
        M_SIZE: Int,
        CDOF_SIZE: Int,
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve contact constraints using Projected CG on CPU."""
        var num_contacts = data.num_contacts

        # Detect joint limits
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
        for j in range(model.num_joints):
            var joint = model.joints[j]
            if joint.jnt_type != JNT_HINGE and joint.jnt_type != JNT_SLIDE:
                continue
            var dof = joint.dof_adr
            var pos = data.qpos[joint.qpos_adr]
            var rmin = joint.range_min
            var rmax = joint.range_max
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](1)
                limit_dist_arr[num_limits] = dist_lo
                K_limit[num_limits] = M_inv[dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](-1)
                limit_dist_arr[num_limits] = dist_hi
                K_limit[num_limits] = M_inv[dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1

        if num_contacts == 0 and num_limits == 0:
            return

        var nc = num_contacts
        if nc > MAX_CONTACTS:
            nc = MAX_CONTACTS

        comptime MC = _max_one[MAX_CONTACTS]()
        comptime JN_SIZE = _max_one[MAX_CONTACTS * NV]()

        # Normal Jacobian rows: J_n[c * NV + i]
        var J_n = InlineArray[Scalar[DTYPE], JN_SIZE](uninitialized=True)
        for i in range(JN_SIZE):
            J_n[i] = Scalar[DTYPE](0)

        # Effective mass (diagonal of A) for preconditioning
        var K_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            K_n[i] = Scalar[DTYPE](0)

        # Normal impulse accumulators
        var lambda_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            lambda_n[i] = Scalar[DTYPE](0)

        # Contact distances (for impedance model)
        var contact_dist = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            contact_dist[i] = Scalar[DTYPE](0)

        # Contact body indices
        var contact_body = InlineArray[Int, MC](uninitialized=True)
        var contact_body_b = InlineArray[Int, MC](uninitialized=True)
        for i in range(MC):
            contact_body[i] = 0
            contact_body_b[i] = -1

        # Right-hand side: b[c] = J_n[c] . qvel_pred + baumgarte_bias[c]
        var rhs = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            rhs[i] = Scalar[DTYPE](0)

        # Phase 1: Precompute contact data
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # MuJoCo solref/solimp impedance model for contacts
        var sr_tc = model.solref_contact[0]
        var sr_dr = model.solref_contact[1]
        var si_dmin = model.solimp_contact[0]
        var si_dmax = model.solimp_contact[1]
        var si_width = model.solimp_contact[2]
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)
        var inv_tc_dr = Scalar[DTYPE](1.0) / (sr_tc * sr_dr)
        var b_vel_coef = Scalar[DTYPE](2.0) * sr_dr * dt / (si_dmax * sr_tc)

        for c in range(nc):
            var contact = data.contacts[c]

            if contact.dist >= Scalar[DTYPE](0):
                K_n[c] = Scalar[DTYPE](1)
                continue

            contact_dist[c] = contact.dist
            contact_body[c] = contact.body_a
            contact_body_b[c] = contact.body_b

            # Compute normal Jacobian row
            compute_contact_jacobian_row[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
            ](
                model,
                data,
                cdof,
                contact.body_a,
                contact.body_b,
                contact.pos_x,
                contact.pos_y,
                contact.pos_z,
                contact.normal_x,
                contact.normal_y,
                contact.normal_z,
                J_row,
            )

            # Store and compute effective mass (diagonal of Delassus matrix)
            var k: Scalar[DTYPE] = 0
            var v_n: Scalar[DTYPE] = 0
            for i in range(NV):
                J_n[c * NV + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * qvel[i]

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            K_n[c] = k

            # MuJoCo impedance model for RHS
            var penetration = -contact.dist
            if penetration > Scalar[DTYPE](0.05):
                penetration = Scalar[DTYPE](0.05)
            var x = penetration / si_width
            if x > Scalar[DTYPE](1.0):
                x = Scalar[DTYPE](1.0)
            var imp = si_dmin + (
                Scalar[DTYPE](3.0) * x * x - Scalar[DTYPE](2.0) * x * x * x
            ) * (si_dmax - si_dmin)
            if imp < Scalar[DTYPE](0.2):
                imp = Scalar[DTYPE](0.2)
            var bias = -imp * penetration * inv_tc_dr - b_vel_coef * v_n

            rhs[c] = v_n + bias

            # Warm start from stored impulses
            lambda_n[c] = contact.impulse_n

        # Apply warm start impulses to velocity
        for c in range(nc):
            if contact_dist[c] >= Scalar[DTYPE](0):
                continue
            if lambda_n[c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c * NV + j_idx]
                    qvel[i] += mi_j_sum * lambda_n[c]

        # Phase 2: Projected CG for normal constraints
        # Solve: A * lambda = -rhs, subject to lambda >= 0
        # where A[c1,c2] = J[c1] @ M_inv @ J[c2]^T

        # CG vectors
        var r = InlineArray[Scalar[DTYPE], MC](uninitialized=True)  # residual
        var p = InlineArray[Scalar[DTYPE], MC](
            uninitialized=True
        )  # search direction
        var Ap = InlineArray[Scalar[DTYPE], MC](uninitialized=True)  # A * p

        for i in range(MC):
            r[i] = Scalar[DTYPE](0)
            p[i] = Scalar[DTYPE](0)
            Ap[i] = Scalar[DTYPE](0)

        # Compute initial residual: r = -rhs - A*lambda
        # (gradient of 0.5*lambda^T*A*lambda + rhs^T*lambda, negated)
        # First compute A*lambda
        for c in range(nc):
            if contact_dist[c] >= Scalar[DTYPE](0):
                continue
            var ax: Scalar[DTYPE] = 0
            for c2 in range(nc):
                if contact_dist[c2] >= Scalar[DTYPE](0):
                    continue
                # A[c,c2] = J[c] @ M_inv @ J[c2]^T
                var a_cc2: Scalar[DTYPE] = 0
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c2 * NV + j_idx]
                    a_cc2 += J_n[c * NV + i] * mi_j_sum
                ax += a_cc2 * lambda_n[c2]
            r[c] = -rhs[c] - ax

        # Project residual: zero out where lambda=0 and r<0 (active constraint)
        for c in range(nc):
            if contact_dist[c] >= Scalar[DTYPE](0):
                r[c] = Scalar[DTYPE](0)
                continue
            if lambda_n[c] <= Scalar[DTYPE](0) and r[c] < Scalar[DTYPE](0):
                r[c] = Scalar[DTYPE](0)

        # Initialize search direction
        for c in range(nc):
            p[c] = r[c]

        var rr: Scalar[DTYPE] = 0
        for c in range(nc):
            rr += r[c] * r[c]

        # CG iterations
        for _ in range(CG_ITERATIONS):
            if rr < Scalar[DTYPE](CG_TOLERANCE):
                break

            # Compute A*p
            for c in range(nc):
                Ap[c] = Scalar[DTYPE](0)
                if contact_dist[c] >= Scalar[DTYPE](0):
                    continue
                for c2 in range(nc):
                    if contact_dist[c2] >= Scalar[DTYPE](0):
                        continue
                    var a_cc2: Scalar[DTYPE] = 0
                    for i in range(NV):
                        var mi_j_sum: Scalar[DTYPE] = 0
                        for j_idx in range(NV):
                            mi_j_sum += (
                                M_inv[i * NV + j_idx] * J_n[c2 * NV + j_idx]
                            )
                        a_cc2 += J_n[c * NV + i] * mi_j_sum
                    Ap[c] += a_cc2 * p[c2]

            # Step size: alpha = r^T*r / (p^T*A*p)
            var pAp: Scalar[DTYPE] = 0
            for c in range(nc):
                pAp += p[c] * Ap[c]

            if pAp < Scalar[DTYPE](1e-14):
                break

            var alpha = rr / pAp

            # Update lambda: lambda += alpha * p, then project >= 0
            var projected = False
            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    continue
                lambda_n[c] = lambda_n[c] + alpha * p[c]
                if lambda_n[c] < Scalar[DTYPE](0):
                    lambda_n[c] = Scalar[DTYPE](0)
                    projected = True

            # Recompute residual after projection (projection breaks conjugacy)
            # r = -rhs - A*lambda
            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    r[c] = Scalar[DTYPE](0)
                    continue
                var ax: Scalar[DTYPE] = 0
                for c2 in range(nc):
                    if contact_dist[c2] >= Scalar[DTYPE](0):
                        continue
                    var a_cc2: Scalar[DTYPE] = 0
                    for i in range(NV):
                        var mi_j_sum: Scalar[DTYPE] = 0
                        for j_idx in range(NV):
                            mi_j_sum += (
                                M_inv[i * NV + j_idx] * J_n[c2 * NV + j_idx]
                            )
                        a_cc2 += J_n[c * NV + i] * mi_j_sum
                    ax += a_cc2 * lambda_n[c2]
                r[c] = -rhs[c] - ax

            # Project residual
            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    continue
                if lambda_n[c] <= Scalar[DTYPE](0) and r[c] < Scalar[DTYPE](0):
                    r[c] = Scalar[DTYPE](0)

            var rr_new: Scalar[DTYPE] = 0
            for c in range(nc):
                rr_new += r[c] * r[c]

            # CG direction update (restart if projection occurred)
            if projected or rr < Scalar[DTYPE](1e-14):
                for c in range(nc):
                    p[c] = r[c]
            else:
                var beta = rr_new / rr
                for c in range(nc):
                    p[c] = r[c] + beta * p[c]

            rr = rr_new

        # Apply solved impulses to velocity
        # First, remove warm-start (we'll apply the full solved lambda)
        for c in range(nc):
            if contact_dist[c] >= Scalar[DTYPE](0):
                continue
            # Remove warm-start contribution
            var warm = data.contacts[c].impulse_n
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c * NV + j_idx]
                    qvel[i] -= mi_j_sum * warm

        # Apply final solved impulses
        for c in range(nc):
            if contact_dist[c] >= Scalar[DTYPE](0):
                continue
            if lambda_n[c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c * NV + j_idx]
                    qvel[i] += mi_j_sum * lambda_n[c]

        # Phase 2b: Joint limit constraints (PGS)
        if num_limits > 0:
            var lr_tc = model.solref_limit[0]
            var lr_dr = model.solref_limit[1]
            var li_dmin = model.solimp_limit[0]
            var li_dmax = model.solimp_limit[1]
            var li_width = model.solimp_limit[2]
            if li_width < Scalar[DTYPE](1e-6):
                li_width = Scalar[DTYPE](1e-6)
            if li_dmax < Scalar[DTYPE](1e-4):
                li_dmax = Scalar[DTYPE](1e-4)
            var l_inv_tc_dr = Scalar[DTYPE](1.0) / (lr_tc * lr_dr)
            var l_b_vel_coef = (
                Scalar[DTYPE](2.0) * lr_dr * dt / (li_dmax * lr_tc)
            )

            for _ in range(CG_ITERATIONS):
                for l in range(num_limits):
                    var dof = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof]
                    var penetration = -limit_dist_arr[l]
                    if penetration < Scalar[DTYPE](0):
                        penetration = Scalar[DTYPE](0)
                    if penetration > Scalar[DTYPE](0.05):
                        penetration = Scalar[DTYPE](0.05)
                    var x_lim = penetration / li_width
                    if x_lim > Scalar[DTYPE](1.0):
                        x_lim = Scalar[DTYPE](1.0)
                    var imp_lim = li_dmin + (
                        Scalar[DTYPE](3.0) * x_lim * x_lim
                        - Scalar[DTYPE](2.0) * x_lim * x_lim * x_lim
                    ) * (li_dmax - li_dmin)
                    if imp_lim < Scalar[DTYPE](0.2):
                        imp_lim = Scalar[DTYPE](0.2)
                    var bias = (
                        -imp_lim * penetration * l_inv_tc_dr
                        - l_b_vel_coef * v_limit
                    )
                    var delta_l = -(v_limit + bias) / (K_limit[l] / imp_lim)
                    var old_lam = lambda_limit[l]
                    lambda_limit[l] = lambda_limit[l] + delta_l
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual = lambda_limit[l] - old_lam
                    for i in range(NV):
                        qvel[i] += M_inv[i * NV + dof] * sign * actual

        # Phase 3: Friction (Coulomb cone) - using PGS iterations
        _solve_friction_pgs_cpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
        ](
            model,
            data,
            cdof,
            M_inv,
            J_n,
            lambda_n,
            contact_dist,
            contact_body_b,
            nc,
            qvel,
        )

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
        M_SIZE: Int,
        CDOF_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
        cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve contact constraints using Projected CG on GPU (per-environment).

        All MC-sized arrays live in workspace (device memory).
        """

        var M_inv = workspace.ptr + env * WS_SIZE + ws_m_inv_offset()
        var solver_ws = workspace.ptr + env * WS_SIZE + ws_solver_offset[NV]()

        comptime MC = _max_one[MAX_CONTACTS]()

        # Common contact block (11*MC)
        var ws_lambda_n = solver_ws + 0 * MC
        var ws_K_n      = solver_ws + 1 * MC
        var ws_c_dist   = solver_ws + 2 * MC
        var ws_c_body   = solver_ws + 3 * MC
        var ws_c_body_b = solver_ws + 4 * MC
        var ws_c_px     = solver_ws + 5 * MC
        var ws_c_py     = solver_ws + 6 * MC
        var ws_c_pz     = solver_ws + 7 * MC
        var ws_c_nx     = solver_ws + 8 * MC
        var ws_c_ny     = solver_ws + 9 * MC
        var ws_c_nz     = solver_ws + 10 * MC
        # CG-specific
        var ws_rhs      = solver_ws + 11 * MC
        var ws_J_n      = solver_ws + 12 * MC                   # MC*NV floats
        var ws_r        = solver_ws + 12 * MC + MC * NV         # residual
        var ws_p        = solver_ws + 13 * MC + MC * NV         # search direction
        var ws_Ap       = solver_ws + 14 * MC + MC * NV         # A*p
        # Friction block (10*MC)
        var ws_lambda_t1 = solver_ws + 15 * MC + MC * NV
        var ws_lambda_t2 = solver_ws + 16 * MC + MC * NV
        var ws_K_t1      = solver_ws + 17 * MC + MC * NV
        var ws_K_t2      = solver_ws + 18 * MC + MC * NV
        var ws_t1x       = solver_ws + 19 * MC + MC * NV
        var ws_t1y       = solver_ws + 20 * MC + MC * NV
        var ws_t1z       = solver_ws + 21 * MC + MC * NV
        var ws_t2x       = solver_ws + 22 * MC + MC * NV
        var ws_t2y       = solver_ws + 23 * MC + MC * NV
        var ws_t2z       = solver_ws + 24 * MC + MC * NV

        # Initialize workspace
        for i in range(MC):
            (ws_lambda_n + i)[] = Scalar[DTYPE](0)
            (ws_K_n + i)[] = Scalar[DTYPE](1)
            (ws_c_dist + i)[] = Scalar[DTYPE](0)
            (ws_c_body + i)[] = Scalar[DTYPE](0)
            (ws_c_body_b + i)[] = Scalar[DTYPE](-1)
            (ws_c_px + i)[] = Scalar[DTYPE](0)
            (ws_c_py + i)[] = Scalar[DTYPE](0)
            (ws_c_pz + i)[] = Scalar[DTYPE](0)
            (ws_c_nx + i)[] = Scalar[DTYPE](0)
            (ws_c_ny + i)[] = Scalar[DTYPE](0)
            (ws_c_nz + i)[] = Scalar[DTYPE](1)
            (ws_rhs + i)[] = Scalar[DTYPE](0)
            (ws_r + i)[] = Scalar[DTYPE](0)
            (ws_p + i)[] = Scalar[DTYPE](0)
            (ws_Ap + i)[] = Scalar[DTYPE](0)
        for i in range(MC * NV):
            (ws_J_n + i)[] = Scalar[DTYPE](0)

        var contacts_off = contacts_offset[NQ, NV, NBODY]()
        var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var num_contacts = Int(
            rebind[Scalar[DTYPE]](state[env, meta_off + META_IDX_NUM_CONTACTS])
        )
        var friction_coef = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_FRICTION]
        )

        # Detect joint limits
        comptime MAX_LIMITS = _max_one[2 * NJOINT]()
        var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
        var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
        var limit_dist_arr = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
        var K_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
        var lambda_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
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
            var jtype = Int(rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_TYPE]))
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_DOF_ADR]))
            var qpos_adr = Int(rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_QPOS_ADR]))
            var rmin = rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_RANGE_MIN])
            var rmax = rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_RANGE_MAX])
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            var pos = rebind[Scalar[DTYPE]](state[env, qpos_off_lim + qpos_adr])
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](1)
                limit_dist_arr[num_limits] = dist_lo
                K_limit[num_limits] = M_inv[dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](-1)
                limit_dist_arr[num_limits] = dist_hi
                K_limit[num_limits] = M_inv[dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1

        if num_contacts == 0 and num_limits == 0:
            return

        var nc = num_contacts
        if nc > MAX_CONTACTS:
            nc = MAX_CONTACTS

        # Read solref/solimp contact from model buffer
        var sr_tc = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0])
        var sr_dr = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1])
        var si_dmin = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_0])
        var si_dmax = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1])
        var si_width = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_2])
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)
        var inv_tc_dr = Scalar[DTYPE](1.0) / (sr_tc * sr_dr)
        var b_vel_coef = Scalar[DTYPE](2.0) * sr_dr * dt / (si_dmax * sr_tc)

        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Phase 1: Read contact data and precompute Jacobians
        for c in range(nc):
            var c_off = contacts_off + c * CONTACT_SIZE
            var body = Int(rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_A]))
            var body_b = Int(rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_B]))
            var dist = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_DIST])

            (ws_c_dist + c)[] = dist
            (ws_c_body + c)[] = Scalar[DTYPE](body)
            (ws_c_body_b + c)[] = Scalar[DTYPE](body_b)

            if dist >= Scalar[DTYPE](0):
                continue

            (ws_c_px + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_POS_X])
            (ws_c_py + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_POS_Y])
            (ws_c_pz + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_POS_Z])
            (ws_c_nx + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NX])
            (ws_c_ny + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NY])
            (ws_c_nz + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NZ])

            compute_contact_jacobian_row_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
            ](
                env, state, model, cdof, body, body_b,
                (ws_c_px + c)[], (ws_c_py + c)[], (ws_c_pz + c)[],
                (ws_c_nx + c)[], (ws_c_ny + c)[], (ws_c_nz + c)[],
                J_row,
            )

            var k: Scalar[DTYPE] = 0
            var v_n: Scalar[DTYPE] = 0
            for i in range(NV):
                (ws_J_n + c * NV + i)[] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * qvel[i]
            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            (ws_K_n + c)[] = k

            # MuJoCo impedance model for RHS
            var penetration = -dist
            if penetration > Scalar[DTYPE](0.05):
                penetration = Scalar[DTYPE](0.05)
            var x = penetration / si_width
            if x > Scalar[DTYPE](1.0):
                x = Scalar[DTYPE](1.0)
            var imp = si_dmin + (Scalar[DTYPE](3.0) * x * x - Scalar[DTYPE](2.0) * x * x * x) * (si_dmax - si_dmin)
            if imp < Scalar[DTYPE](0.2):
                imp = Scalar[DTYPE](0.2)
            var bias = -imp * penetration * inv_tc_dr - b_vel_coef * v_n
            (ws_rhs + c)[] = v_n + bias

            # Warm start
            (ws_lambda_n + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_IMPULSE_N])

            if (ws_lambda_n + c)[] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                    qvel[i] += mi_j_sum * (ws_lambda_n + c)[]

        # Phase 2: Projected CG for normal constraints
        # Compute initial residual: r = -rhs - A*lambda
        for c in range(nc):
            if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                continue
            var ax: Scalar[DTYPE] = 0
            for c2 in range(nc):
                if (ws_c_dist + c2)[] >= Scalar[DTYPE](0):
                    continue
                var a_cc2: Scalar[DTYPE] = 0
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * (ws_J_n + c2 * NV + j_idx)[]
                    a_cc2 += (ws_J_n + c * NV + i)[] * mi_j_sum
                ax += a_cc2 * (ws_lambda_n + c2)[]
            (ws_r + c)[] = -(ws_rhs + c)[] - ax

        # Project residual
        for c in range(nc):
            if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                (ws_r + c)[] = Scalar[DTYPE](0)
                continue
            if (ws_lambda_n + c)[] <= Scalar[DTYPE](0) and (ws_r + c)[] < Scalar[DTYPE](0):
                (ws_r + c)[] = Scalar[DTYPE](0)

        for c in range(nc):
            (ws_p + c)[] = (ws_r + c)[]

        var rr: Scalar[DTYPE] = 0
        for c in range(nc):
            rr += (ws_r + c)[] * (ws_r + c)[]

        # CG iterations
        for _ in range(CG_ITERATIONS):
            if rr < Scalar[DTYPE](CG_TOLERANCE):
                break

            # Compute A*p
            for c in range(nc):
                (ws_Ap + c)[] = Scalar[DTYPE](0)
                if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                    continue
                for c2 in range(nc):
                    if (ws_c_dist + c2)[] >= Scalar[DTYPE](0):
                        continue
                    var a_cc2: Scalar[DTYPE] = 0
                    for i in range(NV):
                        var mi_j_sum: Scalar[DTYPE] = 0
                        for j_idx in range(NV):
                            mi_j_sum += M_inv[i * NV + j_idx] * (ws_J_n + c2 * NV + j_idx)[]
                        a_cc2 += (ws_J_n + c * NV + i)[] * mi_j_sum
                    (ws_Ap + c)[] = (ws_Ap + c)[] + a_cc2 * (ws_p + c2)[]

            var pAp: Scalar[DTYPE] = 0
            for c in range(nc):
                pAp += (ws_p + c)[] * (ws_Ap + c)[]
            if pAp < Scalar[DTYPE](1e-14):
                break

            var alpha = rr / pAp

            var projected = False
            for c in range(nc):
                if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                    continue
                (ws_lambda_n + c)[] = (ws_lambda_n + c)[] + alpha * (ws_p + c)[]
                if (ws_lambda_n + c)[] < Scalar[DTYPE](0):
                    (ws_lambda_n + c)[] = Scalar[DTYPE](0)
                    projected = True

            # Recompute residual after projection
            for c in range(nc):
                if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                    (ws_r + c)[] = Scalar[DTYPE](0)
                    continue
                var ax: Scalar[DTYPE] = 0
                for c2 in range(nc):
                    if (ws_c_dist + c2)[] >= Scalar[DTYPE](0):
                        continue
                    var a_cc2: Scalar[DTYPE] = 0
                    for i in range(NV):
                        var mi_j_sum: Scalar[DTYPE] = 0
                        for j_idx in range(NV):
                            mi_j_sum += M_inv[i * NV + j_idx] * (ws_J_n + c2 * NV + j_idx)[]
                        a_cc2 += (ws_J_n + c * NV + i)[] * mi_j_sum
                    ax += a_cc2 * (ws_lambda_n + c2)[]
                (ws_r + c)[] = -(ws_rhs + c)[] - ax

            for c in range(nc):
                if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                    continue
                if (ws_lambda_n + c)[] <= Scalar[DTYPE](0) and (ws_r + c)[] < Scalar[DTYPE](0):
                    (ws_r + c)[] = Scalar[DTYPE](0)

            var rr_new: Scalar[DTYPE] = 0
            for c in range(nc):
                rr_new += (ws_r + c)[] * (ws_r + c)[]

            if projected or rr < Scalar[DTYPE](1e-14):
                for c in range(nc):
                    (ws_p + c)[] = (ws_r + c)[]
            else:
                var beta = rr_new / rr
                for c in range(nc):
                    (ws_p + c)[] = (ws_r + c)[] + beta * (ws_p + c)[]

            rr = rr_new

        # Remove warm-start and apply final solved impulses
        for c in range(nc):
            if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                continue
            var c_off = contacts_off + c * CONTACT_SIZE
            var warm = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_IMPULSE_N])
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * (ws_J_n + c * NV + j_idx)[]
                    qvel[i] -= mi_j_sum * warm

        for c in range(nc):
            if (ws_c_dist + c)[] >= Scalar[DTYPE](0):
                continue
            if (ws_lambda_n + c)[] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * (ws_J_n + c * NV + j_idx)[]
                    qvel[i] += mi_j_sum * (ws_lambda_n + c)[]

        # Phase 2b: Joint limit constraints (PGS)
        if num_limits > 0:
            var lr_tc = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_0])
            var lr_dr = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_1])
            var li_dmin = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_0])
            var li_dmax = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_1])
            var li_width = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_2])
            if li_width < Scalar[DTYPE](1e-6):
                li_width = Scalar[DTYPE](1e-6)
            if li_dmax < Scalar[DTYPE](1e-4):
                li_dmax = Scalar[DTYPE](1e-4)
            var l_inv_tc_dr = Scalar[DTYPE](1.0) / (lr_tc * lr_dr)
            var l_b_vel_coef = Scalar[DTYPE](2.0) * lr_dr * dt / (li_dmax * lr_tc)

            for _ in range(CG_ITERATIONS):
                for l in range(num_limits):
                    var dof = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof]
                    var penetration = -limit_dist_arr[l]
                    if penetration < Scalar[DTYPE](0):
                        penetration = Scalar[DTYPE](0)
                    if penetration > Scalar[DTYPE](0.05):
                        penetration = Scalar[DTYPE](0.05)
                    var x_lim = penetration / li_width
                    if x_lim > Scalar[DTYPE](1.0):
                        x_lim = Scalar[DTYPE](1.0)
                    var imp_lim = li_dmin + (Scalar[DTYPE](3.0) * x_lim * x_lim - Scalar[DTYPE](2.0) * x_lim * x_lim * x_lim) * (li_dmax - li_dmin)
                    if imp_lim < Scalar[DTYPE](0.2):
                        imp_lim = Scalar[DTYPE](0.2)
                    var bias = -imp_lim * penetration * l_inv_tc_dr - l_b_vel_coef * v_limit
                    var delta_l = -(v_limit + bias) / (K_limit[l] / imp_lim)
                    var old_lam = lambda_limit[l]
                    lambda_limit[l] = lambda_limit[l] + delta_l
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual_l = lambda_limit[l] - old_lam
                    for i in range(NV):
                        qvel[i] += M_inv[i * NV + dof] * sign * actual_l

        # Phase 3: Friction via PGS
        _solve_friction_pgs_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, V_SIZE, M_SIZE, CDOF_SIZE, BATCH, WS_SIZE,
        ](
            env, state, model, workspace, cdof,
            ws_m_inv_offset(),
            ws_solver_offset[NV](),               # contact block at solver_ws + 0
            ws_solver_offset[NV]() + 15 * MC + MC * NV,  # friction block
            nc, friction_coef, contacts_off, qvel,
        )


# =============================================================================
# Shared friction solver (PGS) - CPU
# =============================================================================


fn _solve_friction_pgs_cpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
    J_n: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS * NV]()],
    lambda_n: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    contact_dist: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    contact_body_b: InlineArray[Int, _max_one[MAX_CONTACTS]()],
    nc: Int,
    mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Friction solver using PGS (shared by CG and Newton solvers)."""
    var friction_coef = model.friction
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime JT_SIZE = _max_one[MAX_CONTACTS * NV]()

    var J_t1_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var J_t2_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    var lambda_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var lambda_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    for i in range(MC):
        lambda_t1[i] = Scalar[DTYPE](0)
        lambda_t2[i] = Scalar[DTYPE](0)

    var J_t1_all = InlineArray[Scalar[DTYPE], JT_SIZE](uninitialized=True)
    var J_t2_all = InlineArray[Scalar[DTYPE], JT_SIZE](uninitialized=True)
    var K_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var K_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)

    for i in range(JT_SIZE):
        J_t1_all[i] = Scalar[DTYPE](0)
        J_t2_all[i] = Scalar[DTYPE](0)
    for i in range(MC):
        K_t1[i] = Scalar[DTYPE](1)
        K_t2[i] = Scalar[DTYPE](1)

    for c in range(nc):
        if lambda_n[c] <= Scalar[DTYPE](0):
            continue

        var contact = data.contacts[c]
        var nx = contact.normal_x
        var ny = contact.normal_y
        var nz = contact.normal_z

        # Compute tangent basis
        var t1_x: Scalar[DTYPE]
        var t1_y: Scalar[DTYPE]
        var t1_z: Scalar[DTYPE]

        if abs(nx) < Scalar[DTYPE](0.9):
            t1_x = Scalar[DTYPE](0)
            t1_y = -nz
            t1_z = ny
        else:
            t1_x = nz
            t1_y = Scalar[DTYPE](0)
            t1_z = -nx

        var t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
        if t1_mag > Scalar[DTYPE](1e-10):
            t1_x = t1_x / t1_mag
            t1_y = t1_y / t1_mag
            t1_z = t1_z / t1_mag

        var t2_x = ny * t1_z - nz * t1_y
        var t2_y = nz * t1_x - nx * t1_z
        var t2_z = nx * t1_y - ny * t1_x

        # Compute tangent Jacobian rows
        compute_contact_jacobian_row[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](
            model,
            data,
            cdof,
            contact.body_a,
            contact_body_b[c],
            contact.pos_x,
            contact.pos_y,
            contact.pos_z,
            t1_x,
            t1_y,
            t1_z,
            J_t1_row,
        )
        compute_contact_jacobian_row[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](
            model,
            data,
            cdof,
            contact.body_a,
            contact_body_b[c],
            contact.pos_x,
            contact.pos_y,
            contact.pos_z,
            t2_x,
            t2_y,
            t2_z,
            J_t2_row,
        )

        var k1: Scalar[DTYPE] = 0
        var k2: Scalar[DTYPE] = 0
        for i in range(NV):
            J_t1_all[c * NV + i] = J_t1_row[i]
            J_t2_all[c * NV + i] = J_t2_row[i]
            var mi_j_sum1: Scalar[DTYPE] = 0
            var mi_j_sum2: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum1 += M_inv[i * NV + j_idx] * J_t1_row[j_idx]
                mi_j_sum2 += M_inv[i * NV + j_idx] * J_t2_row[j_idx]
            k1 += J_t1_row[i] * mi_j_sum1
            k2 += J_t2_row[i] * mi_j_sum2

        if k1 < Scalar[DTYPE](1e-10):
            k1 = Scalar[DTYPE](1e-10)
        if k2 < Scalar[DTYPE](1e-10):
            k2 = Scalar[DTYPE](1e-10)
        K_t1[c] = k1
        K_t2[c] = k2

        # Warm start tangent impulses
        lambda_t1[c] = contact.impulse_t1
        lambda_t2[c] = contact.impulse_t2

    # Apply tangent warm start
    for c in range(nc):
        if lambda_n[c] <= Scalar[DTYPE](0):
            continue
        if lambda_t1[c] != Scalar[DTYPE](0) or lambda_t2[c] != Scalar[DTYPE](0):
            for i in range(NV):
                var mi_j_sum1: Scalar[DTYPE] = 0
                var mi_j_sum2: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum1 += (
                        M_inv[i * NV + j_idx] * J_t1_all[c * NV + j_idx]
                    )
                    mi_j_sum2 += (
                        M_inv[i * NV + j_idx] * J_t2_all[c * NV + j_idx]
                    )
                qvel[i] += mi_j_sum1 * lambda_t1[c] + mi_j_sum2 * lambda_t2[c]

    # Friction PGS iterations
    for _ in range(FRICTION_PGS_ITERATIONS):
        for c in range(nc):
            if lambda_n[c] <= Scalar[DTYPE](0):
                continue

            var max_friction = friction_coef * lambda_n[c]

            var v_t1: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t1 += J_t1_all[c * NV + i] * qvel[i]
            var delta_t1 = -v_t1 / K_t1[c]
            var old_t1 = lambda_t1[c]
            lambda_t1[c] = lambda_t1[c] + delta_t1

            var v_t2: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t2 += J_t2_all[c * NV + i] * qvel[i]
            var delta_t2 = -v_t2 / K_t2[c]
            var old_t2 = lambda_t2[c]
            lambda_t2[c] = lambda_t2[c] + delta_t2

            # Coulomb cone clamping
            var t_mag = sqrt(
                lambda_t1[c] * lambda_t1[c] + lambda_t2[c] * lambda_t2[c]
            )
            if t_mag > max_friction:
                var scale = max_friction / t_mag
                lambda_t1[c] = lambda_t1[c] * scale
                lambda_t2[c] = lambda_t2[c] * scale

            var actual_delta_t1 = lambda_t1[c] - old_t1
            var actual_delta_t2 = lambda_t2[c] - old_t2

            for i in range(NV):
                var mi_j_sum1: Scalar[DTYPE] = 0
                var mi_j_sum2: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum1 += (
                        M_inv[i * NV + j_idx] * J_t1_all[c * NV + j_idx]
                    )
                    mi_j_sum2 += (
                        M_inv[i * NV + j_idx] * J_t2_all[c * NV + j_idx]
                    )
                qvel[i] += (
                    mi_j_sum1 * actual_delta_t1 + mi_j_sum2 * actual_delta_t2
                )


# =============================================================================
# Shared friction solver (PGS) - GPU
# =============================================================================


@always_inline
fn _solve_friction_pgs_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    # Offsets into workspace (absolute from env row start)
    m_inv_off: Int,
    contact_ws_off: Int,   # Offset to common contact block (11*MC)
    friction_ws_off: Int,  # Offset to friction block (10*MC)
    nc: Int,
    friction_coef: Scalar[DTYPE],
    contacts_off: Int,
    mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Friction solver using PGS on GPU (shared by CG and Newton solvers).

    Derives all writable pointers from workspace.ptr (preserves mutable origin).
    Contact data is read-only, friction data is read-write.
    """

    comptime MC = _max_one[MAX_CONTACTS]()

    # Derive ALL pointers from workspace.ptr for mutable writes
    var base = workspace.ptr + env * WS_SIZE
    var M_inv = base + m_inv_off

    # Contact block (read-only)
    var ws_lambda_n = base + contact_ws_off + 0 * MC
    var ws_c_body   = base + contact_ws_off + 3 * MC
    var ws_c_body_b = base + contact_ws_off + 4 * MC
    var ws_c_px     = base + contact_ws_off + 5 * MC
    var ws_c_py     = base + contact_ws_off + 6 * MC
    var ws_c_pz     = base + contact_ws_off + 7 * MC
    var ws_c_nx     = base + contact_ws_off + 8 * MC
    var ws_c_ny     = base + contact_ws_off + 9 * MC
    var ws_c_nz     = base + contact_ws_off + 10 * MC

    # Friction block (read-write)
    var lt1  = base + friction_ws_off + 0 * MC
    var lt2  = base + friction_ws_off + 1 * MC
    var kt1  = base + friction_ws_off + 2 * MC
    var kt2  = base + friction_ws_off + 3 * MC
    var _t1x = base + friction_ws_off + 4 * MC
    var _t1y = base + friction_ws_off + 5 * MC
    var _t1z = base + friction_ws_off + 6 * MC
    var _t2x = base + friction_ws_off + 7 * MC
    var _t2y = base + friction_ws_off + 8 * MC
    var _t2z = base + friction_ws_off + 9 * MC

    # Initialize friction workspace
    for i in range(MC):
        (lt1 + i)[] = Scalar[DTYPE](0)
        (lt2 + i)[] = Scalar[DTYPE](0)
        (kt1 + i)[] = Scalar[DTYPE](1)
        (kt2 + i)[] = Scalar[DTYPE](1)
        (_t1x + i)[] = Scalar[DTYPE](0)
        (_t1y + i)[] = Scalar[DTYPE](0)
        (_t1z + i)[] = Scalar[DTYPE](0)
        (_t2x + i)[] = Scalar[DTYPE](0)
        (_t2y + i)[] = Scalar[DTYPE](0)
        (_t2z + i)[] = Scalar[DTYPE](0)

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    # Precompute tangent basis and K_t for active contacts
    for c in range(nc):
        if (ws_lambda_n + c)[] <= Scalar[DTYPE](0):
            continue

        var nx = (ws_c_nx + c)[]
        var ny = (ws_c_ny + c)[]
        var nz = (ws_c_nz + c)[]

        if abs(nx) < Scalar[DTYPE](0.9):
            (_t1x + c)[] = Scalar[DTYPE](0)
            (_t1y + c)[] = -nz
            (_t1z + c)[] = ny
        else:
            (_t1x + c)[] = nz
            (_t1y + c)[] = Scalar[DTYPE](0)
            (_t1z + c)[] = -nx

        var t1_mag = sqrt((_t1x + c)[] * (_t1x + c)[] + (_t1y + c)[] * (_t1y + c)[] + (_t1z + c)[] * (_t1z + c)[])
        if t1_mag > Scalar[DTYPE](1e-10):
            (_t1x + c)[] = (_t1x + c)[] / t1_mag
            (_t1y + c)[] = (_t1y + c)[] / t1_mag
            (_t1z + c)[] = (_t1z + c)[] / t1_mag

        (_t2x + c)[] = ny * (_t1z + c)[] - nz * (_t1y + c)[]
        (_t2y + c)[] = nz * (_t1x + c)[] - nx * (_t1z + c)[]
        (_t2z + c)[] = nx * (_t1y + c)[] - ny * (_t1x + c)[]

        # Compute K_t1
        compute_contact_jacobian_row_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
        ](
            env, state, model, cdof,
            Int((ws_c_body + c)[]), Int((ws_c_body_b + c)[]),
            (ws_c_px + c)[], (ws_c_py + c)[], (ws_c_pz + c)[],
            (_t1x + c)[], (_t1y + c)[], (_t1z + c)[],
            J_row,
        )
        var k1: Scalar[DTYPE] = 0
        for i in range(NV):
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
            k1 += J_row[i] * mi_j_sum
        if k1 < Scalar[DTYPE](1e-10):
            k1 = Scalar[DTYPE](1e-10)
        (kt1 + c)[] = k1

        # Compute K_t2
        compute_contact_jacobian_row_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
        ](
            env, state, model, cdof,
            Int((ws_c_body + c)[]), Int((ws_c_body_b + c)[]),
            (ws_c_px + c)[], (ws_c_py + c)[], (ws_c_pz + c)[],
            (_t2x + c)[], (_t2y + c)[], (_t2z + c)[],
            J_row,
        )
        var k2: Scalar[DTYPE] = 0
        for i in range(NV):
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
            k2 += J_row[i] * mi_j_sum
        if k2 < Scalar[DTYPE](1e-10):
            k2 = Scalar[DTYPE](1e-10)
        (kt2 + c)[] = k2

        # Warm start tangent impulses
        var c_off = contacts_off + c * CONTACT_SIZE
        (lt1 + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_IMPULSE_T1])
        (lt2 + c)[] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_IMPULSE_T2])

    # Friction PGS iterations
    var J_t_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    for _ in range(FRICTION_PGS_ITERATIONS):
        for c in range(nc):
            if (ws_lambda_n + c)[] <= Scalar[DTYPE](0):
                continue

            var max_friction = friction_coef * (ws_lambda_n + c)[]

            # Tangent 1
            compute_contact_jacobian_row_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
            ](
                env, state, model, cdof,
                Int((ws_c_body + c)[]), Int((ws_c_body_b + c)[]),
                (ws_c_px + c)[], (ws_c_py + c)[], (ws_c_pz + c)[],
                (_t1x + c)[], (_t1y + c)[], (_t1z + c)[],
                J_t_row,
            )
            var v_t1: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t1 += J_t_row[i] * qvel[i]

            var delta_t1 = -v_t1 / (kt1 + c)[]
            var old_t1 = (lt1 + c)[]
            (lt1 + c)[] = (lt1 + c)[] + delta_t1

            # Tangent 2
            compute_contact_jacobian_row_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
            ](
                env, state, model, cdof,
                Int((ws_c_body + c)[]), Int((ws_c_body_b + c)[]),
                (ws_c_px + c)[], (ws_c_py + c)[], (ws_c_pz + c)[],
                (_t2x + c)[], (_t2y + c)[], (_t2z + c)[],
                J_t_row,
            )
            var v_t2: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t2 += J_t_row[i] * qvel[i]

            var delta_t2 = -v_t2 / (kt2 + c)[]
            var old_t2 = (lt2 + c)[]
            (lt2 + c)[] = (lt2 + c)[] + delta_t2

            # Coulomb cone clamping
            var t_mag = sqrt(
                (lt1 + c)[] * (lt1 + c)[] + (lt2 + c)[] * (lt2 + c)[]
            )
            if t_mag > max_friction:
                var scale = max_friction / t_mag
                (lt1 + c)[] = (lt1 + c)[] * scale
                (lt2 + c)[] = (lt2 + c)[] * scale

            var actual_t1 = (lt1 + c)[] - old_t1
            var actual_t2 = (lt2 + c)[] - old_t2

            # Apply tangent 1 correction
            compute_contact_jacobian_row_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
            ](
                env, state, model, cdof,
                Int((ws_c_body + c)[]), Int((ws_c_body_b + c)[]),
                (ws_c_px + c)[], (ws_c_py + c)[], (ws_c_pz + c)[],
                (_t1x + c)[], (_t1y + c)[], (_t1z + c)[],
                J_row,
            )
            for i in range(NV):
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                qvel[i] += mi_j_sum * actual_t1

            # Apply tangent 2 correction
            compute_contact_jacobian_row_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
            ](
                env, state, model, cdof,
                Int((ws_c_body + c)[]), Int((ws_c_body_b + c)[]),
                (ws_c_px + c)[], (ws_c_py + c)[], (ws_c_pz + c)[],
                (_t2x + c)[], (_t2y + c)[], (_t2z + c)[],
                J_t_row,
            )
            for i in range(NV):
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_t_row[j_idx]
                qvel[i] += mi_j_sum * actual_t2

    # Store impulses back for warm-starting
    for c in range(nc):
        var c_off = contacts_off + c * CONTACT_SIZE
        state[env, c_off + CONTACT_IDX_IMPULSE_N] = (ws_lambda_n + c)[]
        state[env, c_off + CONTACT_IDX_IMPULSE_T1] = (lt1 + c)[]
        state[env, c_off + CONTACT_IDX_IMPULSE_T2] = (lt2 + c)[]
