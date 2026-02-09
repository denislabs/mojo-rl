"""Newton constraint solver for Generalized Coordinates engine.

Implements MuJoCo-style projected Newton method for contact solving:
1. Form the Delassus matrix: A[c1,c2] = J[c1] * M^-1 * J[c2]^T
2. Minimize 0.5*lambda^T*A*lambda + b^T*lambda subject to lambda >= 0
3. Use Newton steps with active-set identification and line search

The Newton solver has quadratic convergence rate for the active set
and is the most accurate solver for stiff contact problems. It is
more expensive per iteration than PGS or CG but converges in fewer steps.

Friction is solved with PGS iterations (same as MuJoCo's approach).

Reference: MuJoCo Technical Notes, Section on Newton solver.
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

# Import shared friction solver
from .friction_solver import _solve_friction_pgs_cpu, _solve_friction_pgs_gpu

from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    model_joint_offset,
    ws_qvel_pred_offset,
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


# Newton solver parameters
comptime NEWTON_ITERATIONS: Int = 30
comptime NEWTON_TOLERANCE: Float64 = 1e-8
comptime LINESEARCH_ITERATIONS: Int = 20
comptime LINESEARCH_BETA: Float64 = 0.5  # Step shrink factor
comptime LINESEARCH_ARMIJO: Float64 = 1e-4  # Armijo sufficient decrease
# Friction uses PGS iterations
comptime FRICTION_PGS_ITERATIONS_NEWTON: Int = 30


struct NewtonSolver(ConstraintSolver):
    """Projected Newton constraint solver for GC engine.

    Solves the normal constraint QP using Newton's method with
    active-set identification and Armijo line search.

    Advantages over PGS/CG:
    - Quadratic convergence for well-conditioned problems
    - Most accurate for stiff contacts
    - Reliable convergence with line search

    Disadvantages:
    - Most expensive per iteration (solves linear system)
    - Requires forming and solving the reduced Hessian
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Newton solver workspace: 26*MC + MC*NV + MC*MC floats.

        Layout (offsets relative to solver workspace start):
          [0..11*MC)                              Common contact block
          [11*MC..12*MC)                          rhs
          [12*MC..12*MC+MC*NV)                    J_n (normal Jacobian)
          [12*MC+MC*NV..12*MC+MC*NV+MC*MC)        A (Delassus matrix)
          [12*MC+MC*NV+MC*MC..13*MC+MC*NV+MC*MC)  grad
          [13*MC+MC*NV+MC*MC..14*MC+MC*NV+MC*MC)  d (Newton direction)
          [14*MC+MC*NV+MC*MC..15*MC+MC*NV+MC*MC)  lambda_trial
          [15*MC+MC*NV+MC*MC..16*MC+MC*NV+MC*MC)  free_map (Float)
          [16*MC+MC*NV+MC*MC..26*MC+MC*NV+MC*MC)  Friction block (10 arrays)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 26 * MC + MC * NV + MC * MC

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
        """Solve contact constraints using Projected Newton on CPU."""
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
        # Delassus matrix A (nc x nc, stored flat)
        comptime A_SIZE = _max_one[MAX_CONTACTS * MAX_CONTACTS]()

        # Normal Jacobian rows: J_n[c * NV + i]
        var J_n = InlineArray[Scalar[DTYPE], JN_SIZE](uninitialized=True)
        for i in range(JN_SIZE):
            J_n[i] = Scalar[DTYPE](0)

        # Diagonal of Delassus matrix (for preconditioning/fallback)
        var K_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            K_n[i] = Scalar[DTYPE](0)

        # Normal impulse accumulators
        var lambda_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            lambda_n[i] = Scalar[DTYPE](0)

        # Contact distances
        var contact_dist = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            contact_dist[i] = Scalar[DTYPE](0)

        # Contact body indices
        var contact_body = InlineArray[Int, MC](uninitialized=True)
        var contact_body_b = InlineArray[Int, MC](uninitialized=True)
        for i in range(MC):
            contact_body[i] = 0
            contact_body_b[i] = -1

        # RHS vector
        var rhs = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            rhs[i] = Scalar[DTYPE](0)

        # Delassus matrix A[c1 * MAX_CONTACTS + c2]
        var A = InlineArray[Scalar[DTYPE], A_SIZE](uninitialized=True)
        for i in range(A_SIZE):
            A[i] = Scalar[DTYPE](0)

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

            # Warm start
            lambda_n[c] = contact.impulse_n

        # Build full Delassus matrix A[c1,c2] = J[c1] * M^-1 * J[c2]^T
        for c1 in range(nc):
            if contact_dist[c1] >= Scalar[DTYPE](0):
                continue
            for c2 in range(nc):
                if contact_dist[c2] >= Scalar[DTYPE](0):
                    continue
                var a_val: Scalar[DTYPE] = 0
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c2 * NV + j_idx]
                    a_val += J_n[c1 * NV + i] * mi_j_sum
                A[c1 * MAX_CONTACTS + c2] = a_val

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

        # Phase 2: Projected Newton for normal constraints
        # Minimize: f(x) = 0.5 * x^T * A * x + rhs^T * x subject to x >= 0
        var grad = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var d = InlineArray[Scalar[DTYPE], MC](
            uninitialized=True
        )  # Newton direction
        var lambda_trial = InlineArray[Scalar[DTYPE], MC](uninitialized=True)

        for i in range(MC):
            grad[i] = Scalar[DTYPE](0)
            d[i] = Scalar[DTYPE](0)
            lambda_trial[i] = Scalar[DTYPE](0)

        for _ in range(NEWTON_ITERATIONS):
            # Compute gradient: g = A * lambda + rhs
            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    grad[c] = Scalar[DTYPE](0)
                    continue
                var g: Scalar[DTYPE] = rhs[c]
                for c2 in range(nc):
                    if contact_dist[c2] >= Scalar[DTYPE](0):
                        continue
                    g += A[c * MAX_CONTACTS + c2] * lambda_n[c2]
                grad[c] = g

            # Compute projected gradient norm (convergence check)
            # For active constraints (lambda=0, grad>=0), gradient is zero
            var grad_norm: Scalar[DTYPE] = 0
            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    continue
                if lambda_n[c] > Scalar[DTYPE](0) or grad[c] < Scalar[DTYPE](0):
                    grad_norm += grad[c] * grad[c]

            if grad_norm < Scalar[DTYPE](NEWTON_TOLERANCE):
                break

            # Identify free set: constraints where lambda > 0 or gradient < 0
            # For free variables, solve A_FF * d_F = -g_F
            # For active variables (lambda=0, grad>=0), d = 0

            # Count free variables and map indices
            var free_count = 0
            var free_map = InlineArray[Int, MC](uninitialized=True)
            for i in range(MC):
                free_map[i] = -1

            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    continue
                if lambda_n[c] > Scalar[DTYPE](0) or grad[c] < Scalar[DTYPE](0):
                    free_map[c] = free_count
                    free_count += 1

            if free_count == 0:
                break

            # Solve the reduced Newton system: A_FF * d_F = -g_F
            # For small nc, use direct solve via Gauss elimination
            # Build reduced system in d array (reusing memory)
            for c in range(nc):
                d[c] = Scalar[DTYPE](0)

            # Use diagonal preconditioning (Jacobi) as a simple approach
            # For each free variable: d[c] = -grad[c] / A[c,c]
            # Then refine with a few Gauss-Seidel sweeps on the reduced system
            for c in range(nc):
                if free_map[c] < 0:
                    continue
                if K_n[c] > Scalar[DTYPE](1e-14):
                    d[c] = -grad[c] / K_n[c]

            # Gauss-Seidel refinement on the reduced system (5 sweeps)
            for _ in range(5):
                for c in range(nc):
                    if free_map[c] < 0:
                        continue
                    # Compute residual for this row: r_c = -g_c - sum_{c2!=c} A[c,c2] * d[c2]
                    var sum_off_diag: Scalar[DTYPE] = 0
                    for c2 in range(nc):
                        if c2 == c:
                            continue
                        if free_map[c2] < 0:
                            continue
                        sum_off_diag += A[c * MAX_CONTACTS + c2] * d[c2]
                    d[c] = (-grad[c] - sum_off_diag) / A[c * MAX_CONTACTS + c]

            # Line search with Armijo condition
            # f(x) = 0.5 * x^T * A * x + rhs^T * x
            # Compute f(lambda)
            var f_current: Scalar[DTYPE] = 0
            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    continue
                f_current += rhs[c] * lambda_n[c]
                for c2 in range(nc):
                    if contact_dist[c2] >= Scalar[DTYPE](0):
                        continue
                    f_current += (
                        Scalar[DTYPE](0.5)
                        * lambda_n[c]
                        * A[c * MAX_CONTACTS + c2]
                        * lambda_n[c2]
                    )

            # Directional derivative: g^T * d
            var gtd: Scalar[DTYPE] = 0
            for c in range(nc):
                if free_map[c] < 0:
                    continue
                gtd += grad[c] * d[c]

            # Line search
            var step = Scalar[DTYPE](1.0)
            var armijo = Scalar[DTYPE](LINESEARCH_ARMIJO)
            var beta = Scalar[DTYPE](LINESEARCH_BETA)

            for _ in range(LINESEARCH_ITERATIONS):
                # Trial point: lambda_trial = project(lambda + step * d)
                for c in range(nc):
                    lambda_trial[c] = lambda_n[c]
                    if free_map[c] >= 0:
                        lambda_trial[c] = lambda_n[c] + step * d[c]
                    if lambda_trial[c] < Scalar[DTYPE](0):
                        lambda_trial[c] = Scalar[DTYPE](0)

                # Compute f(lambda_trial)
                var f_trial: Scalar[DTYPE] = 0
                for c in range(nc):
                    if contact_dist[c] >= Scalar[DTYPE](0):
                        continue
                    f_trial += rhs[c] * lambda_trial[c]
                    for c2 in range(nc):
                        if contact_dist[c2] >= Scalar[DTYPE](0):
                            continue
                        f_trial += (
                            Scalar[DTYPE](0.5)
                            * lambda_trial[c]
                            * A[c * MAX_CONTACTS + c2]
                            * lambda_trial[c2]
                        )

                # Armijo condition: f(trial) <= f(current) + armijo * step * g^T * d
                if f_trial <= f_current + armijo * step * gtd:
                    break

                step = step * beta

            # Apply step
            for c in range(nc):
                lambda_n[c] = lambda_trial[c]

        # Apply solved impulses to velocity
        # Remove warm-start contribution
        for c in range(nc):
            if contact_dist[c] >= Scalar[DTYPE](0):
                continue
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

            for _ in range(NEWTON_ITERATIONS):
                for l in range(num_limits):
                    var dof_l = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof_l]
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
                        qvel[i] += M_inv[i * NV + dof_l] * sign * actual

        # Phase 3: Friction (Coulomb cone) via PGS
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
    fn solver_threads[NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,]() -> Int:
        return max(_max_one[MAX_CONTACTS](), NV, 2 * NJOINT)

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
        """Solve contact constraints using Projected Newton on GPU.

        All MC-sized arrays live in workspace (device memory).
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        comptime THREADS = Self.solver_threads[NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

        var thread_idx = Int(block_dim.y * block_idx.y + thread_idx.y)

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

        # Newton-specific
        comptime ws_rhs_idx = solver_ws_idx + 11 * MC
        comptime ws_J_n_idx = solver_ws_idx + 12 * MC
        comptime ws_A_idx = solver_ws_idx + 12 * MC + MC * NV
        comptime ws_grad_idx = solver_ws_idx + 12 * MC + MC * NV + MC * MC
        comptime ws_d_idx = solver_ws_idx + 13 * MC + MC * NV + MC * MC
        comptime ws_ltrial_idx = solver_ws_idx + 14 * MC + MC * NV + MC * MC
        comptime ws_fmap_idx = solver_ws_idx + 15 * MC + MC * NV + MC * MC
        comptime ws_t2x_idx = solver_ws_idx + 23 * MC + MC * NV + MC * MC
        comptime ws_t2y_idx = solver_ws_idx + 24 * MC + MC * NV + MC * MC
        comptime ws_t2z_idx = solver_ws_idx + 25 * MC + MC * NV + MC * MC

        # Initialize workspace
        if(thread_idx < MC):
            workspace[env, ws_lambda_n_idx + thread_idx] = 0
            workspace[env, ws_K_n_idx + thread_idx] = 1
            workspace[env, ws_c_dist_idx + thread_idx] = 0
            workspace[env, ws_c_body_idx + thread_idx] = 0
            workspace[env, ws_c_body_b_idx + thread_idx] = -1
            workspace[env, ws_c_px_idx + thread_idx] = 0
            workspace[env, ws_c_py_idx + thread_idx] = 0
            workspace[env, ws_c_pz_idx + thread_idx] = 0
            workspace[env, ws_c_nx_idx + thread_idx] = 0
            workspace[env, ws_c_ny_idx + thread_idx] = 0
            workspace[env, ws_c_nz_idx + thread_idx] = 1
            workspace[env, ws_rhs_idx + thread_idx] = 0
            workspace[env, ws_grad_idx + thread_idx] = 0
            workspace[env, ws_d_idx + thread_idx] = 0
            workspace[env, ws_ltrial_idx + thread_idx] = 0
            workspace[env, ws_fmap_idx + thread_idx] = -1

        if(thread_idx < MC * NV):
            workspace[env, ws_J_n_idx + thread_idx] = 0
        if(thread_idx < MC * MC):
            workspace[env, ws_A_idx + thread_idx] = 0

        barrier()

        if(thread_idx >= 1):
            return

        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](  # global vars are not supported in comptime
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )
        var num_contacts = Int(state[env, meta_off + META_IDX_NUM_CONTACTS])
        var friction_coef = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_FRICTION]
        )

        # Detect joint limits from model/state buffers
        comptime MAX_LIMITS = _max_one[2 * NJOINT]()
        var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
        var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var limit_dist_arr = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var K_limit = InlineArray[workspace.element_type, MAX_LIMITS](
            uninitialized=True
        )
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
            var jtype = Int(model[0, j_off + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(model[0, j_off + JOINT_IDX_DOF_ADR])
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
                K_limit[num_limits] = workspace[env, M_inv_idx + dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](-1)
                limit_dist_arr[num_limits] = dist_hi
                K_limit[num_limits] = workspace[env, M_inv_idx + dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1

        if num_contacts == 0 and num_limits == 0:
            return

        var nc = num_contacts
        if nc > MAX_CONTACTS:
            nc = MAX_CONTACTS

        # Read solref/solimp contact from model buffer
        var sr_tc = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0]
        )
        var sr_dr = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1]
        )
        var si_dmin = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_0]
        )
        var si_dmax = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1]
        )
        var si_width = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_2]
        )
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)
        var inv_tc_dr = Scalar[DTYPE](1.0) / (sr_tc * sr_dr)
        var b_vel_coef = Scalar[DTYPE](2.0) * sr_dr * dt / (si_dmax * sr_tc)

        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Phase 1: Read contact data and build Delassus matrix
        for c in range(nc):
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

            if dist >= Scalar[DTYPE](0):
                continue

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
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * workspace[env, qvel_idx + i]
            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            workspace[env, ws_K_n_idx + c] = k

            # MuJoCo impedance model for RHS
            var penetration = -dist
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
            workspace[env, ws_rhs_idx + c] = v_n + bias

            workspace[env, ws_lambda_n_idx + c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )

            if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, qvel_idx + i] +=  rebind[Scalar[DTYPE]](mi_j_sum) * rebind[
                        Scalar[DTYPE]
                    ](workspace[env, ws_lambda_n_idx + c])

        # Build full Delassus matrix
        for c1 in range(nc):
            if workspace[env, ws_c_dist_idx + c1] >= Scalar[DTYPE](0):
                continue
            for c2 in range(nc):
                if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                    continue
                var a_val: workspace.element_type = 0
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += 
                            workspace[env, M_inv_idx + i * NV + j_idx]
                         * workspace[env, ws_J_n_idx + c2 * NV + j_idx]
                    a_val += workspace[env, ws_J_n_idx + c1 * NV + i] * mi_j_sum
                workspace[env, ws_A_idx + c1 * MAX_CONTACTS + c2] = a_val

        # Phase 2: Projected Newton iterations
        for _ in range(NEWTON_ITERATIONS):
            # Compute gradient: g = A * lambda + rhs
            var grad_norm: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    workspace[env, ws_grad_idx + c] = Scalar[DTYPE](0)
                    continue
                var g: workspace.element_type = workspace[env, ws_rhs_idx + c]
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    g += (
                        workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                        * workspace[env, ws_lambda_n_idx + c2]
                    )
                workspace[env, ws_grad_idx + c] = g

            # Projected gradient norm
            grad_norm = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0) or (
                    workspace[env, ws_grad_idx + c]
                ) < Scalar[DTYPE](0):
                    grad_norm += (
                        workspace[env, ws_grad_idx + c]
                        * workspace[env, ws_grad_idx + c]
                    )

            if grad_norm < Scalar[DTYPE](NEWTON_TOLERANCE):
                break

            # Identify free set
            var free_count = 0
            for c in range(nc):
                workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](-1)
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0) or (
                    workspace[env, ws_grad_idx + c]
                ) < Scalar[DTYPE](0):
                    workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](free_count)
                    free_count += 1

            if free_count == 0:
                break

            # Solve reduced system with Jacobi + GS refinement
            for c in range(nc):
                workspace[env, ws_d_idx + c] = Scalar[DTYPE](0)

            for c in range(nc):
                if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_K_n_idx + c] > Scalar[DTYPE](1e-14):
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c])
                        / workspace[env, ws_K_n_idx + c]
                    )

            for _ in range(5):
                for c in range(nc):
                    if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                        continue
                    var sum_off_diag: workspace.element_type = 0
                    for c2 in range(nc):
                        if c2 == c:
                            continue
                        if workspace[env, ws_fmap_idx + c2] < Scalar[DTYPE](0):
                            continue
                        sum_off_diag += workspace[
                            env, ws_A_idx + c * MAX_CONTACTS + c2
                        ] * (workspace[env, ws_d_idx + c2])
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c]) - sum_off_diag
                    ) / (workspace[env, ws_A_idx + c * MAX_CONTACTS + c])

            # Line search with Armijo condition
            var f_current: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                f_current += (
                    workspace[env, ws_rhs_idx + c]
                    * workspace[env, ws_lambda_n_idx + c]
                )
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    f_current += (
                        Scalar[DTYPE](0.5)
                        * workspace[env, ws_lambda_n_idx + c]
                        * workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
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
                    if workspace[env, ws_ltrial_idx + c] < Scalar[DTYPE](0):
                        workspace[env, ws_ltrial_idx + c] = Scalar[DTYPE](0)

                var f_trial: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                        continue
                    f_trial += (
                        workspace[env, ws_rhs_idx + c]
                        * workspace[env, ws_ltrial_idx + c]
                    )
                    for c2 in range(nc):
                        if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](
                            0
                        ):
                            continue
                        f_trial += (
                            Scalar[DTYPE](0.5)
                            * workspace[env, ws_ltrial_idx + c]
                            * workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                            * workspace[env, ws_ltrial_idx + c2]
                        )

                if f_trial <= f_current + armijo * step * gtd:
                    break

                step = step * beta

            for c in range(nc):
                workspace[env, ws_lambda_n_idx + c] = workspace[
                    env, ws_ltrial_idx + c
                ]

        # Remove warm-start and apply final solved impulses
        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            var c_off = contacts_off + c * CONTACT_SIZE
            var warm = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (workspace[env, M_inv_idx + i * NV + j_idx]) * (
                            workspace[env, ws_J_n_idx + c * NV + j_idx]
                        )
                    workspace[env, qvel_idx + i] -=mi_j_sum * warm

        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (workspace[env, M_inv_idx + i * NV + j_idx]) * (
                            workspace[env, ws_J_n_idx + c * NV + j_idx]
                        )
                    workspace[env, qvel_idx + i] += mi_j_sum * workspace[env, ws_lambda_n_idx + c]

        # Phase 2b: Joint limit constraints (PGS)
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

            for _ in range(NEWTON_ITERATIONS):
                for l in range(num_limits):
                    var dof_l = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * workspace[env, qvel_idx + dof_l]
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
                    lambda_limit[l] +=  rebind[Scalar[DTYPE]](delta_l)
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual_l = lambda_limit[l] - old_lam
                    for i in range(NV):
                        workspace[env, qvel_idx + i] += workspace[env, M_inv_idx + i * NV + dof_l] * sign * actual_l

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
            FRICTION_WS_OFFSET= 16 * MC + MC * NV + MC * MC,
        ](
            env,
            state,
            model,
            workspace,
            nc,
            friction_coef,
            contacts_off,
        )

    @staticmethod
    @always_inline
    fn solve_gpu_old[
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
        dt: Scalar[DTYPE],
    ):
        """Solve contact constraints using Projected Newton on GPU.

        All MC-sized arrays live in workspace (device memory).
        """

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

        # Newton-specific
        comptime ws_rhs_idx = solver_ws_idx + 11 * MC
        comptime ws_J_n_idx = solver_ws_idx + 12 * MC
        comptime ws_A_idx = solver_ws_idx + 12 * MC + MC * NV
        comptime ws_grad_idx = solver_ws_idx + 12 * MC + MC * NV + MC * MC
        comptime ws_d_idx = solver_ws_idx + 13 * MC + MC * NV + MC * MC
        comptime ws_ltrial_idx = solver_ws_idx + 14 * MC + MC * NV + MC * MC
        comptime ws_fmap_idx = solver_ws_idx + 15 * MC + MC * NV + MC * MC
        comptime ws_t2x_idx = solver_ws_idx + 23 * MC + MC * NV + MC * MC
        comptime ws_t2y_idx = solver_ws_idx + 24 * MC + MC * NV + MC * MC
        comptime ws_t2z_idx = solver_ws_idx + 25 * MC + MC * NV + MC * MC

        # Initialize workspace
        for i in range(MC):
            workspace[env, ws_lambda_n_idx + i] = 0
            workspace[env, ws_K_n_idx + i] = 1
            workspace[env, ws_c_dist_idx + i] = 0
            workspace[env, ws_c_body_idx + i] = 0
            workspace[env, ws_c_body_b_idx + i] = -1
            workspace[env, ws_c_px_idx + i] = 0
            workspace[env, ws_c_py_idx + i] = 0
            workspace[env, ws_c_pz_idx + i] = 0
            workspace[env, ws_c_nx_idx + i] = 0
            workspace[env, ws_c_ny_idx + i] = 0
            workspace[env, ws_c_nz_idx + i] = 1
            workspace[env, ws_rhs_idx + i] = 0
            workspace[env, ws_grad_idx + i] = 0
            workspace[env, ws_d_idx + i] = 0
            workspace[env, ws_ltrial_idx + i] = 0
            workspace[env, ws_fmap_idx + i] = -1
        for i in range(MC * NV):
            workspace[env, ws_J_n_idx + i] = 0
        for i in range(MC * MC):
            workspace[env, ws_A_idx + i] = 0

        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var num_contacts = Int(state[env, meta_off + META_IDX_NUM_CONTACTS])
        var friction_coef = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_FRICTION]
        )

        # Detect joint limits from model/state buffers
        comptime MAX_LIMITS = _max_one[2 * NJOINT]()
        var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
        var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var limit_dist_arr = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var K_limit = InlineArray[workspace.element_type, MAX_LIMITS](
            uninitialized=True
        )
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
            var jtype = Int(model[0, j_off + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(model[0, j_off + JOINT_IDX_DOF_ADR])
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
                K_limit[num_limits] = workspace[env, M_inv_idx + dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](-1)
                limit_dist_arr[num_limits] = dist_hi
                K_limit[num_limits] = workspace[env, M_inv_idx + dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1

        if num_contacts == 0 and num_limits == 0:
            return

        var nc = num_contacts
        if nc > MAX_CONTACTS:
            nc = MAX_CONTACTS

        # Read solref/solimp contact from model buffer
        var sr_tc = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0]
        )
        var sr_dr = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1]
        )
        var si_dmin = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_0]
        )
        var si_dmax = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1]
        )
        var si_width = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_2]
        )
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)
        var inv_tc_dr = Scalar[DTYPE](1.0) / (sr_tc * sr_dr)
        var b_vel_coef = Scalar[DTYPE](2.0) * sr_dr * dt / (si_dmax * sr_tc)

        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Phase 1: Read contact data and build Delassus matrix
        for c in range(nc):
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

            if dist >= Scalar[DTYPE](0):
                continue

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
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * workspace[env, qvel_idx + i]
            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            workspace[env, ws_K_n_idx + c] = k

            # MuJoCo impedance model for RHS
            var penetration = -dist
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
            workspace[env, ws_rhs_idx + c] = v_n + bias

            workspace[env, ws_lambda_n_idx + c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )

            if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, qvel_idx + i] +=  rebind[Scalar[DTYPE]](mi_j_sum) * rebind[
                        Scalar[DTYPE]
                    ](workspace[env, ws_lambda_n_idx + c])

        # Build full Delassus matrix
        for c1 in range(nc):
            if workspace[env, ws_c_dist_idx + c1] >= Scalar[DTYPE](0):
                continue
            for c2 in range(nc):
                if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                    continue
                var a_val: workspace.element_type = 0
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += 
                            workspace[env, M_inv_idx + i * NV + j_idx]
                         * workspace[env, ws_J_n_idx + c2 * NV + j_idx]
                    a_val += workspace[env, ws_J_n_idx + c1 * NV + i] * mi_j_sum
                workspace[env, ws_A_idx + c1 * MAX_CONTACTS + c2] = a_val

        # Phase 2: Projected Newton iterations
        for _ in range(NEWTON_ITERATIONS):
            # Compute gradient: g = A * lambda + rhs
            var grad_norm: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    workspace[env, ws_grad_idx + c] = Scalar[DTYPE](0)
                    continue
                var g: workspace.element_type = workspace[env, ws_rhs_idx + c]
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    g += (
                        workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                        * workspace[env, ws_lambda_n_idx + c2]
                    )
                workspace[env, ws_grad_idx + c] = g

            # Projected gradient norm
            grad_norm = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0) or (
                    workspace[env, ws_grad_idx + c]
                ) < Scalar[DTYPE](0):
                    grad_norm += (
                        workspace[env, ws_grad_idx + c]
                        * workspace[env, ws_grad_idx + c]
                    )

            if grad_norm < Scalar[DTYPE](NEWTON_TOLERANCE):
                break

            # Identify free set
            var free_count = 0
            for c in range(nc):
                workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](-1)
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0) or (
                    workspace[env, ws_grad_idx + c]
                ) < Scalar[DTYPE](0):
                    workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](free_count)
                    free_count += 1

            if free_count == 0:
                break

            # Solve reduced system with Jacobi + GS refinement
            for c in range(nc):
                workspace[env, ws_d_idx + c] = Scalar[DTYPE](0)

            for c in range(nc):
                if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_K_n_idx + c] > Scalar[DTYPE](1e-14):
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c])
                        / workspace[env, ws_K_n_idx + c]
                    )

            for _ in range(5):
                for c in range(nc):
                    if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                        continue
                    var sum_off_diag: workspace.element_type = 0
                    for c2 in range(nc):
                        if c2 == c:
                            continue
                        if workspace[env, ws_fmap_idx + c2] < Scalar[DTYPE](0):
                            continue
                        sum_off_diag += workspace[
                            env, ws_A_idx + c * MAX_CONTACTS + c2
                        ] * (workspace[env, ws_d_idx + c2])
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c]) - sum_off_diag
                    ) / (workspace[env, ws_A_idx + c * MAX_CONTACTS + c])

            # Line search with Armijo condition
            var f_current: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                f_current += (
                    workspace[env, ws_rhs_idx + c]
                    * workspace[env, ws_lambda_n_idx + c]
                )
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    f_current += (
                        Scalar[DTYPE](0.5)
                        * workspace[env, ws_lambda_n_idx + c]
                        * workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
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
                    if workspace[env, ws_ltrial_idx + c] < Scalar[DTYPE](0):
                        workspace[env, ws_ltrial_idx + c] = Scalar[DTYPE](0)

                var f_trial: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                        continue
                    f_trial += (
                        workspace[env, ws_rhs_idx + c]
                        * workspace[env, ws_ltrial_idx + c]
                    )
                    for c2 in range(nc):
                        if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](
                            0
                        ):
                            continue
                        f_trial += (
                            Scalar[DTYPE](0.5)
                            * workspace[env, ws_ltrial_idx + c]
                            * workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                            * workspace[env, ws_ltrial_idx + c2]
                        )

                if f_trial <= f_current + armijo * step * gtd:
                    break

                step = step * beta

            for c in range(nc):
                workspace[env, ws_lambda_n_idx + c] = workspace[
                    env, ws_ltrial_idx + c
                ]

        # Remove warm-start and apply final solved impulses
        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            var c_off = contacts_off + c * CONTACT_SIZE
            var warm = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (workspace[env, M_inv_idx + i * NV + j_idx]) * (
                            workspace[env, ws_J_n_idx + c * NV + j_idx]
                        )
                    workspace[env, qvel_idx + i] -=mi_j_sum * warm

        for c in range(nc):
            if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                continue
            if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (workspace[env, M_inv_idx + i * NV + j_idx]) * (
                            workspace[env, ws_J_n_idx + c * NV + j_idx]
                        )
                    workspace[env, qvel_idx + i] += mi_j_sum * workspace[env, ws_lambda_n_idx + c]

        # Phase 2b: Joint limit constraints (PGS)
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

            for _ in range(NEWTON_ITERATIONS):
                for l in range(num_limits):
                    var dof_l = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * workspace[env, qvel_idx + dof_l]
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
                    lambda_limit[l] +=  rebind[Scalar[DTYPE]](delta_l)
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual_l = lambda_limit[l] - old_lam
                    for i in range(NV):
                        workspace[env, qvel_idx + i] += workspace[env, M_inv_idx + i * NV + dof_l] * sign * actual_l

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
            FRICTION_WS_OFFSET= 16 * MC + MC * NV + MC * MC,
        ](
            env,
            state,
            model,
            workspace,
            nc,
            friction_coef,
            contacts_off,
        )
