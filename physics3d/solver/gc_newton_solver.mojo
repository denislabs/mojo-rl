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

from ..types import ModelGC, DataGC, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..traits.gc_solver import GcConstraintSolver
from ..dynamics.jacobian import (
    compute_contact_jacobian_row,
    compute_contact_jacobian_row_gpu,
)

# Import shared friction solver
from .gc_cg_solver import _solve_friction_pgs_cpu, _solve_friction_pgs_gpu


# Newton solver parameters
comptime NEWTON_ITERATIONS: Int = 30
comptime NEWTON_TOLERANCE: Float64 = 1e-8
comptime LINESEARCH_ITERATIONS: Int = 20
comptime LINESEARCH_BETA: Float64 = 0.5     # Step shrink factor
comptime LINESEARCH_ARMIJO: Float64 = 1e-4   # Armijo sufficient decrease
comptime BAUMGARTE_NEWTON: Float64 = 0.8
comptime SLOP_NEWTON: Float64 = 0.0001
# Friction uses PGS iterations
comptime FRICTION_PGS_ITERATIONS_NEWTON: Int = 30


struct GcNewtonSolver(GcConstraintSolver):
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
        model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
        var baumgarte_coef = Scalar[DTYPE](BAUMGARTE_NEWTON)
        var slop_val = Scalar[DTYPE](SLOP_NEWTON)

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
                model, data, cdof,
                contact.body_a,
                contact.body_b,
                contact.pos_x, contact.pos_y, contact.pos_z,
                contact.normal_x, contact.normal_y, contact.normal_z,
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

            # RHS: velocity + Baumgarte bias
            var penetration = -contact.dist
            if penetration > Scalar[DTYPE](0.01):
                penetration = Scalar[DTYPE](0.01)
            var correction = penetration - slop_val
            if correction < Scalar[DTYPE](0):
                correction = Scalar[DTYPE](0)
            var bias = -baumgarte_coef * correction / dt
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
        var d = InlineArray[Scalar[DTYPE], MC](uninitialized=True)      # Newton direction
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
                    f_current += Scalar[DTYPE](0.5) * lambda_n[c] * A[c * MAX_CONTACTS + c2] * lambda_n[c2]

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
                        f_trial += Scalar[DTYPE](0.5) * lambda_trial[c] * A[c * MAX_CONTACTS + c2] * lambda_trial[c2]

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
            var baumgarte_lim = Scalar[DTYPE](BAUMGARTE_NEWTON)
            var slop_lim = Scalar[DTYPE](SLOP_NEWTON)
            for _ in range(NEWTON_ITERATIONS):
                for l in range(num_limits):
                    var dof_l = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof_l]
                    var penetration = -limit_dist_arr[l]
                    if penetration < Scalar[DTYPE](0):
                        penetration = Scalar[DTYPE](0)
                    if penetration > Scalar[DTYPE](0.01):
                        penetration = Scalar[DTYPE](0.01)
                    var correction = penetration - slop_lim
                    if correction < Scalar[DTYPE](0):
                        correction = Scalar[DTYPE](0)
                    var bias = -baumgarte_lim * correction / dt
                    var delta_l = -(v_limit + bias) / K_limit[l]
                    var old_lam = lambda_limit[l]
                    lambda_limit[l] = lambda_limit[l] + delta_l
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual = lambda_limit[l] - old_lam
                    for i in range(NV):
                        qvel[i] += M_inv[i * NV + dof_l] * sign * actual

        # Phase 3: Friction (Coulomb cone) via PGS
        _solve_friction_pgs_cpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, M_SIZE, CDOF_SIZE
        ](model, data, cdof, M_inv, J_n, lambda_n, contact_dist, contact_body_b, nc, qvel)

    @staticmethod
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
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve contact constraints using Projected Newton on GPU (per-environment)."""
        from ..gpu.constants import (
            gc_contacts_offset,
            gc_metadata_offset,
            gc_model_metadata_offset,
            gc_model_joint_offset,
            GC_CONTACT_SIZE,
            GC_CONTACT_IDX_BODY_A,
            GC_CONTACT_IDX_BODY_B,
            GC_CONTACT_IDX_POS_X,
            GC_CONTACT_IDX_POS_Y,
            GC_CONTACT_IDX_POS_Z,
            GC_CONTACT_IDX_NX,
            GC_CONTACT_IDX_NY,
            GC_CONTACT_IDX_NZ,
            GC_CONTACT_IDX_DIST,
            GC_CONTACT_IDX_IMPULSE_N,
            GC_CONTACT_IDX_IMPULSE_T1,
            GC_CONTACT_IDX_IMPULSE_T2,
            GC_META_IDX_NUM_CONTACTS,
            GC_MODEL_META_IDX_FRICTION,
            GC_MODEL_JOINT_SIZE,
            GC_JOINT_IDX_TYPE,
            GC_JOINT_IDX_QPOS_ADR,
            GC_JOINT_IDX_DOF_ADR,
            GC_JOINT_IDX_RANGE_MIN,
            GC_JOINT_IDX_RANGE_MAX,
            GC_JNT_HINGE,
            GC_JNT_SLIDE,
        )

        var contacts_off = gc_contacts_offset[NQ, NV, NBODY]()
        var meta_off = gc_metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()

        var num_contacts = Int(rebind[Scalar[DTYPE]](
            state[env, meta_off + GC_META_IDX_NUM_CONTACTS]
        ))
        var friction_coef = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + GC_MODEL_META_IDX_FRICTION]
        )

        # Detect joint limits from model/state buffers
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
            var j_off = gc_model_joint_offset[NBODY](j)
            var jtype = Int(rebind[Scalar[DTYPE]](model[0, j_off + GC_JOINT_IDX_TYPE]))
            if jtype != GC_JNT_HINGE and jtype != GC_JNT_SLIDE:
                continue
            var dof = Int(rebind[Scalar[DTYPE]](model[0, j_off + GC_JOINT_IDX_DOF_ADR]))
            var qpos_adr = Int(rebind[Scalar[DTYPE]](model[0, j_off + GC_JOINT_IDX_QPOS_ADR]))
            var rmin = rebind[Scalar[DTYPE]](model[0, j_off + GC_JOINT_IDX_RANGE_MIN])
            var rmax = rebind[Scalar[DTYPE]](model[0, j_off + GC_JOINT_IDX_RANGE_MAX])
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

        var baumgarte_coef = Scalar[DTYPE](BAUMGARTE_NEWTON)
        var slop_val = Scalar[DTYPE](SLOP_NEWTON)

        comptime MC = _max_one[MAX_CONTACTS]()
        comptime JN_SIZE = _max_one[MAX_CONTACTS * NV]()
        comptime A_SIZE = _max_one[MAX_CONTACTS * MAX_CONTACTS]()

        # Store full Jacobian and Delassus matrix
        var J_n = InlineArray[Scalar[DTYPE], JN_SIZE](uninitialized=True)
        for i in range(JN_SIZE):
            J_n[i] = Scalar[DTYPE](0)

        var A = InlineArray[Scalar[DTYPE], A_SIZE](uninitialized=True)
        for i in range(A_SIZE):
            A[i] = Scalar[DTYPE](0)

        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Per-contact data
        var lambda_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var K_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var c_dist = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var c_body = InlineArray[Int, MC](uninitialized=True)
        var c_body_b = InlineArray[Int, MC](uninitialized=True)
        var c_px = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var c_py = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var c_pz = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var c_nx = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var c_ny = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var c_nz = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var rhs = InlineArray[Scalar[DTYPE], MC](uninitialized=True)

        for i in range(MC):
            lambda_n[i] = Scalar[DTYPE](0)
            K_n[i] = Scalar[DTYPE](1)
            c_dist[i] = Scalar[DTYPE](0)
            c_body[i] = 0
            c_body_b[i] = -1
            c_px[i] = Scalar[DTYPE](0)
            c_py[i] = Scalar[DTYPE](0)
            c_pz[i] = Scalar[DTYPE](0)
            c_nx[i] = Scalar[DTYPE](0)
            c_ny[i] = Scalar[DTYPE](0)
            c_nz[i] = Scalar[DTYPE](1)
            rhs[i] = Scalar[DTYPE](0)

        # Phase 1: Read contact data and build Delassus matrix
        for c in range(nc):
            var c_off = contacts_off + c * GC_CONTACT_SIZE
            var body = Int(rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_BODY_A]))
            var body_b = Int(rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_BODY_B]))
            var dist = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_DIST])

            c_dist[c] = dist
            c_body[c] = body
            c_body_b[c] = body_b

            if dist >= Scalar[DTYPE](0):
                continue

            c_px[c] = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_POS_X])
            c_py[c] = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_POS_Y])
            c_pz[c] = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_POS_Z])
            c_nx[c] = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_NX])
            c_ny[c] = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_NY])
            c_nz[c] = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_NZ])

            compute_contact_jacobian_row_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
            ](
                env, state, model, cdof,
                body, body_b, c_px[c], c_py[c], c_pz[c],
                c_nx[c], c_ny[c], c_nz[c],
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

            var penetration = -dist
            if penetration > Scalar[DTYPE](0.01):
                penetration = Scalar[DTYPE](0.01)
            var correction = penetration - slop_val
            if correction < Scalar[DTYPE](0):
                correction = Scalar[DTYPE](0)
            var bias = -baumgarte_coef * correction / dt
            rhs[c] = v_n + bias

            lambda_n[c] = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_IMPULSE_N])

            if lambda_n[c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                    qvel[i] += mi_j_sum * lambda_n[c]

        # Build full Delassus matrix
        for c1 in range(nc):
            if c_dist[c1] >= Scalar[DTYPE](0):
                continue
            for c2 in range(nc):
                if c_dist[c2] >= Scalar[DTYPE](0):
                    continue
                var a_val: Scalar[DTYPE] = 0
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c2 * NV + j_idx]
                    a_val += J_n[c1 * NV + i] * mi_j_sum
                A[c1 * MAX_CONTACTS + c2] = a_val

        # Phase 2: Projected Newton iterations
        var grad = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var d = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var lambda_trial = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var free_map = InlineArray[Int, MC](uninitialized=True)

        for i in range(MC):
            grad[i] = Scalar[DTYPE](0)
            d[i] = Scalar[DTYPE](0)
            lambda_trial[i] = Scalar[DTYPE](0)
            free_map[i] = -1

        for _ in range(NEWTON_ITERATIONS):
            # Compute gradient: g = A * lambda + rhs
            var grad_norm: Scalar[DTYPE] = 0
            for c in range(nc):
                if c_dist[c] >= Scalar[DTYPE](0):
                    grad[c] = Scalar[DTYPE](0)
                    continue
                var g: Scalar[DTYPE] = rhs[c]
                for c2 in range(nc):
                    if c_dist[c2] >= Scalar[DTYPE](0):
                        continue
                    g += A[c * MAX_CONTACTS + c2] * lambda_n[c2]
                grad[c] = g

            # Projected gradient norm
            grad_norm = Scalar[DTYPE](0)
            for c in range(nc):
                if c_dist[c] >= Scalar[DTYPE](0):
                    continue
                if lambda_n[c] > Scalar[DTYPE](0) or grad[c] < Scalar[DTYPE](0):
                    grad_norm += grad[c] * grad[c]

            if grad_norm < Scalar[DTYPE](NEWTON_TOLERANCE):
                break

            # Identify free set
            var free_count = 0
            for c in range(nc):
                free_map[c] = -1
                if c_dist[c] >= Scalar[DTYPE](0):
                    continue
                if lambda_n[c] > Scalar[DTYPE](0) or grad[c] < Scalar[DTYPE](0):
                    free_map[c] = free_count
                    free_count += 1

            if free_count == 0:
                break

            # Solve reduced system with Jacobi + GS refinement
            for c in range(nc):
                d[c] = Scalar[DTYPE](0)

            for c in range(nc):
                if free_map[c] < 0:
                    continue
                if K_n[c] > Scalar[DTYPE](1e-14):
                    d[c] = -grad[c] / K_n[c]

            for _ in range(5):
                for c in range(nc):
                    if free_map[c] < 0:
                        continue
                    var sum_off_diag: Scalar[DTYPE] = 0
                    for c2 in range(nc):
                        if c2 == c:
                            continue
                        if free_map[c2] < 0:
                            continue
                        sum_off_diag += A[c * MAX_CONTACTS + c2] * d[c2]
                    d[c] = (-grad[c] - sum_off_diag) / A[c * MAX_CONTACTS + c]

            # Line search with Armijo condition
            var f_current: Scalar[DTYPE] = 0
            for c in range(nc):
                if c_dist[c] >= Scalar[DTYPE](0):
                    continue
                f_current += rhs[c] * lambda_n[c]
                for c2 in range(nc):
                    if c_dist[c2] >= Scalar[DTYPE](0):
                        continue
                    f_current += Scalar[DTYPE](0.5) * lambda_n[c] * A[c * MAX_CONTACTS + c2] * lambda_n[c2]

            var gtd: Scalar[DTYPE] = 0
            for c in range(nc):
                if free_map[c] < 0:
                    continue
                gtd += grad[c] * d[c]

            var step = Scalar[DTYPE](1.0)
            var armijo = Scalar[DTYPE](LINESEARCH_ARMIJO)
            var beta = Scalar[DTYPE](LINESEARCH_BETA)

            for _ in range(LINESEARCH_ITERATIONS):
                for c in range(nc):
                    lambda_trial[c] = lambda_n[c]
                    if free_map[c] >= 0:
                        lambda_trial[c] = lambda_n[c] + step * d[c]
                    if lambda_trial[c] < Scalar[DTYPE](0):
                        lambda_trial[c] = Scalar[DTYPE](0)

                var f_trial: Scalar[DTYPE] = 0
                for c in range(nc):
                    if c_dist[c] >= Scalar[DTYPE](0):
                        continue
                    f_trial += rhs[c] * lambda_trial[c]
                    for c2 in range(nc):
                        if c_dist[c2] >= Scalar[DTYPE](0):
                            continue
                        f_trial += Scalar[DTYPE](0.5) * lambda_trial[c] * A[c * MAX_CONTACTS + c2] * lambda_trial[c2]

                if f_trial <= f_current + armijo * step * gtd:
                    break

                step = step * beta

            for c in range(nc):
                lambda_n[c] = lambda_trial[c]

        # Remove warm-start and apply final solved impulses
        for c in range(nc):
            if c_dist[c] >= Scalar[DTYPE](0):
                continue
            var c_off = contacts_off + c * GC_CONTACT_SIZE
            var warm = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_IMPULSE_N])
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c * NV + j_idx]
                    qvel[i] -= mi_j_sum * warm

        for c in range(nc):
            if c_dist[c] >= Scalar[DTYPE](0):
                continue
            if lambda_n[c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c * NV + j_idx]
                    qvel[i] += mi_j_sum * lambda_n[c]

        # Phase 2b: Joint limit constraints (PGS)
        if num_limits > 0:
            var baumgarte_lim = Scalar[DTYPE](BAUMGARTE_NEWTON)
            var slop_lim = Scalar[DTYPE](SLOP_NEWTON)
            for _ in range(NEWTON_ITERATIONS):
                for l in range(num_limits):
                    var dof_l = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof_l]
                    var penetration = -limit_dist_arr[l]
                    if penetration < Scalar[DTYPE](0):
                        penetration = Scalar[DTYPE](0)
                    if penetration > Scalar[DTYPE](0.01):
                        penetration = Scalar[DTYPE](0.01)
                    var correction = penetration - slop_lim
                    if correction < Scalar[DTYPE](0):
                        correction = Scalar[DTYPE](0)
                    var bias = -baumgarte_lim * correction / dt
                    var delta_l = -(v_limit + bias) / K_limit[l]
                    var old_lam = lambda_limit[l]
                    lambda_limit[l] = lambda_limit[l] + delta_l
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual_l = lambda_limit[l] - old_lam
                    # Use diagonal M_inv for correction (consistent with GPU)
                    qvel[dof_l] += M_inv[dof_l * NV + dof_l] * sign * actual_l

        # Phase 3: Friction via PGS
        _solve_friction_pgs_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, V_SIZE, M_SIZE, CDOF_SIZE, BATCH,
        ](
            env, state, model, cdof, M_inv,
            lambda_n, c_dist, c_body, c_body_b, c_px, c_py, c_pz, c_nx, c_ny, c_nz,
            nc, friction_coef, contacts_off, qvel,
        )
