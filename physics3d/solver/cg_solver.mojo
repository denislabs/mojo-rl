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

# CG solver parameters
comptime CG_ITERATIONS: Int = 30
comptime CG_TOLERANCE: Float64 = 1e-8
comptime BAUMGARTE_CG: Float64 = 0.8
comptime SLOP_CG: Float64 = 0.0001
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

        # Contact distances (for Baumgarte)
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
        var baumgarte_coef = Scalar[DTYPE](BAUMGARTE_CG)
        var slop_val = Scalar[DTYPE](SLOP_CG)

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

            # Compute RHS: velocity + Baumgarte bias
            var penetration = -contact.dist
            if penetration > Scalar[DTYPE](0.01):
                penetration = Scalar[DTYPE](0.01)
            var correction = penetration - slop_val
            if correction < Scalar[DTYPE](0):
                correction = Scalar[DTYPE](0)
            var bias = -baumgarte_coef * correction / dt

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
            for _ in range(CG_ITERATIONS):
                for l in range(num_limits):
                    var dof = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof]
                    var penetration = -limit_dist_arr[l]
                    if penetration < Scalar[DTYPE](0):
                        penetration = Scalar[DTYPE](0)
                    if penetration > Scalar[DTYPE](0.01):
                        penetration = Scalar[DTYPE](0.01)
                    var correction = penetration - slop_val
                    if correction < Scalar[DTYPE](0):
                        correction = Scalar[DTYPE](0)
                    var bias = -baumgarte_coef * correction / dt
                    var delta_l = -(v_limit + bias) / K_limit[l]
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
        """Solve contact constraints using Projected CG on GPU (per-environment).
        """

        var contacts_off = contacts_offset[NQ, NV, NBODY]()
        var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var num_contacts = Int(
            rebind[Scalar[DTYPE]](state[env, meta_off + META_IDX_NUM_CONTACTS])
        )
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

        var baumgarte_coef = Scalar[DTYPE](BAUMGARTE_CG)
        var slop_val = Scalar[DTYPE](SLOP_CG)

        comptime MC = _max_one[MAX_CONTACTS]()
        comptime JN_SIZE = _max_one[MAX_CONTACTS * NV]()

        # Store full Jacobian for A*x products
        var J_n = InlineArray[Scalar[DTYPE], JN_SIZE](uninitialized=True)
        for i in range(JN_SIZE):
            J_n[i] = Scalar[DTYPE](0)

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

        # Phase 1: Read contact data and precompute Jacobians
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

            c_dist[c] = dist
            c_body[c] = body
            c_body_b[c] = body_b

            if dist >= Scalar[DTYPE](0):
                continue

            c_px[c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_POS_X]
            )
            c_py[c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_POS_Y]
            )
            c_pz[c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_POS_Z]
            )
            c_nx[c] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NX])
            c_ny[c] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NY])
            c_nz[c] = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_NZ])

            # Compute normal Jacobian and store
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
                CDOF_SIZE,
                BATCH,
            ](
                env,
                state,
                model,
                cdof,
                body,
                body_b,
                c_px[c],
                c_py[c],
                c_pz[c],
                c_nx[c],
                c_ny[c],
                c_nz[c],
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
            var penetration = -dist
            if penetration > Scalar[DTYPE](0.01):
                penetration = Scalar[DTYPE](0.01)
            var correction = penetration - slop_val
            if correction < Scalar[DTYPE](0):
                correction = Scalar[DTYPE](0)
            var bias = -baumgarte_coef * correction / dt
            rhs[c] = v_n + bias

            # Warm start
            lambda_n[c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )

            if lambda_n[c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                    qvel[i] += mi_j_sum * lambda_n[c]

        # Phase 2: Projected CG for normal constraints
        var r = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var p = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var Ap = InlineArray[Scalar[DTYPE], MC](uninitialized=True)

        for i in range(MC):
            r[i] = Scalar[DTYPE](0)
            p[i] = Scalar[DTYPE](0)
            Ap[i] = Scalar[DTYPE](0)

        # Compute initial residual: r = -rhs - A*lambda
        for c in range(nc):
            if c_dist[c] >= Scalar[DTYPE](0):
                continue
            var ax: Scalar[DTYPE] = 0
            for c2 in range(nc):
                if c_dist[c2] >= Scalar[DTYPE](0):
                    continue
                var a_cc2: Scalar[DTYPE] = 0
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c2 * NV + j_idx]
                    a_cc2 += J_n[c * NV + i] * mi_j_sum
                ax += a_cc2 * lambda_n[c2]
            r[c] = -rhs[c] - ax

        # Project residual
        for c in range(nc):
            if c_dist[c] >= Scalar[DTYPE](0):
                r[c] = Scalar[DTYPE](0)
                continue
            if lambda_n[c] <= Scalar[DTYPE](0) and r[c] < Scalar[DTYPE](0):
                r[c] = Scalar[DTYPE](0)

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
                if c_dist[c] >= Scalar[DTYPE](0):
                    continue
                for c2 in range(nc):
                    if c_dist[c2] >= Scalar[DTYPE](0):
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

            var pAp: Scalar[DTYPE] = 0
            for c in range(nc):
                pAp += p[c] * Ap[c]
            if pAp < Scalar[DTYPE](1e-14):
                break

            var alpha = rr / pAp

            var projected = False
            for c in range(nc):
                if c_dist[c] >= Scalar[DTYPE](0):
                    continue
                lambda_n[c] = lambda_n[c] + alpha * p[c]
                if lambda_n[c] < Scalar[DTYPE](0):
                    lambda_n[c] = Scalar[DTYPE](0)
                    projected = True

            # Recompute residual after projection
            for c in range(nc):
                if c_dist[c] >= Scalar[DTYPE](0):
                    r[c] = Scalar[DTYPE](0)
                    continue
                var ax: Scalar[DTYPE] = 0
                for c2 in range(nc):
                    if c_dist[c2] >= Scalar[DTYPE](0):
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

            for c in range(nc):
                if c_dist[c] >= Scalar[DTYPE](0):
                    continue
                if lambda_n[c] <= Scalar[DTYPE](0) and r[c] < Scalar[DTYPE](0):
                    r[c] = Scalar[DTYPE](0)

            var rr_new: Scalar[DTYPE] = 0
            for c in range(nc):
                rr_new += r[c] * r[c]

            if projected or rr < Scalar[DTYPE](1e-14):
                for c in range(nc):
                    p[c] = r[c]
            else:
                var beta = rr_new / rr
                for c in range(nc):
                    p[c] = r[c] + beta * p[c]

            rr = rr_new

        # Remove warm-start and apply final solved impulses
        for c in range(nc):
            if c_dist[c] >= Scalar[DTYPE](0):
                continue
            var c_off = contacts_off + c * CONTACT_SIZE
            var warm = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_N]
            )
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
            var baumgarte_lim = Scalar[DTYPE](BAUMGARTE_CG)
            var slop_lim = Scalar[DTYPE](SLOP_CG)
            for _ in range(CG_ITERATIONS):
                for l in range(num_limits):
                    var dof = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof]
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
                    qvel[dof] += M_inv[dof * NV + dof] * sign * actual_l

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
            M_SIZE,
            CDOF_SIZE,
            BATCH,
        ](
            env,
            state,
            model,
            cdof,
            M_inv,
            lambda_n,
            c_dist,
            c_body,
            c_body_b,
            c_px,
            c_py,
            c_pz,
            c_nx,
            c_ny,
            c_nz,
            nc,
            friction_coef,
            contacts_off,
            qvel,
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
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
    lambda_n: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    c_dist: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    c_body: InlineArray[Int, _max_one[MAX_CONTACTS]()],
    c_body_b: InlineArray[Int, _max_one[MAX_CONTACTS]()],
    c_px: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    c_py: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    c_pz: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    c_nx: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    c_ny: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    c_nz: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    nc: Int,
    friction_coef: Scalar[DTYPE],
    contacts_off: Int,
    mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Friction solver using PGS on GPU (shared by CG and Newton solvers)."""
    from ..gpu.constants import (
        CONTACT_SIZE,
        CONTACT_IDX_IMPULSE_N,
        CONTACT_IDX_IMPULSE_T1,
        CONTACT_IDX_IMPULSE_T2,
    )

    comptime MC = _max_one[MAX_CONTACTS]()

    var lambda_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var lambda_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var K_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var K_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)

    var t1x = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var t1y = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var t1z = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var t2x = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var t2y = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var t2z = InlineArray[Scalar[DTYPE], MC](uninitialized=True)

    for i in range(MC):
        lambda_t1[i] = Scalar[DTYPE](0)
        lambda_t2[i] = Scalar[DTYPE](0)
        K_t1[i] = Scalar[DTYPE](1)
        K_t2[i] = Scalar[DTYPE](1)
        t1x[i] = Scalar[DTYPE](0)
        t1y[i] = Scalar[DTYPE](0)
        t1z[i] = Scalar[DTYPE](0)
        t2x[i] = Scalar[DTYPE](0)
        t2y[i] = Scalar[DTYPE](0)
        t2z[i] = Scalar[DTYPE](0)

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    # Precompute tangent basis and K_t for active contacts
    for c in range(nc):
        if lambda_n[c] <= Scalar[DTYPE](0):
            continue

        var nx = c_nx[c]
        var ny = c_ny[c]
        var nz = c_nz[c]

        if abs(nx) < Scalar[DTYPE](0.9):
            t1x[c] = Scalar[DTYPE](0)
            t1y[c] = -nz
            t1z[c] = ny
        else:
            t1x[c] = nz
            t1y[c] = Scalar[DTYPE](0)
            t1z[c] = -nx

        var t1_mag = sqrt(t1x[c] * t1x[c] + t1y[c] * t1y[c] + t1z[c] * t1z[c])
        if t1_mag > Scalar[DTYPE](1e-10):
            t1x[c] = t1x[c] / t1_mag
            t1y[c] = t1y[c] / t1_mag
            t1z[c] = t1z[c] / t1_mag

        t2x[c] = ny * t1z[c] - nz * t1y[c]
        t2y[c] = nz * t1x[c] - nx * t1z[c]
        t2z[c] = nx * t1y[c] - ny * t1x[c]

        # Compute K_t1
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
            CDOF_SIZE,
            BATCH,
        ](
            env,
            state,
            model,
            cdof,
            c_body[c],
            c_body_b[c],
            c_px[c],
            c_py[c],
            c_pz[c],
            t1x[c],
            t1y[c],
            t1z[c],
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
        K_t1[c] = k1

        # Compute K_t2
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
            CDOF_SIZE,
            BATCH,
        ](
            env,
            state,
            model,
            cdof,
            c_body[c],
            c_body_b[c],
            c_px[c],
            c_py[c],
            c_pz[c],
            t2x[c],
            t2y[c],
            t2z[c],
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
        K_t2[c] = k2

        # Warm start tangent impulses
        var c_off = contacts_off + c * CONTACT_SIZE
        lambda_t1[c] = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_IMPULSE_T1]
        )
        lambda_t2[c] = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_IMPULSE_T2]
        )

    # Friction PGS iterations
    var J_t_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    for _ in range(FRICTION_PGS_ITERATIONS):
        for c in range(nc):
            if lambda_n[c] <= Scalar[DTYPE](0):
                continue

            var max_friction = friction_coef * lambda_n[c]

            # Tangent 1
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
                CDOF_SIZE,
                BATCH,
            ](
                env,
                state,
                model,
                cdof,
                c_body[c],
                c_body_b[c],
                c_px[c],
                c_py[c],
                c_pz[c],
                t1x[c],
                t1y[c],
                t1z[c],
                J_t_row,
            )
            var v_t1: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t1 += J_t_row[i] * qvel[i]

            var delta_t1 = -v_t1 / K_t1[c]
            var old_t1 = lambda_t1[c]
            lambda_t1[c] = lambda_t1[c] + delta_t1

            # Tangent 2
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
                CDOF_SIZE,
                BATCH,
            ](
                env,
                state,
                model,
                cdof,
                c_body[c],
                c_body_b[c],
                c_px[c],
                c_py[c],
                c_pz[c],
                t2x[c],
                t2y[c],
                t2z[c],
                J_t_row,
            )
            var v_t2: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t2 += J_t_row[i] * qvel[i]

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

            var actual_t1 = lambda_t1[c] - old_t1
            var actual_t2 = lambda_t2[c] - old_t2

            # Apply tangent 1 correction
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
                CDOF_SIZE,
                BATCH,
            ](
                env,
                state,
                model,
                cdof,
                c_body[c],
                c_body_b[c],
                c_px[c],
                c_py[c],
                c_pz[c],
                t1x[c],
                t1y[c],
                t1z[c],
                J_row,
            )
            for i in range(NV):
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                qvel[i] += mi_j_sum * actual_t1

            # Apply tangent 2 correction
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
                CDOF_SIZE,
                BATCH,
            ](
                env,
                state,
                model,
                cdof,
                c_body[c],
                c_body_b[c],
                c_px[c],
                c_py[c],
                c_pz[c],
                t2x[c],
                t2y[c],
                t2z[c],
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
        state[env, c_off + CONTACT_IDX_IMPULSE_N] = lambda_n[c]
        state[env, c_off + CONTACT_IDX_IMPULSE_T1] = lambda_t1[c]
        state[env, c_off + CONTACT_IDX_IMPULSE_T2] = lambda_t2[c]
