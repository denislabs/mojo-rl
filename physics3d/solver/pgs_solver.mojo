"""Projected Gauss-Seidel (PGS) constraint solver for Generalized Coordinates engine.

Implements MuJoCo-style constraint-based contact solving in joint space:
1. For each contact, compute the contact Jacobian row (J_n, J_t1, J_t2)
2. Compute effective constraint mass K = sum(J[i]^2 / M[i])
3. Iteratively solve for contact impulses using PGS
4. Apply impulses to modify predicted velocity: qvel += M^-1 * J^T * delta_lambda

Key features:
- Unilateral normal constraints (lambda_n >= 0)
- Coulomb friction cone clamping
- MuJoCo solref/solimp impedance model for position stabilization
- Warm-starting from previous timestep impulses

Reference: MuJoCo's constraint solver + existing Cartesian PGS in pgs_solver.mojo
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
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    model_joint_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_cdof_offset,
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
)
from ..joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
)

# PGS solver parameters
comptime PGS_ITERATIONS: Int = 30


struct PGSSolver(ConstraintSolver):
    """PGS constraint solver for Generalized Coordinates engine.

    Modifies the predicted (unconstrained) velocity in-place to satisfy
    contact constraints (non-penetration + Coulomb friction).
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """PGS solver workspace: 23 * MC + 6 * MC * NV floats.

        Layout (offsets relative to solver workspace start):
          Scalars (21 * MC):
          [0*MC..1*MC)                   lambda_n   Normal impulse accumulators
          [1*MC..2*MC)                   K_n        Effective mass (diagonal)
          [2*MC..3*MC)                   c_dist     Contact distance
          [3*MC..4*MC)                   c_body     Body A index (as Float)
          [4*MC..5*MC)                   c_body_b   Body B index (as Float)
          [5*MC..6*MC)                   c_px       Contact position X
          [6*MC..7*MC)                   c_py       Contact position Y
          [7*MC..8*MC)                   c_pz       Contact position Z
          [8*MC..9*MC)                   c_nx       Contact normal X
          [9*MC..10*MC)                  c_ny       Contact normal Y
          [10*MC..11*MC)                 c_nz       Contact normal Z
          [11*MC..12*MC)                 lambda_t1  Tangent 1 impulse
          [12*MC..13*MC)                 lambda_t2  Tangent 2 impulse
          [13*MC..14*MC)                 K_t1       Tangent 1 effective mass
          [14*MC..15*MC)                 K_t2       Tangent 2 effective mass
          [15*MC..16*MC)                 t1x        Tangent 1 direction X
          [16*MC..17*MC)                 t1y        Tangent 1 direction Y
          [17*MC..18*MC)                 t1z        Tangent 1 direction Z
          [18*MC..19*MC)                 t2x        Tangent 2 direction X
          [19*MC..20*MC)                 t2y        Tangent 2 direction Y
          [20*MC..21*MC)                 t2z        Tangent 2 direction Z
          Jacobians (3 * MC * NV):
          [21*MC..21*MC+MC*NV)           J_n        Normal Jacobian (MC x NV)
          [21*MC+MC*NV..21*MC+2*MC*NV)   J_t1       Tangent 1 Jacobian (MC x NV)
          [21*MC+2*MC*NV..21*MC+3*MC*NV) J_t2       Tangent 2 Jacobian (MC x NV)
          Precomputed M_inv @ J^T (3 * MC * NV):
          [21*MC+3*MC*NV..21*MC+4*MC*NV) MinvJn     M_inv @ J_n^T (MC x NV)
          [21*MC+4*MC*NV..21*MC+5*MC*NV) MinvJt1    M_inv @ J_t1^T (MC x NV)
          [21*MC+5*MC*NV..21*MC+6*MC*NV) MinvJt2    M_inv @ J_t2^T (MC x NV)
          Precomputed impedance (2 * MC):
          [21*MC+6*MC*NV..22*MC+6*MC*NV) pos_bias   imp*pen*inv_tc_dr per contact
          [22*MC+6*MC*NV..23*MC+6*MC*NV) inv_K_imp  imp/K_n per contact
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 23 * MC + 6 * MC * NV

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
        """Solve contact constraints using PGS on CPU.

        Algorithm:
        1. For each contact, compute J_n (normal Jacobian row) and K_n (effective mass)
        2. For PGS_ITERATIONS:
           a. For each contact: compute velocity error, solve for impulse correction
           b. Clamp normal impulse >= 0 (unilateral)
           c. Apply velocity correction: qvel += M^-1 * J^T * delta
        3. Friction pass: similar PGS for tangent directions with cone clamping
        """
        var num_contacts = data.num_contacts

        if num_contacts == 0:
            return

        # Cap to MAX_CONTACTS
        var nc = num_contacts
        if nc > MAX_CONTACTS:
            nc = MAX_CONTACTS

        # Per-contact data (stored in flat arrays, indexed by contact)
        # We use InlineArrays sized to MAX_CONTACTS
        comptime MC = _max_one[MAX_CONTACTS]()

        # Normal Jacobian rows - stored flat: J_n[c * NV + i]
        comptime JN_SIZE = _max_one[MAX_CONTACTS * NV]()
        var J_n = InlineArray[Scalar[DTYPE], JN_SIZE](uninitialized=True)
        for i in range(JN_SIZE):
            J_n[i] = Scalar[DTYPE](0)

        # Effective mass per contact (diagonal approx)
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

        # Phase 1: Precompute contact data
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        for c in range(nc):
            var contact = data.contacts[c]

            if contact.dist >= Scalar[DTYPE](0):
                # No penetration - skip
                K_n[c] = Scalar[DTYPE](1)  # Avoid div by zero
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

            # Store and compute effective mass: K = J @ M_inv @ J^T
            var k: Scalar[DTYPE] = 0
            for i in range(NV):
                J_n[c * NV + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                k += J_row[i] * mi_j_sum

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            K_n[c] = k

            # Warm start from stored impulses
            lambda_n[c] = contact.impulse_n

        # Apply warm start impulses: qvel += M_inv @ J^T * lambda
        for c in range(nc):
            if contact_dist[c] >= Scalar[DTYPE](0):
                continue
            if lambda_n[c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c * NV + j_idx]
                    qvel[i] += mi_j_sum * lambda_n[c]

        # Phase 2: PGS iterations for normal constraints
        # MuJoCo solref/solimp impedance model
        var sr_tc = model.solref_contact[0]
        var sr_dr = model.solref_contact[1]
        var si_dmin = model.solimp_contact[0]
        var si_dmax = model.solimp_contact[1]
        var si_width = model.solimp_contact[2]
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)

        # Precompute velocity-level correction coefficients from solref
        var inv_tc_dr = Scalar[DTYPE](1.0) / (sr_tc * sr_dr)
        var b_vel_coef = Scalar[DTYPE](2.0) * sr_dr * dt / (si_dmax * sr_tc)

        for _ in range(PGS_ITERATIONS):
            for c in range(nc):
                if contact_dist[c] >= Scalar[DTYPE](0):
                    continue

                # Compute current contact-normal velocity: v_n = J_n . qvel
                var v_n: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_n += J_n[c * NV + i] * qvel[i]

                # MuJoCo impedance model
                var penetration = -contact_dist[c]
                if penetration > Scalar[DTYPE](0.05):
                    penetration = Scalar[DTYPE](0.05)
                # Impedance (Hermite smoothstep)
                var x = penetration / si_width
                if x > Scalar[DTYPE](1.0):
                    x = Scalar[DTYPE](1.0)
                var imp = si_dmin + (
                    Scalar[DTYPE](3.0) * x * x - Scalar[DTYPE](2.0) * x * x * x
                ) * (si_dmax - si_dmin)
                if imp < Scalar[DTYPE](0.2):
                    imp = Scalar[DTYPE](0.2)
                # Velocity-level bias: position correction + velocity damping
                var bias = -imp * penetration * inv_tc_dr - b_vel_coef * v_n

                # PGS update with impedance-scaled effective mass
                var delta = -(v_n + bias) / (K_n[c] / imp)
                var old_lambda = lambda_n[c]
                lambda_n[c] = lambda_n[c] + delta

                # Unilateral clamp: lambda_n >= 0
                if lambda_n[c] < Scalar[DTYPE](0):
                    lambda_n[c] = Scalar[DTYPE](0)

                var actual_delta = lambda_n[c] - old_lambda

                # Apply velocity correction: qvel += M_inv @ J^T * delta
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[c * NV + j_idx]
                    qvel[i] += mi_j_sum * actual_delta

        # Phase 2b: Joint limit constraints (PGS)
        # Detect active joint limits and solve as unilateral constraints
        comptime MAX_LIMITS = _max_one[2 * NJOINT]()
        var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
        var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var limit_dist = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        var K_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
        var lambda_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](
            uninitialized=True
        )
        for i in range(MAX_LIMITS):
            limit_dof[i] = 0
            limit_sign[i] = Scalar[DTYPE](0)
            limit_dist[i] = Scalar[DTYPE](0)
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
            # Skip joints with default (unlimited) ranges
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            # Lower limit: q >= range_min → constraint dist = q - range_min
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](1)  # J[dof] = +1
                limit_dist[num_limits] = dist_lo
                K_limit[num_limits] = M_inv[dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1
            # Upper limit: q <= range_max → constraint dist = range_max - q
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                limit_dof[num_limits] = dof
                limit_sign[num_limits] = Scalar[DTYPE](-1)  # J[dof] = -1
                limit_dist[num_limits] = dist_hi
                K_limit[num_limits] = M_inv[dof * NV + dof]
                if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                    K_limit[num_limits] = Scalar[DTYPE](1e-10)
                num_limits += 1

        if num_limits > 0:
            # MuJoCo solref/solimp for joint limits
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

            for _ in range(PGS_ITERATIONS):
                for l in range(num_limits):
                    var dof = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * qvel[dof]
                    # MuJoCo impedance model for limits
                    var penetration = -limit_dist[l]
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
                    # PGS update
                    var delta = -(v_limit + bias) / (K_limit[l] / imp_lim)
                    var old_lambda = lambda_limit[l]
                    lambda_limit[l] = lambda_limit[l] + delta
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual = lambda_limit[l] - old_lambda
                    # Apply: qvel[i] += M_inv[i, dof] * sign * actual
                    for i in range(NV):
                        qvel[i] += M_inv[i * NV + dof] * sign * actual

        # Phase 3: Friction (Coulomb cone)
        var friction_coef = model.friction

        # Tangent Jacobian rows (computed on-the-fly per iteration)
        var J_t1_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var J_t2_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        var lambda_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var lambda_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for i in range(MC):
            lambda_t1[i] = Scalar[DTYPE](0)
            lambda_t2[i] = Scalar[DTYPE](0)

        # Pre-compute tangent directions and their Jacobians for active contacts
        comptime JT_SIZE = _max_one[MAX_CONTACTS * NV]()
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

            # Choose a vector not parallel to normal
            if abs(nx) < Scalar[DTYPE](0.9):
                # t1 = normalize(cross((1,0,0), normal))
                t1_x = Scalar[DTYPE](0)
                t1_y = -nz
                t1_z = ny
            else:
                # t1 = normalize(cross((0,1,0), normal))
                t1_x = nz
                t1_y = Scalar[DTYPE](0)
                t1_z = -nx

            var t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
            if t1_mag > Scalar[DTYPE](1e-10):
                t1_x = t1_x / t1_mag
                t1_y = t1_y / t1_mag
                t1_z = t1_z / t1_mag

            # t2 = normal x t1
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
                contact.body_b,
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
                contact.body_b,
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
            if lambda_t1[c] != Scalar[DTYPE](0) or lambda_t2[c] != Scalar[
                DTYPE
            ](0):
                for i in range(NV):
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * (
                            J_t1_all[c * NV + j_idx] * lambda_t1[c]
                            + J_t2_all[c * NV + j_idx] * lambda_t2[c]
                        )
                    qvel[i] += mi_j_sum

        # Friction PGS iterations
        for _ in range(PGS_ITERATIONS):
            for c in range(nc):
                if lambda_n[c] <= Scalar[DTYPE](0):
                    continue

                var max_friction = friction_coef * lambda_n[c]

                # Tangent 1
                var v_t1: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_t1 += J_t1_all[c * NV + i] * qvel[i]

                var delta_t1 = -v_t1 / K_t1[c]
                var old_t1 = lambda_t1[c]
                lambda_t1[c] = lambda_t1[c] + delta_t1

                # Tangent 2
                var v_t2: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_t2 += J_t2_all[c * NV + i] * qvel[i]

                var delta_t2 = -v_t2 / K_t2[c]
                var old_t2 = lambda_t2[c]
                lambda_t2[c] = lambda_t2[c] + delta_t2

                # Coulomb cone clamping: |lambda_t| <= mu * lambda_n
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
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * (
                            J_t1_all[c * NV + j_idx] * actual_delta_t1
                            + J_t2_all[c * NV + j_idx] * actual_delta_t2
                        )
                    qvel[i] += mi_j_sum

        # Store impulses back for warm-starting next step
        # Note: data is not mutable here, so warm-start storage happens
        # in the integrator which has mutable access to data.

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
        """Solve contact constraints using PGS on GPU with 2D threading.

        Uses thread_x for environment index, thread_y for contact index.
        Precompute phases (Phase 1, Phase 3) are parallelized across contacts.
        PGS iterations are sequential on thread_y==0 (Gauss-Seidel dependency).
        All threads must hit all barriers (no early returns).
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        # Workspace pointers
        comptime qvel_idx = ws_qvel_pred_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime solver_idx = ws_solver_offset[NV, NBODY]()

        comptime MC = _max_one[MAX_CONTACTS]()

        # Solver workspace layout: 23 * MC + 6 * MC * NV floats
        comptime ws_lambda_n = solver_idx + 0 * MC
        comptime ws_K_n = solver_idx + 1 * MC
        comptime ws_c_dist = solver_idx + 2 * MC
        comptime ws_c_body = solver_idx + 3 * MC
        comptime ws_c_body_b = solver_idx + 4 * MC
        comptime ws_c_px = solver_idx + 5 * MC
        comptime ws_c_py = solver_idx + 6 * MC
        comptime ws_c_pz = solver_idx + 7 * MC
        comptime ws_c_nx = solver_idx + 8 * MC
        comptime ws_c_ny = solver_idx + 9 * MC
        comptime ws_c_nz = solver_idx + 10 * MC
        comptime ws_lambda_t1 = solver_idx + 11 * MC
        comptime ws_lambda_t2 = solver_idx + 12 * MC
        comptime ws_K_t1 = solver_idx + 13 * MC
        comptime ws_K_t2 = solver_idx + 14 * MC
        comptime ws_t1x = solver_idx + 15 * MC
        comptime ws_t1y = solver_idx + 16 * MC
        comptime ws_t1z = solver_idx + 17 * MC
        comptime ws_t2x = solver_idx + 18 * MC
        comptime ws_t2y = solver_idx + 19 * MC
        comptime ws_t2z = solver_idx + 20 * MC
        comptime ws_J_n = solver_idx + 21 * MC
        comptime ws_J_t1 = solver_idx + 21 * MC + MC * NV
        comptime ws_J_t2 = solver_idx + 21 * MC + 2 * MC * NV
        comptime ws_MinvJn = solver_idx + 21 * MC + 3 * MC * NV
        comptime ws_MinvJt1 = solver_idx + 21 * MC + 4 * MC * NV
        comptime ws_MinvJt2 = solver_idx + 21 * MC + 5 * MC * NV
        comptime ws_pos_bias = solver_idx + 21 * MC + 6 * MC * NV
        comptime ws_inv_K_imp = solver_idx + 22 * MC + 6 * MC * NV

        # === PARALLEL: Initialize workspace (each thread handles one slot) ===
        if valid_env:
            workspace[env, ws_lambda_n + contact_tid] = 0
            workspace[env, ws_K_n + contact_tid] = 1
            workspace[env, ws_c_dist + contact_tid] = 0
            workspace[env, ws_c_body + contact_tid] = 0
            workspace[env, ws_c_body_b + contact_tid] = -1
            workspace[env, ws_c_px + contact_tid] = 0
            workspace[env, ws_c_py + contact_tid] = 0
            workspace[env, ws_c_pz + contact_tid] = 0
            workspace[env, ws_c_nx + contact_tid] = 0
            workspace[env, ws_c_ny + contact_tid] = 0
            workspace[env, ws_c_nz + contact_tid] = 1
            workspace[env, ws_lambda_t1 + contact_tid] = 0
            workspace[env, ws_lambda_t2 + contact_tid] = 0
            workspace[env, ws_K_t1 + contact_tid] = 1
            workspace[env, ws_K_t2 + contact_tid] = 1
            workspace[env, ws_t1x + contact_tid] = 0
            workspace[env, ws_t1y + contact_tid] = 0
            workspace[env, ws_t1z + contact_tid] = 0
            workspace[env, ws_t2x + contact_tid] = 0
            workspace[env, ws_t2y + contact_tid] = 0
            workspace[env, ws_t2z + contact_tid] = 0

        # All threads read metadata independently
        var contacts_off = contacts_offset[NQ, NV, NBODY]()
        var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()

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

            # Read solref/solimp for impedance precompute
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

            workspace[env, ws_c_dist + c] = dist
            workspace[env, ws_c_body + c] = Scalar[DTYPE](body)
            workspace[env, ws_c_body_b + c] = Scalar[DTYPE](body_b)

            if dist < Scalar[DTYPE](0):
                workspace[env, ws_c_px + c] = state[
                    env, c_off + CONTACT_IDX_POS_X
                ]
                workspace[env, ws_c_py + c] = state[
                    env, c_off + CONTACT_IDX_POS_Y
                ]
                workspace[env, ws_c_pz + c] = state[
                    env, c_off + CONTACT_IDX_POS_Z
                ]
                workspace[env, ws_c_nx + c] = state[env, c_off + CONTACT_IDX_NX]
                workspace[env, ws_c_ny + c] = state[env, c_off + CONTACT_IDX_NY]
                workspace[env, ws_c_nz + c] = state[env, c_off + CONTACT_IDX_NZ]

                # Compute normal Jacobian
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
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_nx + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_ny + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_nz + c]),
                    J_row,
                )

                # Store J_n, compute MinvJn and K_n
                var k: workspace.element_type = 0
                for i in range(NV):
                    workspace[env, ws_J_n + c * NV + i] = J_row[i]
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, ws_MinvJn + c * NV + i] = mi_j_sum
                    k += J_row[i] * mi_j_sum
                if k < Scalar[DTYPE](1e-10):
                    k = Scalar[DTYPE](1e-10)
                workspace[env, ws_K_n + c] = k

                # Precompute impedance coefficients
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
                workspace[env, ws_pos_bias + c] = imp * penetration * inv_tc_dr
                workspace[env, ws_inv_K_imp + c] = imp / k

                # Store warm start lambda (applied by thread 0 after barrier)
                workspace[env, ws_lambda_n + c] = state[
                    env, c_off + CONTACT_IDX_IMPULSE_N
                ]

        # All threads must hit this barrier
        barrier()

        # === SEQUENTIAL: Warm start + PGS normal + joint limits (thread 0) ===
        if valid_env and contact_tid == 0:
            # Apply warm start: qvel += MinvJn * lambda
            for c in range(nc):
                if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n + c] > Scalar[DTYPE](0):
                    for i in range(NV):
                        workspace[env, qvel_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i]
                            * workspace[env, ws_lambda_n + c]
                        )

            # Phase 2: PGS normal iterations
            var vel_factor = Scalar[DTYPE](1.0) - b_vel_coef
            for _ in range(PGS_ITERATIONS):
                var max_delta: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                        continue
                    var v_n: workspace.element_type = 0
                    for i in range(NV):
                        v_n += (
                            workspace[env, ws_J_n + c * NV + i]
                            * workspace[env, qvel_idx + i]
                        )
                    var delta = (
                        -(v_n * vel_factor - workspace[env, ws_pos_bias + c])
                        * workspace[env, ws_inv_K_imp + c]
                    )
                    var old_lambda = workspace[env, ws_lambda_n + c]
                    workspace[env, ws_lambda_n + c] = (
                        workspace[env, ws_lambda_n + c] + delta
                    )
                    if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                        workspace[env, ws_lambda_n + c] = Scalar[DTYPE](0)
                    var actual_delta = (
                        workspace[env, ws_lambda_n + c] - old_lambda
                    )
                    var abs_delta = abs(actual_delta)
                    if abs_delta > max_delta:
                        max_delta = abs_delta
                    for i in range(NV):
                        workspace[env, qvel_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i]
                            * actual_delta
                        )
                if max_delta < Scalar[DTYPE](1e-4):
                    break

            # Phase 2b: Joint limit constraints
            comptime MAX_LIMITS = _max_one[2 * NJOINT]()
            var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
            var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            var limit_dist_arr = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            var K_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](
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
            var qpos_off = 0
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
                var pos = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr])
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
                    lim_pos_bias[l] = imp_lim * penetration * l_inv_tc_dr
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

                for _ in range(PGS_ITERATIONS):
                    var max_lim_delta: Scalar[DTYPE] = 0
                    for l in range(num_limits):
                        var v_limit = (
                            limit_sign[l]
                            * workspace[env, qvel_idx + limit_dof[l]]
                        )
                        var delta_l = (
                            -(v_limit * l_vel_factor - lim_pos_bias[l])
                            * lim_inv_K_imp[l]
                        )
                        var old_lam = lambda_limit[l]
                        lambda_limit[l] = lambda_limit[l] + rebind[
                            Scalar[DTYPE]
                        ](delta_l)
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

        # All threads must hit this barrier
        barrier()

        # === PARALLEL PHASE 3: Each thread precomputes tangent for one contact ===
        if valid_env and contact_tid < nc:
            var c = contact_tid
            if workspace[env, ws_lambda_n + c] > 0:
                var nx = workspace[env, ws_c_nx + c]
                var ny = workspace[env, ws_c_ny + c]
                var nz = workspace[env, ws_c_nz + c]

                # Tangent basis
                if abs(nx) < 0.9:
                    workspace[env, ws_t1x + c] = 0
                    workspace[env, ws_t1y + c] = -nz
                    workspace[env, ws_t1z + c] = ny
                else:
                    workspace[env, ws_t1x + c] = nz
                    workspace[env, ws_t1y + c] = 0
                    workspace[env, ws_t1z + c] = -nx

                var t1_mag = sqrt(
                    workspace[env, ws_t1x + c] * workspace[env, ws_t1x + c]
                    + workspace[env, ws_t1y + c] * workspace[env, ws_t1y + c]
                    + workspace[env, ws_t1z + c] * workspace[env, ws_t1z + c]
                )
                if t1_mag > Scalar[DTYPE](1e-10):
                    workspace[env, ws_t1x + c] /= t1_mag
                    workspace[env, ws_t1y + c] /= t1_mag
                    workspace[env, ws_t1z + c] /= t1_mag

                workspace[env, ws_t2x + c] = (
                    ny * workspace[env, ws_t1z + c]
                    - nz * workspace[env, ws_t1y + c]
                )
                workspace[env, ws_t2y + c] = (
                    nz * workspace[env, ws_t1x + c]
                    - nx * workspace[env, ws_t1z + c]
                )
                workspace[env, ws_t2z + c] = (
                    nx * workspace[env, ws_t1y + c]
                    - ny * workspace[env, ws_t1x + c]
                )

                # Compute J_t1, K_t1, MinvJt1
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
                    Int(workspace[env, ws_c_body + c]),
                    Int(workspace[env, ws_c_body_b + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t1x + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t1y + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t1z + c]),
                    J_row,
                )
                var k1: workspace.element_type = 0
                for i in range(NV):
                    workspace[env, ws_J_t1 + c * NV + i] = J_row[i]
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, ws_MinvJt1 + c * NV + i] = mi_j_sum
                    k1 += J_row[i] * mi_j_sum
                if k1 < Scalar[DTYPE](1e-10):
                    k1 = Scalar[DTYPE](1e-10)
                workspace[env, ws_K_t1 + c] = k1

                # Compute J_t2, K_t2, MinvJt2
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
                    Int(workspace[env, ws_c_body + c]),
                    Int(workspace[env, ws_c_body_b + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t2x + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t2y + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t2z + c]),
                    J_row,
                )
                var k2: workspace.element_type = 0
                for i in range(NV):
                    workspace[env, ws_J_t2 + c * NV + i] = J_row[i]
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, ws_MinvJt2 + c * NV + i] = mi_j_sum
                    k2 += J_row[i] * mi_j_sum
                if k2 < Scalar[DTYPE](1e-10):
                    k2 = Scalar[DTYPE](1e-10)
                workspace[env, ws_K_t2 + c] = k2

                # Store warm start tangent impulses
                var c_off = contacts_off + c * CONTACT_SIZE
                workspace[env, ws_lambda_t1 + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_IMPULSE_T1]
                )
                workspace[env, ws_lambda_t2 + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_IMPULSE_T2]
                )

        # All threads must hit this barrier
        barrier()

        # === SEQUENTIAL: Friction PGS + impulse store (thread 0 only) ===
        if valid_env and contact_tid == 0:
            # Friction PGS iterations
            for _ in range(PGS_ITERATIONS):
                var max_fric_delta: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                        continue

                    var max_friction = (
                        friction_coef * workspace[env, ws_lambda_n + c]
                    )

                    var v_t1: workspace.element_type = 0
                    for i in range(NV):
                        v_t1 += (
                            workspace[env, ws_J_t1 + c * NV + i]
                            * workspace[env, qvel_idx + i]
                        )
                    var delta_t1 = -v_t1 / workspace[env, ws_K_t1 + c]
                    var old_t1 = workspace[env, ws_lambda_t1 + c]
                    workspace[env, ws_lambda_t1 + c] = (
                        workspace[env, ws_lambda_t1 + c] + delta_t1
                    )

                    var v_t2: workspace.element_type = 0
                    for i in range(NV):
                        v_t2 += (
                            workspace[env, ws_J_t2 + c * NV + i]
                            * workspace[env, qvel_idx + i]
                        )
                    var delta_t2 = -v_t2 / workspace[env, ws_K_t2 + c]
                    var old_t2 = workspace[env, ws_lambda_t2 + c]
                    workspace[env, ws_lambda_t2 + c] = (
                        workspace[env, ws_lambda_t2 + c] + delta_t2
                    )

                    # Coulomb cone clamping
                    var t_mag = sqrt(
                        workspace[env, ws_lambda_t1 + c]
                        * workspace[env, ws_lambda_t1 + c]
                        + workspace[env, ws_lambda_t2 + c]
                        * workspace[env, ws_lambda_t2 + c]
                    )
                    if t_mag > max_friction:
                        var scale = max_friction / t_mag
                        workspace[env, ws_lambda_t1 + c] = (
                            workspace[env, ws_lambda_t1 + c] * scale
                        )
                        workspace[env, ws_lambda_t2 + c] = (
                            workspace[env, ws_lambda_t2 + c] * scale
                        )

                    var actual_t1 = workspace[env, ws_lambda_t1 + c] - old_t1
                    var actual_t2 = workspace[env, ws_lambda_t2 + c] - old_t2

                    var abs_t1 = abs(actual_t1)
                    var abs_t2 = abs(actual_t2)
                    if abs_t1 > max_fric_delta:
                        max_fric_delta = abs_t1
                    if abs_t2 > max_fric_delta:
                        max_fric_delta = abs_t2

                    for i in range(NV):
                        workspace[env, qvel_idx + i] += (
                            workspace[env, ws_MinvJt1 + c * NV + i] * actual_t1
                            + workspace[env, ws_MinvJt2 + c * NV + i]
                            * actual_t2
                        )
                if max_fric_delta < Scalar[DTYPE](1e-4):
                    break

            # Store impulses back to state buffer for warm-starting
            for c in range(nc):
                var c_off = contacts_off + c * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_IMPULSE_N] = workspace[
                    env, ws_lambda_n + c
                ]
                state[env, c_off + CONTACT_IDX_IMPULSE_T1] = workspace[
                    env, ws_lambda_t1 + c
                ]
                state[env, c_off + CONTACT_IDX_IMPULSE_T2] = workspace[
                    env, ws_lambda_t2 + c
                ]
