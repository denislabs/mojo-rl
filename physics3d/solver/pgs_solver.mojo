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
from gpu import thread_idx, block_idx, block_dim
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
        """PGS solver workspace: 21 * MC + 3 * MC * NV floats.

        Layout (offsets relative to solver workspace start):
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
          [21*MC..21*MC+MC*NV)           J_n        Normal Jacobian (MC x NV)
          [21*MC+MC*NV..21*MC+2*MC*NV)   J_t1       Tangent 1 Jacobian (MC x NV)
          [21*MC+2*MC*NV..21*MC+3*MC*NV) J_t2       Tangent 2 Jacobian (MC x NV)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 21 * MC + 3 * MC * NV

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
        return 1

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
        """Solve contact constraints using PGS on GPU (per-environment).

        All MC-sized arrays live in workspace (device memory) to minimize
        register spilling. Only J_row/J_t_row (NV-sized, temporary) stay
        as InlineArrays.
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var thread_idx = Int(block_dim.y * block_idx.y + thread_idx.y)

        if thread_idx > 0:
            return

        # Workspace pointers
        comptime qvel_idx = ws_qvel_pred_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime solver_idx = ws_solver_offset[NV, NBODY]()

        comptime MC = _max_one[MAX_CONTACTS]()

        # Solver workspace layout: 21 * MC + 3 * MC * NV floats
        # Common contact block (11*MC)
        comptime ws_lambda_n = solver_idx + 0 * MC
        comptime ws_K_n = solver_idx + 1 * MC
        comptime ws_c_dist = solver_idx + 2 * MC
        comptime ws_c_body = solver_idx + 3 * MC  # body A as Float
        comptime ws_c_body_b = solver_idx + 4 * MC  # body B as Float
        comptime ws_c_px = solver_idx + 5 * MC
        comptime ws_c_py = solver_idx + 6 * MC
        comptime ws_c_pz = solver_idx + 7 * MC
        comptime ws_c_nx = solver_idx + 8 * MC
        comptime ws_c_ny = solver_idx + 9 * MC
        comptime ws_c_nz = solver_idx + 10 * MC
        # Friction block (10*MC)
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
        # Jacobian storage (3 * MC * NV) — precomputed once, read in PGS iters
        comptime ws_J_n = solver_idx + 21 * MC
        comptime ws_J_t1 = solver_idx + 21 * MC + MC * NV
        comptime ws_J_t2 = solver_idx + 21 * MC + 2 * MC * NV

        # Initialize workspace
        for i in range(MC):
            workspace[env, ws_lambda_n + i] = 0
            workspace[env, ws_K_n + i] = 1
            workspace[env, ws_c_dist + i] = 0
            workspace[env, ws_c_body + i] = 0
            workspace[env, ws_c_body_b + i] = -1
            workspace[env, ws_c_px + i] = 0
            workspace[env, ws_c_py + i] = 0
            workspace[env, ws_c_pz + i] = 0
            workspace[env, ws_c_nx + i] = 0
            workspace[env, ws_c_ny + i] = 0
            workspace[env, ws_c_nz + i] = 1
            workspace[env, ws_lambda_t1 + i] = 0
            workspace[env, ws_lambda_t2 + i] = 0
            workspace[env, ws_K_t1 + i] = 1
            workspace[env, ws_K_t2 + i] = 1
            workspace[env, ws_t1x + i] = 0
            workspace[env, ws_t1y + i] = 0
            workspace[env, ws_t1z + i] = 0
            workspace[env, ws_t2x + i] = 0
            workspace[env, ws_t2y + i] = 0
            workspace[env, ws_t2z + i] = 0

        var contacts_off = contacts_offset[NQ, NV, NBODY]()
        var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )
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
        var qpos_off = 0  # qpos starts at offset 0 in state buffer
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
            # Lower limit
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
            # Upper limit
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

        # Phase 1: Read contact data and precompute K_n
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

            workspace[env, ws_c_dist + c] = dist
            workspace[env, ws_c_body + c] = Scalar[DTYPE](body)
            workspace[env, ws_c_body_b + c] = Scalar[DTYPE](body_b)

            if dist >= Scalar[DTYPE](0):
                continue

            workspace[env, ws_c_px + c] = state[env, c_off + CONTACT_IDX_POS_X]
            workspace[env, ws_c_py + c] = state[env, c_off + CONTACT_IDX_POS_Y]
            workspace[env, ws_c_pz + c] = state[env, c_off + CONTACT_IDX_POS_Z]
            workspace[env, ws_c_nx + c] = state[env, c_off + CONTACT_IDX_NX]
            workspace[env, ws_c_ny + c] = state[env, c_off + CONTACT_IDX_NY]
            workspace[env, ws_c_nz + c] = state[env, c_off + CONTACT_IDX_NZ]

            # Compute normal Jacobian and effective mass
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

            # Store J_row in workspace and compute K = J @ M_inv @ J^T
            var k: workspace.element_type = 0
            for i in range(NV):
                workspace[env, ws_J_n + c * NV + i] = J_row[i]
                var mi_j_sum: workspace.element_type = 0
                for j_idx in range(NV):
                    mi_j_sum += (
                        workspace[env, M_inv_idx + i * NV + j_idx]
                        * J_row[j_idx]
                    )
                k += J_row[i] * mi_j_sum
            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            workspace[env, ws_K_n + c] = k

            # Warm start
            workspace[env, ws_lambda_n + c] = state[
                env, c_off + CONTACT_IDX_IMPULSE_N
            ]

            # Apply warm start: qvel += M_inv @ J^T * lambda
            if workspace[env, ws_lambda_n + c] > Scalar[DTYPE](0):
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, qvel_idx + i] += (
                        mi_j_sum * workspace[env, ws_lambda_n + c]
                    )

        # Phase 2: PGS normal iterations (reads J_n from workspace)
        for _ in range(PGS_ITERATIONS):
            for c in range(nc):
                if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                    continue

                # Contact-normal velocity: v_n = J_n . qvel
                var v_n: workspace.element_type = 0
                for i in range(NV):
                    v_n += (
                        workspace[env, ws_J_n + c * NV + i]
                        * workspace[env, qvel_idx + i]
                    )

                # MuJoCo impedance model
                var penetration = -workspace[env, ws_c_dist + c]
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

                var delta = -(v_n + bias) / (workspace[env, ws_K_n + c] / imp)
                var old_lambda = workspace[env, ws_lambda_n + c]
                workspace[env, ws_lambda_n + c] = (
                    workspace[env, ws_lambda_n + c] + delta
                )
                if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                    workspace[env, ws_lambda_n + c] = Scalar[DTYPE](0)

                var actual_delta = workspace[env, ws_lambda_n + c] - old_lambda
                # Apply velocity correction: qvel += M_inv @ J_n^T * delta
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * workspace[env, ws_J_n + c * NV + j_idx]
                        )
                    workspace[env, qvel_idx + i] += mi_j_sum * actual_delta

        # Phase 2b: Joint limit constraints (PGS)
        if num_limits > 0:
            # Read solref/solimp limit from model buffer
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

            for _ in range(PGS_ITERATIONS):
                for l in range(num_limits):
                    var dof = limit_dof[l]
                    var sign = limit_sign[l]
                    var v_limit = sign * workspace[env, qvel_idx + dof]
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
                    lambda_limit[l] = lambda_limit[l] + rebind[Scalar[DTYPE]](
                        delta_l
                    )
                    if lambda_limit[l] < Scalar[DTYPE](0):
                        lambda_limit[l] = Scalar[DTYPE](0)
                    var actual_l = lambda_limit[l] - old_lam
                    for i in range(NV):
                        workspace[env, qvel_idx + i] += (
                            workspace[env, M_inv_idx + i * NV + dof]
                            * sign
                            * actual_l
                        )

        # Phase 3: Friction
        # Precompute tangent basis and K_t for active contacts
        for c in range(nc):
            if workspace[env, ws_lambda_n + c] <= 0:
                continue

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

            # Compute K_t1 and store J_t1 in workspace
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
                k1 += J_row[i] * mi_j_sum
            if k1 < Scalar[DTYPE](1e-10):
                k1 = Scalar[DTYPE](1e-10)
            workspace[env, ws_K_t1 + c] = k1

            # Compute K_t2 and store J_t2 in workspace
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
                k2 += J_row[i] * mi_j_sum
            if k2 < Scalar[DTYPE](1e-10):
                k2 = Scalar[DTYPE](1e-10)
            workspace[env, ws_K_t2 + c] = k2

            # Warm start tangent impulses
            var c_off = contacts_off + c * CONTACT_SIZE
            workspace[env, ws_lambda_t1 + c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_T1]
            )
            workspace[env, ws_lambda_t2 + c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_IMPULSE_T2]
            )

        # Friction PGS iterations (reads J_t1/J_t2 from workspace)
        for _ in range(PGS_ITERATIONS):
            for c in range(nc):
                if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                    continue

                var max_friction = (
                    friction_coef * workspace[env, ws_lambda_n + c]
                )

                # Tangent 1 velocity from stored J_t1
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

                # Tangent 2 velocity from stored J_t2
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

                # Apply tangent corrections: qvel += M_inv @ (J_t1^T*dt1 + J_t2^T*dt2)
                for i in range(NV):
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += workspace[
                            env, M_inv_idx + i * NV + j_idx
                        ] * (
                            workspace[env, ws_J_t1 + c * NV + j_idx]
                            * actual_t1
                            + workspace[env, ws_J_t2 + c * NV + j_idx]
                            * actual_t2
                        )
                    workspace[env, qvel_idx + i] += mi_j_sum

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
