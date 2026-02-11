"""Constraint builder for physics3d solvers.

Extracts all constraint setup (contact normals, friction, joint limits) into
a single builder function. Solvers become pure iterative algorithms consuming
pre-built ConstraintData.

This consolidates code previously duplicated across PGS, CG, and Newton solvers.
"""

from math import sqrt
from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..dynamics.jacobian import compute_contact_jacobian_row
from ..gpu.constants import MAX_POS_CORRECTION_VEL
from .constraint_data import (
    ConstraintRow,
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)


fn _compute_impedance[
    DTYPE: DType,
](
    penetration: Scalar[DTYPE],
    si_dmin: Scalar[DTYPE],
    si_dmax: Scalar[DTYPE],
    si_width: Scalar[DTYPE],
    inv_tc_dr: Scalar[DTYPE],
    b_vel_coef: Scalar[DTYPE],
    v_n: Scalar[DTYPE],
    K: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute MuJoCo impedance model bias and inv_K_imp.

    Returns (bias, inv_K_imp) where:
    - bias = -(pos_correction + vel_damping)
    - inv_K_imp = imp / K (for PGS update: delta = -(v + bias) * inv_K_imp)
    """
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
    var inv_K_imp = imp / K
    return (bias, inv_K_imp)


fn build_constraints[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
    qvel: InlineArray[Scalar[DTYPE], V_SIZE],
    dt: Scalar[DTYPE],
    mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
):
    """Build all constraints from contacts and joint limits.

    Populates constraints with:
    1. Contact normal rows (with Jacobian, K, impedance bias, warm-start)
    2. Contact friction rows (t1, t2 per active normal, with Coulomb coupling)
    3. Joint limit rows (for active HINGE/SLIDE limits)

    After this call, solvers can iterate over constraints.rows without
    knowing anything about contacts, Jacobians, or impedance.
    """
    var num_contacts = data.num_contacts
    var nc = num_contacts
    if nc > MAX_CONTACTS:
        nc = MAX_CONTACTS

    comptime MC = _max_one[MAX_CONTACTS]()

    # Read solref/solimp for contacts
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
    var friction_coef = model.friction

    var row_idx = 0
    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    # =========================================================================
    # Phase 1: Contact normal constraints
    # =========================================================================
    # Track which contacts are active (penetrating) for friction phase
    var contact_active = InlineArray[Int, MC](uninitialized=True)
    var contact_normal_row = InlineArray[Int, MC](uninitialized=True)
    for i in range(MC):
        contact_active[i] = 0
        contact_normal_row[i] = -1

    for c in range(nc):
        var contact = data.contacts[c]

        if contact.dist >= Scalar[DTYPE](0):
            continue

        if row_idx >= MAX_ROWS:
            break

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

        # Store J and compute K = J @ M_inv @ J^T, MinvJT
        var k: Scalar[DTYPE] = 0
        var v_n: Scalar[DTYPE] = 0
        for i in range(NV):
            constraints.J[row_idx * NV + i] = J_row[i]
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
            constraints.MinvJT[row_idx * NV + i] = mi_j_sum
            k += J_row[i] * mi_j_sum
            v_n += J_row[i] * qvel[i]

        if k < Scalar[DTYPE](1e-10):
            k = Scalar[DTYPE](1e-10)

        # Compute impedance
        var penetration = -contact.dist
        var imp_result = _compute_impedance[DTYPE](
            penetration, si_dmin, si_dmax, si_width, inv_tc_dr, b_vel_coef, v_n, k
        )

        # Fill constraint row
        constraints.rows[row_idx].K = k
        constraints.rows[row_idx].bias = imp_result[0]
        constraints.rows[row_idx].inv_K_imp = imp_result[1]
        constraints.rows[row_idx].lo = Scalar[DTYPE](0)
        constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
        constraints.rows[row_idx].lambda_val = contact.impulse_n
        constraints.rows[row_idx].constraint_type = CNSTR_NORMAL
        constraints.rows[row_idx].friction_parent = -1
        constraints.rows[row_idx].friction_coef = friction_coef
        constraints.rows[row_idx].source_contact_idx = c
        constraints.rows[row_idx].source_dof = -1
        constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)

        contact_active[c] = 1
        contact_normal_row[c] = row_idx
        row_idx += 1

    constraints.num_normals = row_idx

    # =========================================================================
    # Phase 2: Contact friction constraints (t1, t2 per active normal)
    # =========================================================================
    var friction_start = row_idx

    for c in range(nc):
        if contact_active[c] == 0:
            continue

        var contact = data.contacts[c]
        var normal_row = contact_normal_row[c]
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

        # Tangent 1
        if row_idx + 1 >= MAX_ROWS:
            break

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
            J_row,
        )

        var k1: Scalar[DTYPE] = 0
        for i in range(NV):
            constraints.J[row_idx * NV + i] = J_row[i]
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
            constraints.MinvJT[row_idx * NV + i] = mi_j_sum
            k1 += J_row[i] * mi_j_sum
        if k1 < Scalar[DTYPE](1e-10):
            k1 = Scalar[DTYPE](1e-10)

        constraints.rows[row_idx].K = k1
        constraints.rows[row_idx].bias = Scalar[DTYPE](0)
        constraints.rows[row_idx].inv_K_imp = Scalar[DTYPE](0)
        constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
        constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
        constraints.rows[row_idx].lambda_val = contact.impulse_t1
        constraints.rows[row_idx].constraint_type = CNSTR_FRICTION_T1
        constraints.rows[row_idx].friction_parent = normal_row
        constraints.rows[row_idx].friction_coef = friction_coef
        constraints.rows[row_idx].source_contact_idx = c
        constraints.rows[row_idx].source_dof = -1
        constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
        row_idx += 1

        # Tangent 2
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
            J_row,
        )

        var k2: Scalar[DTYPE] = 0
        for i in range(NV):
            constraints.J[row_idx * NV + i] = J_row[i]
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
            constraints.MinvJT[row_idx * NV + i] = mi_j_sum
            k2 += J_row[i] * mi_j_sum
        if k2 < Scalar[DTYPE](1e-10):
            k2 = Scalar[DTYPE](1e-10)

        constraints.rows[row_idx].K = k2
        constraints.rows[row_idx].bias = Scalar[DTYPE](0)
        constraints.rows[row_idx].inv_K_imp = Scalar[DTYPE](0)
        constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
        constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
        constraints.rows[row_idx].lambda_val = contact.impulse_t2
        constraints.rows[row_idx].constraint_type = CNSTR_FRICTION_T2
        constraints.rows[row_idx].friction_parent = normal_row
        constraints.rows[row_idx].friction_coef = friction_coef
        constraints.rows[row_idx].source_contact_idx = c
        constraints.rows[row_idx].source_dof = -1
        constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
        row_idx += 1

    constraints.num_friction = row_idx - friction_start

    # =========================================================================
    # Phase 3: Joint limit constraints
    # =========================================================================
    var limits_start = row_idx

    # Read solref/solimp for limits
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
    var l_b_vel_coef = Scalar[DTYPE](2.0) * lr_dr * dt / (li_dmax * lr_tc)

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

        # Lower limit: q >= range_min → constraint dist = q - range_min
        var dist_lo = pos - rmin
        if dist_lo < Scalar[DTYPE](0.01) and row_idx < MAX_ROWS:
            var sign = Scalar[DTYPE](1)  # J[dof] = +1
            var K_lim = M_inv[dof * NV + dof]
            if K_lim < Scalar[DTYPE](1e-10):
                K_lim = Scalar[DTYPE](1e-10)

            # Jacobian: J[dof] = sign, all others zero
            for i in range(NV):
                constraints.J[row_idx * NV + i] = Scalar[DTYPE](0)
                constraints.MinvJT[row_idx * NV + i] = M_inv[i * NV + dof] * sign
            constraints.J[row_idx * NV + dof] = sign

            # Impedance for limit
            var penetration = -dist_lo
            if penetration < Scalar[DTYPE](0):
                penetration = Scalar[DTYPE](0)
            var v_lim = sign * qvel[dof]
            var imp_result = _compute_impedance[DTYPE](
                penetration, li_dmin, li_dmax, li_width, l_inv_tc_dr, l_b_vel_coef, v_lim, K_lim
            )

            constraints.rows[row_idx].K = K_lim
            constraints.rows[row_idx].bias = imp_result[0]
            constraints.rows[row_idx].inv_K_imp = imp_result[1]
            constraints.rows[row_idx].lo = Scalar[DTYPE](0)
            constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
            constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
            constraints.rows[row_idx].constraint_type = CNSTR_LIMIT
            constraints.rows[row_idx].friction_parent = -1
            constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
            constraints.rows[row_idx].source_contact_idx = -1
            constraints.rows[row_idx].source_dof = dof
            constraints.rows[row_idx].limit_sign = sign
            row_idx += 1

        # Upper limit: q <= range_max → constraint dist = range_max - q
        var dist_hi = rmax - pos
        if dist_hi < Scalar[DTYPE](0.01) and row_idx < MAX_ROWS:
            var sign = Scalar[DTYPE](-1)  # J[dof] = -1
            var K_lim = M_inv[dof * NV + dof]
            if K_lim < Scalar[DTYPE](1e-10):
                K_lim = Scalar[DTYPE](1e-10)

            for i in range(NV):
                constraints.J[row_idx * NV + i] = Scalar[DTYPE](0)
                constraints.MinvJT[row_idx * NV + i] = M_inv[i * NV + dof] * sign
            constraints.J[row_idx * NV + dof] = sign

            var penetration = -dist_hi
            if penetration < Scalar[DTYPE](0):
                penetration = Scalar[DTYPE](0)
            var v_lim = sign * qvel[dof]
            var imp_result = _compute_impedance[DTYPE](
                penetration, li_dmin, li_dmax, li_width, l_inv_tc_dr, l_b_vel_coef, v_lim, K_lim
            )

            constraints.rows[row_idx].K = K_lim
            constraints.rows[row_idx].bias = imp_result[0]
            constraints.rows[row_idx].inv_K_imp = imp_result[1]
            constraints.rows[row_idx].lo = Scalar[DTYPE](0)
            constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
            constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
            constraints.rows[row_idx].constraint_type = CNSTR_LIMIT
            constraints.rows[row_idx].friction_parent = -1
            constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
            constraints.rows[row_idx].source_contact_idx = -1
            constraints.rows[row_idx].source_dof = dof
            constraints.rows[row_idx].limit_sign = sign
            row_idx += 1

    constraints.num_limits = row_idx - limits_start
    constraints.num_rows = row_idx


fn writeback_impulses[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Write solved impulses back to data.contacts for warm-starting.

    This replaces per-solver impulse writeback code. Loops over all
    constraint rows and writes impulse_n/t1/t2 based on constraint_type.
    """
    for r in range(constraints.num_rows):
        var row = constraints.rows[r]
        var c = row.source_contact_idx
        if c < 0:
            continue

        if row.constraint_type == CNSTR_NORMAL:
            data.contacts[c].impulse_n = row.lambda_val
        elif row.constraint_type == CNSTR_FRICTION_T1:
            data.contacts[c].impulse_t1 = row.lambda_val
        elif row.constraint_type == CNSTR_FRICTION_T2:
            data.contacts[c].impulse_t2 = row.lambda_val
